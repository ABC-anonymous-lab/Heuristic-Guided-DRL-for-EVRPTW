import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


# ---------------------------
# Utilities
# ---------------------------

def build_batch_edge_index_and_attr_from_dist(
    dist: torch.Tensor,            # [B, N, N]
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = device or dist.device
    B, N, _ = dist.shape
 
    dist_max = dist.view(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1)
    dist_norm = dist / (dist_max + 1e-8)                      # [B, N, N]
 
    mask = ~torch.eye(N, dtype=torch.bool, device=device)     # [N, N], no self-loops
    src_local, dst_local = mask.nonzero(as_tuple=True)
    E_per_graph = src_local.size(0)
 
    offsets = (torch.arange(B, device=device) * N).view(B, 1)         # [B, 1]
    src = (src_local.view(1, -1) + offsets).reshape(-1)               # [B*E_per_graph]
    dst = (dst_local.view(1, -1) + offsets).reshape(-1)               # [B*E_per_graph]
 
    edge_index = torch.stack([src, dst], dim=0)                       # [2, B*E_per_graph]
 
    batch_arange = torch.arange(B, device=device).view(B, 1).expand(B, E_per_graph).reshape(-1)
    edge_attr = dist_norm[batch_arange, src_local.repeat(B), dst_local.repeat(B)].unsqueeze(-1) 
    batch_index = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1)  # [B*N]
 
    return edge_index, edge_attr, batch_index


def build_fully_connected_edge_index_and_attr_from_dist(
    dist: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = device or dist.device
    N = dist.size(0)

    dist_norm = dist / (dist.max() + 1e-8)

    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    src, dst = mask.nonzero(as_tuple=True)

    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = dist_norm[src, dst].unsqueeze(-1)

    return edge_index, edge_attr


# ----------------
# Encoder
# ----------------
class PaperGATEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_dim: int = 1,
        d_model: int = 128,
        d_edge: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_edge = d_edge
        self.n_layers = n_layers
        self.dropout = dropout

        self.node_proj = nn.Linear(node_in_dim, d_model)
        self.node_bn = nn.BatchNorm1d(d_model)

        self.edge_proj = nn.Linear(edge_dim, d_edge)
        self.edge_bn = nn.BatchNorm1d(d_edge)

        self.convs = nn.ModuleList()
        self.layer_bns = nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=d_model,
                    out_channels=d_model,
                    heads=n_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=d_edge,
                )
            )
            self.layer_bns.append(nn.BatchNorm1d(d_model))


    def forward_batch(
        self,
        node_feat: torch.Tensor,       # [B*N, node_in_dim]
        edge_index: torch.Tensor,      # [2, E]
        edge_attr: torch.Tensor,       # [E, edge_dim]
        batch_index: torch.Tensor,     # [B*N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        h = self.node_proj(node_feat)
        h = self.node_bn(h)
    
        e = self.edge_proj(edge_attr)
        e = self.edge_bn(e)
    
        for conv, bn in zip(self.convs, self.layer_bns):
            h_res = h
    
            h = conv(h, edge_index, edge_attr=e)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res
            h = bn(h)
    
        h_graph = global_mean_pool(h, batch_index)   # [B, d_model]
    
        return h, h_graph
 

    def forward(
        self,
        node_feat: torch.Tensor,      # [N, node_in_dim]
        edge_index: torch.Tensor,     # [2, E]
        edge_attr: torch.Tensor,      # [E, edge_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Input node embedding
        h = self.node_proj(node_feat)
        h = self.node_bn(h)

        # Input edge embedding
        e = self.edge_proj(edge_attr)
        e = self.edge_bn(e)

        # GAT layers
        for conv, bn in zip(self.convs, self.layer_bns):
            h_res = h

            h = conv(
                h,
                edge_index,
                edge_attr=e,
            )

            h = F.relu(h)

            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )

            # residual connection
            h = h + h_res
            h = bn(h)

        # Eq. (10): graph embedding by average pooling
        h_graph = h.mean(dim=0, keepdim=True)

        return h, h_graph


# ---------------------------
# Decoder
# ---------------------------
class PaperDualLayerAttentionDecoder(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        tanh_clip: float = 10.0,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_v = d_model // n_heads
        self.tanh_clip = tanh_clip

        # Eq. (11): W_x maps [x_last || q_t || b_t] to d_model
        self.W_x = nn.Linear(d_model + 1, d_model)

        # Eq. (12): upper-layer projections
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Eq. (14): output projection after concatenating H heads
        self.W_F = nn.Linear(d_model, d_model, bias=False)
        self.W_K_lower = nn.Linear(d_model, d_model, bias=False)


    def build_context(
        self,
        h_graph: torch.Tensor,
        h_last: torch.Tensor,
        b_t: torch.Tensor,
    ) -> torch.Tensor:
        if b_t.dim() == 1:
            b_t = b_t.unsqueeze(-1)

        B = h_last.size(0)

        if h_graph.size(0) == 1 and B > 1:
            h_graph = h_graph.expand(B, -1)

        global_part = torch.cat([h_last, b_t], dim=-1)
        c1 = h_graph + self.W_x(global_part)

        return c1


    def forward(
        self,
        h: torch.Tensor,             # [N, d_model], node embeddings
        h_graph: torch.Tensor,       # [1, d_model]
        last_node_idx: torch.Tensor, # [B]
        b_t: torch.Tensor,           # [B]
        mask: torch.Tensor,          # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = mask.size(0)
        N = h.size(0)

        h_last = h[last_node_idx]    # [B, d_model]
        c1 = self.build_context(
            h_graph=h_graph,
            h_last=h_last,
            b_t=b_t,
        )

        # ---------------------------
        # Upper layer: multi-head attention
        # ---------------------------

        # Eq. (12)
        q = self.W_Q(c1)             # [B, d_model]
        k = self.W_K(h)              # [N, d_model]
        v = self.W_V(h)              # [N, d_model]

        # Split into H heads
        q = q.view(B, self.n_heads, self.d_v)                 # [B, H, d_v]
        k = k.view(N, self.n_heads, self.d_v).transpose(0, 1) # [H, N, d_v]
        v = v.view(N, self.n_heads, self.d_v).transpose(0, 1) # [H, N, d_v]

        # Eq. (13): u_i,t^1 = q^T k_i / sqrt(d_v)
        scores = torch.einsum("bhd,hnd->bhn", q, k)
        scores = scores / math.sqrt(self.d_v)                 # [B, H, N]

        scores = scores.masked_fill(
            ~mask.unsqueeze(1),
            torch.finfo(scores.dtype).min,
        )

        attn = torch.softmax(scores, dim=-1)                  # [B, H, N]

        # Eq. (14): concatenate H head outputs
        head_out = torch.einsum("bhn,hnd->bhd", attn, v)       # [B, H, d_v]
        head_out = head_out.reshape(B, self.d_model)          # [B, d_model]

        c2 = self.W_F(head_out)                               # [B, d_model]

        # ---------------------------
        # Lower layer: single-head attention
        # ---------------------------

        k_lower = self.W_K_lower(h)                           # [N, d_model]

        # Eq. (15)
        logits = torch.matmul(c2, k_lower.t())
        logits = logits / math.sqrt(self.d_v)
        logits = self.tanh_clip * torch.tanh(logits)

        logits = logits.masked_fill(
            ~mask,
            torch.finfo(logits.dtype).min,
        )

        # Eq. (16)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-12))

        return probs, log_probs


# -------------
# Policy
# -------------

class GATTransformerPolicy(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_edge: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 1,
    ):
        super().__init__()

        node_in_dim = 3

        self.encoder = PaperGATEncoder(
            node_in_dim=node_in_dim,
            edge_dim=edge_dim,
            d_model=d_model,
            d_edge=d_edge,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.decoder = PaperDualLayerAttentionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            tanh_clip=10.0,
        )

    def obs_to_instance(self, obs, device="cpu"):
        inst = {
            "xy": torch.tensor(
                obs["xy"],
                device=device,
                dtype=torch.float32,
            ),
            "node_type": torch.tensor(
                obs["node_type"],
                device=device,
                dtype=torch.long,
            ),
            "dist": torch.tensor(
                obs["dist"],
                device=device,
                dtype=torch.float32,
            ),
            "batt_cur": torch.tensor(
                obs["battery_norm"],
                device=device,
                dtype=torch.float32,
            ),
        }

        return inst


    def build_paper_node_feat(self, obs_inst):
        xy = obs_inst["xy"]                       # [N, 2]
        node_type = obs_inst["node_type"]         # [N]

        paper_type = torch.empty_like(
            node_type,
            dtype=xy.dtype,
            device=xy.device,
        )

        # 0 = depot, 1 = customer, 2 = station
        paper_type[node_type == 0] = -1.0
        paper_type[node_type == 1] = 0.0
        paper_type[node_type == 2] = 1.0

        paper_type = paper_type.view(-1, 1)

        node_feat = torch.cat(
            [xy, paper_type],
            dim=-1,
        )  # [N, 3]

        return node_feat

    # ---------------------------
    # Encoder
    # ---------------------------

    def encode(self, obs_inst):
        node_feat = self.build_paper_node_feat(obs_inst)

        dist = obs_inst["dist"]
        edge_index, edge_attr = build_fully_connected_edge_index_and_attr_from_dist(
            dist,
            device=dist.device,
        )

        h, h_graph = self.encoder(
            node_feat=node_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return h, h_graph, edge_index, edge_attr

    # -------------
    # Decoder
    # -------------

    def forward_step(
        self,
        h: torch.Tensor,
        h_graph: torch.Tensor,
        obs_inst,
        last_node_idx: int,
        mask_np,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = h.size(0)
        device = h.device

        last_node_idx = torch.tensor([last_node_idx], device=device, dtype=torch.long)
        b_t = obs_inst["batt_cur"].view(1).to(device=device, dtype=h.dtype)

        mask = torch.tensor(
            mask_np,
            device=device,
            dtype=torch.bool,
        ).view(1, N)

        probs, log_probs = self.decoder(
            h=h,
            h_graph=h_graph,
            last_node_idx=last_node_idx,
            b_t=b_t,
            mask=mask,
        )

        return probs.squeeze(0), log_probs.squeeze(0)


    def obs_list_to_instance_batch(self, obs_list: List[dict], device="cpu"):
        return {
            "xy": torch.tensor(
                np.stack([o["xy"] for o in obs_list]),
                device=device, dtype=torch.float32,
            ),                                                          # [B, N, 2]
            "node_type": torch.tensor(
                np.stack([o["node_type"] for o in obs_list]),
                device=device, dtype=torch.long,
            ),                                                           # [B, N]
            "dist": torch.tensor(
                np.stack([o["dist"] for o in obs_list]),
                device=device, dtype=torch.float32,
            ),                                                           # [B, N, N]
            "batt_cur": torch.tensor(
                [o["battery_norm"] for o in obs_list],
                device=device, dtype=torch.float32,
            ),                                                           # [B]
        }
 
    
    def build_paper_node_feat_batch(self, batch_inst):
        xy = batch_inst["xy"]                  # [B, N, 2]
        node_type = batch_inst["node_type"]    # [B, N]
    
        paper_type = torch.empty_like(node_type, dtype=xy.dtype, device=xy.device)
        # 0 = depot, 1 = customer, 2 = station.
        paper_type[node_type == 0] = -1.0
        paper_type[node_type == 1] = 0.0
        paper_type[node_type == 2] = 1.0
    
        paper_type = paper_type.unsqueeze(-1)   # [B, N, 1]
    
        node_feat = torch.cat([xy, paper_type], dim=-1)   # [B, N, 3]
        return node_feat
    
    
    def encode_batch(self, batch_inst):
        B, N, _ = batch_inst["xy"].shape
        device = batch_inst["xy"].device
    
        node_feat = self.build_paper_node_feat_batch(batch_inst)     # [B, N, 3]
        node_feat_flat = node_feat.view(B * N, -1)                   # [B*N, 3]
    
        edge_index, edge_attr, batch_index = build_batch_edge_index_and_attr_from_dist(
            batch_inst["dist"], device=device
        )
    
        h_flat, h_graph_batch = self.encoder.forward_batch(
            node_feat_flat, edge_index, edge_attr, batch_index
        )                                                              # h_flat: [B*N, d_model]
                                                                       # h_graph_batch: [B, d_model]
    
        h_batch = h_flat.view(B, N, -1)                               # [B, N, d_model]
    
        return h_batch, h_graph_batch, edge_index, edge_attr