import math
from typing import List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv




# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────
class HeteroGATEncoder(nn.Module):
    def __init__(
        self,
        in_dim:   int,
        d_model:  int   = 128,
        edge_dim: int   = 1,
        n_heads:  int   = 4,
        n_layers: int   = 4,
        dropout:  float = 0.1,
        n_types:  int   = 3,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_types = n_types
        self.dropout = dropout

        self.type_proj = nn.ModuleList([
            nn.Linear(in_dim, d_model) for _ in range(n_types)
        ])

        self.convs = nn.ModuleList([
            GATv2Conv(
                in_channels=d_model,
                out_channels=d_model,
                heads=n_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=edge_dim,
            )
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

    def forward(self, x, node_type, edge_index, edge_attr):
        h = x.new_zeros((x.size(0), self.d_model))

        for t, proj in enumerate(self.type_proj):
            mask = node_type == t
            if mask.any():
                h[mask] = proj(x[mask])

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h     = norm(h)
            h     = conv(h, edge_index, edge_attr=edge_attr)
            h     = F.gelu(h)
            h     = F.dropout(h, p=self.dropout, training=self.training)
            h     = h_res + h

        return h  # [N, d] or [B*N, d]

# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────
class PointerDecoder(nn.Module):
    def __init__(
        self,
        d_model:    int   = 128,
        ctx_dim:    int   = 128,
        cand_dim:   int   = 4,
        tanh_clip:  float = 6.0,
    ):
        super().__init__()
        self.d_model   = d_model
        self.tanh_clip = tanh_clip
        self.cand_dim  = cand_dim

        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.cand_mlp = nn.Sequential(
            nn.Linear(d_model + cand_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.value_head = nn.Sequential(
            nn.Linear(ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self.key_proj   = nn.Linear(d_model, d_model, bias=False)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)


    def masked_softmax(
        self, logits: torch.Tensor, mask: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        neg_inf = torch.finfo(logits.dtype).min
        logits  = logits.masked_fill(~mask, neg_inf)
        return F.softmax(logits, dim=dim)


    def forward(self, h, ctx, cand_dyn, mask):
        batched = h.dim() == 3
        if not batched:
            h        = h.unsqueeze(0)         # [1, N, d]
            cand_dyn = cand_dyn.unsqueeze(0)  # [1, N, cand_dim]

        q = self.query_proj(self.ctx_mlp(ctx))           # [B, d]
        q = q.unsqueeze(1)                               # [B, 1, d]

        hk = torch.cat([h, cand_dyn], dim=-1)            # [B, N, d+cand_dim]
        hk = self.cand_mlp(hk)                           # [B, N, d]
        k  = self.key_proj(hk)                           # [B, N, d]

        logits = torch.bmm(q, k.transpose(1, 2))         # [B, 1, N]
        logits = logits.squeeze(1) / math.sqrt(self.d_model)  # [B, N]
        logits = self.tanh_clip * torch.tanh(logits)

        if mask.dim() == 1:
            mask = mask.unsqueeze(0)                     # [1, N]

        probs     = self.masked_softmax(logits, mask, dim=-1) # [B, N]
        log_probs = torch.log(probs.clamp_min(1e-12))

        if not batched:
            probs     = probs.squeeze(0)
            log_probs = log_probs.squeeze(0)
            logits    = logits.squeeze(0)

        return probs, log_probs, logits


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────
class PPOGATPointerPolicyEnergy(nn.Module):
    def __init__(
        self,
        d_model:   int   = 128,
        n_heads:   int   = 8,
        n_layers:  int   = 3,
        dropout:   float = 0.1,
        edge_dim:  int   = 1,
        tanh_clip: float = 6.0,
    ):
        super().__init__()

        in_dim = 2 + 2  # xy + tw

        self.encoder = HeteroGATEncoder(
            in_dim=in_dim,
            d_model=d_model,
            edge_dim=edge_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.ctx_dim  = d_model * 3 + 3
        self.cand_dim = 4  # batt_arrival, time_arrival, dist_norm, visited

        self.decoder = PointerDecoder(
            d_model=d_model,
            ctx_dim=self.ctx_dim,
            cand_dim=self.cand_dim,
            tanh_clip=tanh_clip,
        )

    def obs_to_instance(self, obs, device="cpu"):
        tw = np.stack(
            [obs["ready_norm"], obs["due_norm"]], axis=1
        ).astype(np.float32)
        return {
            "xy":        torch.tensor(obs["xy"],        device=device, dtype=torch.float32),
            "tw":        torch.tensor(tw,               device=device, dtype=torch.float32),
            "node_type": torch.tensor(obs["node_type"], device=device, dtype=torch.long),
            "dist":      torch.tensor(obs["dist"],      device=device, dtype=torch.float32),
            "t_cur":     torch.tensor(obs["time_norm"],     device=device, dtype=torch.float32),
            "batt_cur":  torch.tensor(obs["battery_norm"],  device=device, dtype=torch.float32),
            "veh_left":  torch.tensor(obs["vehicles_left"], device=device, dtype=torch.float32),
            "batt_arrival": torch.tensor(obs["batt_arrival"], device=device, dtype=torch.float32),
            "time_arrival_at_all_other_node": torch.tensor(
                obs["time_arrival_at_all_other_node"], device=device, dtype=torch.float32
            ),
            "visited": torch.tensor(obs["visited_customer"], device=device, dtype=torch.float32),
        }


    def obs_list_to_instance_batch(self, obs_list: List[dict], device="cpu"):
        tws = [
            np.stack([o["ready_norm"], o["due_norm"]], axis=1).astype(np.float32)
            for o in obs_list
        ]
        return {
            "xy":        torch.tensor(
                             np.stack([o["xy"] for o in obs_list]),
                             device=device, dtype=torch.float32),          # [B, N, 2]
            "tw":        torch.tensor(
                             np.stack(tws),
                             device=device, dtype=torch.float32),          # [B, N, 2]
            "node_type": torch.tensor(
                             np.stack([o["node_type"] for o in obs_list]),
                             device=device, dtype=torch.long),             # [B, N]
            "dist":      torch.tensor(
                             np.stack([o["dist"] for o in obs_list]),
                             device=device, dtype=torch.float32),          # [B, N, N]
            "t_cur":     torch.tensor(
                             [o["time_norm"]     for o in obs_list],
                             device=device, dtype=torch.float32),
            "batt_cur":  torch.tensor(
                             [o["battery_norm"]  for o in obs_list],
                             device=device, dtype=torch.float32),
            "veh_left":  torch.tensor(
                             [o["vehicles_left"] for o in obs_list],
                             device=device, dtype=torch.float32),
            "batt_arrival": torch.tensor(
                             np.stack([o["batt_arrival"] for o in obs_list]),
                             device=device, dtype=torch.float32),          # [B, N]
            "time_arrival_at_all_other_node": torch.tensor(
                             np.stack([o["time_arrival_at_all_other_node"] for o in obs_list]),
                             device=device, dtype=torch.float32),          # [B, N]
            "visited":   torch.tensor(
                             np.stack([o["visited_customer"] for o in obs_list]),
                             device=device, dtype=torch.float32),          # [B, N]
        }


    def build_static_node_feat(self, obs_inst):
        return torch.cat([obs_inst["xy"], obs_inst["tw"]], dim=-1)

    def build_static_node_feat_batch(self, batch_inst):
        return torch.cat([batch_inst["xy"], batch_inst["tw"]], dim=-1)

    def build_global_feat(self, obs_inst):
        return torch.stack([
            obs_inst["t_cur"],
            obs_inst["batt_cur"],
            obs_inst["veh_left"],
        ]).view(1, 3)

    def build_global_feat_batch(self, batch_inst):
        return torch.stack([
            batch_inst["t_cur"],
            batch_inst["batt_cur"],
            batch_inst["veh_left"],
        ], dim=-1)  # [B, 3]

    def build_candidate_dyn(self, obs_inst, last_node_idx):
        batt_arrival = obs_inst["batt_arrival"].view(-1, 1)
        time_arrival = obs_inst["time_arrival_at_all_other_node"].view(-1, 1)
        dist_norm    = (obs_inst["dist"][last_node_idx] /
                        (obs_inst["dist"].max() + 1e-8)).view(-1, 1)
        visited      = obs_inst["visited"].view(-1, 1)
        return torch.cat([batt_arrival, time_arrival, dist_norm, visited], dim=-1)

    def build_candidate_dyn_batch(self, batch_inst, last_node_indices: torch.Tensor):
        B = batch_inst["batt_arrival"].size(0)

        batt_arrival = batch_inst["batt_arrival"].unsqueeze(-1)           # [B, N, 1]
        time_arrival = batch_inst["time_arrival_at_all_other_node"].unsqueeze(-1)  # [B, N, 1]

        dist = batch_inst["dist"]                                         # [B, N, N]
        dist_max = dist.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)   # [B, 1, 1]
        dist_norm = dist / dist_max                                       # [B, N, N]

        idx = last_node_indices.view(B, 1, 1).expand(B, 1, dist_norm.size(-1))
        dist_from_last = dist_norm.gather(1, idx).squeeze(1)             # [B, N]
        dist_from_last = dist_from_last.unsqueeze(-1)                    # [B, N, 1]

        visited = batch_inst["visited"].unsqueeze(-1)                    # [B, N, 1]

        return torch.cat([batt_arrival, time_arrival, dist_from_last, visited], dim=-1)


    def build_fully_connected_edge_index_and_attr_from_dist(
        self, dist: torch.Tensor, device
    ):
        N = dist.size(0)
        dist_norm = dist / (dist.max() + 1e-8)
        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        src, dst = mask.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0)         # [2, E]
        edge_attr  = dist_norm[src, dst].unsqueeze(-1)      # [E, 1]
        return edge_index, edge_attr


    def build_batch_edge_index_and_attr_from_dist(
        self, dist_batch: torch.Tensor, device
    ):

        B, N, _ = dist_batch.shape

        all_edge_index = []
        all_edge_attr  = []

        base_ei, _ = self.build_fully_connected_edge_index_and_attr_from_dist(
            dist_batch[0], device
        )

        for b in range(B):
            dist      = dist_batch[b]                        # [N, N]
            dist_norm = dist / (dist.max() + 1e-8)
            src, dst  = base_ei[0], base_ei[1]
            ei        = base_ei + b * N
            ea        = dist_norm[src, dst].unsqueeze(-1)    # [E, 1]
            all_edge_index.append(ei)
            all_edge_attr.append(ea)

        edge_index = torch.cat(all_edge_index, dim=1)        # [2, B*E]
        edge_attr  = torch.cat(all_edge_attr,  dim=0)        # [B*E, 1]
        batch_vec  = torch.arange(B, device=device).repeat_interleave(N)  # [B*N]
        return edge_index, edge_attr, batch_vec


    def encode_static(self, obs_inst):
        xy        = obs_inst["xy"]
        dist      = obs_inst["dist"]
        node_type = obs_inst["node_type"]
        device    = xy.device

        node_feat  = self.build_static_node_feat(obs_inst)
        edge_index, edge_attr = self.build_fully_connected_edge_index_and_attr_from_dist(
            dist, device=device
        )
        h = self.encoder(node_feat, node_type, edge_index, edge_attr)
        return h, edge_index, edge_attr


    def encode_static_batch(self, batch_inst):

        B, N, _   = batch_inst["xy"].shape
        device    = batch_inst["xy"].device

        node_feat  = self.build_static_node_feat_batch(batch_inst)  # [B, N, 4]
        node_feat_flat = node_feat.view(B * N, -1)                  # [B*N, 4]

        node_type_flat = batch_inst["node_type"].view(B * N)        # [B*N]

        edge_index, edge_attr, _ = self.build_batch_edge_index_and_attr_from_dist(
            batch_inst["dist"], device=device
        )

        h_flat = self.encoder(
            node_feat_flat, node_type_flat, edge_index, edge_attr
        )                                                            # [B*N, d]

        h_batch = h_flat.view(B, N, -1)                             # [B, N, d]
        return h_batch


    def forward_step(self, h_static, obs_inst, last_node_idx, mask_np):
        if h_static is None:
            raise ValueError("h_static is None. You probably forgot to call encode_static() before forward_step().")

        if not torch.is_tensor(h_static):
            raise TypeError(f"h_static must be torch.Tensor, got {type(h_static)}")
        device = h_static.device
        dtype  = h_static.dtype
        N      = h_static.size(0)

        visited   = obs_inst["visited"].to(device=device, dtype=torch.bool).view(N)
        alive     = ~visited
        alive_2d  = alive.unsqueeze(-1)
        alive_f   = alive_2d.to(dtype)

        global_feat = self.build_global_feat(obs_inst).to(device=device, dtype=dtype)
        cand_dyn    = self.build_candidate_dyn(obs_inst, last_node_idx).to(device=device, dtype=dtype)

        alive_count  = alive_f.sum().clamp_min(1.0)
        h_graph_mean = (h_static * alive_f).sum(dim=0, keepdim=True) / alive_count

        if alive.any():
            h_for_max   = h_static.masked_fill(~alive_2d, torch.finfo(dtype).min)
            h_graph_max = h_for_max.max(dim=0, keepdim=True).values
        else:
            h_graph_max = h_static.max(dim=0, keepdim=True).values

        h_last = h_static[last_node_idx].view(1, -1)

        ctx   = torch.cat([h_graph_mean, h_graph_max, h_last, global_feat], dim=-1)
        value = self.decoder.value_head(ctx)

        mask  = torch.as_tensor(mask_np, device=device, dtype=torch.bool).view(1, N)
        probs, log_probs, logits = self.decoder(h_static, ctx, cand_dyn, mask)

        return (
            probs.squeeze(0),
            log_probs.squeeze(0),
            value.squeeze(0),
            logits.squeeze(0),
        )


    def evaluate_action(
        self, h_static, obs_inst, last_node_idx, mask_np, action
    ):
        """Log-probability, entropy and value of a given action (single graph; used by decode_mode="serial")."""
        probs, log_probs, value, logits = self.forward_step(
            h_static, obs_inst, last_node_idx, mask_np
        )
        dist = torch.distributions.Categorical(probs=probs)
        if not torch.is_tensor(action):
            action = torch.tensor(action, device=probs.device, dtype=torch.long)
        return dist.log_prob(action), dist.entropy(), value


    def forward_step_batch(
        self,
        h_static_batch: torch.Tensor,
        batch_inst:     dict,
        last_node_indices: torch.Tensor,
        mask_batch:     torch.Tensor,
    ):

        device = h_static_batch.device
        dtype  = h_static_batch.dtype
        B, N, d = h_static_batch.shape

        visited  = batch_inst["visited"].to(device=device, dtype=torch.bool)  # [B, N]
        alive    = ~visited                                                    # [B, N]
        alive_f  = alive.to(dtype).unsqueeze(-1)                              # [B, N, 1]

        global_feat = self.build_global_feat_batch(batch_inst).to(dtype)      # [B, 3]
        cand_dyn    = self.build_candidate_dyn_batch(
            batch_inst, last_node_indices
        ).to(dtype)                                                            # [B, N, 4]

        alive_count  = alive_f.sum(dim=1, keepdim=True).clamp_min(1.0)       # [B, 1, 1]
        h_graph_mean = (h_static_batch * alive_f).sum(dim=1) / alive_count.squeeze(-1)  # [B, d]

        neg_inf_mask = ~alive.unsqueeze(-1).expand_as(h_static_batch)
        h_for_max    = h_static_batch.masked_fill(neg_inf_mask, torch.finfo(dtype).min)
        h_graph_max  = h_for_max.max(dim=1).values                           # [B, d]

        h_graph_max_masked = h_for_max.max(dim=1).values                 # [B, d]
        any_alive_per_graph = alive.any(dim=1)                           # [B]
        h_graph_max = torch.where(
            any_alive_per_graph.unsqueeze(-1),
            h_graph_max_masked,
            h_static_batch.max(dim=1).values,
        )

        # last node embedding: each graph uses its own last_node_idx
        idx    = last_node_indices.view(B, 1, 1).expand(B, 1, d)
        h_last = h_static_batch.gather(1, idx).squeeze(1)                    # [B, d]

        ctx = torch.cat(
            [h_graph_mean, h_graph_max, h_last, global_feat], dim=-1
        )                                                                     # [B, ctx_dim]

        values = self.decoder.value_head(ctx)                                 # [B, 1]

        probs, log_probs, logits = self.decoder(
            h_static_batch, ctx, cand_dyn, mask_batch
        )                                                                     # all [B, N]

        return probs, log_probs, values, logits