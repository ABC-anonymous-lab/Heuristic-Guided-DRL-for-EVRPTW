from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv





# ---------------------------
# Encoder
# ---------------------------
class HeteroGATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        edge_dim: int = 1,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_types: int = 3,
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


    def forward(self, x: torch.Tensor, node_type: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:        
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

        return h


# ---------------------------
# Decoder
# ---------------------------
class QDecoder(nn.Module):
    def __init__(self, d_model=128, ctx_dim=128, cand_dim=4):
        super().__init__()
        input_dim = d_model + ctx_dim + cand_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )


    def forward(self, h_static, ctx, cand_dyn):
        batched = h_static.dim() == 3
        if not batched:
            h_static = h_static.unsqueeze(0)   # [1, N, d]
            cand_dyn = cand_dyn.unsqueeze(0)   # [1, N, cand_dim]

        B, N, _ = h_static.shape
        ctx_expand = ctx.unsqueeze(1).expand(B, N, -1)   # [B, N, ctx_dim]

        feat = torch.cat([h_static, ctx_expand, cand_dyn], dim=-1)   # [B, N, d+ctx_dim+cand_dim]
        q_values = self.mlp(feat).squeeze(-1)                        # [B, N]

        if not batched:
            q_values = q_values.squeeze(0)    # [N]
        return q_values


# ----------------
# Policy
# ----------------
class DDQNGATPolicyEnergy(nn.Module):
    def __init__(
            self, 
            d_model:   int   = 128,
            n_heads:   int   = 8,
            n_layers:  int   = 3,
            dropout:   float = 0.1,
            edge_dim:  int   = 1,
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

        self.ctx_dim = d_model * 3 + 3
        self.cand_dim = 4  # batt_arrival, time_arrival, dist_norm, visited

        self.decoder = QDecoder(
            d_model=d_model,
            ctx_dim=self.ctx_dim,
            cand_dim=self.cand_dim,
        )


    @contextmanager
    def eval_context(self):
        training = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(training)


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


    def obs_list_to_instance_batch(self, obs_list, device="cpu"):
        tws = [
            np.stack([o["ready_norm"], o["due_norm"]], axis=1).astype(np.float32)
            for o in obs_list
        ]
        return {
            "xy": torch.tensor(
                np.stack([o["xy"] for o in obs_list]),
                device=device, dtype=torch.float32),          # [B, N, 2]
            "tw": torch.tensor(
                np.stack(tws),
                device=device, dtype=torch.float32),          # [B, N, 2]
            "node_type": torch.tensor(
                np.stack([o["node_type"] for o in obs_list]),
                device=device, dtype=torch.long),             # [B, N]
            "dist": torch.tensor(
                np.stack([o["dist"] for o in obs_list]),
                device=device, dtype=torch.float32),          # [B, N, N]
            "t_cur": torch.tensor(
                [o["time_norm"] for o in obs_list],
                device=device, dtype=torch.float32),          # [B]
            "batt_cur": torch.tensor(
                [o["battery_norm"] for o in obs_list],
                device=device, dtype=torch.float32),          # [B]
            "veh_left": torch.tensor(
                [o["vehicles_left"] for o in obs_list],
                device=device, dtype=torch.float32),          # [B]
            "batt_arrival": torch.tensor(
                np.stack([o["batt_arrival"] for o in obs_list]),
                device=device, dtype=torch.float32),          # [B, N]
            "time_arrival_at_all_other_node": torch.tensor(
                np.stack([o["time_arrival_at_all_other_node"] for o in obs_list]),
                device=device, dtype=torch.float32),          # [B, N]
            "visited": torch.tensor(
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
        ])
    

    def build_global_feat_batch(self, batch_inst):
        return torch.stack([
            batch_inst["t_cur"],
            batch_inst["batt_cur"],
            batch_inst["veh_left"],
        ], dim=-1)


    def build_candidate_dyn(self, obs_inst, last_node_idx: int):
        batt_arrival = obs_inst["batt_arrival"].unsqueeze(-1)                      # [N, 1]
        time_arrival = obs_inst["time_arrival_at_all_other_node"].unsqueeze(-1)    # [N, 1]
 
        dist = obs_inst["dist"]                                                    # [N, N]
        dist_max = dist.amax().clamp_min(1e-8)                                     # scalar
        dist_norm = dist / dist_max                                                # [N, N]
 
        dist_from_last = dist_norm[last_node_idx].unsqueeze(-1)                    # [N, 1]
        visited = obs_inst["visited"].unsqueeze(-1)                                # [N, 1]

        return torch.cat([batt_arrival, time_arrival, dist_from_last, visited], dim=-1)  # [N, 4]


    def build_candidate_dyn_batch(self, batch_inst, last_node_indices: torch.Tensor):
        B = batch_inst["batt_arrival"].size(0)

        batt_arrival = batch_inst["batt_arrival"].unsqueeze(-1)           # [B, N, 1]
        time_arrival = batch_inst["time_arrival_at_all_other_node"].unsqueeze(-1)  # [B, N, 1]

        dist = batch_inst["dist"]                                         # [B, N, N]
        dist_max = dist.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)    # [B, 1, 1]
        dist_norm = dist / dist_max                                       # [B, N, N]
        
        idx = last_node_indices.view(B, 1, 1).expand(B, 1, dist_norm.size(-1))
        dist_from_last = dist_norm.gather(1, idx).squeeze(1).unsqueeze(-1)    # [B, N, 1]
        
        visited = batch_inst["visited"].unsqueeze(-1)                     # [B, N, 1]

        return torch.cat([batt_arrival, time_arrival, dist_from_last, visited], dim=-1)


    def build_fully_connected_edge_index_and_attr_from_dist(self, dist: torch.Tensor, device):
        N = dist.size(0)

        dist_norm = dist / (dist.max() + 1e-8)

        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        src, dst = mask.nonzero(as_tuple=True)

        edge_index = torch.stack([src, dst], dim=0)         # [2, E]
        edge_attr = dist_norm[src, dst].unsqueeze(-1)       # [E, 1]
        return edge_index, edge_attr


    def build_batch_edge_index_and_attr_from_dist(self, dist_batch: torch.Tensor, device):
        B, N, _ = dist_batch.shape

        all_edge_index = []
        all_edge_attr = []

        base_ei, _ = self.build_fully_connected_edge_index_and_attr_from_dist(
            dist_batch[0], device
        )  # [2, E_single]

        for b in range(B):
            dist = dist_batch[b]                              # [N, N]
            dist_norm = dist / (dist.max() + 1e-8)
            src, dst = base_ei[0], base_ei[1]
            ei = base_ei + b * N
            ea = dist_norm[src, dst].unsqueeze(-1)             # [E, 1]
            all_edge_index.append(ei)
            all_edge_attr.append(ea)

        edge_index = torch.cat(all_edge_index, dim=1)          # [2, B*E]
        edge_attr = torch.cat(all_edge_attr, dim=0)            # [B*E, 1]
        batch_vec = torch.arange(B, device=device).repeat_interleave(N)  # [B*N]
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
        B, N, _ = batch_inst["xy"].shape
        device = batch_inst["xy"].device

        node_feat = self.build_static_node_feat_batch(batch_inst)   # [B, N, 7]
        node_feat_flat = node_feat.view(B * N, -1)                  # [B*N, 7]

        node_type_flat = batch_inst["node_type"].view(B * N)        # [B*N]

        edge_index, edge_attr, _ = self.build_batch_edge_index_and_attr_from_dist(
            batch_inst["dist"], device=device
        )

        h_flat = self.encoder(
            node_feat_flat, node_type_flat, edge_index, edge_attr
        )                                                            # [B*N, d]

        h_batch = h_flat.view(B, N, -1)                              # [B, N, d]
        return h_batch


    def forward_step(
        self,
        h_static: torch.Tensor,
        obs_inst: dict,
        last_node_idx: int,
        mask_np=None,
    ):
        """
        Single-graph version of forward_step_batch, with the same alive mask:
        h_graph_mean / h_graph_max are taken over unvisited (alive) nodes only.
        """
        device = h_static.device
        dtype = h_static.dtype
        N, d = h_static.shape
 
        visited = obs_inst["visited"].to(device=device, dtype=torch.bool)    # [N]
        alive   = ~visited                                                   # [N]
        alive_f = alive.to(dtype).unsqueeze(-1)                              # [N, 1]
 
        global_feat = self.build_global_feat(obs_inst).to(device=device, dtype=dtype)   # [3]
        cand_dyn = self.build_candidate_dyn(obs_inst, last_node_idx).to(device=device, dtype=dtype)  # [N, 4]
 
        alive_count  = alive_f.sum(dim=0, keepdim=True).clamp_min(1.0)     # [1, 1]
        h_graph_mean = (h_static * alive_f).sum(dim=0) / alive_count.squeeze(-1)  # [d]
 
        neg_inf_mask = ~alive.unsqueeze(-1).expand_as(h_static)            # [N, d]
        h_for_max    = h_static.masked_fill(neg_inf_mask, torch.finfo(dtype).min)

        # If no node is alive (e.g. the last steps before done), the masked max would
        # be finfo.min and blow up the Q-values; fall back to the max over all nodes.
        if alive.any():
            h_graph_max = h_for_max.max(dim=0).values                      # [d]
        else:
            h_graph_max = h_static.max(dim=0).values                       # [d]
 
        h_last = h_static[last_node_idx]                                   # [d]
 
        ctx = torch.cat([h_graph_mean, h_graph_max, h_last, global_feat], dim=-1)  # [ctx_dim]
        ctx = ctx.unsqueeze(0)                                              # [1, ctx_dim]
 
        q_values = self.decoder(h_static, ctx, cand_dyn)   # h_static [N, d], cand_dyn [N, 4]: single-graph branch of QDecoder
 
        if mask_np is not None:
            mask = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
            q_values = q_values.masked_fill(~mask, -1e9)
 
        return q_values  # [N]


    def forward_step_batch(
        self,
        h_static_batch: torch.Tensor,    # [B, N, d]
        batch_inst: dict,
        last_node_indices: torch.Tensor, # [B] int
        mask_batch: torch.Tensor = None, # [B, N] bool, None = no mask
    ):
        device = h_static_batch.device
        dtype = h_static_batch.dtype
        B, N, d = h_static_batch.shape

        visited  = batch_inst["visited"].to(device=device, dtype=torch.bool)  # [B, N]
        alive    = ~visited                                                    # [B, N]
        alive_f  = alive.to(dtype).unsqueeze(-1)                              # [B, N, 1]
        
        global_feat = self.build_global_feat_batch(batch_inst).to(device=device,dtype=dtype)     # [B, 3]
        cand_dyn = self.build_candidate_dyn_batch(batch_inst, last_node_indices).to(device=device, dtype=dtype)  # [B, N, 4]

        alive_count  = alive_f.sum(dim=1, keepdim=True).clamp_min(1.0)       # [B, 1, 1]
        h_graph_mean = (h_static_batch * alive_f).sum(dim=1) / alive_count.squeeze(-1)  # [B, d]

        neg_inf_mask = ~alive.unsqueeze(-1).expand_as(h_static_batch)
        h_for_max    = h_static_batch.masked_fill(neg_inf_mask, torch.finfo(dtype).min)
        h_graph_max_masked = h_for_max.max(dim=1).values                     # [B, d]

        any_alive_per_graph = alive.any(dim=1)                               # [B]
        h_graph_max_fallback = h_static_batch.max(dim=1).values              # [B, d]
        h_graph_max = torch.where(
            any_alive_per_graph.unsqueeze(-1),                               # [B, 1], broadcast to [B, d]
            h_graph_max_masked,
            h_graph_max_fallback,
        )                                                                    # [B, d]

        idx    = last_node_indices.view(B, 1, 1).expand(B, 1, d)
        h_last = h_static_batch.gather(1, idx).squeeze(1)                    # [B, d]

        ctx = torch.cat(
            [h_graph_mean, h_graph_max, h_last, global_feat], dim=-1
        )

        q_values = self.decoder(h_static_batch, ctx, cand_dyn)

        if mask_batch is not None:
            q_values = q_values.masked_fill(~mask_batch, -1e9)

        return q_values