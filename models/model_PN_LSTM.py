from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
LSTMState = Tuple[Tensor, Tensor]
EncodedState = Dict[str, Tensor]


# ---------------
# Encoder
# ---------------
class Lin2022Structure2VecEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 4,
        global_dim: int = 3,
        d_model: int = 128,
        edge_dim: int = 1,
        n_layers: int = 4,
        dropout: float = 0.0,
        aggr: str = "mean",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.global_dim = global_dim
        self.d_model = d_model
        self.edge_dim = edge_dim
        self.n_layers = n_layers
        self.dropout = dropout

        assert aggr in ("mean", "sum"), f"aggr must be 'mean' or 'sum', got {aggr}"
        self.aggr = aggr

        # X_i^t -> x_hat_i
        self.node_proj = nn.Linear(in_dim, d_model)

        # G^t -> g_hat
        self.global_proj = nn.Linear(global_dim, d_model)

        # w(i,j) -> edge hidden
        self.edge_proj = nn.Linear(edge_dim, d_model)

        # theta_1 - theta_4
        self.theta_x = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_layers)
        ])
        self.theta_g = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_layers)
        ])
        self.theta_neighbor = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_layers)
        ])
        self.theta_edge = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])

    @staticmethod
    def _aggregate_sum(src_values: Tensor, dst_index: Tensor, num_nodes: int) -> Tensor:
        out = src_values.new_zeros((num_nodes, src_values.size(-1)))
        out.index_add_(0, dst_index, src_values)
        return out

    def _aggregate(self, src_values: Tensor, dst_index: Tensor, num_nodes: int) -> Tensor:
        out = self._aggregate_sum(src_values, dst_index, num_nodes)
        if self.aggr == "mean":
            deg = src_values.new_zeros(num_nodes)
            deg.index_add_(
                0, dst_index,
                torch.ones(dst_index.size(0), device=src_values.device, dtype=src_values.dtype),
            )
            out = out / deg.clamp_min(1.0).unsqueeze(-1)
        return out

    def _expand_global_embedding(
        self,
        global_feat: Tensor,
        num_nodes: int,
        batch: Optional[Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)

        global_feat = global_feat.to(device=device, dtype=dtype)
        g_hat = self.global_proj(global_feat)  # [1,d] or [B,d]

        if g_hat.size(0) == 1:
            return g_hat.expand(num_nodes, -1)

        if batch is None:
            raise ValueError("batch is required when global_feat has shape [B, global_dim].")

        return g_hat[batch]

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        global_feat: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        num_nodes = x.size(0)
        x_hat = self.node_proj(x)
        g_hat_node = self._expand_global_embedding(
            global_feat=global_feat,
            num_nodes=num_nodes,
            batch=batch,
            device=x.device,
            dtype=x.dtype,
        )

        # mu_i^(0) = x_hat_i
        mu = x_hat
        src = edge_index[0]
        dst = edge_index[1]

        # sum_{j in N(i)} ReLU(theta_5 w(i,j)), degree-normalized when aggr="mean"
        edge_hidden = F.relu(self.edge_proj(edge_attr))
        edge_aggr = self._aggregate(edge_hidden, dst, num_nodes)

        for layer in range(self.n_layers):
            neighbor_aggr = self._aggregate(mu[src], dst, num_nodes)

            mu = F.relu(
                self.theta_x[layer](x_hat)
                + self.theta_g[layer](g_hat_node)
                + self.theta_neighbor[layer](neighbor_aggr)
                + self.theta_edge[layer](edge_aggr)
            )
            mu = self.layer_norms[layer](mu)

            if self.dropout > 0.0:
                mu = F.dropout(mu, p=self.dropout, training=self.training)

        return mu, x_hat


# ---------------
# Attention
# ---------------

class Lin2022PointerAttention(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        mask_value: float = -1e9,
        tanh_clip: float = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.mask_value = mask_value

        # v_i^t = theta_v tanh(theta_u [mu_i^t ; h^t])
        self.theta_u = nn.Linear(2 * d_model, d_model)
        self.theta_v = nn.Linear(d_model, 1, bias=False)

        # g_i^t = theta_g tanh(theta_c [mu_i^t ; c^t])
        self.theta_c = nn.Linear(2 * d_model, d_model)
        self.theta_g = nn.Linear(d_model, 1, bias=False)

        self.tanh_clip = tanh_clip

    def _masked_scores(self, scores: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        if valid_mask is None:
            return scores

        valid_mask = valid_mask.to(device=scores.device, dtype=torch.bool)
        if (~valid_mask.any(dim=-1)).any():
            raise RuntimeError("Lin2022PointerAttention: no valid action in mask.")

        return scores.masked_fill(~valid_mask, self.mask_value)


    def forward(
        self,
        mu: Tensor,
        h_t: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        single = False
        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
            single = True
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
        if valid_mask is not None and valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)

        B, N, d = mu.shape
        h_expand = h_t.unsqueeze(1).expand(B, N, d)

        v_input = torch.cat([mu, h_expand], dim=-1)
        v = self.theta_v(torch.tanh(self.theta_u(v_input))).squeeze(-1)  # [B,N]
        v_masked = self._masked_scores(v, valid_mask)
        attn_weights = F.softmax(v_masked, dim=-1)

        context = torch.bmm(attn_weights.unsqueeze(1), mu).squeeze(1)  # [B,d]

        c_expand = context.unsqueeze(1).expand(B, N, d)
        g_input = torch.cat([mu, c_expand], dim=-1)
        logits = self.theta_g(torch.tanh(self.theta_c(g_input))).squeeze(-1)  # [B,N]
        if self.tanh_clip is not None:
            logits = self.tanh_clip * torch.tanh(logits / math.sqrt(d)) # add root(d)
        logits = self._masked_scores(logits, valid_mask)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        out = {
            "probs": probs,
            "log_probs": log_probs,
            "logits": logits,
            "context": context,
            "attn_weights": attn_weights,
        }

        if single:
            out = {k: v.squeeze(0) for k, v in out.items()}

        return out


# ---------------
# LSTM Decoder
# ---------------

class Lin2022LSTMDecoder(nn.Module):
    def __init__(self, d_model: int = 128, tanh_clip=None):
        super().__init__()
        self.d_model = d_model
        self.lstm_cell = nn.LSTMCell(input_size=d_model, hidden_size=d_model)
        self.attention = Lin2022PointerAttention(d_model=d_model, tanh_clip=tanh_clip)

    def init_state(
        self,
        batch_size: int,
        device: Union[str, torch.device],
        dtype: torch.dtype = torch.float32,
    ) -> LSTMState:
        h = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        return h, c

    def forward_step(
        self,
        current_node_emb: Tensor,
        lstm_state: Optional[LSTMState],
        mu: Tensor,
        valid_mask: Tensor,
    ) -> Dict[str, Union[Tensor, LSTMState]]:
        if current_node_emb.dim() == 1:
            current_node_emb = current_node_emb.unsqueeze(0)
        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)

        B = current_node_emb.size(0)
        if lstm_state is None:
            lstm_state = self.init_state(
                batch_size=B,
                device=current_node_emb.device,
                dtype=current_node_emb.dtype,
            )

        h_prev, c_prev = lstm_state
        h_t, c_t = self.lstm_cell(current_node_emb, (h_prev, c_prev))

        attn_out = self.attention(mu=mu, h_t=h_t, valid_mask=valid_mask)
        attn_out["lstm_state"] = (h_t, c_t)
        return attn_out


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE Policy
# ─────────────────────────────────────────────────────────────────────────────
class PNLSTMPolicy(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
        edge_dim: int = 1,
        global_dim: int = 3,
        aggr: str = "mean",
        tanh_clip=None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.in_dim = 5

        self.encoder = Lin2022Structure2VecEncoder(
            in_dim=self.in_dim,
            global_dim=global_dim,
            d_model=d_model,
            edge_dim=edge_dim,
            n_layers=n_layers,
            dropout=dropout,
            aggr=aggr,
        )
        self.decoder = Lin2022LSTMDecoder(d_model=d_model, tanh_clip=tanh_clip)


    def obs_to_instance(self, obs: dict, device: Union[str, torch.device] = "cpu") -> Dict[str, Tensor]:
        tw = np.stack([obs["ready_norm"], obs["due_norm"]], axis=1).astype(np.float32)
        return {
            "xy": torch.tensor(obs["xy"], device=device, dtype=torch.float32),
            "tw": torch.tensor(tw, device=device, dtype=torch.float32),
            "node_type": torch.tensor(obs["node_type"], device=device, dtype=torch.long),
            "dist": torch.tensor(obs["dist"], device=device, dtype=torch.float32),
            "t_cur": torch.tensor(obs["time_norm"], device=device, dtype=torch.float32),
            "batt_cur": torch.tensor(obs["battery_norm"], device=device, dtype=torch.float32),
            "veh_left": torch.tensor(obs["vehicles_left"], device=device, dtype=torch.float32),
            "visited": torch.tensor(obs["visited_customer"], device=device, dtype=torch.float32),
        }


    def build_node_feat(self, obs_inst: Dict[str, Tensor]) -> Tensor:
        xy = obs_inst["xy"]
        tw = obs_inst["tw"]
        node_type = obs_inst["node_type"]
        visited = obs_inst["visited"].to(dtype=xy.dtype)

        is_customer = node_type == 1
        unserved = (is_customer & (visited < 0.5)).to(dtype=xy.dtype).unsqueeze(-1)
        return torch.cat([xy, tw, unserved], dim=-1)


    def build_global_feat(self, obs_inst: Dict[str, Tensor]) -> Tensor:
        return torch.stack([
            obs_inst["t_cur"],
            obs_inst["batt_cur"],
            obs_inst["veh_left"],
        ]).view(1, self.global_dim)


    def build_fully_connected_edge_index_and_attr_from_dist(
        self,
        dist: Tensor,
        device: Union[str, torch.device],
    ) -> Tuple[Tensor, Tensor]:
        N = dist.size(0)
        dist = dist.to(device=device)
        dist_norm = dist / dist.max().clamp_min(1e-8)

        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        src, dst = mask.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = dist_norm[src, dst].unsqueeze(-1)
        return edge_index, edge_attr

    def encode_state(self, obs_inst: Dict[str, Tensor]) -> Tuple[EncodedState, Tensor, Tensor]:
        device = obs_inst["xy"].device
        node_feat = self.build_node_feat(obs_inst)
        global_feat = self.build_global_feat(obs_inst).to(device=device, dtype=node_feat.dtype)
        edge_index, edge_attr = self.build_fully_connected_edge_index_and_attr_from_dist(
            obs_inst["dist"], device=device
        )

        mu, x_hat = self.encoder(
            x=node_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_feat=global_feat,
            batch=None,
        )
        return {"mu": mu, "x_hat": x_hat}, edge_index, edge_attr




    def forward_step(
        self,
        encoded: EncodedState,
        last_node_idx: Union[int, Tensor],
        mask: Union[np.ndarray, Tensor, List[bool]],
        lstm_state: Optional[LSTMState] = None,
    ) -> Dict[str, Union[Tensor, LSTMState]]:
        mu = encoded["mu"]
        x_hat = encoded["x_hat"]
        device = mu.device
        N = mu.size(0)

        if torch.is_tensor(last_node_idx):
            last_node_idx = int(last_node_idx.item())

        current_node_emb = x_hat[last_node_idx].view(1, -1)
        valid_mask = torch.as_tensor(mask, device=device, dtype=torch.bool).view(1, N)

        out = self.decoder.forward_step(
            current_node_emb=current_node_emb,
            lstm_state=lstm_state,
            mu=mu.unsqueeze(0),
            valid_mask=valid_mask,
        )

        # Squeeze the batch dimension for single-graph convenience
        result = {}
        for key, value in out.items():
            if key == "lstm_state":
                result[key] = value
            else:
                result[key] = value.squeeze(0)
        return result


    def sample_action(
        self,
        encoded: EncodedState,
        last_node_idx: Union[int, Tensor],
        mask: Union[np.ndarray, Tensor, List[bool]],
        lstm_state: Optional[LSTMState] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, LSTMState, Dict[str, Tensor]]:
        out = self.forward_step(encoded, last_node_idx, mask, lstm_state=lstm_state)
        probs = out["probs"]
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        next_state = out["lstm_state"]
        aux = {
            "probs": out["probs"],
            "log_probs": out["log_probs"],
            "logits": out["logits"],
            "context": out["context"],
            "attn_weights": out["attn_weights"],
        }
        return action, log_prob, entropy, next_state, aux

    def greedy_action(
        self,
        encoded: EncodedState,
        last_node_idx: Union[int, Tensor],
        mask: Union[np.ndarray, Tensor, List[bool]],
        lstm_state: Optional[LSTMState] = None,
    ) -> Tuple[Tensor, Tensor, LSTMState, Dict[str, Tensor]]:

        out = self.forward_step(encoded, last_node_idx, mask, lstm_state=lstm_state)
        probs = out["probs"]
        action = torch.argmax(probs, dim=-1)
        log_prob = out["log_probs"].gather(0, action.view(1)).squeeze(0)
        next_state = out["lstm_state"]
        aux = {
            "probs": out["probs"],
            "log_probs": out["log_probs"],
            "logits": out["logits"],
            "context": out["context"],
            "attn_weights": out["attn_weights"],
        }
        return action, log_prob, next_state, aux