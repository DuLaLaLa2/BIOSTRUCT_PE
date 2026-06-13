import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v1_struct_transformer import (
    ProteinGraphSequence,
    RotaryPositionEmbedding,
    calculate_specificity,
    collate_windows,
    evaluate_metrics,
    find_best_threshold_by_mcc,
    load_graph_sequences,
    make_window_loader,
    move_batch_to_device,
    safe_ap,
    safe_auc,
    set_seed,
    split_sequences_by_protein,
)


class FixedAbsoluteSequenceBias(nn.Module):
    """Fixed, non-learned sequence-distance bias: -scale * |i-j| / distance_scale."""

    def __init__(self, distance_scale: float = 128.0, bias_scale: float = 1.0):
        super().__init__()
        self.distance_scale = float(distance_scale)
        self.bias_scale = float(bias_scale)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        distance = (positions.unsqueeze(2) - positions.unsqueeze(1)).abs().float()
        distance = distance / max(self.distance_scale, 1.0)
        return (-self.bias_scale * distance).unsqueeze(1)


class PairwiseStructuralBiasV2(nn.Module):
    """RBF(C-alpha distance), direction i->j, and contact-only structural bias."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        rbf_centers: Optional[Sequence[float]] = None,
        rbf_sigma: float = 2.0,
        contact_cutoff: float = 8.0,
    ):
        super().__init__()
        centers = torch.tensor(
            list(rbf_centers or [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]),
            dtype=torch.float32,
        )
        self.register_buffer("rbf_centers", centers, persistent=False)
        self.rbf_sigma = float(rbf_sigma)
        self.contact_cutoff = float(contact_cutoff)

        feature_dim = centers.numel() + 3 + 1
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        coords = coords.to(dtype=torch.float32)
        diff = coords.unsqueeze(1) - coords.unsqueeze(2)
        distance = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)
        direction = diff / distance.unsqueeze(-1)

        centers = self.rbf_centers.to(device=coords.device, dtype=coords.dtype)
        rbf = torch.exp(-((distance.unsqueeze(-1) - centers) / self.rbf_sigma).pow(2))
        contact = (distance <= self.contact_cutoff).to(coords.dtype)

        features = torch.cat([rbf, direction, contact.unsqueeze(-1)], dim=-1)
        valid_coords = torch.isfinite(coords).all(dim=-1) & mask.bool()
        pair_mask = valid_coords.unsqueeze(2) & valid_coords.unsqueeze(1)
        features = features.masked_fill(~pair_mask.unsqueeze(-1), 0.0)

        bias = self.mlp(features).permute(0, 3, 1, 2)
        return bias.masked_fill(~pair_mask.unsqueeze(1), 0.0)


class V2StructureAwareMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        seq_distance_scale: float,
        seq_bias_scale: float,
        use_rope: bool = True,
        use_struct_bias: bool = True,
        struct_hidden_dim: int = 64,
        contact_cutoff: float = 8.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionEmbedding(self.head_dim) if use_rope else None
        self.seq_bias = FixedAbsoluteSequenceBias(
            distance_scale=seq_distance_scale,
            bias_scale=seq_bias_scale,
        )
        self.struct_bias = (
            PairwiseStructuralBiasV2(
                num_heads=num_heads,
                hidden_dim=struct_hidden_dim,
                dropout=dropout,
                contact_cutoff=contact_cutoff,
            )
            if use_struct_bias
            else None
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            q, k = self.rope(q, k, positions)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        scores = scores + self.seq_bias(positions).to(device=scores.device, dtype=scores.dtype)
        if self.struct_bias is not None:
            scores = scores + self.struct_bias(coords, mask).to(dtype=scores.dtype)

        key_padding_mask = ~mask.bool()
        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(key_padding_mask[:, None, :, None], 0.0)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.out_proj(out)
        return out.masked_fill(~mask.unsqueeze(-1), 0.0)


class V2StructureAwareEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        seq_distance_scale: float,
        seq_bias_scale: float,
        use_rope: bool,
        use_struct_bias: bool,
        struct_hidden_dim: int,
        contact_cutoff: float,
    ):
        super().__init__()
        self.self_attn = V2StructureAwareMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            seq_distance_scale=seq_distance_scale,
            seq_bias_scale=seq_bias_scale,
            use_rope=use_rope,
            use_struct_bias=use_struct_bias,
            struct_hidden_dim=struct_hidden_dim,
            contact_cutoff=contact_cutoff,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, mask=mask, positions=positions, coords=coords))
        x = x + self.dropout(self.ffn(self.norm2(x))).masked_fill(~mask.unsqueeze(-1), 0.0)
        return x.masked_fill(~mask.unsqueeze(-1), 0.0)


class V2StructureAwareTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        dim_feedforward: int = 384,
        dropout: float = 0.2,
        seq_distance_scale: float = 128.0,
        seq_bias_scale: float = 1.0,
        use_rope: bool = True,
        use_struct_bias: bool = True,
        struct_hidden_dim: int = 64,
        contact_cutoff: float = 8.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.config = {
            "input_dim": int(input_dim),
            "d_model": int(d_model),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
            "seq_distance_scale": float(seq_distance_scale),
            "seq_bias_scale": float(seq_bias_scale),
            "use_rope": bool(use_rope),
            "use_struct_bias": bool(use_struct_bias),
            "struct_hidden_dim": int(struct_hidden_dim),
            "contact_cutoff": float(contact_cutoff),
        }

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                V2StructureAwareEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    seq_distance_scale=seq_distance_scale,
                    seq_bias_scale=seq_bias_scale,
                    use_rope=use_rope,
                    use_struct_bias=use_struct_bias,
                    struct_hidden_dim=struct_hidden_dim,
                    contact_cutoff=contact_cutoff,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(self.input_norm(x))
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            h = layer(h, mask=mask, positions=positions, coords=pos)
        logits = self.classifier(self.output_norm(h)).squeeze(-1)
        return logits


class MaskedStandardFocalLoss(nn.Module):
    """Standard binary focal loss with alpha/gamma, without BCE pos_weight."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid_logits = logits[mask]
        valid_targets = targets[mask].float()
        if valid_logits.numel() == 0:
            return logits.sum() * 0.0

        ce_loss = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            reduction="none",
        )
        probs = torch.sigmoid(valid_logits)
        p_t = torch.where(valid_targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(
            valid_targets > 0.5,
            torch.full_like(valid_targets, self.alpha),
            torch.full_like(valid_targets, 1.0 - self.alpha),
        )
        focal = (1.0 - p_t).clamp_min(1e-6).pow(self.gamma)
        return (alpha_t * focal * ce_loss).mean()


def checkpoint_model_config(
    input_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    dim_feedforward: int,
    dropout: float,
    seq_distance_scale: float,
    seq_bias_scale: float,
    use_rope: bool,
    use_struct_bias: bool,
    struct_hidden_dim: int,
    contact_cutoff: float,
) -> Dict[str, object]:
    return {
        "input_dim": int(input_dim),
        "d_model": int(d_model),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "dim_feedforward": int(dim_feedforward),
        "dropout": float(dropout),
        "seq_distance_scale": float(seq_distance_scale),
        "seq_bias_scale": float(seq_bias_scale),
        "use_rope": bool(use_rope),
        "use_struct_bias": bool(use_struct_bias),
        "struct_hidden_dim": int(struct_hidden_dim),
        "contact_cutoff": float(contact_cutoff),
    }


def build_model_from_config(config: Dict[str, object]) -> V2StructureAwareTransformer:
    return V2StructureAwareTransformer(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        seq_distance_scale=float(config.get("seq_distance_scale", 128.0)),
        seq_bias_scale=float(config.get("seq_bias_scale", 1.0)),
        use_rope=bool(config.get("use_rope", True)),
        use_struct_bias=bool(config.get("use_struct_bias", True)),
        struct_hidden_dim=int(config.get("struct_hidden_dim", 64)),
        contact_cutoff=float(config.get("contact_cutoff", 8.0)),
    )


@torch.no_grad()
def predict_sequences(
    model: nn.Module,
    sequences: Sequence[ProteinGraphSequence],
    batch_size: int,
    max_len: int,
    stride: int,
    device: torch.device,
    num_workers: int = 0,
) -> Dict[str, np.ndarray]:
    model.eval()
    loader = make_window_loader(
        sequences=sequences,
        batch_size=batch_size,
        max_len=max_len,
        stride=stride,
        shuffle=False,
        num_workers=num_workers,
    )
    prob_sums = [torch.zeros(seq.length, dtype=torch.float32) for seq in sequences]
    counts = [torch.zeros(seq.length, dtype=torch.float32) for seq in sequences]

    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        logits = model(
            x=device_batch["x"],
            mask=device_batch["mask"],
            positions=device_batch["positions"],
            pos=device_batch["pos"],
        )
        probs = torch.sigmoid(logits).detach().cpu()
        mask = batch["mask"]

        for row, seq_idx in enumerate(batch["seq_indices"]):
            start = batch["starts"][row]
            end = batch["ends"][row]
            length = end - start
            row_mask = mask[row, :length].float()
            prob_sums[seq_idx][start:end] += probs[row, :length] * row_mask
            counts[seq_idx][start:end] += row_mask

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_positions: List[np.ndarray] = []
    all_sample_ids: List[np.ndarray] = []
    for seq_idx, seq in enumerate(sequences):
        probs = (prob_sums[seq_idx] / counts[seq_idx].clamp_min(1.0)).numpy()
        all_probs.append(probs)
        all_positions.append(seq.positions.cpu().numpy())
        all_sample_ids.append(np.asarray([seq.sample_id] * seq.length))
        if seq.y is not None:
            all_labels.append((seq.y.cpu().numpy() > 0.5).astype(np.int64))

    result = {
        "prob": np.concatenate(all_probs, axis=0),
        "position": np.concatenate(all_positions, axis=0),
        "sample_id": np.concatenate(all_sample_ids, axis=0),
    }
    if all_labels:
        result["label"] = np.concatenate(all_labels, axis=0)
    return result
