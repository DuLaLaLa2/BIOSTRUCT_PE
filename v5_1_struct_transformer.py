import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.rotary_dim = int(head_dim - (head_dim % 2))
        if self.rotary_dim <= 0:
            raise ValueError("head_dim must contain at least two rotary dimensions.")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = positions.to(device=q.device, dtype=torch.float32)
        freqs = torch.einsum("bt,d->btd", pos, self.inv_freq.to(device=q.device))
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        def apply(x: torch.Tensor) -> torch.Tensor:
            x_rot = x[..., : self.rotary_dim]
            x_pass = x[..., self.rotary_dim :]
            rotated = (x_rot * cos) + (rotate_half(x_rot) * sin)
            return torch.cat((rotated, x_pass), dim=-1) if x_pass.numel() else rotated

        return apply(q), apply(k)


class RelativeSequenceBias(nn.Module):
    def __init__(self, num_heads: int, max_relative_position: int = 128):
        super().__init__()
        if max_relative_position <= 0:
            raise ValueError("max_relative_position must be positive.")
        self.num_heads = int(num_heads)
        self.max_relative_position = int(max_relative_position)
        self.bias = nn.Embedding(2 * self.max_relative_position + 1, self.num_heads)
        nn.init.zeros_(self.bias.weight)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        rel = positions.unsqueeze(2) - positions.unsqueeze(1)
        rel = rel.clamp(-self.max_relative_position, self.max_relative_position)
        rel = rel + self.max_relative_position
        bias = self.bias(rel)
        return bias.permute(0, 3, 1, 2)


class PairwiseStructureFeatures(nn.Module):
    def __init__(
        self,
        rbf_centers: Optional[Sequence[float]] = None,
        rbf_sigma: float = 2.0,
        contact_cutoff: float = 8.0,
        max_sequence_separation: int = 128,
    ):
        super().__init__()
        centers = torch.tensor(
            list(rbf_centers or [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]),
            dtype=torch.float32,
        )
        self.register_buffer("rbf_centers", centers, persistent=False)
        self.rbf_sigma = float(rbf_sigma)
        self.contact_cutoff = float(contact_cutoff)
        self.max_sequence_separation = int(max_sequence_separation)
        self.feature_dim = centers.numel() + 3 + 2

    def forward(
        self,
        coords: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = coords.to(dtype=torch.float32)
        diff = coords.unsqueeze(1) - coords.unsqueeze(2)
        distance = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)
        direction = diff / distance.unsqueeze(-1)

        centers = self.rbf_centers.to(device=coords.device, dtype=coords.dtype)
        rbf = torch.exp(-((distance.unsqueeze(-1) - centers) / self.rbf_sigma).pow(2))
        seq_sep = (positions.unsqueeze(2) - positions.unsqueeze(1)).abs().to(coords.dtype)
        seq_sep = (seq_sep / max(float(self.max_sequence_separation), 1.0)).clamp(max=1.0)
        contact = (distance <= self.contact_cutoff).to(coords.dtype)

        features = torch.cat(
            [
                rbf,
                direction,
                seq_sep.unsqueeze(-1),
                contact.unsqueeze(-1),
            ],
            dim=-1,
        )

        valid_coords = torch.isfinite(coords).all(dim=-1) & mask.bool()
        pair_mask = valid_coords.unsqueeze(2) & valid_coords.unsqueeze(1)
        features = features.masked_fill(~pair_mask.unsqueeze(-1), 0.0)
        return features, pair_mask, distance


class StructuralPositionEncoding(nn.Module):
    """
    Residue-level conditional positional encoding from 3D neighbors.

    It aggregates hidden states and geometry features from contact/top-k spatial
    neighbors, then returns a small additive position signal for each residue.
    """

    def __init__(
        self,
        d_model: int,
        feature_dim: int,
        hidden_dim: int = 128,
        top_k: int = 16,
        contact_cutoff: float = 10.0,
        dropout: float = 0.1,
        initial_scale: float = 0.05,
    ):
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.top_k = int(top_k)
        self.contact_cutoff = float(contact_cutoff)

        self.content_proj = nn.Linear(d_model, d_model)
        self.geom_score = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.geom_proj = nn.Linear(feature_dim, d_model)
        self.out = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def _neighbor_mask(
        self,
        pair_mask: torch.Tensor,
        distance: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = distance.shape
        eye = torch.eye(seq_len, device=distance.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = pair_mask & ~eye

        contact_mask = (distance <= self.contact_cutoff) & valid_pair
        masked_distance = distance.masked_fill(~valid_pair, torch.inf)
        k = min(self.top_k, seq_len)
        _, topk_index = torch.topk(masked_distance, k=k, dim=-1, largest=False)
        topk_mask = torch.zeros_like(valid_pair)
        topk_mask.scatter_(dim=-1, index=topk_index, value=True)
        topk_mask = topk_mask & torch.isfinite(masked_distance)

        neighbor_mask = contact_mask | topk_mask
        has_neighbor = neighbor_mask.any(dim=-1, keepdim=True)
        self_mask = eye & mask.unsqueeze(1) & mask.unsqueeze(2)
        return torch.where(has_neighbor, neighbor_mask, self_mask)

    def forward(
        self,
        h: torch.Tensor,
        pair_features: torch.Tensor,
        pair_mask: torch.Tensor,
        distance: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_mask = self._neighbor_mask(pair_mask, distance, mask)
        logits = self.geom_score(pair_features).squeeze(-1)
        logits = logits.masked_fill(~neighbor_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1).masked_fill(~neighbor_mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        content_summary = torch.matmul(weights, self.content_proj(h))
        geom_summary = torch.einsum("bij,bijf->bif", weights, pair_features)
        geom_summary = self.geom_proj(geom_summary)
        encoding = self.out(torch.cat([content_summary, geom_summary], dim=-1))
        return (self.scale * encoding).masked_fill(~mask.unsqueeze(-1), 0.0)


class PairwiseStructuralBias(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        bias = self.mlp(features).permute(0, 3, 1, 2)
        return bias.masked_fill(~pair_mask.unsqueeze(1), 0.0)


class StructureAwareMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        max_relative_position: int,
        use_rope: bool = True,
        use_struct_bias: bool = True,
        struct_feature_dim: int = 17,
        struct_hidden_dim: int = 64,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.use_rope = bool(use_rope)
        self.use_struct_bias = bool(use_struct_bias)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionEmbedding(self.head_dim) if self.use_rope else None
        self.seq_bias = RelativeSequenceBias(
            num_heads=num_heads,
            max_relative_position=max_relative_position,
        )
        self.struct_bias = (
            PairwiseStructuralBias(
                feature_dim=struct_feature_dim,
                num_heads=num_heads,
                hidden_dim=struct_hidden_dim,
                dropout=dropout,
            )
            if self.use_struct_bias
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
        pair_features: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            q, k = self.rope(q, k, positions)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        scores = scores + self.seq_bias(positions).to(dtype=scores.dtype)
        if self.struct_bias is not None:
            scores = scores + self.struct_bias(pair_features, pair_mask).to(dtype=scores.dtype)

        key_padding_mask = ~mask.bool()
        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(key_padding_mask[:, None, :, None], 0.0)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        out = self.out_proj(out)
        return out.masked_fill(~mask.unsqueeze(-1), 0.0)


class StructureAwareEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        max_relative_position: int,
        use_rope: bool,
        use_struct_bias: bool,
        struct_feature_dim: int,
        struct_hidden_dim: int,
    ):
        super().__init__()
        self.self_attn = StructureAwareMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_relative_position=max_relative_position,
            use_rope=use_rope,
            use_struct_bias=use_struct_bias,
            struct_feature_dim=struct_feature_dim,
            struct_hidden_dim=struct_hidden_dim,
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
        pair_features: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(
            self.self_attn(
                h,
                mask=mask,
                positions=positions,
                pair_features=pair_features,
                pair_mask=pair_mask,
            )
        )
        ffn_out = self.dropout(self.ffn(self.norm2(x))).masked_fill(~mask.unsqueeze(-1), 0.0)
        x = x + ffn_out
        return x.masked_fill(~mask.unsqueeze(-1), 0.0)


class V5_1StructPEGTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_relative_position: int = 128,
        use_rope: bool = True,
        use_struct_bias: bool = True,
        struct_hidden_dim: int = 64,
        contact_cutoff: float = 8.0,
        use_struct_peg: bool = True,
        struct_peg_hidden_dim: int = 128,
        struct_peg_top_k: int = 16,
        struct_peg_cutoff: float = 10.0,
        struct_peg_scale: float = 0.05,
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
            "max_relative_position": int(max_relative_position),
            "use_rope": bool(use_rope),
            "use_struct_bias": bool(use_struct_bias),
            "struct_hidden_dim": int(struct_hidden_dim),
            "contact_cutoff": float(contact_cutoff),
            "use_struct_peg": bool(use_struct_peg),
            "struct_peg_hidden_dim": int(struct_peg_hidden_dim),
            "struct_peg_top_k": int(struct_peg_top_k),
            "struct_peg_cutoff": float(struct_peg_cutoff),
            "struct_peg_scale": float(struct_peg_scale),
        }

        self.pair_features = PairwiseStructureFeatures(
            contact_cutoff=contact_cutoff,
            max_sequence_separation=max_relative_position,
        )
        self.use_struct_peg = bool(use_struct_peg)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.struct_peg = (
            StructuralPositionEncoding(
                d_model=d_model,
                feature_dim=self.pair_features.feature_dim,
                hidden_dim=struct_peg_hidden_dim,
                top_k=struct_peg_top_k,
                contact_cutoff=struct_peg_cutoff,
                dropout=dropout,
                initial_scale=struct_peg_scale,
            )
            if self.use_struct_peg
            else None
        )
        self.layers = nn.ModuleList(
            [
                StructureAwareEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_relative_position=max_relative_position,
                    use_rope=use_rope,
                    use_struct_bias=use_struct_bias,
                    struct_feature_dim=self.pair_features.feature_dim,
                    struct_hidden_dim=struct_hidden_dim,
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

        pair_features, pair_mask, distance = self.pair_features(pos, positions, mask)
        if self.struct_peg is not None:
            h = h + self.struct_peg(
                h=h,
                pair_features=pair_features,
                pair_mask=pair_mask,
                distance=distance,
                mask=mask,
            )
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)

        for layer in self.layers:
            h = layer(
                h,
                mask=mask,
                positions=positions,
                pair_features=pair_features,
                pair_mask=pair_mask,
            )
        logits = self.classifier(self.output_norm(h)).squeeze(-1)
        return logits


def checkpoint_model_config(
    input_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    dim_feedforward: int,
    dropout: float,
    max_relative_position: int,
    use_rope: bool,
    use_struct_bias: bool,
    struct_hidden_dim: int,
    contact_cutoff: float,
    use_struct_peg: bool,
    struct_peg_hidden_dim: int,
    struct_peg_top_k: int,
    struct_peg_cutoff: float,
    struct_peg_scale: float,
) -> Dict[str, object]:
    return {
        "input_dim": int(input_dim),
        "d_model": int(d_model),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "dim_feedforward": int(dim_feedforward),
        "dropout": float(dropout),
        "max_relative_position": int(max_relative_position),
        "use_rope": bool(use_rope),
        "use_struct_bias": bool(use_struct_bias),
        "struct_hidden_dim": int(struct_hidden_dim),
        "contact_cutoff": float(contact_cutoff),
        "use_struct_peg": bool(use_struct_peg),
        "struct_peg_hidden_dim": int(struct_peg_hidden_dim),
        "struct_peg_top_k": int(struct_peg_top_k),
        "struct_peg_cutoff": float(struct_peg_cutoff),
        "struct_peg_scale": float(struct_peg_scale),
    }


def build_model_from_config(config: Dict[str, object]) -> V5_1StructPEGTransformer:
    return V5_1StructPEGTransformer(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_relative_position=int(config.get("max_relative_position", 128)),
        use_rope=bool(config.get("use_rope", True)),
        use_struct_bias=bool(config.get("use_struct_bias", True)),
        struct_hidden_dim=int(config.get("struct_hidden_dim", 64)),
        contact_cutoff=float(config.get("contact_cutoff", 8.0)),
        use_struct_peg=bool(config.get("use_struct_peg", True)),
        struct_peg_hidden_dim=int(config.get("struct_peg_hidden_dim", 128)),
        struct_peg_top_k=int(config.get("struct_peg_top_k", 16)),
        struct_peg_cutoff=float(config.get("struct_peg_cutoff", 10.0)),
        struct_peg_scale=float(config.get("struct_peg_scale", 0.05)),
    )
