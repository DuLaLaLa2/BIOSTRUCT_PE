import math
from typing import Dict

import torch
import torch.nn as nn

from v1_struct_transformer import (
    MaskedBCEWithLogitsLoss,
    MaskedFocalBCEWithLogitsLoss,
    PairwiseStructuralBias,
    ProteinGraphSequence,
    RotaryPositionEmbedding,
    calculate_specificity,
    collate_windows,
    compute_pos_weight,
    evaluate_metrics,
    find_best_threshold_by_mcc,
    load_graph_sequences,
    make_window_loader,
    move_batch_to_device,
    predict_sequences,
    safe_ap,
    safe_auc,
    set_seed,
    split_sequences_by_protein,
)


class SPEConv1DSequenceBias(nn.Module):
    """Sequential position bias using Conv1D over signed relative offsets r=i-j."""

    def __init__(
        self,
        num_heads: int,
        max_relative_position: int = 128,
        kernel_size: int = 7,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.num_heads = int(num_heads)
        self.max_relative_position = int(max_relative_position)
        self.kernel_size = int(kernel_size)

        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, num_heads, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        rel = positions.unsqueeze(2) - positions.unsqueeze(1)
        rel = rel.clamp(-self.max_relative_position, self.max_relative_position)
        rel = rel.float() / max(float(self.max_relative_position), 1.0)

        batch, seq_len, _ = rel.shape
        # Treat each query residue row r_i,* as a 1D relative-position signal over key residues.
        signal = rel.reshape(batch * seq_len, 1, seq_len)
        bias = self.net(signal)
        bias = bias.view(batch, seq_len, self.num_heads, seq_len)
        return bias.permute(0, 2, 1, 3)


class V1_1StructureAwareMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        max_relative_position: int,
        spe_kernel_size: int = 7,
        spe_hidden_dim: int = 32,
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
        self.use_rope = bool(use_rope)
        self.use_struct_bias = bool(use_struct_bias)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionEmbedding(self.head_dim) if self.use_rope else None
        self.seq_bias = SPEConv1DSequenceBias(
            num_heads=num_heads,
            max_relative_position=max_relative_position,
            kernel_size=spe_kernel_size,
            hidden_dim=spe_hidden_dim,
            dropout=dropout,
        )
        self.struct_bias = (
            PairwiseStructuralBias(
                num_heads=num_heads,
                hidden_dim=struct_hidden_dim,
                dropout=dropout,
                contact_cutoff=contact_cutoff,
                max_sequence_separation=max_relative_position,
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
        coords: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            q, k = self.rope(q, k, positions)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        scores = scores + self.seq_bias(positions).to(dtype=scores.dtype)
        if self.struct_bias is not None:
            scores = scores + self.struct_bias(coords, positions, mask).to(dtype=scores.dtype)

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


class V1_1StructureAwareEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        max_relative_position: int,
        spe_kernel_size: int,
        spe_hidden_dim: int,
        use_rope: bool,
        use_struct_bias: bool,
        struct_hidden_dim: int,
        contact_cutoff: float,
    ):
        super().__init__()
        self.self_attn = V1_1StructureAwareMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_relative_position=max_relative_position,
            spe_kernel_size=spe_kernel_size,
            spe_hidden_dim=spe_hidden_dim,
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


class V1_1StructureAwareTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_relative_position: int = 128,
        spe_kernel_size: int = 7,
        spe_hidden_dim: int = 32,
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
            "max_relative_position": int(max_relative_position),
            "spe_kernel_size": int(spe_kernel_size),
            "spe_hidden_dim": int(spe_hidden_dim),
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
                V1_1StructureAwareEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_relative_position=max_relative_position,
                    spe_kernel_size=spe_kernel_size,
                    spe_hidden_dim=spe_hidden_dim,
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


def checkpoint_model_config(
    input_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    dim_feedforward: int,
    dropout: float,
    max_relative_position: int,
    spe_kernel_size: int,
    spe_hidden_dim: int,
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
        "max_relative_position": int(max_relative_position),
        "spe_kernel_size": int(spe_kernel_size),
        "spe_hidden_dim": int(spe_hidden_dim),
        "use_rope": bool(use_rope),
        "use_struct_bias": bool(use_struct_bias),
        "struct_hidden_dim": int(struct_hidden_dim),
        "contact_cutoff": float(contact_cutoff),
    }


def build_model_from_config(config: Dict[str, object]) -> V1_1StructureAwareTransformer:
    return V1_1StructureAwareTransformer(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_relative_position=int(config["max_relative_position"]),
        spe_kernel_size=int(config.get("spe_kernel_size", 7)),
        spe_hidden_dim=int(config.get("spe_hidden_dim", 32)),
        use_rope=bool(config.get("use_rope", True)),
        use_struct_bias=bool(config.get("use_struct_bias", True)),
        struct_hidden_dim=int(config.get("struct_hidden_dim", 64)),
        contact_cutoff=float(config.get("contact_cutoff", 8.0)),
    )
