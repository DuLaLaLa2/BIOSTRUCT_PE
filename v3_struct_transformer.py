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


class ConvolutionalRandomFourierRelativeBias(nn.Module):
    """
    SPE-inspired relative sequence bias.

    Short-range offsets use an exact learnable table.
    Longer-range offsets are generated from fixed random Fourier bases and a
    small 1D convolutional network over the relative-distance axis.
    """

    def __init__(
        self,
        num_heads: int,
        local_relative_position: int = 128,
        num_random_features: int = 32,
        conv_hidden_dim: int = 64,
        conv_kernel_size: int = 17,
        max_global_distance: int = 768,
        fourier_scale: float = 1.0,
    ):
        super().__init__()
        if local_relative_position <= 0:
            raise ValueError("local_relative_position must be positive.")
        if num_random_features <= 0:
            raise ValueError("num_random_features must be positive.")
        if conv_hidden_dim <= 0:
            raise ValueError("conv_hidden_dim must be positive.")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be a positive odd integer.")

        self.num_heads = int(num_heads)
        self.local_relative_position = int(local_relative_position)
        self.num_random_features = int(num_random_features)
        self.max_global_distance = max(int(max_global_distance), self.local_relative_position + 1)

        self.local_bias = nn.Embedding(2 * self.local_relative_position + 1, self.num_heads)
        nn.init.zeros_(self.local_bias.weight)

        frequencies = torch.randn(self.num_random_features, dtype=torch.float32) * float(fourier_scale)
        phases = torch.rand(self.num_random_features, dtype=torch.float32) * (2.0 * math.pi)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.register_buffer("phases", phases, persistent=False)

        self.global_bias_net = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * self.num_random_features,
                out_channels=int(conv_hidden_dim),
                kernel_size=int(conv_kernel_size),
                padding=int(conv_kernel_size) // 2,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=int(conv_hidden_dim),
                out_channels=self.num_heads,
                kernel_size=int(conv_kernel_size),
                padding=int(conv_kernel_size) // 2,
            ),
        )
        nn.init.zeros_(self.global_bias_net[-1].weight)
        nn.init.zeros_(self.global_bias_net[-1].bias)

    def _random_fourier_features(self, offsets: torch.Tensor) -> torch.Tensor:
        normalized = offsets / max(float(self.max_global_distance), 1.0)
        projection = (
            2.0 * math.pi * normalized.unsqueeze(-1) * self.frequencies.unsqueeze(0)
            + self.phases.unsqueeze(0)
        )
        basis = torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)
        return basis * math.sqrt(1.0 / float(self.num_random_features))

    def _global_bias_table(self, max_abs_relative: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        offsets = torch.arange(
            -max_abs_relative,
            max_abs_relative + 1,
            device=device,
            dtype=torch.float32,
        )
        basis = self._random_fourier_features(offsets)
        conv_input = basis.transpose(0, 1).unsqueeze(0)
        bias = self.global_bias_net(conv_input)
        return bias.squeeze(0).transpose(0, 1).to(dtype=dtype)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        rel = positions.unsqueeze(2) - positions.unsqueeze(1)
        abs_rel = rel.abs()
        max_abs_relative = max(int(abs_rel.max().item()), self.local_relative_position)

        local_bucket = rel.clamp(-self.local_relative_position, self.local_relative_position)
        local_bucket = local_bucket + self.local_relative_position
        local_bias = self.local_bias(local_bucket)

        global_bias_table = self._global_bias_table(
            max_abs_relative=max_abs_relative,
            device=positions.device,
            dtype=local_bias.dtype,
        )
        global_bucket = (rel + max_abs_relative).long()
        global_bias = global_bias_table[global_bucket]

        use_local = abs_rel <= self.local_relative_position
        bias = torch.where(use_local.unsqueeze(-1), local_bias, global_bias)
        return bias.permute(0, 3, 1, 2)


class PairwiseStructuralBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
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

        feature_dim = centers.numel() + 3 + 2
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, coords: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
        bias = self.mlp(features).permute(0, 3, 1, 2)
        return bias.masked_fill(~pair_mask.unsqueeze(1), 0.0)


class StructureAwareMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        local_relative_position: int,
        num_random_features: int,
        spe_hidden_dim: int,
        spe_conv_kernel_size: int,
        max_global_distance: int,
        fourier_scale: float,
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
        self.seq_bias = ConvolutionalRandomFourierRelativeBias(
            num_heads=num_heads,
            local_relative_position=local_relative_position,
            num_random_features=num_random_features,
            conv_hidden_dim=spe_hidden_dim,
            conv_kernel_size=spe_conv_kernel_size,
            max_global_distance=max_global_distance,
            fourier_scale=fourier_scale,
        )
        self.struct_bias = (
            PairwiseStructuralBias(
                num_heads=num_heads,
                hidden_dim=struct_hidden_dim,
                dropout=dropout,
                contact_cutoff=contact_cutoff,
                max_sequence_separation=local_relative_position,
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


class StructureAwareEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        local_relative_position: int,
        num_random_features: int,
        spe_hidden_dim: int,
        spe_conv_kernel_size: int,
        max_global_distance: int,
        fourier_scale: float,
        use_rope: bool,
        use_struct_bias: bool,
        struct_hidden_dim: int,
        contact_cutoff: float,
    ):
        super().__init__()
        self.self_attn = StructureAwareMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            local_relative_position=local_relative_position,
            num_random_features=num_random_features,
            spe_hidden_dim=spe_hidden_dim,
            spe_conv_kernel_size=spe_conv_kernel_size,
            max_global_distance=max_global_distance,
            fourier_scale=fourier_scale,
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


class V3StructureAwareTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        local_relative_position: int = 128,
        num_random_features: int = 32,
        spe_hidden_dim: int = 64,
        spe_conv_kernel_size: int = 17,
        max_global_distance: int = 768,
        fourier_scale: float = 1.0,
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
            "local_relative_position": int(local_relative_position),
            "num_random_features": int(num_random_features),
            "spe_hidden_dim": int(spe_hidden_dim),
            "spe_conv_kernel_size": int(spe_conv_kernel_size),
            "max_global_distance": int(max_global_distance),
            "fourier_scale": float(fourier_scale),
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
                StructureAwareEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    local_relative_position=local_relative_position,
                    num_random_features=num_random_features,
                    spe_hidden_dim=spe_hidden_dim,
                    spe_conv_kernel_size=spe_conv_kernel_size,
                    max_global_distance=max_global_distance,
                    fourier_scale=fourier_scale,
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
    local_relative_position: int,
    num_random_features: int,
    spe_hidden_dim: int,
    spe_conv_kernel_size: int,
    max_global_distance: int,
    fourier_scale: float,
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
        "local_relative_position": int(local_relative_position),
        "num_random_features": int(num_random_features),
        "spe_hidden_dim": int(spe_hidden_dim),
        "spe_conv_kernel_size": int(spe_conv_kernel_size),
        "max_global_distance": int(max_global_distance),
        "fourier_scale": float(fourier_scale),
        "use_rope": bool(use_rope),
        "use_struct_bias": bool(use_struct_bias),
        "struct_hidden_dim": int(struct_hidden_dim),
        "contact_cutoff": float(contact_cutoff),
    }


def build_model_from_config(config: Dict[str, object]) -> V3StructureAwareTransformer:
    return V3StructureAwareTransformer(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        local_relative_position=int(config["local_relative_position"]),
        num_random_features=int(config.get("num_random_features", 32)),
        spe_hidden_dim=int(config.get("spe_hidden_dim", 64)),
        spe_conv_kernel_size=int(config.get("spe_conv_kernel_size", 17)),
        max_global_distance=int(config.get("max_global_distance", 768)),
        fourier_scale=float(config.get("fourier_scale", 1.0)),
        use_rope=bool(config.get("use_rope", True)),
        use_struct_bias=bool(config.get("use_struct_bias", True)),
        struct_hidden_dim=int(config.get("struct_hidden_dim", 64)),
        contact_cutoff=float(config.get("contact_cutoff", 8.0)),
    )
