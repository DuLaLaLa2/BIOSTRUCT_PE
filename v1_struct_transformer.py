import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch_geometric.data import Data

    torch.serialization.add_safe_globals([Data])
except Exception:
    Data = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ProteinGraphSequence:
    sample_id: str
    sequence: str
    x: torch.Tensor
    y: Optional[torch.Tensor]
    pos: torch.Tensor
    positions: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.x.size(0))

    @property
    def has_positive(self) -> int:
        if self.y is None:
            return 0
        return int((self.y > 0.5).any().item())


def _as_sample_id(graph, index: int) -> str:
    sample_id = getattr(graph, "sample_id", None)
    if sample_id is None:
        sample_id = getattr(graph, "protein_id", None)
    if sample_id is None:
        sample_id = f"protein_{index}"
    return str(sample_id)


def _as_sequence(graph, length: int) -> str:
    sequence = getattr(graph, "sequence", None)
    if sequence is None:
        return ""
    sequence = str(sequence)
    return sequence if len(sequence) == length else sequence[:length]


def load_graph_sequences(path: str, require_y: bool = True) -> List[ProteinGraphSequence]:
    """Load list-style PyG protein graphs with x/y/pos fields."""

    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    graphs = saved["data"] if isinstance(saved, dict) and "data" in saved else saved
    if not isinstance(graphs, (list, tuple)):
        raise ValueError(
            f"{path} is not a list of per-protein graphs. "
            "Use 495.pt/117.pt style files for v1 structural Transformer."
        )

    sequences: List[ProteinGraphSequence] = []
    for idx, graph in enumerate(graphs):
        if not hasattr(graph, "x") or graph.x is None:
            raise ValueError(f"Graph {idx} in {path} is missing x.")
        if not hasattr(graph, "pos") or graph.pos is None:
            raise ValueError(f"Graph {idx} in {path} is missing pos coordinates.")

        x = graph.x.detach().cpu().float()
        pos = graph.pos.detach().cpu().float()
        if x.ndim != 2:
            raise ValueError(f"Graph {idx} has invalid x shape {tuple(x.shape)}.")
        if pos.shape != (x.size(0), 3):
            raise ValueError(
                f"Graph {idx} has mismatched pos shape {tuple(pos.shape)} for x {tuple(x.shape)}."
            )

        if hasattr(graph, "y") and graph.y is not None:
            y = graph.y.detach().cpu().view(-1).float()
            if y.numel() != x.size(0):
                raise ValueError(
                    f"Graph {idx} has mismatched x/y lengths: x={x.size(0)}, y={y.numel()}."
                )
        elif require_y:
            raise ValueError(f"Graph {idx} in {path} is missing y.")
        else:
            y = None

        positions = torch.arange(x.size(0), dtype=torch.long)
        sequences.append(
            ProteinGraphSequence(
                sample_id=_as_sample_id(graph, idx),
                sequence=_as_sequence(graph, x.size(0)),
                x=x,
                y=y,
                pos=pos,
                positions=positions,
            )
        )

    if not sequences:
        raise ValueError(f"No protein graphs were loaded from {path}.")
    return sequences


def split_sequences_by_protein(
    sequences: Sequence[ProteinGraphSequence],
    train_ratio: float = 0.85,
    seed: int = 42,
) -> Tuple[List[ProteinGraphSequence], List[ProteinGraphSequence]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")

    indices = np.arange(len(sequences))
    labels = np.asarray([seq.has_positive for seq in sequences], dtype=np.int64)
    stratify = None
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) == 2 and counts.min() >= 2:
        stratify = labels

    try:
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except Exception:
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        split = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))
        train_idx, val_idx = shuffled[:split], shuffled[split:]

    train = [sequences[int(i)] for i in train_idx]
    val = [sequences[int(i)] for i in val_idx]
    train.sort(key=lambda item: item.sample_id)
    val.sort(key=lambda item: item.sample_id)
    return train, val


class ProteinWindowDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[ProteinGraphSequence],
        max_len: int = 512,
        stride: int = 384,
    ):
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        self.sequences = list(sequences)
        self.max_len = int(max_len)
        self.stride = int(stride)
        self.windows: List[Tuple[int, int, int]] = []

        for seq_idx, seq in enumerate(self.sequences):
            length = seq.length
            if length <= self.max_len:
                self.windows.append((seq_idx, 0, length))
                continue

            starts = list(range(0, max(1, length - self.max_len + 1), self.stride))
            last_start = max(0, length - self.max_len)
            if starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                end = min(length, start + self.max_len)
                self.windows.append((seq_idx, start, end))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        seq_idx, start, end = self.windows[index]
        seq = self.sequences[seq_idx]
        return {
            "seq_idx": seq_idx,
            "start": start,
            "end": end,
            "full_len": seq.length,
            "sample_id": seq.sample_id,
            "x": seq.x[start:end],
            "y": None if seq.y is None else seq.y[start:end],
            "pos": seq.pos[start:end],
            "positions": seq.positions[start:end],
        }


def collate_windows(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    batch_size = len(items)
    max_len = max(int(item["x"].size(0)) for item in items)
    input_dim = int(items[0]["x"].size(1))

    x = torch.zeros(batch_size, max_len, input_dim, dtype=torch.float32)
    y = torch.zeros(batch_size, max_len, dtype=torch.float32)
    pos = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    positions = torch.zeros(batch_size, max_len, dtype=torch.long)
    lengths = torch.tensor([int(item["full_len"]) for item in items], dtype=torch.long)

    starts: List[int] = []
    ends: List[int] = []
    seq_indices: List[int] = []
    sample_ids: List[str] = []

    for row, item in enumerate(items):
        length = int(item["x"].size(0))
        x[row, :length] = item["x"]
        pos[row, :length] = item["pos"]
        positions[row, :length] = item["positions"]
        mask[row, :length] = True
        if item["y"] is not None:
            y[row, :length] = item["y"].float()
        starts.append(int(item["start"]))
        ends.append(int(item["end"]))
        seq_indices.append(int(item["seq_idx"]))
        sample_ids.append(str(item["sample_id"]))

    return {
        "x": x,
        "y": y,
        "pos": pos,
        "mask": mask,
        "positions": positions,
        "lengths": lengths,
        "starts": starts,
        "ends": ends,
        "seq_indices": seq_indices,
        "sample_ids": sample_ids,
    }


def make_window_loader(
    sequences: Sequence[ProteinGraphSequence],
    batch_size: int,
    max_len: int,
    stride: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ProteinWindowDataset(sequences, max_len=max_len, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_windows,
        pin_memory=torch.cuda.is_available(),
    )


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.num_heads = int(num_heads)
        self.max_relative_position = int(max_relative_position)
        self.relative_bias = nn.Embedding(2 * self.max_relative_position + 1, self.num_heads)
        nn.init.zeros_(self.relative_bias.weight)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        rel = positions.unsqueeze(2) - positions.unsqueeze(1)
        rel = rel.clamp(-self.max_relative_position, self.max_relative_position)
        bucket = rel + self.max_relative_position
        return self.relative_bias(bucket).permute(0, 3, 1, 2)


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
        max_relative_position: int,
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
        self.seq_bias = RelativeSequenceBias(num_heads, max_relative_position=max_relative_position)
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
        struct_hidden_dim: int,
        contact_cutoff: float,
    ):
        super().__init__()
        self.self_attn = StructureAwareMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_relative_position=max_relative_position,
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


class V1StructureAwareTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        dim_feedforward: int = 384,
        dropout: float = 0.2,
        max_relative_position: int = 128,
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
                    max_relative_position=max_relative_position,
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


class MaskedFocalBCEWithLogitsLoss(nn.Module):
    """Standard binary focal loss with alpha/gamma, masked over real residues."""

    def __init__(self, pos_weight: float = 1.0, gamma: float = 2.0, alpha: float = 0.25):
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


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid_logits = logits[mask]
        valid_targets = targets[mask].float()
        if valid_logits.numel() == 0:
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=self.pos_weight.to(device=valid_logits.device),
        )


def compute_pos_weight(sequences: Sequence[ProteinGraphSequence]) -> float:
    labels = [seq.y for seq in sequences if seq.y is not None]
    if not labels:
        return 1.0
    y = torch.cat(labels)
    num_pos = int((y > 0.5).sum().item())
    num_neg = int((y <= 0.5).sum().item())
    if num_pos == 0:
        raise ValueError("Training data has no positive residues.")
    return float(num_neg) / max(float(num_pos), 1.0)


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_prob))


def find_best_threshold_by_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import matthews_corrcoef

    best_thr = 0.5
    best_mcc = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= threshold).astype(np.int64)
        mcc = float(matthews_corrcoef(y_true, pred))
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(threshold)
    return best_thr, best_mcc


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "AUC": safe_auc(y_true, y_prob),
        "AP": safe_ap(y_true, y_prob),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": calculate_specificity(y_true, y_pred),
    }


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    for key in ["x", "y", "pos", "mask", "positions", "lengths"]:
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


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
    }


def build_model_from_config(config: Dict[str, object]) -> V1StructureAwareTransformer:
    return V1StructureAwareTransformer(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_relative_position=int(config["max_relative_position"]),
        use_rope=bool(config.get("use_rope", True)),
        use_struct_bias=bool(config.get("use_struct_bias", True)),
        struct_hidden_dim=int(config.get("struct_hidden_dim", 64)),
        contact_cutoff=float(config.get("contact_cutoff", 8.0)),
    )
