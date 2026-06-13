import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def load_pyg_data(path: str, require_y: bool = True):
    saved = torch.load(path, weights_only=False)
    data = saved["data"] if isinstance(saved, dict) and "data" in saved else saved

    if not hasattr(data, "x") or data.x is None:
        raise ValueError(f"{path} is missing data.x")

    data.x = data.x.float()

    if hasattr(data, "y") and data.y is not None:
        data.y = data.y.view(-1).long()
        if data.x.size(0) != data.y.size(0):
            raise ValueError(
                f"{path} has mismatched x/y node counts: "
                f"x={data.x.size(0)}, y={data.y.size(0)}"
            )
    elif require_y:
        raise ValueError(f"{path} is missing data.y")

    return data.cpu()


@dataclass
class ProteinSequence:
    sample_id: str
    x: torch.Tensor
    y: Optional[torch.Tensor]
    positions: torch.Tensor
    node_indices: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.x.size(0))

    @property
    def has_positive(self) -> int:
        if self.y is None:
            return 0
        return int((self.y == 1).any().item())


def _metadata_lists(data) -> Tuple[List[str], List[int]]:
    residue_sample_ids = getattr(data, "residue_sample_ids", None)
    residue_positions = getattr(data, "residue_positions", None)

    if residue_sample_ids is None:
        residue_sample_ids = ["protein_0"] * data.x.size(0)
    if residue_positions is None:
        residue_positions = list(range(data.x.size(0)))

    usable = min(len(residue_sample_ids), len(residue_positions), data.x.size(0))
    if usable == 0:
        raise ValueError("No residue metadata is available.")

    if usable < data.x.size(0):
        print(
            "Warning: residue metadata is shorter than data.x. "
            f"Using the first {usable} nodes and ignoring {data.x.size(0) - usable} nodes. "
            "This usually means AWGAN synthetic nodes are present without sequence context."
        )

    ids = [str(v) for v in residue_sample_ids[:usable]]
    positions = [int(v) for v in residue_positions[:usable]]
    return ids, positions


def build_protein_sequences(data, mask: Optional[torch.Tensor] = None) -> List[ProteinSequence]:
    sample_ids, residue_positions = _metadata_lists(data)
    usable = len(sample_ids)

    if mask is not None:
        mask = mask[:usable].bool().cpu()
    else:
        mask = torch.ones(usable, dtype=torch.bool)

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, sample_id in enumerate(sample_ids):
        if bool(mask[idx]):
            groups[sample_id].append(idx)

    sequences: List[ProteinSequence] = []
    for sample_id, indices in groups.items():
        indices.sort(key=lambda i: (residue_positions[i], i))
        node_indices = torch.tensor(indices, dtype=torch.long)
        pos = torch.tensor([residue_positions[i] for i in indices], dtype=torch.long)
        x = data.x[node_indices].float()
        y = data.y[node_indices].long() if hasattr(data, "y") and data.y is not None else None
        sequences.append(
            ProteinSequence(
                sample_id=sample_id,
                x=x,
                y=y,
                positions=pos,
                node_indices=node_indices,
            )
        )

    sequences.sort(key=lambda item: item.sample_id)
    if not sequences:
        raise ValueError("No protein sequences were built from the provided data.")
    return sequences


def split_sequences_by_protein(
    sequences: Sequence[ProteinSequence],
    train_ratio: float = 0.85,
    seed: int = 42,
) -> Tuple[List[ProteinSequence], List[ProteinSequence]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")

    indices = np.arange(len(sequences))
    labels = np.array([seq.has_positive for seq in sequences], dtype=np.int64)

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
        sequences: Sequence[ProteinSequence],
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
            "positions": seq.positions[start:end],
            "node_indices": seq.node_indices[start:end],
        }


def collate_windows(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    batch_size = len(items)
    max_len = max(int(item["x"].size(0)) for item in items)
    input_dim = int(items[0]["x"].size(1))

    x = torch.zeros(batch_size, max_len, input_dim, dtype=torch.float32)
    y = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    positions = torch.zeros(batch_size, max_len, dtype=torch.long)
    node_indices = torch.full((batch_size, max_len), -1, dtype=torch.long)
    lengths = torch.tensor([int(item["full_len"]) for item in items], dtype=torch.long)

    starts: List[int] = []
    ends: List[int] = []
    seq_indices: List[int] = []
    sample_ids: List[str] = []

    for row, item in enumerate(items):
        length = int(item["x"].size(0))
        x[row, :length] = item["x"]
        if item["y"] is not None:
            y[row, :length] = item["y"].float()
        positions[row, :length] = item["positions"]
        node_indices[row, :length] = item["node_indices"]
        mask[row, :length] = True
        starts.append(int(item["start"]))
        ends.append(int(item["end"]))
        seq_indices.append(int(item["seq_idx"]))
        sample_ids.append(str(item["sample_id"]))

    return {
        "x": x,
        "y": y,
        "mask": mask,
        "positions": positions,
        "lengths": lengths,
        "node_indices": node_indices,
        "starts": starts,
        "ends": ends,
        "seq_indices": seq_indices,
        "sample_ids": sample_ids,
    }


def make_window_loader(
    sequences: Sequence[ProteinSequence],
    batch_size: int,
    max_len: int,
    stride: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ProteinWindowDataset(sequences, max_len=max_len, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_windows,
    )


class TimeAbsolutePositionEncoding(nn.Module):
    """tAPE: length-aware absolute PE for variable-length protein sequences."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        positions: torch.Tensor,
        lengths: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, seq_len = positions.shape
        dim = self.d_model
        device = positions.device

        pos = positions.to(device=device, dtype=torch.float32)
        lengths_f = lengths.to(device=device, dtype=torch.float32).clamp_min(1.0)
        lengths_f = lengths_f.view(batch, 1)

        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(dim, 1))
        )
        scaled_pos = pos.unsqueeze(-1) * (float(dim) / lengths_f).unsqueeze(-1)

        pe = torch.zeros(batch, seq_len, dim, device=device, dtype=dtype)
        pe[..., 0::2] = torch.sin(scaled_pos * div_term).to(dtype=dtype)
        if dim > 1:
            pe[..., 1::2] = torch.cos(scaled_pos * div_term[: pe[..., 1::2].size(-1)]).to(
                dtype=dtype
            )

        return self.dropout(pe.to(dtype=dtype))


class RelativeResidueBias(nn.Module):
    """Signed clipped relative-distance bias with a fixed ALiBi-like near-residue prior."""

    def __init__(self, num_heads: int, max_relative_position: int = 64):
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_relative_position = int(max_relative_position)
        self.relative_bias = nn.Embedding(2 * self.max_relative_position + 1, self.num_heads)
        nn.init.zeros_(self.relative_bias.weight)

        slopes = torch.pow(2.0, -torch.linspace(0.0, 3.0, steps=self.num_heads))
        self.register_buffer("alibi_slopes", slopes.view(1, self.num_heads, 1, 1))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        device = positions.device
        rel = positions.unsqueeze(2) - positions.unsqueeze(1)
        rel_clipped = rel.clamp(-self.max_relative_position, self.max_relative_position)
        bucket = rel_clipped + self.max_relative_position

        learned = self.relative_bias(bucket).permute(0, 3, 1, 2)
        distance = rel.abs().float().unsqueeze(1)
        alibi = -distance / max(float(self.max_relative_position), 1.0)
        alibi = alibi * self.alibi_slopes.to(device=device)
        return learned + alibi.to(dtype=learned.dtype)


class TUPEMultiheadAttention(nn.Module):
    """Untied positional attention: content-content plus tAPE position-position scores."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads

        self.q_content = nn.Linear(d_model, d_model)
        self.k_content = nn.Linear(d_model, d_model)
        self.v_content = nn.Linear(d_model, d_model)
        self.q_position = nn.Linear(d_model, d_model)
        self.k_position = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        position_encoding: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q_c = self._split_heads(self.q_content(x))
        k_c = self._split_heads(self.k_content(x))
        v_c = self._split_heads(self.v_content(x))

        q_p = self._split_heads(self.q_position(position_encoding))
        k_p = self._split_heads(self.k_position(position_encoding))

        content_scores = torch.matmul(q_c, k_c.transpose(-2, -1))
        position_scores = torch.matmul(q_p, k_p.transpose(-2, -1))
        scores = (content_scores + position_scores) / math.sqrt(float(self.head_dim))

        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(key_padding_mask[:, None, :, None], 0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_c)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class TUPETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = TUPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
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
        position_encoding: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out = self.self_attn(
            h,
            position_encoding=position_encoding,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ProteinResidueTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        dim_feedforward: int = 384,
        dropout: float = 0.2,
        max_relative_position: int = 64,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_encoding = TimeAbsolutePositionEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                TUPETransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
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
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(self.input_norm(x))
        pos = self.position_encoding(
            positions=positions.to(device=x.device),
            lengths=lengths.to(device=x.device),
            dtype=h.dtype,
        )

        key_padding_mask = ~mask.bool()

        for layer in self.layers:
            h = layer(h, position_encoding=pos, key_padding_mask=key_padding_mask)

        logits = self.classifier(self.output_norm(h)).squeeze(-1)
        return logits


class MaskedFocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, gamma: float = 1.5):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid_logits = logits[mask]
        valid_targets = targets[mask].float()
        if valid_logits.numel() == 0:
            return logits.sum() * 0.0

        bce = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=self.pos_weight.to(device=valid_logits.device),
            reduction="none",
        )
        probs = torch.sigmoid(valid_logits)
        pt = torch.where(valid_targets > 0.5, probs, 1.0 - probs)
        focal = (1.0 - pt).clamp_min(1e-6).pow(self.gamma)
        return (focal * bce).mean()


def compute_pos_weight(sequences: Sequence[ProteinSequence]) -> float:
    labels = [seq.y for seq in sequences if seq.y is not None]
    if not labels:
        return 1.0
    y = torch.cat(labels)
    num_pos = int((y == 1).sum().item())
    num_neg = int((y == 0).sum().item())
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
    for key in ["x", "y", "mask", "positions", "lengths", "node_indices"]:
        moved[key] = batch[key].to(device)
    return moved


@torch.no_grad()
def predict_sequences(
    model: nn.Module,
    sequences: Sequence[ProteinSequence],
    batch_size: int,
    max_len: int,
    stride: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    loader = make_window_loader(
        sequences=sequences,
        batch_size=batch_size,
        max_len=max_len,
        stride=stride,
        shuffle=False,
    )

    prob_sums = [torch.zeros(seq.length, dtype=torch.float32) for seq in sequences]
    counts = [torch.zeros(seq.length, dtype=torch.float32) for seq in sequences]

    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        logits = model(
            x=device_batch["x"],
            mask=device_batch["mask"],
            positions=device_batch["positions"],
            lengths=device_batch["lengths"],
        )
        probs = torch.sigmoid(logits).detach().cpu()
        mask = batch["mask"]

        for row, seq_idx in enumerate(batch["seq_indices"]):
            start = batch["starts"][row]
            end = batch["ends"][row]
            length = end - start
            row_probs = probs[row, :length]
            row_mask = mask[row, :length].float()
            prob_sums[seq_idx][start:end] += row_probs * row_mask
            counts[seq_idx][start:end] += row_mask

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_node_indices: List[np.ndarray] = []
    all_positions: List[np.ndarray] = []
    all_sample_ids: List[np.ndarray] = []

    for seq_idx, seq in enumerate(sequences):
        denom = counts[seq_idx].clamp_min(1.0)
        probs = (prob_sums[seq_idx] / denom).numpy()
        all_probs.append(probs)
        if seq.y is not None:
            all_labels.append(seq.y.cpu().numpy())
        all_node_indices.append(seq.node_indices.cpu().numpy())
        all_positions.append(seq.positions.cpu().numpy())
        all_sample_ids.append(np.asarray([seq.sample_id] * seq.length))

    result = {
        "prob": np.concatenate(all_probs, axis=0),
        "node_index": np.concatenate(all_node_indices, axis=0),
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
) -> Dict[str, object]:
    return {
        "input_dim": int(input_dim),
        "d_model": int(d_model),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "dim_feedforward": int(dim_feedforward),
        "dropout": float(dropout),
        "max_relative_position": int(max_relative_position),
    }


def build_model_from_config(config: Dict[str, object]) -> ProteinResidueTransformer:
    return ProteinResidueTransformer(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_relative_position=int(config["max_relative_position"]),
    )
