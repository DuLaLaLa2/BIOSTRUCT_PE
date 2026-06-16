import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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


def _as_positions(graph, length: int) -> torch.Tensor:
    residue_positions = getattr(graph, "residue_positions", None)
    if residue_positions is None:
        return torch.arange(length, dtype=torch.long)
    if torch.is_tensor(residue_positions):
        values = residue_positions.detach().cpu().view(-1).long()
    else:
        values = torch.tensor(list(residue_positions), dtype=torch.long)
    if values.numel() < length:
        return torch.arange(length, dtype=torch.long)
    return values[:length]


def load_graph_sequences(path: str, require_y: bool = True) -> List[ProteinGraphSequence]:
    """Load per-protein PyG graphs with x/y/pos fields."""

    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    graphs = saved["data"] if isinstance(saved, dict) and "data" in saved else saved
    if not isinstance(graphs, (list, tuple)):
        raise ValueError(
            f"{path} is not a list of per-protein graphs. "
            "Use 495.pt/117.pt style files for the structural Transformer."
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

        positions = _as_positions(graph, x.size(0))
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
        max_len: int = 768,
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


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    for key in ["x", "y", "pos", "mask", "positions", "lengths"]:
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved
