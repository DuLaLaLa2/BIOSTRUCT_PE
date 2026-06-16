from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v3_data import ProteinGraphSequence, make_window_loader, move_batch_to_device


class MaskedFocalBCEWithLogitsLoss(nn.Module):
    """Standard binary focal loss with alpha/gamma, masked over real residues."""

    def __init__(self, pos_weight: float = 1.0, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid_logits = logits[mask]
        valid_targets = targets[mask].float()
        if valid_logits.numel() == 0:
            return logits.sum() * 0.0
        ce_loss = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=self.pos_weight.to(device=valid_logits.device),
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
