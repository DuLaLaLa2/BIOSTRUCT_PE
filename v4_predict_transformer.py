import os

import numpy as np
import torch

from v4_data import load_graph_sequences
from v4_metrics import evaluate_metrics, predict_sequences
from v4_struct_transformer import build_model_from_config

# Edit these values in VSCode, then click Run.
MODEL_PATH = "v4_best_struct_transformer.pt"
DATA_PATH = "117.pt"
OUTPUT_PATH = "v4_prediction_results.npz"

BATCH_SIZE = 2
NUM_WORKERS = 0
USE_CUDA = True

THRESHOLD_OVERRIDE = None
MAX_LEN_OVERRIDE = None
STRIDE_OVERRIDE = None


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("MODEL_PATH is not a valid v4 checkpoint.")

    model = build_model_from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    threshold = (
        float(THRESHOLD_OVERRIDE)
        if THRESHOLD_OVERRIDE is not None
        else float(checkpoint.get("threshold", 0.5))
    )
    max_len = int(MAX_LEN_OVERRIDE or checkpoint.get("max_len", 768))
    stride = int(STRIDE_OVERRIDE or checkpoint.get("stride", 384))

    sequences = load_graph_sequences(DATA_PATH, require_y=False)
    result = predict_sequences(
        model=model,
        sequences=sequences,
        batch_size=BATCH_SIZE,
        max_len=max_len,
        stride=stride,
        device=device,
        num_workers=NUM_WORKERS,
    )
    pred = (result["prob"] >= threshold).astype(np.int64)

    save_dir = os.path.dirname(os.path.abspath(OUTPUT_PATH))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    save_items = {
        "prob": result["prob"],
        "pred": pred,
        "sample_id": result["sample_id"],
        "position": result["position"],
        "threshold": np.asarray([threshold], dtype=np.float32),
    }
    if "label" in result:
        save_items["label"] = result["label"]

    np.savez_compressed(OUTPUT_PATH, **save_items)
    print(f"Saved predictions: {OUTPUT_PATH}")
    print(f"Residues: {len(result['prob'])}")
    print(f"Threshold: {threshold:.2f}")

    if "label" in result:
        metrics = evaluate_metrics(result["label"], result["prob"], threshold=threshold)
        print("\nMetrics")
        print(f"AUC:         {metrics['AUC']:.4f}")
        print(f"AP:          {metrics['AP']:.4f}")
        print(f"MCC:         {metrics['MCC']:.4f}")
        print(f"F1:          {metrics['F1']:.4f}")
        print(f"Precision:   {metrics['Precision']:.4f}")
        print(f"Recall:      {metrics['Recall']:.4f}")
        print(f"Specificity: {metrics['Specificity']:.4f}")


if __name__ == "__main__":
    main()
