import argparse
import os

import numpy as np
import torch

from v1_struct_transformer import (
    build_model_from_config,
    evaluate_metrics,
    load_graph_sequences,
    predict_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a v1 structural-bias Transformer checkpoint."
    )
    parser.add_argument("--data", default="117.pt")
    parser.add_argument("--model", default="v1_best_struct_transformer.pt")
    parser.add_argument("--output", default="v1_prediction_results.npz")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Device: {device}")

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("The model file must be a v1_train_transformer.py checkpoint.")

    model = build_model_from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )
    max_len = int(args.max_len or checkpoint.get("max_len", 512))
    stride = int(args.stride or checkpoint.get("stride", 384))

    sequences = load_graph_sequences(args.data, require_y=False)
    result = predict_sequences(
        model=model,
        sequences=sequences,
        batch_size=args.batch_size,
        max_len=max_len,
        stride=stride,
        device=device,
        num_workers=args.num_workers,
    )
    pred = (result["prob"] >= threshold).astype(np.int64)

    save_dir = os.path.dirname(os.path.abspath(args.output))
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

    np.savez_compressed(args.output, **save_items)
    print(f"Saved predictions: {args.output}")
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
