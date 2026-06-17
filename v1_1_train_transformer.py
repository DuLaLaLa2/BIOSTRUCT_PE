import argparse
import os
from typing import Dict

import torch

from v1_1_struct_transformer import (
    MaskedBCEWithLogitsLoss,
    MaskedFocalBCEWithLogitsLoss,
    build_model_from_config,
    checkpoint_model_config,
    compute_pos_weight,
    evaluate_metrics,
    find_best_threshold_by_mcc,
    load_graph_sequences,
    make_window_loader,
    move_batch_to_device,
    predict_sequences,
    set_seed,
    split_sequences_by_protein,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v1.1 RoPE + SPE-Conv1D positional-bias Transformer."
    )
    parser.add_argument("--train-data", default="495.pt")
    parser.add_argument("--test-data", default="117.pt")
    parser.add_argument("--save-model", default="v1_1_best_struct_transformer.pt")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-relative-position", type=int, default=128)
    parser.add_argument("--spe-kernel-size", type=int, default=7)
    parser.add_argument("--spe-hidden-dim", type=int, default=32)
    parser.add_argument("--struct-hidden-dim", type=int, default=64)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--loss", choices=["focal", "bce"], default="focal")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-struct-bias", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def save_checkpoint(path: str, checkpoint: Dict[str, object]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(checkpoint, path)


def print_metrics(prefix: str, metrics: Dict[str, float], threshold: float) -> None:
    print(
        f"{prefix} | "
        f"AUC {metrics['AUC']:.4f} | "
        f"AP {metrics['AP']:.4f} | "
        f"MCC {metrics['MCC']:.4f} | "
        f"F1 {metrics['F1']:.4f} | "
        f"Precision {metrics['Precision']:.4f} | "
        f"Recall {metrics['Recall']:.4f} | "
        f"Spec {metrics['Specificity']:.4f} | "
        f"Thr {threshold:.2f}"
    )


def build_loss(name: str, pos_weight: float, alpha: float, gamma: float) -> torch.nn.Module:
    if name == "bce":
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    return MaskedFocalBCEWithLogitsLoss(pos_weight=pos_weight, alpha=alpha, gamma=gamma)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Device: {device}")

    all_sequences = load_graph_sequences(args.train_data, require_y=True)
    train_sequences, val_sequences = split_sequences_by_protein(
        all_sequences,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    input_dim = int(train_sequences[0].x.size(1))
    pos_weight = compute_pos_weight(train_sequences)

    print("\nData summary")
    print(f"Train data: {args.train_data}")
    print(f"Total proteins: {len(all_sequences)}")
    print(f"Train proteins: {len(train_sequences)}")
    print(f"Val proteins: {len(val_sequences)}")
    print(f"Input dim: {input_dim}")
    print(f"Positive class weight for BCE ablation: {pos_weight:.4f}")

    train_loader = make_window_loader(
        sequences=train_sequences,
        batch_size=args.batch_size,
        max_len=args.max_len,
        stride=args.stride,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model_config = checkpoint_model_config(
        input_dim=input_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_relative_position=args.max_relative_position,
        spe_kernel_size=args.spe_kernel_size,
        spe_hidden_dim=args.spe_hidden_dim,
        use_rope=not args.no_rope,
        use_struct_bias=not args.no_struct_bias,
        struct_hidden_dim=args.struct_hidden_dim,
        contact_cutoff=args.contact_cutoff,
    )
    model = build_model_from_config(model_config).to(device)
    criterion = build_loss(
        args.loss,
        pos_weight=pos_weight,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    best_val_mcc = -1.0
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}
    stale_epochs = 0

    print("\nModel summary")
    print(model_config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nStart training")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                x=batch["x"],
                mask=batch["mask"],
                positions=batch["positions"],
                pos=batch["pos"],
            )
            loss = criterion(logits, batch["y"], batch["mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item())

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)

        val_result = predict_sequences(
            model=model,
            sequences=val_sequences,
            batch_size=args.batch_size,
            max_len=args.max_len,
            stride=args.stride,
            device=device,
            num_workers=args.num_workers,
        )
        val_labels = val_result["label"]
        val_probs = val_result["prob"]
        threshold, val_mcc = find_best_threshold_by_mcc(val_labels, val_probs)
        val_metrics = evaluate_metrics(val_labels, val_probs, threshold=threshold)

        print(
            f"Epoch {epoch:03d} | Loss {train_loss:.4f} | "
            f"Val AUC {val_metrics['AUC']:.4f} | "
            f"Val AP {val_metrics['AP']:.4f} | "
            f"Val MCC {val_metrics['MCC']:.4f} | "
            f"Thr {threshold:.2f}"
        )

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_threshold = threshold
            best_metrics = val_metrics
            stale_epochs = 0
            save_checkpoint(
                args.save_model,
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "threshold": best_threshold,
                    "val_metrics": best_metrics,
                    "max_len": args.max_len,
                    "stride": args.stride,
                    "train_data": args.train_data,
                    "loss": args.loss,
                    "focal_alpha": args.focal_alpha,
                    "focal_gamma": args.focal_gamma,
                    "pos_weight": pos_weight,
                    "architecture": (
                        "v1.1 RoPE Transformer with SPE Conv1D(r=i-j) sequence bias and "
                        "C-alpha RBF/direction/contact structural attention bias"
                    ),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping.")
                break

    print("\nBest validation")
    print_metrics("Val", best_metrics, best_threshold)
    print(f"Saved model: {args.save_model}")

    if args.test_data:
        print(f"\nIndependent test: {args.test_data}")
        checkpoint = torch.load(args.save_model, map_location=device, weights_only=False)
        model = build_model_from_config(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_sequences = load_graph_sequences(args.test_data, require_y=True)
        test_result = predict_sequences(
            model=model,
            sequences=test_sequences,
            batch_size=args.batch_size,
            max_len=args.max_len,
            stride=args.stride,
            device=device,
            num_workers=args.num_workers,
        )
        test_metrics = evaluate_metrics(
            test_result["label"],
            test_result["prob"],
            threshold=float(checkpoint["threshold"]),
        )
        print_metrics("Test", test_metrics, float(checkpoint["threshold"]))


if __name__ == "__main__":
    main()
