import os
from typing import Dict

import torch

from v3_data import (
    load_graph_sequences,
    make_window_loader,
    move_batch_to_device,
    set_seed,
    split_sequences_by_protein,
)
from v3_metrics import (
    MaskedBCEWithLogitsLoss,
    MaskedFocalBCEWithLogitsLoss,
    compute_pos_weight,
    evaluate_metrics,
    find_best_threshold_by_mcc,
    predict_sequences,
)
from v3_struct_transformer import build_model_from_config, checkpoint_model_config

# Edit these values in VSCode, then click Run.
TRAIN_DATA_PATH = "495.pt"
SAVE_MODEL_PATH = "v3_best_struct_transformer.pt"

TRAIN_RATIO = 0.85
MAX_LEN = 768
STRIDE = 384
BATCH_SIZE = 2
EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 0
USE_CUDA = True

D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DIM_FEEDFORWARD = 512
DROPOUT = 0.2

LOCAL_RELATIVE_POSITION = 128
NUM_RANDOM_FEATURES = 32
SPE_HIDDEN_DIM = 64
SPE_CONV_KERNEL_SIZE = 17
MAX_GLOBAL_DISTANCE = MAX_LEN
FOURIER_SCALE = 1.0

USE_ROPE = True
USE_STRUCT_BIAS = True
STRUCT_HIDDEN_DIM = 64
CONTACT_CUTOFF = 8.0

LOSS_NAME = "focal"  # "focal" or "bce"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


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
    return MaskedFocalBCEWithLogitsLoss(
        pos_weight=pos_weight,
        alpha=alpha,
        gamma=gamma,
    )


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    print(f"Device: {device}")

    all_sequences = load_graph_sequences(TRAIN_DATA_PATH, require_y=True)
    train_sequences, val_sequences = split_sequences_by_protein(
        all_sequences,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )
    input_dim = int(train_sequences[0].x.size(1))
    pos_weight = compute_pos_weight(train_sequences)

    print("\nData summary")
    print(f"Train data: {TRAIN_DATA_PATH}")
    print(f"Total proteins: {len(all_sequences)}")
    print(f"Train proteins: {len(train_sequences)}")
    print(f"Val proteins: {len(val_sequences)}")
    print(f"Input dim: {input_dim}")
    print(f"Positive class weight for BCE ablation: {pos_weight:.4f}")

    train_loader = make_window_loader(
        sequences=train_sequences,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        stride=STRIDE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    model_config = checkpoint_model_config(
        input_dim=input_dim,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        local_relative_position=LOCAL_RELATIVE_POSITION,
        num_random_features=NUM_RANDOM_FEATURES,
        spe_hidden_dim=SPE_HIDDEN_DIM,
        spe_conv_kernel_size=SPE_CONV_KERNEL_SIZE,
        max_global_distance=MAX_GLOBAL_DISTANCE,
        fourier_scale=FOURIER_SCALE,
        use_rope=USE_ROPE,
        use_struct_bias=USE_STRUCT_BIAS,
        struct_hidden_dim=STRUCT_HIDDEN_DIM,
        contact_cutoff=CONTACT_CUTOFF,
    )
    model = build_model_from_config(model_config).to(device)
    criterion = build_loss(
        LOSS_NAME,
        pos_weight=pos_weight,
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(EPOCHS, 1),
    )

    best_val_mcc = -1.0
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}
    stale_epochs = 0

    print("\nModel summary")
    print(model_config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nStart training")
    for epoch in range(1, EPOCHS + 1):
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
            batch_size=BATCH_SIZE,
            max_len=MAX_LEN,
            stride=STRIDE,
            device=device,
            num_workers=NUM_WORKERS,
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
                SAVE_MODEL_PATH,
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "threshold": best_threshold,
                    "val_metrics": best_metrics,
                    "max_len": MAX_LEN,
                    "stride": STRIDE,
                    "train_data": TRAIN_DATA_PATH,
                    "loss": LOSS_NAME,
                    "focal_alpha": FOCAL_ALPHA,
                    "focal_gamma": FOCAL_GAMMA,
                    "pos_weight": pos_weight,
                    "architecture": (
                        "v3 RoPE Transformer with exact local bias and "
                        "SPE-inspired convolutional random-Fourier long-range bias"
                    ),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print("Early stopping.")
                break

    print("\nBest validation")
    print_metrics("Val", best_metrics, best_threshold)
    print(f"Saved model: {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    main()
