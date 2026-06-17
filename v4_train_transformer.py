import os
from typing import Dict, Optional

import torch

from v4_data import (
    load_graph_sequences,
    make_window_loader,
    move_batch_to_device,
    set_seed,
    split_sequences_by_protein,
)
from v4_metrics import (
    MaskedBCEWithLogitsLoss,
    MaskedFocalBCEWithLogitsLoss,
    compute_pos_weight,
    evaluate_metrics,
    find_best_threshold_by_mcc,
    predict_sequences,
)
from v4_struct_transformer import build_model_from_config, checkpoint_model_config

# Edit these values in VSCode, then click Run.
TRAIN_DATA_PATH = "495.pt"
SAVE_MODEL_PATH = "v4_best_struct_transformer.pt"

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
INITIAL_GLOBAL_SCALE = 0.05

USE_ROPE = True
USE_STRUCT_BIAS = True
STRUCT_HIDDEN_DIM = 64
GATE_HIDDEN_DIM = 64
CONTACT_CUTOFF = 8.0

LOSS_NAME = "focal"  # "focal" or "bce"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


def get_default_train_config() -> Dict[str, object]:
    return {
        "TRAIN_DATA_PATH": TRAIN_DATA_PATH,
        "SAVE_MODEL_PATH": SAVE_MODEL_PATH,
        "TRAIN_RATIO": TRAIN_RATIO,
        "MAX_LEN": MAX_LEN,
        "STRIDE": STRIDE,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "PATIENCE": PATIENCE,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "SEED": SEED,
        "NUM_WORKERS": NUM_WORKERS,
        "USE_CUDA": USE_CUDA,
        "D_MODEL": D_MODEL,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_HEADS": NUM_HEADS,
        "DIM_FEEDFORWARD": DIM_FEEDFORWARD,
        "DROPOUT": DROPOUT,
        "LOCAL_RELATIVE_POSITION": LOCAL_RELATIVE_POSITION,
        "NUM_RANDOM_FEATURES": NUM_RANDOM_FEATURES,
        "SPE_HIDDEN_DIM": SPE_HIDDEN_DIM,
        "SPE_CONV_KERNEL_SIZE": SPE_CONV_KERNEL_SIZE,
        "MAX_GLOBAL_DISTANCE": MAX_GLOBAL_DISTANCE,
        "FOURIER_SCALE": FOURIER_SCALE,
        "INITIAL_GLOBAL_SCALE": INITIAL_GLOBAL_SCALE,
        "USE_ROPE": USE_ROPE,
        "USE_STRUCT_BIAS": USE_STRUCT_BIAS,
        "STRUCT_HIDDEN_DIM": STRUCT_HIDDEN_DIM,
        "GATE_HIDDEN_DIM": GATE_HIDDEN_DIM,
        "CONTACT_CUTOFF": CONTACT_CUTOFF,
        "LOSS_NAME": LOSS_NAME,
        "FOCAL_ALPHA": FOCAL_ALPHA,
        "FOCAL_GAMMA": FOCAL_GAMMA,
    }


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


def run_training(config_overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    cfg = get_default_train_config()
    if config_overrides:
        cfg.update(config_overrides)

    train_data_path = str(cfg["TRAIN_DATA_PATH"])
    save_model_path = str(cfg["SAVE_MODEL_PATH"])
    train_ratio = float(cfg["TRAIN_RATIO"])
    max_len = int(cfg["MAX_LEN"])
    stride = int(cfg["STRIDE"])
    batch_size = int(cfg["BATCH_SIZE"])
    epochs = int(cfg["EPOCHS"])
    patience = int(cfg["PATIENCE"])
    learning_rate = float(cfg["LEARNING_RATE"])
    weight_decay = float(cfg["WEIGHT_DECAY"])
    seed = int(cfg["SEED"])
    num_workers = int(cfg["NUM_WORKERS"])
    use_cuda = bool(cfg["USE_CUDA"])

    d_model = int(cfg["D_MODEL"])
    num_layers = int(cfg["NUM_LAYERS"])
    num_heads = int(cfg["NUM_HEADS"])
    dim_feedforward = int(cfg["DIM_FEEDFORWARD"])
    dropout = float(cfg["DROPOUT"])

    local_relative_position = int(cfg["LOCAL_RELATIVE_POSITION"])
    num_random_features = int(cfg["NUM_RANDOM_FEATURES"])
    spe_hidden_dim = int(cfg["SPE_HIDDEN_DIM"])
    spe_conv_kernel_size = int(cfg["SPE_CONV_KERNEL_SIZE"])
    max_global_distance = int(cfg["MAX_GLOBAL_DISTANCE"])
    fourier_scale = float(cfg["FOURIER_SCALE"])
    initial_global_scale = float(cfg["INITIAL_GLOBAL_SCALE"])

    use_rope = bool(cfg["USE_ROPE"])
    use_struct_bias = bool(cfg["USE_STRUCT_BIAS"])
    struct_hidden_dim = int(cfg["STRUCT_HIDDEN_DIM"])
    gate_hidden_dim = int(cfg["GATE_HIDDEN_DIM"])
    contact_cutoff = float(cfg["CONTACT_CUTOFF"])

    loss_name = str(cfg["LOSS_NAME"])
    focal_alpha = float(cfg["FOCAL_ALPHA"])
    focal_gamma = float(cfg["FOCAL_GAMMA"])

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"Device: {device}")

    all_sequences = load_graph_sequences(train_data_path, require_y=True)
    train_sequences, val_sequences = split_sequences_by_protein(
        all_sequences,
        train_ratio=train_ratio,
        seed=seed,
    )
    input_dim = int(train_sequences[0].x.size(1))
    pos_weight = compute_pos_weight(train_sequences)

    print("\nData summary")
    print(f"Train data: {train_data_path}")
    print(f"Total proteins: {len(all_sequences)}")
    print(f"Train proteins: {len(train_sequences)}")
    print(f"Val proteins: {len(val_sequences)}")
    print(f"Input dim: {input_dim}")
    print(f"Positive class weight for BCE ablation: {pos_weight:.4f}")

    train_loader = make_window_loader(
        sequences=train_sequences,
        batch_size=batch_size,
        max_len=max_len,
        stride=stride,
        shuffle=True,
        num_workers=num_workers,
    )

    model_config = checkpoint_model_config(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
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
        gate_hidden_dim=gate_hidden_dim,
        contact_cutoff=contact_cutoff,
        initial_global_scale=initial_global_scale,
    )
    model = build_model_from_config(model_config).to(device)
    criterion = build_loss(
        loss_name,
        pos_weight=pos_weight,
        alpha=focal_alpha,
        gamma=focal_gamma,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )

    best_val_mcc = -1.0
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}
    stale_epochs = 0

    print("\nModel summary")
    print(model_config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nStart training")
    for epoch in range(1, epochs + 1):
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
            batch_size=batch_size,
            max_len=max_len,
            stride=stride,
            device=device,
            num_workers=num_workers,
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
                save_model_path,
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "threshold": best_threshold,
                    "val_metrics": best_metrics,
                    "max_len": max_len,
                    "stride": stride,
                    "train_data": train_data_path,
                    "loss": loss_name,
                    "focal_alpha": focal_alpha,
                    "focal_gamma": focal_gamma,
                    "pos_weight": pos_weight,
                    "architecture": (
                        "v4 contact-gated SPE with exact local relative bias, "
                        "structural bias, and structure-gated long-range random-Fourier bias"
                    ),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping.")
                break

    print("\nBest validation")
    print_metrics("Val", best_metrics, best_threshold)
    print(f"Saved model: {save_model_path}")
    return {
        "save_model_path": save_model_path,
        "best_threshold": best_threshold,
        "best_val_mcc": best_val_mcc,
        "best_metrics": best_metrics,
        "model_config": model_config,
        "train_data_path": train_data_path,
        "max_len": max_len,
        "stride": stride,
    }


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
