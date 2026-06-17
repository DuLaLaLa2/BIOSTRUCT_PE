import csv
import itertools
import os
import shutil
from typing import Dict, List

from v3_train_transformer import run_training

# Edit these values in VSCode, then click Run.
TRAIN_DATA_PATH = "495.pt"
SWEEP_OUTPUT_DIR = "v3_sweeps"
BEST_MODEL_ALIAS_PATH = "v3_best_struct_transformer.pt"

LOCAL_RELATIVE_POSITION_CANDIDATES = [64, 128, 192]
NUM_RANDOM_FEATURES_CANDIDATES = [16, 32, 64]
SPE_CONV_KERNEL_SIZE_CANDIDATES = [9, 17]

# Keep the backbone fixed during this sweep so the conclusion is only about SPE.
TRAIN_RATIO = 0.85
MAX_LEN = 768
STRIDE = 384
BATCH_SIZE = 2
EPOCHS = 40
PATIENCE = 8
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
SPE_HIDDEN_DIM = 64
FOURIER_SCALE = 1.0
USE_ROPE = True
USE_STRUCT_BIAS = True
STRUCT_HIDDEN_DIM = 64
CONTACT_CUTOFF = 8.0
LOSS_NAME = "focal"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        "rank",
        "experiment_name",
        "checkpoint_path",
        "local_relative_position",
        "num_random_features",
        "spe_conv_kernel_size",
        "AUC",
        "AP",
        "MCC",
        "F1",
        "Precision",
        "Recall",
        "Specificity",
        "threshold",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": index,
                    "experiment_name": row["experiment_name"],
                    "checkpoint_path": row["checkpoint_path"],
                    "local_relative_position": row["local_relative_position"],
                    "num_random_features": row["num_random_features"],
                    "spe_conv_kernel_size": row["spe_conv_kernel_size"],
                    "AUC": f"{row['AUC']:.4f}",
                    "AP": f"{row['AP']:.4f}",
                    "MCC": f"{row['MCC']:.4f}",
                    "F1": f"{row['F1']:.4f}",
                    "Precision": f"{row['Precision']:.4f}",
                    "Recall": f"{row['Recall']:.4f}",
                    "Specificity": f"{row['Specificity']:.4f}",
                    "threshold": f"{row['threshold']:.2f}",
                }
            )


def main() -> None:
    os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(SWEEP_OUTPUT_DIR, "v3_spe_sweep_summary.csv")

    experiment_grid = list(
        itertools.product(
            LOCAL_RELATIVE_POSITION_CANDIDATES,
            NUM_RANDOM_FEATURES_CANDIDATES,
            SPE_CONV_KERNEL_SIZE_CANDIDATES,
        )
    )
    print(f"Total experiments: {len(experiment_grid)}")

    results: List[Dict[str, object]] = []
    for index, (local_relative_position, num_random_features, spe_conv_kernel_size) in enumerate(
        experiment_grid,
        start=1,
    ):
        experiment_name = (
            f"v3_lrp{local_relative_position}_rf{num_random_features}_k{spe_conv_kernel_size}"
        )
        checkpoint_path = os.path.join(SWEEP_OUTPUT_DIR, f"{experiment_name}.pt")

        print("\n" + "=" * 80)
        print(f"[{index}/{len(experiment_grid)}] {experiment_name}")
        print("=" * 80)

        result = run_training(
            {
                "TRAIN_DATA_PATH": TRAIN_DATA_PATH,
                "SAVE_MODEL_PATH": checkpoint_path,
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
                "LOCAL_RELATIVE_POSITION": local_relative_position,
                "NUM_RANDOM_FEATURES": num_random_features,
                "SPE_HIDDEN_DIM": SPE_HIDDEN_DIM,
                "SPE_CONV_KERNEL_SIZE": spe_conv_kernel_size,
                "MAX_GLOBAL_DISTANCE": MAX_LEN,
                "FOURIER_SCALE": FOURIER_SCALE,
                "USE_ROPE": USE_ROPE,
                "USE_STRUCT_BIAS": USE_STRUCT_BIAS,
                "STRUCT_HIDDEN_DIM": STRUCT_HIDDEN_DIM,
                "CONTACT_CUTOFF": CONTACT_CUTOFF,
                "LOSS_NAME": LOSS_NAME,
                "FOCAL_ALPHA": FOCAL_ALPHA,
                "FOCAL_GAMMA": FOCAL_GAMMA,
            }
        )
        metrics = result["best_metrics"]
        results.append(
            {
                "experiment_name": experiment_name,
                "checkpoint_path": checkpoint_path,
                "local_relative_position": local_relative_position,
                "num_random_features": num_random_features,
                "spe_conv_kernel_size": spe_conv_kernel_size,
                "AUC": float(metrics["AUC"]),
                "AP": float(metrics["AP"]),
                "MCC": float(metrics["MCC"]),
                "F1": float(metrics["F1"]),
                "Precision": float(metrics["Precision"]),
                "Recall": float(metrics["Recall"]),
                "Specificity": float(metrics["Specificity"]),
                "threshold": float(result["best_threshold"]),
            }
        )

        results.sort(key=lambda item: (item["MCC"], item["AP"], item["AUC"]), reverse=True)
        write_summary_csv(summary_path, results)

    best_result = results[0]
    shutil.copyfile(best_result["checkpoint_path"], BEST_MODEL_ALIAS_PATH)

    print("\nSweep ranking")
    for index, row in enumerate(results[:10], start=1):
        print(
            f"{index:02d}. {row['experiment_name']} | "
            f"MCC {row['MCC']:.4f} | "
            f"AP {row['AP']:.4f} | "
            f"AUC {row['AUC']:.4f} | "
            f"F1 {row['F1']:.4f}"
        )

    print("\nBest configuration")
    print(f"LOCAL_RELATIVE_POSITION = {best_result['local_relative_position']}")
    print(f"NUM_RANDOM_FEATURES = {best_result['num_random_features']}")
    print(f"SPE_CONV_KERNEL_SIZE = {best_result['spe_conv_kernel_size']}")
    print(f"Best checkpoint: {best_result['checkpoint_path']}")
    print(f"Best alias copied to: {BEST_MODEL_ALIAS_PATH}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
