import os
import random
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def log_message(message, log_file):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")


def plot_training_history(history, output_dir):
    figure_path = os.path.join(output_dir, "loss_curve.png")

    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    return figure_path


def plot_feature_ablation_results(
    runs_dir="runs",
    output_file="feature_ablation_metrics.png",
    experiments=None,
    metrics=None,
    metric_labels=None,
):
    if experiments is None:
        experiments = {
            "EVO": "evo",
            "STR+DYN": "str_dyn",
            "EVO+STR+DYN": "evo_str_dyn",
            "TANDEM": "tandem-33"
        }

    if metrics is None:
        metrics = [
            "test_accuracy",
            "test_auc",
            "test_precision",
            "test_recall",
            "test_specificity",
            "test_f1",
            "test_auprc",
        ]

    if metric_labels is None:
        metric_labels = ["Accuracy", "AUROC", "Precision", "Recall", "Specificity", "F1", "AUPRC"]

    rows = []
    for label, folder in experiments.items():
        metrics_file = os.path.join(runs_dir, folder, "metrics.json")
        with open(metrics_file) as f:
            run_metrics = json.load(f)

        rows.append({
            "label": label,
            "values": [run_metrics[metric] for metric in metrics],
        })

    x = np.arange(len(metrics))
    width = min(0.18, 0.8 / len(rows))
    offsets = (np.arange(len(rows)) - (len(rows) - 1) / 2) * width
    faces = ["white", "0.75", "white", "0.75"]
    hatches = ["", "", "//", "//"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in enumerate(rows):
        bars = ax.bar(
            x + offsets[i],
            row["values"],
            width=width,
            color=faces[i % len(faces)],
            edgecolor="black",
            linewidth=1.2,
            hatch=hatches[i % len(hatches)],
            label=row["label"],
        )
        for bar, value in zip(bars, row["values"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Metric value", fontsize=14)
    ax.set_ylim(0.5, 1.05)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("R20000 Feature Ablation", fontsize=16, pad=15)
    ax.legend(fontsize=11, frameon=False, ncol=2)

    plt.tight_layout()
    figure_path = os.path.join(runs_dir, output_file)
    plt.savefig(figure_path, dpi=300)
    plt.close()

    return figure_path
