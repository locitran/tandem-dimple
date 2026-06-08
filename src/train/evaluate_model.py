import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    from .data import prepare_external_test_arrays, prepare_r20000_test_arrays
    from .evaluate import evaluate_arrays
    from .model import DNN
except ImportError:
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from src.train.data import prepare_external_test_arrays, prepare_r20000_test_arrays
    from src.train.evaluate import evaluate_arrays
    from src.train.model import DNN


device = "cuda" if torch.cuda.is_available() else "cpu"


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_trained_model(run_dir):
    config_file = os.path.join(run_dir, "train_config.json")
    checkpoint_file = os.path.join(run_dir, "best_model.pt")

    with open(config_file) as f:
        config = json.load(f)

    model = DNN(
        input_dim=config["input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        output_dim=config["output_dim"],
        dropout_rate=config["dropout_rate"],
    ).to(device)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_weights = torch.tensor(config["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return model, criterion, config


def evaluate_subgroups(model, arrays, criterion, group_key="exposure_group"):
    if group_key not in arrays:
        return {}

    groups = np.asarray(arrays[group_key]).astype(str)
    y = np.asarray(arrays["y"])
    subgroup_results = {}

    for group in sorted(np.unique(groups)):
        mask = groups == group
        if not np.any(mask):
            continue

        metrics = evaluate_arrays(model, arrays["x"][mask], y[mask], criterion)
        subgroup_results[group] = {
            "n": int(mask.sum()),
            "n_benign": int(np.sum(y[mask] == 0)),
            "n_pathogenic": int(np.sum(y[mask] == 1)),
            "positive_rate": float(np.mean(y[mask] == 1)),
            **metrics,
        }

    return subgroup_results


def evaluate_run(
    folder,
    test_set="r20000",
    test_feat_path=None,
    runs_dir="runs",
    output_file=None,
):
    run_dir = os.path.join(runs_dir, folder)
    model, criterion, config = load_trained_model(run_dir)

    if test_set == "r20000":
        arrays = prepare_r20000_test_arrays(
            feat_names=config["feat_names"],
            input_dim=config["input_dim"],
        )
        output_file = output_file or "metrics.json"
    else:
        if test_feat_path is None:
            raise ValueError("test_feat_path is required for external test sets.")
        arrays = prepare_external_test_arrays(
            test_feat_path=test_feat_path,
            feat_names=config["feat_names"],
            input_dim=config["input_dim"],
        )
        output_file = output_file or f"{test_set}_metrics.json"

    if arrays["y"] is None:
        raise ValueError("Evaluation requires labels, but this test set has no labels.")

    metrics = evaluate_arrays(model, arrays["x"], arrays["y"], criterion)
    output = {
        "folder": folder,
        "test_set": test_set,
        "n": int(len(arrays["y"])),
        "n_benign": int(np.sum(arrays["y"] == 0)),
        "n_pathogenic": int(np.sum(arrays["y"] == 1)),
        "positive_rate": float(np.mean(arrays["y"] == 1)),
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_specificity": metrics["specificity"],
        "test_f1": metrics["f1"],
        "test_auc": metrics["auc"],
        "test_auprc": metrics["auprc"],
    }

    subgroup_results = evaluate_subgroups(model, arrays, criterion)
    if subgroup_results:
        output["subgroups"] = subgroup_results

    output_path = os.path.join(run_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(to_jsonable(output), f, indent=2)

    print(f"Saved evaluation metrics: {output_path}")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained TANDEM DNN model.")
    parser.add_argument("--folder", required=True, help="Run folder under runs/, for example evo.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders.")
    parser.add_argument("--test-set", default="r20000", help="Name of test set: r20000, gjb2, ryr1, etc.")
    parser.add_argument("--test-feat-path", default=None, help="Feature CSV for external test sets.")
    parser.add_argument("--output-file", default=None, help="Output JSON filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_run(
        folder=args.folder,
        test_set=args.test_set,
        test_feat_path=args.test_feat_path,
        runs_dir=args.runs_dir,
        output_file=args.output_file,
    )
