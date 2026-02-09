#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


ACC_R20000 = [
    "P29033", "P07101", "Q8IWU9", "P00439", "Q9UHC9", "O15118", "P22304",
    "P30613", "P14618", "P35520", "P11509", "Q16696", "P05181", "P20813",
    "P11712", "P10632", "P33261", "P00966", "P11413", "Q93099", "P15848",
    "P54802", "P60484", "Q06124", "P09619", "P16234", "P10721", "P07333",
    "P78504", "Q9NR61", "P52701", "Q15831", "P08559", "P06400", "P00156",
    "P78527", "Q14353", "Q13224", "Q12879", "Q9H251", "P51648", "P30838",
    "P18074", "O94759", "Q8TD43", "P00813", "O14733", "P36507", "P45985",
    "Q02750", "Q96L73", "O43240", "P06870", "P07288", "P20151", "O60259",
    "Q9Y5K2", "P23946", "P07477", "Q92876", "P46597", "P03891",
]


def _read_csv(path, name):
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Test plot functions from src.model.plot.")
    parser.add_argument("--root", type=Path, default=None, help="Root of tandem repo (default: inferred).")
    parser.add_argument("--tandem-eval", type=Path, default=None, help="Path to TANDEM evaluations.csv.")
    parser.add_argument("--rhapsodydnn-eval", type=Path, default=None, help="Path to RhapsodyDNN evaluations.csv.")
    parser.add_argument("--tandem-gjb2-dir", type=Path, default=None, help="Dir with before/after_transfer.csv for GJB2.")
    parser.add_argument("--tandem-ryr1-dir", type=Path, default=None, help="Dir with before/after_transfer.csv for RYR1.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to save plots.")
    parser.add_argument("--show-bar-values", action="store_true", help="Annotate bar heights (default: on).")
    parser.add_argument("--no-sigstars", action="store_true", help="Disable significance brackets (default: on).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance alpha.")
    args = parser.parse_args()

    root = args.root or Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import matplotlib
    matplotlib.use("Agg")

    from src.model.plot import pl_gene_general_performance, pl_gene_specific_performance
    from src.utils.settings import ROOT_DIR

    root_dir = Path(ROOT_DIR)
    tandem_eval = args.tandem_eval or (root_dir / "models" / "TANDEM" / "evaluations.csv")
    rhapsodydnn_eval = args.rhapsodydnn_eval or (root_dir / "models" / "RhapsodyDNN" / "evaluations.csv")
    tandem_gjb2_dir = args.tandem_gjb2_dir or (root_dir / "models" / "TANDEM_GJB2")
    tandem_ryr1_dir = args.tandem_ryr1_dir or (root_dir / "models" / "TANDEM_RYR1")

    tandem = _read_csv(Path(tandem_eval), "TANDEM evaluations")
    rhapsodyDNN = _read_csv(Path(rhapsodydnn_eval), "RhapsodyDNN evaluations")

    tf_gjb2_before = _read_csv(tandem_gjb2_dir / "before_transfer.csv", "TANDEM_GJB2 before_transfer")
    tf_gjb2_after = _read_csv(tandem_gjb2_dir / "after_transfer.csv", "TANDEM_GJB2 after_transfer")
    tf_ryr1_before = _read_csv(tandem_ryr1_dir / "before_transfer.csv", "TANDEM_RYR1 before_transfer")
    tf_ryr1_after = _read_csv(tandem_ryr1_dir / "after_transfer.csv", "TANDEM_RYR1 after_transfer")

    alphamissense_GJB2 = _read_csv(root_dir / "data" / "GJB2" / "alphamissense-predictions.csv", "AlphaMissense GJB2")
    alphamissense_RYR1 = _read_csv(root_dir / "data" / "RYR1" / "alphamissense-predictions.csv", "AlphaMissense RYR1")
    alphamissense_R20000 = _read_csv(root_dir / "data" / "R20000" / "preds_from_alm.csv", "AlphaMissense R20000")

    rhapsody_GJB2 = _read_csv(root_dir / "data" / "GJB2" / "rhapsody-predictions.csv", "Rhapsody GJB2")
    rhapsody_RYR1 = _read_csv(root_dir / "data" / "RYR1" / "rhapsody-predictions.csv", "Rhapsody RYR1")
    rhapsody_R20000 = _read_csv(root_dir / "data" / "R20000" / "preds_from_rhd.csv", "Rhapsody R20000")

    alphamissense_GJB2 = alphamissense_GJB2.dropna(subset=["labels"])
    alphamissense_RYR1 = alphamissense_RYR1.dropna(subset=["labels"])
    alphamissense_R20000 = alphamissense_R20000.dropna(subset=["labels"])
    rhapsody_GJB2 = rhapsody_GJB2.dropna(subset=["labels"])
    rhapsody_RYR1 = rhapsody_RYR1.dropna(subset=["labels"])
    rhapsody_R20000 = rhapsody_R20000.dropna(subset=["labels"])

    alphamissense_R20000["UniprotID"] = alphamissense_R20000["SAV_coords"].str.split().str[0]
    alphamissense_R20000 = alphamissense_R20000[alphamissense_R20000["UniprotID"].isin(ACC_R20000)].copy()

    rhapsody_R20000["UniprotID"] = rhapsody_R20000["SAV_coords"].str.split().str[0]
    rhapsody_R20000 = rhapsody_R20000[rhapsody_R20000["UniprotID"].isin(ACC_R20000)].copy()

    alphamissense_GJB2_metrics = get_metrics(alphamissense_GJB2["labels"], alphamissense_GJB2["p_variant"], threshold=0.452)
    alphamissense_RYR1_metrics = get_metrics(alphamissense_RYR1["labels"], alphamissense_RYR1["p_variant"], threshold=0.452)
    alphamissense_R20000_metrics = get_metrics(alphamissense_R20000["labels"], alphamissense_R20000["Pathogenicity"], threshold=0.452)
    rhapsody_GJB2_metrics = get_metrics(rhapsody_GJB2["labels"], rhapsody_GJB2["prob"], threshold=0.5)
    rhapsody_RYR1_metrics = get_metrics(rhapsody_RYR1["labels"], rhapsody_RYR1["prob"], threshold=0.5)
    rhapsody_R20000_metrics = get_metrics(rhapsody_R20000["labels"], rhapsody_R20000["path. prob."], threshold=0.5)


    GJB2_test_set = ['P29033 100 H Q', 'P29033 50 D N', 'P29033 115 F V', 'P29033 44 W S', 'P29033 4 G V']
    RYR1_test_set = ['P21817 2458 R H','P21817 4753 A T', 'P21817 933 A T', 'P21817 816 P L', 'P21817 2321 I V', 
                     'P21817 2458 R C',  'P21817 3815 M L', 'P21817 2355 R W', 'P21817 530 R H'] # seed 0
    
    alphamissense_GJB2_test_metrics = get_metrics(
        alphamissense_GJB2[alphamissense_GJB2['SAV_coords'].isin(GJB2_test_set)]['labels'], 
        alphamissense_GJB2[alphamissense_GJB2['SAV_coords'].isin(GJB2_test_set)]['p_variant'],
        threshold=0.452)
    alphamissense_RYR1_test_metrics = get_metrics(
        alphamissense_RYR1[alphamissense_RYR1['SAV_coords'].isin(RYR1_test_set)]['labels'],
        alphamissense_RYR1[alphamissense_RYR1['SAV_coords'].isin(RYR1_test_set)]['p_variant'], 
        threshold=0.452)
    rhapsody_GJB2_test_metrics = get_metrics(
        rhapsody_GJB2[rhapsody_GJB2['SAV_coords'].isin(GJB2_test_set)]['labels'],
        rhapsody_GJB2[rhapsody_GJB2['SAV_coords'].isin(GJB2_test_set)]['prob'], 
        threshold=0.5)
    rhapsody_RYR1_test_metrics = get_metrics(rhapsody_RYR1[rhapsody_RYR1['SAV_coords'].isin(RYR1_test_set)]['labels'], rhapsody_RYR1[rhapsody_RYR1['SAV_coords'].isin(RYR1_test_set)]['prob'],  threshold=0.5)

    out_dir = args.out_dir or (root_dir / "logs" / "plot_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    param = {"edgecolor": "black", "width": 0.2}
    txt_abv_bar = 0.5

    pl_gene_general_performance(
        tandem=tandem,
        rhapsodyDNN=rhapsodyDNN,
        rhapsody_R20000_metrics=rhapsody_R20000_metrics,
        alphamissense_R20000_metrics=alphamissense_R20000_metrics,
        rhapsody_GJB2_metrics=rhapsody_GJB2_metrics,
        alphamissense_GJB2_metrics=alphamissense_GJB2_metrics,
        rhapsody_RYR1_metrics=rhapsody_RYR1_metrics,
        alphamissense_RYR1_metrics=alphamissense_RYR1_metrics,
        param=param,
        txt_abv_bar=txt_abv_bar,
        alpha=args.alpha,
        show_bar_values=True,
        show_sigstars=False,
        save_path=out_dir / "gene_general_performance.png",
    )

    pl_gene_specific_performance(
        tf_gjb2_after=tf_gjb2_after,
        tf_ryr1_after=tf_ryr1_after,
        tf_gjb2_before=tf_gjb2_before,
        tf_ryr1_before=tf_ryr1_before,
        rhapsody_R20000_metrics=rhapsody_R20000_metrics,
        alphamissense_R20000_metrics=alphamissense_R20000_metrics,
        rhapsody_GJB2_test_metrics=rhapsody_GJB2_test_metrics,
        alphamissense_GJB2_test_metrics=alphamissense_GJB2_test_metrics,
        rhapsody_RYR1_test_metrics=rhapsody_RYR1_test_metrics,
        alphamissense_RYR1_test_metrics=alphamissense_RYR1_test_metrics,
        txt_abv_bar=txt_abv_bar,
        alpha=args.alpha,
        show_bar_values=True,
        show_sigstars=False,
        save_path=out_dir / "gene_specific_performance.png",
    )

    print(f"Plot functions executed successfully. Saved to: {out_dir}")

if __name__ == "__main__":
    main()
