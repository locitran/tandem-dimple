import os
import sys

import numpy as np
import pandas as pd

addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, addpath)
os.chdir(addpath)

from src.main import run
from src.utils.logger import LOGGER
from src.utils.settings import ROOT_DIR

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(TEST_DIR, "verified_results", "inference")

UNIREF90 = os.path.join(ROOT_DIR, "data", "consurf", "uniref90.fasta")

CASES = [
    {
        "name": "multi_protein_default_structures",
        "query": [
            "O00189 271 R H",
            "O00194 138 P L",
            "O00194 92 A T",
            "O00204 240 V I",
            "O00204 51 L S",
            "O00206 175 T A",
            "O00206 188 Q R",
            "O00206 246 C S",
        ],
        "custom_PDB": None,
        "pred_csv": os.path.join(REF_DIR, "multi_protein_default_structures_Main_Predictions.csv"),
        "feat_csv": os.path.join(REF_DIR, "multi_protein_default_structures_features.csv"),
        "info": "eight-SAV inference using TANDEM default structure resolution",
    },
]


def _run_case(case):
    """Run one inference case and return prediction/features dataframes."""

    job_name = f"test_inference_check_{case['name']}"
    job_dir = os.path.join(ROOT_DIR, "jobs", job_name)

    run(
        query=case["query"],
        custom_PDB=case["custom_PDB"],
        job_name=job_name,
        refresh=True,
        uniref90=UNIREF90,
    )

    pred_path = os.path.join(job_dir, "Main_Predictions.csv")
    feat_path = os.path.join(job_dir, "features.csv")

    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Missing prediction output: {pred_path}")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"Missing feature output: {feat_path}")

    return pd.read_csv(pred_path), pd.read_csv(feat_path)


def _check_predictions(new_df, ref_df, case_name):
    """Check the saved and rerun inference prediction tables match."""

    assert list(new_df.columns) == list(ref_df.columns), (
        f"Prediction column mismatch for {case_name}: "
        f"{list(new_df.columns)} != {list(ref_df.columns)}"
    )
    assert len(new_df) == len(ref_df), (
        f"Prediction row-count mismatch for {case_name}: "
        f"{len(new_df)} != {len(ref_df)}"
    )
    np.testing.assert_array_equal(
        new_df["SAV"].to_numpy(),
        ref_df["SAV"].to_numpy(),
        err_msg=f"SAV order mismatch for {case_name}",
    )

    mismatch_mask = new_df["TANDEM"].to_numpy() != ref_df["TANDEM"].to_numpy()
    mismatch_lines = []
    if mismatch_mask.any():
        for idx in np.where(mismatch_mask)[0]:
            mismatch_line = (
                f"{new_df.iloc[idx]['SAV']}: "
                f"new={new_df.iloc[idx]['TANDEM']} | ref={ref_df.iloc[idx]['TANDEM']}"
            )
            mismatch_lines.append(mismatch_line)
    return mismatch_lines


def _check_features(new_df, ref_df, case_name):
    """Check the saved and rerun feature matrices match."""

    assert list(new_df.columns) == list(ref_df.columns), (
        f"Feature column mismatch for {case_name}: "
        f"{list(new_df.columns)} != {list(ref_df.columns)}"
    )
    assert len(new_df) == len(ref_df), (
        f"Feature row-count mismatch for {case_name}: "
        f"{len(new_df)} != {len(ref_df)}"
    )

    feature_key = "SAV_coords" if "SAV_coords" in new_df.columns else new_df.columns[0]
    mismatch_lines = []

    for column in new_df.columns:
        new_values = new_df[column].to_numpy()
        ref_values = ref_df[column].to_numpy()

        if pd.api.types.is_numeric_dtype(new_df[column]) and pd.api.types.is_numeric_dtype(ref_df[column]):
            new_numeric = new_values.astype(float)
            ref_numeric = ref_values.astype(float)
            nan_mask = np.isnan(new_numeric) & np.isnan(ref_numeric)
            abs_diff = np.abs(new_numeric - ref_numeric)
            baseline = np.abs(ref_numeric)
            percent_diff = np.zeros_like(abs_diff)

            nonzero_mask = baseline > 0
            percent_diff[nonzero_mask] = abs_diff[nonzero_mask] / baseline[nonzero_mask]

            # For zero-valued references, any non-zero drift is treated as a mismatch.
            mismatch_mask = (~nan_mask) & (
                ((nonzero_mask) & (percent_diff >= 0.01)) |
                ((~nonzero_mask) & (abs_diff > 0))
            )

            if np.any(mismatch_mask):
                mismatch_idx = np.where(mismatch_mask)[0]
                for idx in mismatch_idx:
                    if baseline[idx] > 0:
                        percent_text = f"{percent_diff[idx] * 100:.3f}%"
                    else:
                        percent_text = "ref=0"
                    mismatch_lines.append(
                        f"{feature_key}={new_df.iloc[idx][feature_key]} | column={column} | "
                        f"new={new_numeric[idx]} | ref={ref_numeric[idx]} | diff={percent_text}"
                    )
        else:
            equal_mask = new_values == ref_values
            if not np.all(equal_mask):
                mismatch_idx = np.where(~equal_mask)[0]
                for idx in mismatch_idx:
                    mismatch_lines.append(
                        f"{feature_key}={new_df.iloc[idx][feature_key]} | column={column} | "
                        f"new={new_values[idx]} | ref={ref_values[idx]}"
                    )

    return mismatch_lines


def test_inference_matches_verified_results():
    """Compare inference outputs against verified saved CSV files."""

    results = []

    for case in CASES:
        case_label = case["name"]
        try:
            assert os.path.isfile(case["pred_csv"]), f"Missing verified prediction file: {case['pred_csv']}"
            assert os.path.isfile(case["feat_csv"]), f"Missing verified feature file: {case['feat_csv']}"

            new_pred_df, new_feat_df = _run_case(case)
            ref_pred_df = pd.read_csv(case["pred_csv"])
            ref_feat_df = pd.read_csv(case["feat_csv"])

            prediction_mismatches = _check_predictions(new_pred_df, ref_pred_df, case_label)
            feature_mismatches = _check_features(new_feat_df, ref_feat_df, case_label)

            if prediction_mismatches or feature_mismatches:
                error_parts = []
                if prediction_mismatches:
                    prediction_message = (
                        f"TANDEM prediction mismatch for {case_label}.\n"
                        f"Differing SAVs ({len(prediction_mismatches)}):\n"
                        + "\n".join(prediction_mismatches)
                    )
                    error_parts.append(prediction_message)
                if feature_mismatches:
                    feature_message = (
                        f"Feature mismatch for {case_label}.\n"
                        f"All differing values ({len(feature_mismatches)}):\n"
                        + "\n".join(feature_mismatches)
                    )
                    error_parts.append(feature_message)
                raise AssertionError("\n\n".join(error_parts))
        except Exception as exc:
            LOGGER.warn(f"FAILED: {case_label} ({case['info']}) - {exc}")
            results.append({"case": case_label, "info": case["info"], "passed": False, "error": str(exc)})
            continue

        LOGGER.info(f"PASSED: {case_label} ({case['info']})")
        results.append({"case": case_label, "info": case["info"], "passed": True, "error": ""})

    n_total = len(results)
    n_passed = sum(result["passed"] for result in results)
    n_failed = n_total - n_passed

    LOGGER.info(f"Inference unit test summary: {n_total} total, {n_passed} passed, {n_failed} failed")
    for result in results:
        status = "PASSED" if result["passed"] else "FAILED"
        suffix = "" if result["passed"] else f" - {result['error']}"
        LOGGER.info(f"  {status}: {result['case']} ({result['info']}){suffix}")

    if n_failed:
        raise AssertionError(
            f"Inference verification failed for {n_failed}/{n_total} case(s). "
            "See log for the full summary."
        )


if __name__ == "__main__":
    test_inference_matches_verified_results()
