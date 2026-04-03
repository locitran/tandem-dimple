import os
import sys
import numpy as np
import pandas as pd

addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, addpath)
os.chdir(addpath)
from src.download import fetchPDB
from src.features.consurf import calcConSurf_v2
from src.utils.logger import LOGGER

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(TEST_DIR, "verified_results", "consurf")

CASES = [
    {"pdbfile": "4TVX", "id": "4TVX", "chid": "A", "csv": os.path.join(REF_DIR, "4TVX_A_features.csv"), "info": "available in database"},
    {"pdbfile": "2ZW3", "id": "2ZW3", "chid": "A", "csv": os.path.join(REF_DIR, "2ZW3_A_features.csv"), "info": "available in database"},
    {"pdbfile": "2KZK", "id": "2KZK", "chid": "A", "csv": os.path.join(REF_DIR, "2KZK_A_features.csv"), "info": "chain not available in database"},
    {"pdbfile": "4TVX", "id": "Q46897", "chid": "A", "csv": os.path.join(REF_DIR, "Q46897_A_features.csv"),"info": "UniProt accession number"},
]

def _get_df(case, out_dir):
    """Run calcConSurf_v2 and return a tidy dataframe for comparison."""
    folder = os.path.join(out_dir, case["id"])
    pdb = fetchPDB(case["pdbfile"], format="pdb", compressed=False, folder=folder)
    if pdb is None or not os.path.isfile(pdb):
        raise FileNotFoundError(f"Could not resolve PDB file for {case['pdbfile']}")

    feats = calcConSurf_v2(
        pdbfile=pdb, id=case["id"], chid=case["chid"], folder=folder,
        uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta'
    )
    df = pd.DataFrame(feats)
    df.insert(0, "residue_index", range(1, len(df) + 1))
    return df


def _check(new_features, ref_features, pdb_id, chid):
    """Check that the new result matches the verified saved result."""
    assert list(new_features.columns) == list(ref_features.columns), (
        f"Column mismatch for {pdb_id} chain {chid}: "
        f"{list(new_features.columns)} != {list(ref_features.columns)}"
    )
    assert len(new_features) == len(ref_features), (
        f"Row-count mismatch for {pdb_id} chain {chid}: "
        f"{len(new_features)} != {len(ref_features)}"
    )
    assert new_features["residue_index"].tolist() == ref_features["residue_index"].tolist(), (
        f"Residue index mismatch for {pdb_id} chain {chid}"
    )

    np.testing.assert_allclose(
        new_features["consurf"].to_numpy(dtype=float),
        ref_features["consurf"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
        err_msg=f"ConSurf score mismatch for {pdb_id} chain {chid}",
    )
    np.testing.assert_allclose(
        new_features["ACNR"].to_numpy(dtype=float),
        ref_features["ACNR"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
        err_msg=f"ACNR mismatch for {pdb_id} chain {chid}",
    )
    np.testing.assert_array_equal(
        new_features["consurf_color"].to_numpy(),
        ref_features["consurf_color"].to_numpy(),
        err_msg=f"ConSurf color mismatch for {pdb_id} chain {chid}",
    )


def test_calcConSurf_v2_matches_verified_results():
    """Compare calcConSurf_v2 outputs against verified CSV files."""
    results = []

    for case in CASES:
        case_label = f"{case['id']} chain {case['chid']}"
        try:
            assert os.path.isfile(case["csv"]), f"Missing verified file: {case['csv']}"
            new_features = _get_df(case, REF_DIR)
            ref_features = pd.read_csv(case["csv"])
            _check(new_features, ref_features, case["id"], case["chid"])
        except Exception as exc:
            LOGGER.warn(f"FAILED: {case_label} ({case['info']}) - {exc}")
            results.append({"case": case_label, "info": case["info"], "passed": False, "error": str(exc)})
            continue

        LOGGER.info(f"PASSED: {case_label} ({case['info']})")
        results.append({"case": case_label, "info": case["info"], "passed": True, "error": ""})

    n_total = len(results)
    n_passed = sum(result["passed"] for result in results)
    n_failed = n_total - n_passed

    LOGGER.info(f"ConSurf unit test summary: {n_total} total, {n_passed} passed, {n_failed} failed")
    for result in results:
        status = "PASSED" if result["passed"] else "FAILED"
        suffix = "" if result["passed"] else f" - {result['error']}"
        LOGGER.info(f"  {status}: {result['case']} ({result['info']}){suffix}")

    if n_failed:
        raise AssertionError(
            f"ConSurf verification failed for {n_failed}/{n_total} case(s). "
            "See log for the full summary."
        )

if __name__ == "__main__":
    test_calcConSurf_v2_matches_verified_results()
