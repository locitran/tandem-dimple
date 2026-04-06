import os
import shutil
import sys

import pandas as pd

addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, addpath)  # /tandem/
os.chdir(addpath)

from src.main import run
from src.utils.logger import LOGGER
from src.utils.settings import ROOT_DIR

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TEST_DIR, "verified_results", "inference")
os.makedirs(OUTPUT_DIR, exist_ok=True)

UNIREF90 = '/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta'

INFERENCE_CASES = [
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
        "info": "eight-SAV inference using TANDEM default structure resolution",
    },
]


def main():
    logfile = os.path.join(OUTPUT_DIR, "inference.log")
    LOGGER.start(logfile)

    for case in INFERENCE_CASES:
        job_name = f"test_inference_{case['name']}"
        job_dir = os.path.join(ROOT_DIR, "jobs", job_name)

        try:
            run(
                query=case["query"],
                custom_PDB=case["custom_PDB"],
                job_name=job_name,
                refresh=True,
                uniref90=UNIREF90,
            )
        except Exception:
            import traceback
            msg = traceback.format_exc()
            LOGGER.warn(msg)
            LOGGER.close(logfile)
            raise

        pred_src = os.path.join(job_dir, "Main_Predictions.csv")
        feat_src = os.path.join(job_dir, "features.csv")

        pred_dst = os.path.join(OUTPUT_DIR, f"{case['name']}_Main_Predictions.csv")
        feat_dst = os.path.join(OUTPUT_DIR, f"{case['name']}_features.csv")

        assert os.path.isfile(pred_src), f"Missing inference output: {pred_src}"
        assert os.path.isfile(feat_src), f"Missing feature output: {feat_src}"

        shutil.copy2(pred_src, pred_dst)
        shutil.copy2(feat_src, feat_dst)

        pred_df = pd.read_csv(pred_dst)
        feat_df = pd.read_csv(feat_dst)

        assert list(pred_df.columns) == ["SAV", "TANDEM"]
        assert len(pred_df) == len(case["query"])
        assert len(feat_df) == len(case["query"])

    LOGGER.close(logfile)


if __name__ == "__main__":
    main()
