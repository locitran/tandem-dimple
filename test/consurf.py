import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)

import pandas as pd

from src.download import fetchPDB
from src.features.consurf import calcConSurf_v2
from src.utils.logger import LOGGER

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TEST_DIR, "verified_results", "consurf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONSURF_CASES = [
    # {"pdbfile": "4TVX", "id": "4TVX", "chid": "A", "info": "available in database"},
    # {"pdbfile": "2ZW3", "id": "2ZW3", "chid": "A", "info": "available in database"},
    # {"pdbfile": "2KZK", "id": "2KZK", "chid": "A", "info": "chain not available in database"},
    {"pdbfile": "4TVX", "id": "Q46897", "chid": "A", "info": "UniProt accession number"},
]

def main():
    logfile = os.path.join(OUTPUT_DIR, 'consurf.log')
    LOGGER.start(logfile)
    for case in CONSURF_CASES:
        folder = os.path.join(OUTPUT_DIR, case["id"])

        if not os.path.isfile(case["pdbfile"]):
            pdbfile = fetchPDB(case["pdbfile"], folder=folder)
        else:
            pdbfile = case["pdbfile"]

        try:
            features = calcConSurf_v2(
                pdbfile=pdbfile,id=case["id"],chid=case["chid"],folder=folder,
                uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta'
            )
        except:
            import traceback
            msg = traceback.format_exc()
            LOGGER.warn(msg)
            LOGGER.close(logfile)
            raise

        df = pd.DataFrame(features)
        df.insert(0, "residue_index", range(1, len(df) + 1))
        outfile = os.path.join(OUTPUT_DIR, f"{case['id']}_{case['chid']}_features.csv")
        df.to_csv(outfile, index=False)

        assert os.path.isfile(outfile)
        assert list(df.columns) == ["residue_index", "consurf", "ACNR", "consurf_color"]
        assert len(df) > 0
    
    LOGGER.close(logfile)

if __name__ == "__main__":
    main()
