import os
import shutil
import subprocess
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
from src.download import fetchPDB
from src.features.consurf import calcConSurf_v2
from src.utils.logger import LOGGER

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "verified_results", "consurf_patch")
os.makedirs(OUTPUT_DIR, exist_ok=True)
COLOR_SCRIPT = os.path.join(addpath, "src", "tools", "color_consurf.py")

PATCH = [
    {"pdbfile": "4DZD", "id": "4DZD", "chid": "A"},
]

def save_pymol_session(pdbfile, psefile, obj_name):
    pymol = shutil.which("pymol")
    if pymol is None:
        LOGGER.warn("PyMOL not found in PATH. Did not create .pse file.")
        return
    command = (
        f'run {COLOR_SCRIPT}; '
        f'load {pdbfile}, {obj_name}; '
        f'colour_consurf {obj_name}; '
        f'save {psefile}; '
        f'quit'
    )
    subprocess.run([pymol, "-cq", "-d", command], check=True)
    if not os.path.isfile(psefile):
        raise FileNotFoundError(f"PyMOL finished but .pse file was not created: {psefile}")

def main():
    logfile = os.path.join(OUTPUT_DIR, 'consurf.log')
    LOGGER.start(logfile)
    for case in PATCH:
        folder = os.path.join(OUTPUT_DIR, case["id"])
        os.makedirs(folder, exist_ok=True)

        if not os.path.isfile(case["pdbfile"]):
            pdbfile = fetchPDB(case["pdbfile"], folder=folder)
        else:
            pdbfile = case["pdbfile"]

        try:
            calcConSurf_v2(
                pdbfile=pdbfile,id=case["id"],chid=case["chid"],folder=folder, write_consurf=True,
                uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta',
            )
        except:
            import traceback
            msg = traceback.format_exc()
            LOGGER.warn(msg)
            LOGGER.close(logfile)
            raise

        outfile = os.path.join(folder, f"{case['id']}_{case['chid']}_consurf.pdb",)
        LOGGER.info(f"ConSurf scores patched to coordinate file: {outfile}")
        
        try:
            psefile = os.path.join(folder, f"{case['id']}_{case['chid']}_consurf.pse")
            save_pymol_session(outfile, psefile, f"{case['id']}_{case['chid']}")
            LOGGER.info(f"PyMOL session saved to: {psefile}")
        except Exception:
            import traceback
            msg = traceback.format_exc()
            LOGGER.warn(msg)
    
    LOGGER.close(logfile)

if __name__ == "__main__":
    main()
