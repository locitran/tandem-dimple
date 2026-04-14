import json
import os 
import shutil
import datetime
import numpy as np
import traceback
from .core import Tandem
from .utils.settings import ROOT_DIR
from .utils.logger import LOGGER
from .utils.settings import TANDEM_v1dot1
from .utils.user_log import UserLog

__all__ = ['run']
FILE_EXPLANATION_TEMPLATE = os.path.join(os.path.dirname(__file__), "File_Explanation.txt")

def run(
    query,
    labels=None,
    custom_PDB=None,
    pretrained_model_folder=TANDEM_v1dot1,
    job_name='tandem-dimple',
    features=None, 
    config=None, 
    featSet=None,
    refresh=False,
    pkl_folder=os.path.join(ROOT_DIR, 'data'),
    uniref90=None,
    log_time=False,
):
    """
    query: 
        1. Single amino acid variant(s) <UniProtID> <resid> <wt> <mt>
        2. <UniprotID> <resid>
        3. <UniProtID>
    labels: 
        1: pathogenic; 0: benign
    custom_PDB: 
        1. uploaded coordinate file
        2. AlphaFold DB ID
        3. PDB ID
    pretrained_model_folder: 
        1. TANDEM_v1dot1, default TANDEM foundation models
        2. TANDEM_v1dot1_GJB2, TANDEM-DIMPLE for GJB2
        3. TANDEM_v1dot1_RYR1, TANDEM-DIMPLE for GJB2
        4. New pre-trained models
    """

    job_directory = os.path.join(ROOT_DIR, 'jobs', job_name)
    os.makedirs(job_directory, exist_ok=True)
    if os.path.isfile(FILE_EXPLANATION_TEMPLATE):
        shutil.copy2(FILE_EXPLANATION_TEMPLATE, os.path.join(job_directory, "File_Explanation.txt"))
    # Refresh user-facing log on each run for the same job_name.
    userlog_path = os.path.join(job_directory, "user_log.jsonl")
    with open(userlog_path, "w", encoding="utf-8"):
        pass
    userlog = UserLog(userlog_path, defaults={"job_name": job_name},)
    
    ## LOGGER
    LOGGER._times = {}
    LOGGER._reports = {}
    LOGGER._report_times = {}
    logfile = os.path.join(job_directory, 'log.txt')
    LOGGER.start(logfile)
    LOGGER.info(f"Job name: {job_name} started at {datetime.datetime.now()}")
    LOGGER.info(f"Job directory: {job_directory}")
    LOGGER.timeit("_runtime")
    userlog.emit(level="info", stage="job", message=f"Job '{job_name}' started.")
    mode_name = "training" if labels is not None else "inferencing"
    userlog.emit(level="info", stage="job", message=f"Submitted mode: {mode_name}.")

    try:
        os.makedirs(pkl_folder, exist_ok=True) ## Save feature pickles
        t = Tandem( # Set up the Tandem object
            query, 
            refresh=refresh,
            job_directory=job_directory, 
            folder=pkl_folder,
            uniref90=uniref90,
            userlog=userlog,
        )
        t.getSAVs(filename='SAVs.txt', folder=job_directory)
        t.setFeatSet(featSet)
        t.setCustomPDB(custom_PDB)
        t.getUniprot2PDBmap(filename='Uniprot2PDB.txt', folder=job_directory)
        t.getFeatMatrix(withSAVs=True, filename='features.csv', folder=job_directory)    

        if isinstance(features, np.ndarray):  
            t.featMatrix = features
            t.getFeatMatrix(withSAVs=True, filename='features.csv', folder=job_directory)    

        if labels:
            userlog.emit(level="info", stage="training", message="Transfer learning started.")
            t.setLabels(labels)
            t.setConfig(config)
            t.train()
            userlog.emit(level="info", stage="training", message="Transfer learning completed.")
        else:
            userlog.emit(level="info", stage="prediction", message="Inference started.")
            t.getPredictions(models=pretrained_model_folder, folder=job_directory, filename='Main_Predictions')
            t.plotSHAP(folder=job_directory)
            userlog.emit(level="info", stage="prediction", message="Inference completed.")

        for label in LOGGER._reports:
            LOGGER.info(f"  {label}: {LOGGER._reports[label]:.2f}s ({LOGGER._report_times[label]} time(s))")
        LOGGER.report('Run time elapsed in %.2fs.', "_runtime")

        if log_time:
            log_time_file = os.path.join(job_directory, 'log_time.json')
            log_time_data = {}
            for label in sorted(LOGGER._reports):
                log_time_data[label] = {
                    "seconds": round(float(LOGGER._reports[label]), 6),
                    "count": int(LOGGER._report_times.get(label, 1)),
                }
            if getattr(t, "Uniprot2PDBmap", None) is not None:
                for field in ('Uniprot_sequence_length', 'Asymmetric_PDB_length', 
                    "Asymmetric_PDB_resolved_length", "BioUnit_PDB_length", "OPM_PDB_length"):
                    values = np.asarray(t.Uniprot2PDBmap[field]).tolist()
                    log_time_data[field] = values
            with open(log_time_file, "w", encoding="utf-8") as f:
                json.dump(log_time_data, f, indent=2)

        userlog.emit(level="info", stage="job", message=f"Job '{job_name}' completed successfully.")
        return t
    except Exception as e:
        msg = traceback.format_exc()
        LOGGER.warn(msg)
        action="Please check log.txt for detailed traceback."
        userlog.emit(level="error", stage="job", message=f"Job '{job_name}' failed: {e}", action=action, context={"error": str(e)},)
        raise
    finally:
        LOGGER.close(logfile)
