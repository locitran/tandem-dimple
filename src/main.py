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

def toSAV_coords(SAVs):
    """
    >>> a = ['P29033 Y217E', 'P29033 Y217F', 'P29033 Y217T']
    >>> toSAV_coords(a)
    ['P29033 217 Y E', 'P29033 217 Y F', 'P29033 217 Y T']
    """
    out = []
    for s in SAVs:
        acc, wt_resid_mt = s.split()
        wt = wt_resid_mt[0]
        mt = wt_resid_mt[-1]
        resid = wt_resid_mt[0+1:-1]
        out.append(f"{acc} {resid} {wt} {mt}")
    return out

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
    logfile = os.path.join(job_directory, 'log.txt')
    LOGGER.start(logfile)
    LOGGER.info(f"Job name: {job_name} started at {datetime.datetime.now()}")
    LOGGER.info(f"Job directory: {job_directory}")
    LOGGER.timeit("_runtime")
    userlog.emit("info", "JOB_STARTED", "job", f"Job '{job_name}' started.")

    try:
        ## Save feature pickles
        os.makedirs(pkl_folder, exist_ok=True)

        # Set up the Tandem object
        t = Tandem(
            query, 
            refresh=refresh,
            job_directory=job_directory, 
            folder=pkl_folder,
            uniref90=uniref90,
            userlog=userlog,
        )
        t.getSAVs(filename='SAVs.txt', folder=job_directory)
        t.setFeatSet(featSet)
        
        # if isinstance(features, np.ndarray):
            # t.setFeatureMatrix(features)
        #     userlog.emit("info", "FEATURE_MATRIX_PROVIDED", "features", "Using precomputed feature matrix from caller input.",)
        # else:
        t.setCustomPDB(custom_PDB)
        t.getUniprot2PDBmap(filename='Uniprot2PDB.txt', folder=job_directory)
        t.getFeatMatrix(withSAVs=True, filename='features.csv', folder=job_directory)    

        if isinstance(features, np.ndarray):  
            t.featMatrix = features
            t.getFeatMatrix(withSAVs=True, filename='features.csv', folder=job_directory)    

        if labels:
            userlog.emit("info", "TRAINING_STARTED", "training", "Transfer learning started.")
            t.setLabels(labels)
            t.setConfig(config)
            t.train()
            userlog.emit("info", "TRAINING_COMPLETED", "training", "Transfer learning completed.")
        else:
            userlog.emit("info", "PREDICTION_STARTED", "prediction", "Inference started.")
            t.getPredictions(models=pretrained_model_folder, folder=job_directory, filename='Main_Predictions')
            t.plotSHAP(folder=job_directory)
            userlog.emit("info", "PREDICTION_COMPLETED", "prediction", "Inference completed.")

        for label in LOGGER._reports:
            LOGGER.info(f"  {label}: {LOGGER._reports[label]:.2f}s ({LOGGER._report_times[label]} time(s))")
        LOGGER.report('Run time elapsed in %.2fs.', "_runtime")
        userlog.emit("info", "JOB_COMPLETED", "job", f"Job '{job_name}' completed successfully.")
        return t
    except Exception as e:
        msg = traceback.format_exc()
        LOGGER.warn(msg)
        action="Please check log.txt for detailed traceback."
        userlog.emit("error", "JOB_FAILED", "job", f"Job '{job_name}' failed: {e}", action=action, context={"error": str(e)},)
        raise
    finally:
        LOGGER.close(logfile)
