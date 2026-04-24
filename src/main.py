import os 
import shutil
import datetime
import numpy as np
import traceback
from .core import Tandem
from .utils.settings import ROOT_DIR
from .utils.logger import LOGGER
from .utils.settings import TANDEM_v1dot1
from .utils.user_log import UserLog, USERLOG_MESSAGES, VALIDATING_STAGE, MAPPING_STAGE, FEATURE_STAGE, MODEL_STAGE, REPORT_STAGE

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
    report_path = os.path.join(job_directory, "process_log.txt")
    
    ## LOGGER
    LOGGER._times = {}
    LOGGER._reports = {}
    LOGGER._report_times = {}
    logfile = os.path.join(job_directory, 'log.txt')
    LOGGER.start(logfile)
    LOGGER.info(f"Job name: {job_name} started at {datetime.datetime.now()}")
    LOGGER.info(f"Job directory: {job_directory}")
    LOGGER.timeit("_runtime")
    current_stage = VALIDATING_STAGE

    try:
        userlog.timeit(label="_Tandem")
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
        userlog.report(label="_Tandem", stage=VALIDATING_STAGE, file='SAVs.txt')

        current_stage = MAPPING_STAGE
        userlog.timeit(label="_getUniprot2PDBmap")
        t.getUniprot2PDBmap(filename='Uniprot2PDB.txt', folder=job_directory)
        userlog.report(label="_getUniprot2PDBmap", stage=MAPPING_STAGE, file='Uniprot2PDB.txt')

        current_stage = FEATURE_STAGE
        userlog.timeit(label="_getFeatMatrix")
        t.getFeatMatrix(withSAVs=True, filename='features.txt', folder=job_directory)    

        if isinstance(features, np.ndarray):  
            t.featMatrix = features
            t.getFeatMatrix(withSAVs=True, filename='features.txt', folder=job_directory)    
        userlog.report(label="_getFeatMatrix", stage=FEATURE_STAGE, file='features.txt')

        current_stage = MODEL_STAGE
        if labels:
            userlog.timeit(label="_training")
            t.setLabels(labels)
            t.setConfig(config)
            t.train()
            userlog.report(label="_training", stage=MODEL_STAGE, file='test_evaluation.txt')
        else:
            userlog.timeit("_prediction")
            t.getPredictions(models=pretrained_model_folder, folder=job_directory, filename='Main_Predictions.txt')
            t.plotSHAP(folder=job_directory)
            userlog.report("_prediction", stage=MODEL_STAGE, file='Main_Predictions.txt')

        current_stage = REPORT_STAGE
        userlog.timeit("_prediction")
        for label in LOGGER._reports:
            LOGGER.info(f"  {label}: {LOGGER._reports[label]:.2f}s ({LOGGER._report_times[label]} time(s))")
        LOGGER.report('Run time elapsed in %.2fs.', "_runtime")

        if log_time:
            log_time_file = os.path.join(job_directory, 'log_time.json')
            extra_data = {}
            if getattr(t, "Uniprot2PDBmap", None) is not None:
                for field in ('Uniprot_sequence_length', 'Asymmetric_PDB_length', 
                    "Asymmetric_PDB_resolved_length", "BioUnit_PDB_length", "OPM_PDB_length"):
                    values = np.asarray(t.Uniprot2PDBmap[field]).tolist()
                    extra_data[field] = values
            LOGGER.dump_time(log_time_file, extra_data=extra_data)
        userlog.report("_prediction", stage=REPORT_STAGE, file='report.txt')
        userlog.dump_report(report_path)
        return t
    except Exception as e:
        msg = traceback.format_exc()
        LOGGER.warn(msg)
        job_failed = USERLOG_MESSAGES["JOB_FAILED"]
        userlog.emit(level="error", stage=current_stage,
            message=job_failed["message"].format(job_name=job_name, error=str(e)),
            action=job_failed["action"],
            context={"error": str(e)},
        )
        userlog.dump_report(report_path)
        raise
    finally:
        LOGGER.close(logfile)
