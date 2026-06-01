from flask import Flask, request, jsonify
import fcntl
import os 
import pandas as pd 
import threading
import traceback
from src.main import run as run_tandem
from src.features.Uniprot import SAV2SAV_coord
from src.utils.logger import LOGGER
from src.utils.settings import ROOT_DIR
from src.features import TANDEM_FEATS

tandem_jobs = os.path.join(ROOT_DIR, 'jobs')
app = Flask(__name__)
tandem_job_lock = threading.Lock()

@app.route("/run_tandem_job", methods=["POST"])
def run_tandem_job():
    params = request.get_json(silent=True)
    if not params:
        return jsonify({"error": "No JSON received"}), 400

    if not tandem_job_lock.acquire(blocking=False):
        return jsonify({"error": "Tandem container is busy"}), 409

    job_lock = None
    try:
        session_id = params["session_id"]
        job_name   = params["job_name"]
        job_directory = os.path.join(tandem_jobs, session_id, job_name)
        os.makedirs(job_directory, exist_ok=True)
        job_lock = open(os.path.join(job_directory, ".execution.lock"), "a")
        completion_marker = os.path.join(job_directory, ".execution.complete")
        try:
            fcntl.flock(job_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return jsonify({"error": "Job is already executing"}), 409
        if os.path.exists(completion_marker):
            return jsonify({"error": "Job output already exists"}), 409

        model      = params["model"]
        GJB2_test  = params.get("GJB2_test", None)
        refresh    = params.get("refresh", False)
        query = SAV2SAV_coord(params["SAV"])

        if model in ["TANDEM", "TANDEM-DIMPLE for GJB2", "TANDEM-DIMPLE for RYR1"]:
            pretrained_model_folder = os.path.join(ROOT_DIR, 'models', model)
        else:
            pretrained_model_folder = os.path.join(tandem_jobs, session_id, model)
        
        if GJB2_test:
            df = pd.read_csv(f'{ROOT_DIR}/data/GJB2/final_features.csv')
            fm = df[df['SAV_coords'].isin(query)][TANDEM_FEATS['v1.1']]
            fm = fm.to_records(index=False)
        else:
            fm = None

        run_tandem(
            query=query,
            labels=params["label"],
            features=fm,
            custom_PDB=params["STR"],
            job_name=f"{session_id}/{job_name}",
            pretrained_model_folder=pretrained_model_folder,
            refresh=refresh,
            uniref90="/tandem/data/consurf/uniref90.fasta",
        )
        with open(completion_marker, "w") as marker:
            marker.write("completed\n")
        LOGGER.info(f"Inference result: ok")

        return jsonify({"output": 'ok'})
    
    except Exception as e:
        msg = traceback.format_exc()
        LOGGER.info(msg)
        LOGGER.error(f"Error in inference: {e}")

        return jsonify({"error": str(e)}), 500
    
    finally:
        if job_lock is not None:
            try:
                fcntl.flock(job_lock.fileno(), fcntl.LOCK_UN)
            finally:
                job_lock.close()
        tandem_job_lock.release()

@app.route("/health")
def health():
    return "OK", 200

@app.route("/available")
def available():
    if not tandem_job_lock.acquire(blocking=False):
        return jsonify({"available": False}), 409

    tandem_job_lock.release()
    return jsonify({"available": True}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
