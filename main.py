from flask import Flask, request, jsonify
import os 
import json
import traceback
from src.main import run as run_tandem
from src.main import toSAV_coords
from src.utils.logger import LOGGER
from src.utils.settings import ROOT_DIR

tandem_jobs = os.path.join(ROOT_DIR, 'jobs')
app = Flask(__name__)

@app.route("/run_tandem_job", methods=["POST"])
def run_tandem_job():
    params = request.get_json()

    LOGGER.info(f"Received input: {params}")

    if not params:
        LOGGER.error("No JSON received")

        return jsonify({"error": "No JSON received"}), 400

    try:
        session_id = params["session_id"]
        job_name   = params["job_name"]
        model      = params["model"]
        if model in ["TANDEM", "TANDEM-DIMPLE for GJB2", "TANDEM-DIMPLE for RYR1"]:
            pretrained_model_folder = os.path.join(ROOT_DIR, 'models', model)
        else:
            pretrained_model_folder = os.path.join(tandem_jobs, session_id, model)
            
        run_tandem(
            query=toSAV_coords(params["SAV"]),
            labels=params["label"],
            custom_PDB=params["label"],
            job_name=f"{session_id}/{job_name}",
            pretrained_model_folder=pretrained_model_folder,
        )
        LOGGER.info(f"Inference result: ok")

        with open(f"{tandem_jobs}/{session_id}/{job_name}/params.json", "w") as f:
            json.dump(params, f, indent=4)

        return jsonify({"output": 'ok'})
    
    except Exception as e:
        msg = traceback.format_exc()
        LOGGER.info(msg)
        LOGGER.error(f"Error in inference: {e}")

        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
