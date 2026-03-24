from flask import Flask, request, jsonify
import os 
import pandas as pd 
import traceback
from src.main import run as run_tandem
from src.features.Uniprot import SAV2SAV_coord
from src.utils.logger import LOGGER
from src.utils.settings import ROOT_DIR
from src.features import TANDEM_FEATS

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
        GJB2_test  = params.get("GJB2_test", None)
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
            refresh=params["refresh"],
            uniref90="/tandem/data/consurf/uniref90.fasta",
        )
        LOGGER.info(f"Inference result: ok")

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
