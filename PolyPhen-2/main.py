import os
import subprocess

from flask import Flask, jsonify, request

PPH = os.path.realpath(os.environ.get("PPH", "/pph2/polyphen-2.2.3"))
RUN_PPH = os.path.realpath(os.environ.get("RUN_PPH", os.path.join(PPH, "bin", "run_pph.pl")))
CACHE_ROOT = os.path.realpath(
    os.environ.get("NONHUMAN_CACHE_DIR", os.path.join(os.path.dirname(PPH), "cached_nonhuman"))
)
PORT = int(os.environ.get("PORT", "5001"))

app = Flask(__name__)

def safe_path_name(name):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name)

@app.route("/run_pph2", methods=["POST"])
def run_pph2():
    payload = request.get_json(silent=True) or {}
    input_file = payload.get("input_file", None)
    job_dir = payload.get("job_dir", None)
    nonhuman = bool(payload.get("nonhuman", False))
    fasta_file = payload.get("fasta_file", None)
    config_dir = payload.get("config_dir", os.path.join(PPH, "nonhuman_config"))
    timeout = 3600

    if not input_file:
        return jsonify({"error": "Provide input_file."}), 400

    if not os.path.exists(input_file):
        return jsonify({"error": "Input file not found: {}".format(input_file)}), 400

    if nonhuman:
        if not fasta_file:
            return jsonify({"error": "Provide fasta_file for nonhuman PolyPhen-2 runs."}), 400
        if not os.path.exists(fasta_file):
            return jsonify({"error": "FASTA file not found: {}".format(fasta_file)}), 400
        if not os.path.isdir(config_dir):
            return jsonify({"error": "PolyPhen-2 config directory not found: {}".format(config_dir)}), 400

    input_dir = os.path.dirname(input_file)
    input_name = os.path.basename(input_file)
    input_stem, input_ext = os.path.splitext(input_name)
    cache_dir = None

    out_dir = input_dir if not job_dir else job_dir
    output = os.path.join(out_dir, "{}-pph2output{}".format(input_stem, input_ext))
    log = os.path.join(out_dir, "{}-pph2log{}".format(input_stem, input_ext))
    cmd = [RUN_PPH]
    if nonhuman:
        accession = os.path.splitext(os.path.basename(fasta_file))[0]
        if not accession:
            return jsonify({"error": "Cannot read accession from FASTA file path: {}".format(fasta_file)}), 400
        cache_dir = os.path.join(CACHE_ROOT, safe_path_name(accession))
        os.makedirs(cache_dir, exist_ok=True)
        cmd.extend(["-c", config_dir, "-s", fasta_file, "-d", cache_dir])
    cmd.append(input_file)

    try:
        with open(output, "w") as stdout, open(log, "w") as stderr:
            result = subprocess.run(cmd, cwd=PPH, stdout=stdout, stderr=stderr,
                universal_newlines=True, timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "PolyPhen-2 timed out after {} seconds.".format(timeout)}), 504
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    response = {
        "command": cmd,
        "returncode": result.returncode,
        "input_file": input_file,
        "output": output,
        "log": log,
        "nonhuman": nonhuman,
        "cache_dir": cache_dir,
    }
    status_code = 200 if result.returncode == 0 else 500
    return jsonify(response), status_code

@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "pph": PPH,
            "run_pph": RUN_PPH,
            "run_pph_exists": os.path.exists(RUN_PPH),
            "nonhuman_cache_root": CACHE_ROOT,
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
