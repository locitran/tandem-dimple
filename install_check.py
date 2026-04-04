import importlib
import os
import shutil
import subprocess
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PYTHON_IMPORTS = [
    ("flask", "flask"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("tensorflow", "tensorflow"),
    ("sklearn", "scikit-learn"),
    ("shap", "shap"),
    ("prody", "ProDy"),
    ("Bio", "biopython"),
    ("openmm", "OpenMM"),
    ("propka", "propka"),
    ("pymongo", "pymongo"),
    ("requests", "requests"),
    ("ml_collections", "ml_collections"),
    ("fpdf", "fpdf"),
    ("gdown", "gdown"),
    ("more_itertools", "more_itertools"),
    ("rcsbapi", "rcsb-api"),
    ("pyRONN", "pyRONN"),
]

MODULE_IMPORTS = [
    ("main", "tandem main"),
    ("src.main", "src.main"),
    ("src.core", "src.core"),
    ("src.features.Uniprot", "src.features.Uniprot"),
]

CLI_COMMANDS = [
    "python",
    "mmseqs",
    "blastp",
    "hmmscan",
    "mafft",
    "clustalw",
    "muscle",
    "cd-hit",
]

LOCAL_EXECUTABLES = [
    os.path.join(ROOT_DIR, "src", "features", "bin", "mkdssp"),
    os.path.join(ROOT_DIR, "src", "features", "bin", "naccess"),
    os.path.join(ROOT_DIR, "src", "features", "bin", "hbplus"),
]


def check_import(module_name, label):
    try:
        importlib.import_module(module_name)
        return True, f"OK import: {label}"
    except Exception as exc:
        return False, f"FAIL import: {label} -> {exc}"


def check_command(command_name):
    resolved = shutil.which(command_name)
    if resolved:
        return True, f"OK command: {command_name} -> {resolved}"
    return False, f"FAIL command: {command_name} not found on PATH"


def check_local_executable(path):
    relative_path = os.path.relpath(path, ROOT_DIR)
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return True, f"OK bundled executable: {relative_path}"
    if os.path.isfile(path):
        return False, f"FAIL bundled executable: {relative_path} is not executable"
    return False, f"FAIL bundled executable: {relative_path} is missing"


def run_checks():
    print("=== TANDEM installation check ===")
    failures = []

    for module_name, label in PYTHON_IMPORTS:
        ok, message = check_import(module_name, label)
        print(message)
        if not ok:
            failures.append(message)

    for module_name, label in MODULE_IMPORTS:
        ok, message = check_import(module_name, label)
        print(message)
        if not ok:
            failures.append(message)

    for command_name in CLI_COMMANDS:
        ok, message = check_command(command_name)
        print(message)
        if not ok:
            failures.append(message)

    for path in LOCAL_EXECUTABLES:
        ok, message = check_local_executable(path)
        print(message)
        if not ok:
            failures.append(message)

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    if pip_check.returncode == 0:
        print("OK dependency check: pip check")
    else:
        message = "FAIL dependency check: pip check\n" + (pip_check.stdout or pip_check.stderr).strip()
        print(message)
        failures.append(message)

    if failures:
        print("\nTANDEM installation check failed.")
        print("The Docker image is missing one or more required imports or executables.")
        return 1

    print("\nTANDEM installation successful.")
    print("All checked Python imports and executable dependencies are available.")
    return 0


if __name__ == "__main__":
    sys.exit(run_checks())
