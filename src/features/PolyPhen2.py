# -*- coding: utf-8 -*-
"""This module defines functions for querying the PolyPhen-2 online tool,
parsing its output and deriving features that will be used by the Rhapsody
classifiers.
"""

import os
import requests
import numpy as np
from ..utils.logger import LOGGER

__all__ = ['PP2_FEATS', 'printSAVlist', 'calcPolyPhen2']

PP2_FEATS = ['wt_PSIC', 'Delta_PSIC']
"""List of features derived from PolyPhen-2's output."""

def printSAVlist(input_SAVs, filename):
    if isinstance(input_SAVs, str):
        input_SAVs = [input_SAVs]
    with open(filename, 'w', 1) as f:
        for i, line in enumerate(input_SAVs):
            m = f'error in SAV {i}: '
            assert isinstance(line, str), f'{m} not a string'
            assert len(line) < 25, f'{m} too many characters'
            print(line.upper(), file=f)
    LOGGER.info(f'SAVs saved to {filename}')
    return filename

def parsePolyPhen2(file):
    assert os.path.exists(file), "parsePolyPhen2 input is a file."

    with open(file, 'r') as f:
        lines = f.readlines()
    
    # find header line
    header = next((l.strip() for l in lines if l.strip() and l[0] == "#"), None)
    if header is None:
        raise ValueError("Cannot find PolyPhen-2 header line starting with '#'.")
    # parse header columns
    pph2_columns = [w.strip().lstrip("#") for w in header.split("\t")]
    # keep only non-empty non-header lines
    lines = [l for l in lines if l.strip() and l[0] != '#']
    if not lines:
        msg = (
            "PolyPhen-2's output is empty. Please check file 'pph2-log.txt' "
            "in the output folder for error messages from PolyPhen-2. \n"
            "Typical errors include: \n"
            "1) query contains *non-human* variants \n"
            "2) variants' format is incorrect (e.g. "
            '"UniprotID pos wt_aa mut_aa") \n'
            "3) wild-type amino acids are in the wrong position on the "
            "sequence (please refer to Uniprot's canonical isoform) \n"
            "4) Uniprot accession number is not recognized by PolyPhen-2. \n"
        )
        raise RuntimeError(msg)
    # define a structured array
    pl_dtype = np.dtype([(col, 'U25') for col in pph2_columns])
    parsed_lines = np.zeros(len(lines), dtype=pl_dtype)
    # fill structured array
    n_cols = len(pph2_columns)
    for i, line in enumerate(lines):
        # parse line
        words = [w.strip() for w in line.split('\t')]
        # check format
        n_words = len(words)
        if n_words == n_cols - 1:
            # manually insert null 'other' column
            words.append('?')
        elif n_words != n_cols:
            msg = 'Incorrect number of columns: {}'.format(n_words)
            raise ValueError(msg)
        # import to structured array
        parsed_lines[i] = tuple(words)
    LOGGER.info("PolyPhen-2's output parsed.")
    return parsed_lines

def calcPolyPhen2(SAV_coords, filename='SAVs.txt', folder='.', timeout=3700):
    """Run PolyPhen-2 through the local PolyPhen-2 container service."""
    """
    from src.features.PolyPhen2 import calcPolyPhen2, parsePolyPhen2

    output_file = 'SAVs-pph2output.txt'
    parse = parsePolyPhen2(output_file)
    f = calcPolyPhen2(SAV_coords)
    SAV_coords = ["Q8TDI8 2 S P",
        "Q8TDI8 4 K Q",
        "Q8TDI8 8 I V",
        "Q8TDI8 8 I N",
        "O00255 176 R Q",
        "O00255 177 D Y",
        "Q9P2D1 72 Y C",
        "Q9P2D1 86 P R",
        ]
    """
    service_url='http://polyphen2:5001/run_pph2'
    _dtype = np.dtype([('wtPSIC', 'f'), ('deltaPSIC', 'f')])
    os.makedirs(folder, exist_ok=True)

    # Print SAVs to a shared path visible from the tandem and polyphen2 containers.
    SAV_file = printSAVlist(SAV_coords, f'{folder}/{filename}')
    LOGGER.timeit('_pph2')
    payload = {"input_file": os.path.abspath(SAV_file), "job_dir": folder}

    try:
        LOGGER.info(f"Submitting query to local PolyPhen-2 service: {service_url}")
        response = requests.post(service_url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to connect to local PolyPhen-2 service: {exc}") from exc
    finally:
        if os.path.exists(SAV_file):
            os.remove(SAV_file)

    if response.status_code != 200:
        error_text = response.text.strip()
        try:
            error_json = response.json()
            error_text = error_json.get("error", error_text)
        except ValueError:
            pass
        raise RuntimeError(f"Local PolyPhen-2 service failed: {error_text}")

    try:
        response_json = response.json()
    except ValueError as exc:
        raise RuntimeError("Local PolyPhen-2 service returned invalid JSON.") from exc

    output_file = response_json.get("output")
    log_file = response_json.get("log")
    returncode = response_json.get("returncode")

    if returncode != 0:
        message = "Local PolyPhen-2 finished with a non-zero exit code."
        if log_file:
            message += f" Please check '{log_file}'."
        raise RuntimeError(message)

    if not output_file or not os.path.exists(output_file):
        raise RuntimeError("Local PolyPhen-2 output file was not created.")

    LOGGER.info("PolyPhen-2 is running through the local container service...")
    parsed_lines = parsePolyPhen2(output_file)

    f_l = parsed_lines[['Score1', 'dScore']]
    f_t = [tuple(np.nan if x == '' else x for x in l) for l in f_l]
    features = np.array(f_t, dtype=_dtype)

    if log_file and os.path.exists(log_file):
        LOGGER.info(f"PolyPhen-2 log saved to {log_file}")
    LOGGER.report("PolyPhen-2 features have been calculated in %.2fs.", '_pph2')
    return features
