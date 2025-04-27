#!/bin/bash
# This file is copyrighted from Alphafold2 
# https://github.com/google-deepmind/alphafold/blob/main/scripts/download_uniref90.sh
set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v aria2c &> /dev/null ; then
    echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
    exit 1
fi

DOWNLOAD_DIR="$1"
SOURCE_URL="https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"
BASENAME=$(basename "${SOURCE_URL}")

# Start timer
SECONDS=0

mkdir --parents "${DOWNLOAD_DIR}"
aria2c "${SOURCE_URL}" --dir="${DOWNLOAD_DIR}"

pushd "${DOWNLOAD_DIR}"
gunzip "${BASENAME}"
popd

# Display elapsed time
duration=$SECONDS
echo "Download and extraction completed in $(($duration / 60)) minutes and $(($duration % 60)) seconds."