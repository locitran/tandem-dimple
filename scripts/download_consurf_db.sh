#!/bin/bash
set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

# Check if gdown is installed
if ! command -v gdown &> /dev/null ; then
    echo "Error: gdown could not be found. Please install gdown (pip install gdown)."
    exit 1
fi

DOWNLOAD_DIR="$1"
GGDRIVE_ID="17IFFwGVHrJuUET3J8kEM9sqq2D0Z6Fco"

mkdir --parents "${DOWNLOAD_DIR}"

# Start timer
SECONDS=0
gdown "$GGDRIVE_ID" -O "${DOWNLOAD_DIR}/2024-10-08.tar.gz"
duration=$SECONDS
echo "Download completed in $(($duration / 60)) minutes and $(($duration % 60)) seconds."

pushd "${DOWNLOAD_DIR}/"
tar -xvf "2024-10-08.tar.gz"
# Remove the tar file after extraction
rm "2024-10-08.tar.gz"
popd

# Display elapsed time
duration=$SECONDS
echo "Extraction completed in $(($duration / 60)) minutes and $(($duration % 60)) seconds."