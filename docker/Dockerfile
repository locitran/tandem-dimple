# =========================
# Dockerfile for TANDEM-DIMPLE
# =========================

# docker build -t tandem -f docker/Dockerfile .

# Base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /temp
COPY requirements.txt /temp/requirements.txt
COPY pyRONN /temp/pyRONN

# Install system dependencies (for ConSurf tools)
RUN apt update --quiet \
    && apt install --yes --quiet software-properties-common \
    && apt install --yes --quiet cd-hit prottest ncbi-blast+ \ 
    && apt install --yes --quiet hmmer mafft clustalw muscle \ 
    && apt install --yes --quiet gcc g++ python3.11-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment and install Python dependencies
RUN conda create -n tandem python=3.11.11 \
    && echo "source activate tandem" > ~/.bashrc \
    && /bin/bash -c "source activate tandem && pip install -r requirements.txt" \
    && conda install -c conda-forge -c bioconda mmseqs2

# Download ConSurf database (optional step: comment out if not needed inside container)
# RUN bash scripts/download_consurf_db.sh data/consurf/db

# Download UniRef90 genetic database (optional step: huge file ~47GB; probably better to mount externally)
# RUN bash scripts/download_uniref90.sh data/consurf

# Download Pfam database
# RUN bash scripts/download_pfam.sh data/pfamdb

# Default command: activate conda environment and show help
CMD ["/bin/bash"]