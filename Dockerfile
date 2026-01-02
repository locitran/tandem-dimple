# Base image with Conda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /tandem

# --- Step 1: Copy only dependency declarations first ---
COPY ./requirements.txt ./requirements.txt

# Optional: copy only environment-relevant parts early
COPY ./pyRONN ./pyRONN

# --- Step 2: Install system dependencies ---
RUN apt update --quiet \
    && apt install --yes --quiet software-properties-common \
    && apt install --yes --quiet cd-hit prottest ncbi-blast+ \
    && apt install --yes --quiet hmmer mafft clustalw muscle \
    && apt install --yes --quiet gcc g++ python3.11-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Step 3: Create Conda environment + install Python deps ---
RUN conda create -n tandem python=3.11.11 \
    && echo "source activate tandem" > ~/.bashrc \
    && /bin/bash -c "source activate tandem && pip install flask && pip install -r requirements.txt" \
    && conda install -n tandem -c conda-forge -c bioconda mmseqs2

# --- Step 4: Runtime environment variables ---
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=tandem
ENV CONDA_PREFIX=/opt/conda/envs/tandem
ENV PATH=/opt/conda/envs/tandem/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/tandem/lib:$LD_LIBRARY_PATH

# --- Step 5: Run server ---
EXPOSE 5000
CMD ["conda", "run", "--no-capture-output", "-n", "tandem", "python", "main.py"]
