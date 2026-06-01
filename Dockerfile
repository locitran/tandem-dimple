# Base image with Conda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /tandem

# --- Step 1: Copy only dependency declarations first ---
COPY ./requirements.txt ./requirements.txt
COPY ./install_check.py ./install_check.py

# Optional: copy only environment-relevant parts early
COPY ./pyRONN ./pyRONN

# --- Step 2: Install system dependencies ---
RUN apt update --quiet \
    && apt install --yes --quiet software-properties-common \
    && apt install --yes --quiet cd-hit prottest ncbi-blast+ \
    && apt install --yes --quiet hmmer mafft clustalw muscle \
    && apt install --yes --quiet gcc g++ python3.11-dev libgfortran5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Step 3: Create Conda environment + install Python deps ---
# Remove any checked-in pyRONN binaries so pip builds a fresh extension for this image.
RUN find ./pyRONN/ronn -maxdepth 1 -name 'libronn*.so' -delete

RUN conda create -n tandem python=3.11.11 \
    && echo "source activate tandem" > ~/.bashrc \
    && /bin/bash -c "source activate tandem && pip install --upgrade pip && pip install -r requirements.txt" \
    && conda install -n tandem -c conda-forge -c bioconda mmseqs2

# --- Step 4: Copy application source and verify installation ---
COPY . .

# --- Step 5: Runtime environment variables ---
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=tandem
ENV CONDA_PREFIX=/opt/conda/envs/tandem
ENV PATH=/opt/conda/envs/tandem/bin:$PATH

RUN conda run --no-capture-output -n tandem python install_check.py

# --- Step 6: Run server ---
EXPOSE 5000
CMD ["conda", "run", "--no-capture-output", "-n", "tandem", "python", "main.py"]
