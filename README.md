This repository contains the source code for the TANDEM-DIMPLE project. 

Github repository: https://github.com/locitran/tandem-dimple.git. Fetch the code from the repository:
```bash
mkdir tandem
cd tandem
git clone https://github.com/locitran/tandem-dimple.git .
```

# Installation dependencies

We assume you have already installed Anaconda or Miniconda. If not, please install it first.

```bash
conda create -n tandem python=3.11.11
conda activate tandem
pip install -r requirements.txt
```

## ConSurf database

We have precomputed the [ConSurf database](https://consurfdb.tau.ac.il/) and stored it in my Google Drive accessible [here](https://drive.google.com/file/d/17IFFwGVHrJuUET3J8kEM9sqq2D0Z6Fco/view?usp=drive_link).

```bash
bash scripts/download_consurf_db.sh data/consurf/db # 2.5G, ~2m
```

## ConSurf Tool

In case user provides custom structure or no records found in ConSurf database, we need to use ConSurf tool. And to run ConSurf tool we need genetic (sequence) database, e.g. UniRef90 (default).

### 1. ConSurf Tool dependencies

```bash
conda install -c conda-forge -c bioconda mmseqs2
sudo apt install cd-hit
sudo apt install prottest 
sudo apt install ncbi-blast+
sudo apt install hmmer
sudo apt install mafft
sudo apt install clustalw
sudo apt install muscle
```

### 2. Download genetic database

```bash
bash scripts/download_uniref90.sh data/consurf # 90G, ~127m
```

## Pfam database

To generate Entropy and ranked_MI, we need Pfam database.

```bash
bash scripts/download_pfam.sh data/pfamdb # 1.5G, ~1.5m
```

# Test the installation
```bash
python test/input_as_list_SAVs.py
```

# Docker 

We have provided a Dockerfile to build the docker image `docker build -t tandem -f docker/Dockerfile`. 

The image is lack of databases, so you need to mount the databases to the container. 
Also, scripts are mounted to the container. 

```bash
# cwd: path/to/tandem
# pfamdb: path/to/tandem/data/pfamdb
# consurf: path/to/tandem/data/consurf
docker run -it \
  -v .:/tandem \
  -w /tandem \
  tandem:latest bash \
  -c "source activate tandem && python test/input_as_list_SAVs.py"
```

```bibtex
@article{Loci2025,
  author  = {Loci Tran, Chen-Hua Lu, Pei-Lung Chen, Lee-Wei Yang},
  journal = {Bioarchiv},
  title   = {Predicting the pathogenicity of SAVs Transfer-leArNing-ready and Dynamics-Empowered Model for DIsease-specific Missense Pathogenicity Level Estimation},
  year    = {2025},
  volume  = {*.*},
  number  = {*.*},
  pages   = {*.*},
  doi     = {*.*}
}
```   




 File "/tandem/src/features/SEQ.py", line 65, in _searchPfam
    hmmscan_file = run_hmmscan(fasta_file)
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/tandem/src/features/Pfam.py", line 117, in run_hmmscan
    stdout=open(out, 'w'), # Redirect standard output to the file
           ^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/tandem/src/features/tmp/a6ddb390-2651-11f0-94b2-0242ac110006_hmmscan_out'

