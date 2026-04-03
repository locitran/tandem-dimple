#!/bin/bash
#SBATCH --job-name=notebook              # Job name
#SBATCH --mail-type=BEGIN                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=quangloctrandinh1998vn@gmail.com  # Where to send mail
#SBATCH --ntasks=1                       # Run on a single CPU
#SBATCH --cpus-per-task=1               # Cores
#SBATCH --gres=gpu:1g.5gb:0             # Request GPU "generic resources"
#SBATCH --mem=15gb                       # Job memory request
#SBATCH --output=notebook.log            # Standard output and error log
#SBATCH --partition=COMPUTE1Q            # The partition that job submit to
#SBATCH --account=YangLab                # The account name

# Get an available port
port=$(getAvailablePort)

# Port forward to the login node
/usr/bin/ssh -N -f -R $port:localhost:$port yang_loci@a100


# notebook_link -i notebook.log

conda activate tandem
echo $port
###################################################################################################
python -m notebook --no-browser --allow-root --port $port --NotebookApp.allow_remote_access=True
###################################################################################################


###################################################################################################
# Pytorch environment

# singularity exec \
# --bind /mnt/nas_1/YangLab/loci/NativeEnsembleWeb_copy:/loci \
# --nv /mnt/nas_1/YangLab/loci/NativeEnsembleWeb_copy/images/pytorch_1.11.0-cuda11.3-cudnn8-devel.sif bash

# # Start Jupyter Notebook
# python -m notebook --no-browser --allow-root --port $port --NotebookApp.allow_remote_access=True
###################################################################################################

# singularity exec --bind /mnt/nas_1/YangLab/loci/NativeEnsembleWeb_copy/improve:/improve \
#                  --env MLM_LICENSE_FILE=/improve/license.lic \
#                  -w /mnt/nas_1/YangLab/loci/NativeEnsembleWeb_copy/images/improve1.2 bash -c "
#     source improve.sh
#     python -m notebook --no-browser --allow-root --port $port --NotebookApp.allow_remote_access=True
#     "
