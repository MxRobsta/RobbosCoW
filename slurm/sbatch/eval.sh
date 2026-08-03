#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=75G
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --output=slurm/logs/eval/%j.out
#SBATCH --mail-user=rwhsutherland1@sheffield.ac.uk
#SBATCH --mail-type=ALL

module load GCC/12.3.0
module load libsndfile
module load FFmpeg
export SSL_CERT_FILE="/etc/ssl/certs/ca-bundle.crt"

apptainer exec container/env.sif uv run whisper_inference.py dataset=cpc3 subsets="$1"
