#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=75G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --output=slurm/logs/eval/%j.out
#SBATCH --mail-user=rwhsutherland1@sheffield.ac.uk
#SBATCH --mail-type=ALL

module load GCC/14.3.0
module load libsndfile
module load FFmpeg
export SSL_CERT_FILE="/etc/ssl/certs/ca-bundle.crt"

source .venv/bin/activate

python3 train.py model=$1 dataset=$2 train.lr=$3 exp_name=$1.$2.$3
