#!/bin/bash
#SBATCH --job-name=task_vector_task
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --output=<path to log folder>/%x.%j.out


# single CPU only script
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.7.0


cd <path to project>
pip install -r requirements.txt

echo "Python version: $(python --version)"
nvidia-smi

python -m main --model_name microsoft/phi-2 --dataset_name antonym --icl_examples 4 --batchsize 32
