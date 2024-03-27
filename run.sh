#!/bin/bash
#SBATCH --job-name=general_task_vector
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=/home1/p313544/slurm_logs/%x.%j.out


# single GPU only script
module purge
# module load Python/3.9.6-GCCcore-11.2.0
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0

echo "Python version: $(python --version)"
nvidia-smi
pwd

# User's vars
# All scripts must be in the PATH_TO_PRJ/scripts directory!
PATH_TO_PRJ=/home1/p313544/Documents/general-task-vectors

# checkpoint save path
export PATH_TO_STORAGE=/scratch/p313544/storage_cache/

cd $PATH_TO_PRJ
source .venv/bin/activate


echo "Executing python script..."


(yes | python -m main \
    --model_name stabilityai/stablelm-2-zephyr-1_6b \
    --dataset_name XS_test \
    --mean_support 70 \
    --aie_support 20 \
    --icl_examples 4 \
    --batch_size 1 \
    --load_in_8bit \
    --use_local_backups \
)




# performance notes
# on Tesla T4 (~ 16GB)
#   1B 8bit model + 16 batchsize (up to 60% of memory)
#   1B 8bit model + 32 batchsize (up to 80% of memory) (safe spot)
