#!/bin/bash
#SBATCH --job-name=task_vector_task
#SBATCH --time=2:00:00
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

python -m main --model_name microsoft/phi-2 --dataset_name antonym --icl_examples 4 --batch_size 32 --load_in_8bit
python -m main --model_name microsoft/phi-2 --dataset_name sentiment --icl_examples 4 --batch_size 32 --load_in_8bit

(yes | python -m main --model_name stabilityai/stablelm-2-zephyr-1_6b --dataset_name antonym --icl_examples 10 --batch_size 32 --load_in_8bit)

# python -m main --model_name EleutherAI/pythia-1B --dataset_name sentiment --icl_examples 4 --batch_size 32 --load_in_8bit
# performance notes
# on Tesla T4 (~ 16GB)
#   1B 8bit model + 16 batchsize (up to 60% of memory)
#   1B 8bit model + 32 batchsize (up to 80% of memory) (safe spot)
