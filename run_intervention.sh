#!/bin/bash
#SBATCH --time=0-120:00 # D-HH:MM
#SBATCH --account=def-zhijing
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load arrow/21.0.0

pip install -r requirements.txt





# Set paths
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae"

CONFIG_FILE="/home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/evaluation_config.yaml"
# Run evaluation
echo "Starting evaluation..."
python /home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/sae_vs_steering_evaluation.py \
    --config $CONFIG_FILE

echo "Evaluation completed!"
echo "Results saved in: /home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/evaluation_results/"