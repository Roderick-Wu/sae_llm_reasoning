#!/bin/bash
#SBATCH --time=0-120:00 # D-HH:MM
#SBATCH --account=def-zhijing
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:4
module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load arrow/21.0.0

pip install -r requirements.txt



# SAE Training Script
# Train Sparse Autoencoders for reasoning feature extraction

echo "Starting SAE training for reasoning feature extraction..."

# Set paths
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae"

# Create output directories
mkdir -p /home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/outputs
mkdir -p /home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/outputs/checkpoints
mkdir -p /home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/outputs/analysis

# Training configuration
MODEL_NAME="Llama-3.1-8B"
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"  # Focus on later layers first
EPOCHS=50
BATCH_SIZE=256
LEARNING_RATE=1e-4
SPARSITY_PENALTY=1e-3
EXPANSION_FACTOR=4

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layers: $LAYERS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Sparsity Penalty: $SPARSITY_PENALTY"
echo "  Expansion Factor: $EXPANSION_FACTOR"

# Run training
python /home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/sae_reasoning_trainer.py \
    --model_name $MODEL_NAME \
    --layers $LAYERS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --sparsity_penalty $SPARSITY_PENALTY \
    --expansion_factor $EXPANSION_FACTOR

echo "SAE training completed!"

# Run evaluation
echo "Starting evaluation..."
python /home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/sae_vs_steering_evaluation.py \
    --model_name $MODEL_NAME \
    --datasets mmlu-pro \
    --layers $LAYERS \
    --scales 0.1 0.3 0.5

echo "Evaluation completed!"
echo "Results saved in: /home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/evaluation_results/"