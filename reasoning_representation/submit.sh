#!/bin/bash
#SBATCH --time=0-120:00 # D-HH:MM
#SBATCH --account=def-zhijing
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:4
module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load arrow/21.0.0

pip install numpy
pip install pandas
pip install torch
pip install transformers
pip install nltk
pip install rouge
pip install datasets
pip install seaborn
pip install scikit-learn
#pip install textstat
#pip install spacy

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python LiReFs_storing_hs.py

#python reasoning_representation/Intervention/features_intervention.py
cd Intervention
python features_intervention.py 

# salloc --account=def-zhijing --mem=64G --gpus-per-node=h100:4
# python analysis/steering_vector_effectiveness.py --hs_cache_dir /home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/outputs --output_dir ./analysis_results/effectiveness --model_name Llama-3.1-8B


