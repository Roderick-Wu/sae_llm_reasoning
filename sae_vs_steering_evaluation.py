#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from dataclasses import dataclass, field
from sae_reasoning_trainer import SparseAutoencoder, SAEConfig
import warnings
import yaml

# Import utility functions 
from utils import (
    load_dataset, 
    load_prompt_template, 
    get_prediction, 
    generate_questions,
    form_options,
    form_options_ceval
)

warnings.filterwarnings('ignore')

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    # Model and paths
    model_name: str = "Llama-3.1-8B"
    model_dir: str = "/home/wuroderi/projects/def-zhijing/wuroderi/models"
    sae_models_path: str = "/home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/outputs/final_results/models"
    hs_cache_path: str = "/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/outputs"
    dataset_dir: str = "/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset"
    output_dir: str = "/home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/evaluation_results"
    
    # Evaluation parameters
    # Dataset types:
    # - Reasoning: gsm8k, gsm-symbolic, mgsm  
    # - Memory: popqa, c-eval-h
    # - Mixed (needs reasoning_threshold): mmlu-pro
    test_datasets: List[str] = field(default_factory=lambda: ['gsm8k', 'popqa', 'mmlu-pro'])
    reasoning_threshold: float = 0.5  # Only used for mmlu-pro to split reasoning vs memory
    intervention_scales: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.8, 1.0])
    layers_to_evaluate: List[int] = field(default_factory=lambda: [10, 15, 20, 25])
    
    # Generation parameters
    max_new_tokens: int = 200
    batch_size: int = 8
    device: str = "cuda"


class FeatureExtractor:
    """Load and manage SAE models for feature extraction"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Load SAE models
        self.sae_models = {}
        self.load_sae_models()
        
        # Load hidden states and metadata
        self.load_data()
        
    def load_sae_models(self):
        """Load trained SAE models"""
        print("Loading SAE models...")
        
        for layer in self.config.layers_to_evaluate:
            model_path = os.path.join(self.config.sae_models_path, f'sae_layer_{layer}.pt')
            if os.path.exists(model_path):
                sae = None
                
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                # Extract config parameters
                sae_config = checkpoint.get('config', None)
                if sae_config is not None:
                    input_dim = sae_config.hidden_dim
                    dict_size = sae_config.hidden_dim * sae_config.expansion_factor
                    sparsity_penalty = sae_config.sparsity_penalty
                else:
                    # Use defaults if config not available
                    input_dim = 4096  # Default for Llama-3.1-8B
                    dict_size = input_dim * 4  # Default expansion factor
                    sparsity_penalty = 1e-3
                    print(f"  Warning: Config not found for layer {layer}, using defaults")
                    
                # Create SAE and load state dict
                sae = SparseAutoencoder(input_dim, dict_size, sparsity_penalty)
                sae.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Loaded layer {layer} with config")
                
                sae.to(self.device)
                sae.eval()
                self.sae_models[layer] = sae
            else:
                print(f"Warning: SAE model not found for layer {layer}: {model_path}")
    
    def load_data(self):
        """Load hidden states and reasoning metadata"""
        print("Loading evaluation data...")
        
        # Load hidden states
        hs_cache_path = os.path.join(self.config.hs_cache_path, f'{self.config.model_name}-base_hs_cache_no_cot.pt')
        self.hs_cache = torch.load(hs_cache_path, map_location='cpu', weights_only=False)
        
        # Load reasoning scores
        with open(os.path.join(self.config.dataset_dir, 'mmlu-pro-3000samples.json'), 'r') as f:
            self.samples = json.load(f)
        
        self.reasoning_scores = torch.tensor([sample['memory_reason_score'] for sample in self.samples])
        self.labels = (self.reasoning_scores > self.config.reasoning_threshold).float()
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Reasoning: {self.labels.sum().item()}, Memory: {(1-self.labels).sum().item()}")
    
    def extract_sae_features(self, layer: int, top_k: int = 10) -> Dict[str, torch.Tensor]:
        """Extract top reasoning features from SAE"""
        if layer not in self.sae_models:
            raise ValueError(f"SAE model not found for layer {layer}")
        
        sae = self.sae_models[layer]
        hidden_states = self.hs_cache[layer].squeeze(1).to(self.device)
        
        with torch.no_grad():
            sparse_acts, _ = sae(hidden_states)
        
        # Find top reasoning features
        reasoning_mask = self.labels.bool()
        memory_mask = ~reasoning_mask
        
        reasoning_acts = sparse_acts[reasoning_mask]
        memory_acts = sparse_acts[memory_mask]
        
        # Compute differential activation
        reasoning_mean = torch.mean(reasoning_acts, dim=0)
        memory_mean = torch.mean(memory_acts, dim=0)
        differential = reasoning_mean - memory_mean
        
        # Get top features
        top_indices = torch.topk(differential, k=top_k)[1] # (values, indices)[1]
        
        # Extract feature directions from decoder
        feature_directions = sae.decoder.weight[:, top_indices].T  # Transpose to flip to [top_k, hidden_dim]
        
        # Normalize all feature directions to unit vectors
        feature_directions = F.normalize(feature_directions, dim=-1)
        
        return {
            'feature_directions': feature_directions.cpu(),
            'feature_indices': top_indices.cpu(),
            'differential_scores': differential[top_indices].cpu(),
            'reasoning_activations': reasoning_mean[top_indices].cpu(),
            'memory_activations': memory_mean[top_indices].cpu()
        }
    



class InterventionEvaluator:
    """Evaluate reasoning interventions using different feature extraction methods"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.load_model()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(config)
        
        # Results storage
        self.results = {}

        self.generate_token_count = 200
    
    def load_model(self):
        """Load the language model for evaluation"""
        print(f"Loading model {self.config.model_name}...")
        
        model_path = os.path.join(self.config.model_dir, self.config.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            trust_remote_code=True
        )
        self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set padding token
        if 'llama' in self.config.model_name.lower():
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.tokenizer.padding_side = "left"
        
        print("Model loaded successfully!")
    

    

    
    def _get_model_architecture_info(self):
        """Get model architecture information for utils functions"""
        if 'gptj' in self.model.config.model_type.lower():
            layer_name = 'transformer.h'
            mlp_name = 'mlp'
            attn_name = 'attn'
        elif 'qwen' in self.model.config.model_type.lower() and 'qwen2' not in self.model.config.model_type.lower():
            layer_name = 'transformer.h'
            mlp_name = 'mlp'
            attn_name = 'attn'
        else:
            layer_name = 'model.layers'
            mlp_name = 'mlp'
            attn_name = 'self_attn'
            
        model_layers_num = int(getattr(self.model.config, 'num_hidden_layers', 
                                     getattr(self.model.config, 'n_layer', 32)))
        
        return layer_name, attn_name, mlp_name, model_layers_num
    
    def _map_dataset_name(self, dataset_name: str) -> str:
        """Map dataset names to utils format"""
        dataset_mapping = {
            'gsm8k': 'GSM8k',
            'mmlu-pro': 'MMLU-Pro', 
            'popqa': 'PopQA',
            'mgsm': 'MGSM',
            'gsm-symbolic': 'GSM-symbolic',
            'c-eval-h': 'C-Eval-H'
        }
        return dataset_mapping.get(dataset_name, 'MMLU-Pro')
    

     
    def sae_evaluation_on_dataset(self, val_sampled_data, prompts_cot=None, prompts_no_cot=None, 
                                 run_in_fewshot=True, run_in_cot=True, intervention_layers_dict=None, 
                                 intervention_type=None, 
                                 batch_size=4, scale=0.1, ds_name='MMLU-Pro'):
        """Custom evaluation function for SAE interventions (based on utils.evaluation_on_dataset)
        
        Args:
            intervention_layers_dict: Dict mapping layer_idx -> feature_directions for multi-layer intervention
        """
        
        queries_batch = []  
        entry_batch = []
        
        for ix, entry in tqdm(enumerate(val_sampled_data)):
            
            # Build query based on dataset type (same as utils version)
            if ds_name == 'MMLU-Pro':
                prefix_cot = prompts_cot[entry['category']]
                prefix_no_cot = prompts_no_cot[entry['category']]
                
                if run_in_fewshot:
                    if run_in_cot:
                        query = prefix_cot + 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + "\n\nA: Let's think step by step. "
                    else:
                        query = prefix_no_cot + 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + "\n\nA: "
                else:
                    query = 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + '\n' + "\nA: "
                    
            elif ds_name in ['GSM8k', 'GSM-symbolic', 'MGSM']:
                if run_in_fewshot:
                    if run_in_cot:
                        query = prompts_cot.format(TARGET_QUESTION=entry['question'])
                    else:
                        query = prompts_no_cot.format(TARGET_QUESTION=entry['question'])
                else:
                    query = 'Q: ' + entry['question'] + "\n\nA: "

            elif ds_name in ['PopQA']:
                query = 'Q: ' + entry['question'] + "\n\nA: "

            elif ds_name in ['C-Eval-H']:
                prefix_cot = prompts_cot[entry['subject']]
                if run_in_fewshot:
                    if run_in_cot:
                        query = prefix_cot + '问题: ' + entry['question'] + '\n' + form_options_ceval([entry['A'], entry['B'], entry['C'], entry['D']]) + '\n' + '让我们一步一步思考，'

            queries_batch.append(query)
            entry_batch.append(entry)

            if len(queries_batch) == batch_size or ix == len(val_sampled_data) - 1:
                
                # Generate responses with or without SAE intervention
                if intervention_layers_dict is not None:
                    responses = self.generate_with_sae_intervention(
                        queries_batch, intervention_layers_dict, 
                        intervention_type, scale
                    )
                else:
                    # Baseline generation without intervention
                    responses = generate_questions(
                        model=self.model, tokenizer=self.tokenizer, 
                        questions=queries_batch, n_new_tokens=self.generate_token_count
                    )
                    
                # Process responses and compute correctness (same as utils version)
                if ds_name == 'MMLU-Pro': 
                    for answer, entry in zip(responses, entry_batch):
                        entry['solution'] = answer
                        prediction = get_prediction(answer, ds_name)
                        entry['model_predict_correctness'] = (entry["answer"] == prediction)
                        
                        # Store intervention response if applicable
                        if intervention_layers_dict is not None:
                            entry['pred_with_intervention'] = answer

                elif ds_name in ['GSM8k', 'GSM-symbolic']:  
                    for answer, entry in zip(responses, entry_batch):
                        entry['solution'] = answer
                        prediction = get_prediction(answer, ds_name)
                        entry['model_predict_correctness'] = (entry["final_answer"] == prediction)
                        
                        if intervention_layers_dict is not None:
                            entry['pred_with_intervention'] = answer

                elif ds_name in ['MGSM']:  
                    for answer, entry in zip(responses, entry_batch):
                        prediction = get_prediction(answer, ds_name)
                        entry['model_predict_correctness'] = (float(entry['answer']) == prediction)
                        
                        if intervention_layers_dict is not None:
                            entry['pred_with_intervention'] = answer
                
                elif ds_name in ['C-Eval-H']:  
                    for answer, entry in zip(responses, entry_batch):
                        prediction = get_prediction(answer, ds_name)
                        entry['model_predict_correctness'] = (entry['answer'] == prediction)
                        
                        if intervention_layers_dict is not None:
                            entry['pred_with_intervention'] = answer

                elif ds_name == 'PopQA':
                    for answer, entry in zip(responses, entry_batch):
                        entry['solution'] = answer
                        possible_answers = json.loads(entry['possible_answers'])
                        is_correct = False
                        for pa in possible_answers:
                            if pa in answer or pa.lower() in answer or pa.capitalize() in answer:
                                is_correct = True
                        entry['model_predict_correctness'] = is_correct
                        
                        if intervention_layers_dict is not None:
                            entry['pred_with_intervention'] = answer
                        
                queries_batch = []
                entry_batch = []
        
        # Compute performance metrics after processing all data
        if ds_name == 'MMLU-Pro':
            # Split MMLU-Pro into reasoning vs memory based on scores
            reason_data = [entry for entry in val_sampled_data if entry.get('memory_reason_score', 0) > 0.5]
            memory_data = [entry for entry in val_sampled_data if entry.get('memory_reason_score', 0) <= 0.5]
            
            # Compute accuracies
            memory_correct = sum(1 for entry in memory_data if entry.get('model_predict_correctness', False))
            reason_correct = sum(1 for entry in reason_data if entry.get('model_predict_correctness', False))
            
            memory_accuracy = memory_correct / len(memory_data) if len(memory_data) > 0 else 0.0
            reason_accuracy = reason_correct / len(reason_data) if len(reason_data) > 0 else 0.0
            
            return {
                'memory_accuracy': memory_accuracy,
                'reason_accuracy': reason_accuracy
            }
        else:
            # For other datasets, compute overall reasoning accuracy
            correct_predictions = sum(1 for entry in val_sampled_data if entry.get('model_predict_correctness', False))
            total_predictions = len(val_sampled_data)
            reason_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            return {
                'reason_accuracy': reason_accuracy
            }
    
    def generate_with_sae_intervention(self, questions, intervention_layers_dict, 
                                     intervention_type, scale):
        """Generate text with SAE feature interventions
        
        Args:
            questions: List of input questions
            intervention_layers_dict: Dict mapping layer_idx -> feature_directions for that layer
            intervention_type: Type of intervention ('projected', 'raw', 'activation_patching')
            scale: Intervention scale
        """
        
        inputs = self.tokenizer(questions, return_tensors="pt", padding="longest", 
                               return_token_type_ids=False).to(self.device)
        input_length = inputs.input_ids.size(1)

        # Create SAE intervention hooks for all layers
        hooks = self.set_sae_intervention_hooks_multi_layer(
            intervention_layers_dict, intervention_type, scale
        )
        
        # Generate with hooks
        gen_tokens = self.model.generate(**inputs, max_new_tokens=self.generate_token_count, do_sample=False)
        
        self.remove_hooks(hooks)

        gen_text = self.tokenizer.batch_decode(gen_tokens[:, input_length:], skip_special_tokens=True)
        
        return gen_text
    
    def set_sae_intervention_hooks(self, layer, directions, intervention_type, scale, sae_model=None):
        """Set up hooks for SAE intervention at a specific layer"""
        
        hooks = []
        
        def create_sae_intervention_hook(directions, intervention_type, scale, sae_model=None):
            def hook_fn(module, input, output):
                # Handle tuple output (common in transformers)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                # Apply intervention to all positions (not just last token like steering)
                if intervention_type == 'projected':
                    # Use decoder weights as directions (existing approach)
                    directions_tensor = directions['feature_directions'].to(hidden_states.device).to(hidden_states.dtype)
                    
                    # Apply each feature direction (already normalized at extraction)
                    for direction in directions_tensor:
                        # Extra normalization for safety (already normalized but doesn't hurt)
                        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                        projection = (hidden_states @ direction).unsqueeze(-1) * direction
                        hidden_states = hidden_states + scale * projection
                        
                elif intervention_type == 'raw':
                    # Use feature directions directly without projection
                    # Scale each feature direction by its corresponding reasoning activation value
                    feature_directions = directions['feature_directions'].to(hidden_states.device).to(hidden_states.dtype)
                    reasoning_activations = directions['reasoning_activations'].to(hidden_states.device).to(hidden_states.dtype)
                    
                    # Add each feature direction (now normalized) scaled by activation
                    for i, (direction, activation) in enumerate(zip(feature_directions, reasoning_activations)):
                        # Ensure normalization (already done at extraction, but double-check)
                        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                        # Scale the normalized direction by activation strength and intervention scale
                        scaled_direction = direction * activation * scale
                        hidden_states = hidden_states + scaled_direction
                        
                elif intervention_type == 'activation_patching':
                    # Activation patching: encode to sparse, modify top-k features, decode back
                    if sae_model is None:
                        raise ValueError("SAE model required for activation patching intervention")
                    
                    # Get original shape
                    original_shape = hidden_states.shape  # [batch, seq, hidden]
                    
                    # Flatten to [batch*seq, hidden] for SAE processing
                    flat_hidden = hidden_states.view(-1, hidden_states.size(-1))
                    
                    # Encode to sparse representation
                    sparse_acts = sae_model.encode(flat_hidden)  # [batch*seq, dict_size]
                    
                    # Get top-k feature indices and values for modification
                    top_indices = directions.get('feature_indices', None)
                    reasoning_activations = directions['reasoning_activations'].to(hidden_states.device).to(hidden_states.dtype)
                    
                    if top_indices is not None:
                        # Convert to tensor if needed
                        if not isinstance(top_indices, torch.Tensor):
                            top_indices = torch.tensor(top_indices, device=sparse_acts.device)
                        top_indices = top_indices.to(sparse_acts.device)
                        
                        # Apply intervention to top-k features
                        for i, feat_idx in enumerate(top_indices):
                            if i < len(reasoning_activations):
                                # Option 1: Multiply existing activations by scale factor
                                sparse_acts[:, feat_idx] = sparse_acts[:, feat_idx] * (1 + scale * reasoning_activations[i])
                                
                                # Option 2: Set to specific value (uncomment to use instead)
                                # sparse_acts[:, feat_idx] = scale * reasoning_activations[i]
                    
                    # Decode back to hidden space
                    modified_hidden = sae_model.decode(sparse_acts)
                    
                    # Reshape back to original shape
                    hidden_states = modified_hidden.view(original_shape)
                
                # Update output
                if isinstance(output, tuple):
                    output = (hidden_states,) + output[1:]
                else:
                    output = hidden_states
                    
                return output
            
            return hook_fn
        
        # Get the target layer for intervention
        layer_name, _, _, _ = self._get_model_architecture_info()
        
        # Navigate to the specific layer
        attributes = layer_name.split(".")
        current_obj = self.model
        for attr in attributes:
            current_obj = getattr(current_obj, attr)
        target_layer = current_obj[layer]
        
        # Register hook on the layer (post-forward hook)
        hook = target_layer.register_forward_hook(
            create_sae_intervention_hook(directions, intervention_type, scale, sae_model)
        )
        hooks.append(hook)
        
        return hooks
    
    def set_sae_intervention_hooks_multi_layer(self, intervention_layers_dict, intervention_type, scale):
        """Set up hooks for SAE intervention across multiple layers simultaneously
        
        Args:
            intervention_layers_dict: Dict mapping layer_idx -> directions dict for that layer
            intervention_type: Type of intervention ('projected', 'raw', 'activation_patching')
            scale: Intervention scale
        """
        hooks = []
        
        def create_sae_intervention_hook(directions, intervention_type, scale, sae_model=None):
            def hook_fn(module, input, output):
                # Handle tuple output (common in transformers)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                # Apply intervention to all positions
                if intervention_type == 'projected':
                    # Use decoder weights as directions (existing approach)
                    directions_tensor = directions['feature_directions'].to(hidden_states.device).to(hidden_states.dtype)
                    
                    # Apply each feature direction (already normalized at extraction)
                    for direction in directions_tensor:
                        # Extra normalization for safety (already normalized but doesn't hurt)
                        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                        projection = (hidden_states @ direction).unsqueeze(-1) * direction
                        hidden_states = hidden_states + scale * projection
                        
                elif intervention_type == 'raw':
                    # Use feature directions directly without projection
                    feature_directions = directions['feature_directions'].to(hidden_states.device).to(hidden_states.dtype)
                    reasoning_activations = directions['reasoning_activations'].to(hidden_states.device).to(hidden_states.dtype)
                    
                    # Add each feature direction (now normalized) scaled by activation
                    for i, (direction, activation) in enumerate(zip(feature_directions, reasoning_activations)):
                        # Ensure normalization (already done at extraction, but double-check)
                        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                        # Scale the normalized direction by activation strength and intervention scale
                        scaled_direction = direction * activation * scale
                        hidden_states = hidden_states + scaled_direction
                        
                elif intervention_type == 'activation_patching':
                    # Activation patching: encode to sparse, modify top-k features, decode back
                    if sae_model is None:
                        raise ValueError("SAE model required for activation patching intervention")
                    
                    # Get original shape
                    original_shape = hidden_states.shape  # [batch, seq, hidden]
                    
                    # Flatten to [batch*seq, hidden] for SAE processing
                    flat_hidden = hidden_states.view(-1, hidden_states.size(-1))
                    
                    # Encode to sparse representation
                    sparse_acts = sae_model.encode(flat_hidden)  # [batch*seq, dict_size]
                    
                    # Get top-k feature indices and values for modification
                    top_indices = directions.get('feature_indices', None)
                    reasoning_activations = directions['reasoning_activations'].to(hidden_states.device).to(hidden_states.dtype)
                    
                    if top_indices is not None:
                        # Convert to tensor if needed
                        if not isinstance(top_indices, torch.Tensor):
                            top_indices = torch.tensor(top_indices, device=sparse_acts.device)
                        top_indices = top_indices.to(sparse_acts.device)
                        
                        # Apply intervention to top-k features
                        for i, feat_idx in enumerate(top_indices):
                            if i < len(reasoning_activations):
                                # Multiply existing activations by scale factor
                                sparse_acts[:, feat_idx] = sparse_acts[:, feat_idx] * (1 + scale * reasoning_activations[i])
                    
                    # Decode back to hidden space
                    modified_hidden = sae_model.decode(sparse_acts)
                    
                    # Reshape back to original shape
                    hidden_states = modified_hidden.view(original_shape)
                
                # Update output
                if isinstance(output, tuple):
                    output = (hidden_states,) + output[1:]
                else:
                    output = hidden_states
                    
                return output
            
            return hook_fn
        
        # Get the target layers for intervention
        layer_name, _, _, _ = self._get_model_architecture_info()
        
        # Navigate to the layers
        attributes = layer_name.split(".")
        current_obj = self.model
        for attr in attributes:
            current_obj = getattr(current_obj, attr)
        
        # Register hooks for each layer in the dict
        for layer_idx, directions in intervention_layers_dict.items():
            target_layer = current_obj[layer_idx]
            
            # Get SAE model for activation patching (if needed)
            sae_model = None
            if intervention_type == 'activation_patching' and layer_idx in self.feature_extractor.sae_models:
                sae_model = self.feature_extractor.sae_models[layer_idx]
            
            # Register hook on this layer
            hook = target_layer.register_forward_hook(
                create_sae_intervention_hook(directions, intervention_type, scale, sae_model)
            )
            hooks.append(hook)
        
        return hooks
    
    def remove_hooks(self, hooks):
        """Remove registered hooks"""
        for hook in hooks:
            hook.remove()
    

    
    def _convert_tensors_to_json(self, obj):
        """Recursively convert tensors to JSON-serializable format"""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_tensors_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_tensors_to_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation following features_intervention.py pattern"""
        
        print("Starting comprehensive evaluation...")
        
        # Load datasets and templates using utils functions (consistent with features_intervention.py)
        for dataset_name in self.config.test_datasets:
            print(f"\n{'='*60}")
            print(f"Running on {dataset_name} on {self.config.model_name} with SAE Features Intervention")
            print(f"{'='*60}")
            
            # Map to utils dataset names
            utils_dataset_name = self._map_dataset_name(dataset_name)
            
            # For MMLU-Pro, use the cached samples with reasoning scores
            # For other datasets, load fresh data
            if utils_dataset_name == 'MMLU-Pro':
                # Use cached samples that have reasoning scores
                ds_data = random.sample(self.feature_extractor.samples, 200)
            else:
                # Load fresh dataset for other datasets
                ds_data = load_dataset(ds_name=utils_dataset_name, dataset_dir=self.config.dataset_dir, split='test')
                ds_data = random.sample(ds_data, 200)
                
            prompt_template, prompt_template_no_cot = load_prompt_template(ds_name=utils_dataset_name, dataset_dir=self.config.dataset_dir)
            
            # Get model architecture info for utils functions
            layer_name, attn_name, mlp_name, model_layers_num = self._get_model_architecture_info()
            
            # Initialize results for this dataset
            self.results[dataset_name] = {}
            
            # Baseline evaluation (no intervention)
            print(f'****Running baseline evaluation on {dataset_name} with SAE pipeline')
            baseline_performance = self.sae_evaluation_on_dataset(
                val_sampled_data=ds_data, 
                prompts_cot=prompt_template, 
                prompts_no_cot=prompt_template_no_cot,
                run_in_fewshot=True, 
                run_in_cot=True,
                intervention_layers_dict=None,
                intervention_type=None,
                batch_size=self.config.batch_size, 
                ds_name=utils_dataset_name,
                scale=0.0
            )
            
            # Store baseline performance
            self.results[dataset_name]['baseline'] = {
                'performance': baseline_performance,
                'responses': []  # Baseline responses not stored for space
            }
            
            print(f'Baseline performance for {dataset_name}: {baseline_performance}')
            
            # Extract SAE features from ALL layers we want to evaluate
            print(f'Extracting SAE features from layers: {self.config.layers_to_evaluate}')
            all_layers_features = {}
            for layer in self.config.layers_to_evaluate:
                if layer not in self.feature_extractor.sae_models:
                    print(f"Skipping layer {layer} (SAE model not available)")
                    continue
                
                print(f'  Extracting features from layer {layer}')
                sae_features = self.feature_extractor.extract_sae_features(layer, top_k=5)
                all_layers_features[layer] = sae_features
            
            print(f'Successfully extracted features from {len(all_layers_features)} layers')
            
            # Multi-layer intervention evaluation
            self.results[dataset_name]['multi_layer'] = {}
            
            for intervention_type in ['projected', 'raw', 'activation_patching']:
                for scale in self.config.intervention_scales:
                    print(f'\n*** Testing {intervention_type} intervention with scale {scale} on ALL layers simultaneously ***')
                     
                    # Create a copy of data for this intervention
                    ds_data_copy = copy.deepcopy(ds_data)
                    
                    # Run evaluation with multi-layer SAE intervention
                    intervention_performance = self.sae_evaluation_on_dataset(
                        val_sampled_data=ds_data_copy, 
                        prompts_cot=prompt_template, 
                        prompts_no_cot=prompt_template_no_cot,
                        run_in_fewshot=True, 
                        run_in_cot=True,
                        intervention_layers_dict=all_layers_features,  # Pass all layers!
                        intervention_type=intervention_type,
                        batch_size=self.config.batch_size, 
                        ds_name=utils_dataset_name,
                        scale=scale
                    )
                    
                    # Store results
                    self.results[dataset_name]['multi_layer'][f'{intervention_type}_scale_{scale}'] = {
                        'performance': intervention_performance,
                        'layers_used': list(all_layers_features.keys()),
                        'num_layers': len(all_layers_features),
                        'responses': [item.get('pred_with_intervention', '') for item in ds_data_copy]
                    }
                    
                    print(f'  Performance: {intervention_performance}')
        
        # Save results and generate report
        self.save_results()
        self.generate_comparison_report()
    
    def save_results(self):
        """Save evaluation results"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save raw results
        results_path = os.path.join(self.config.output_dir, 'evaluation_results.json')
        
        # Convert results to JSON-serializable format
        json_results = self._convert_tensors_to_json(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report for SAE intervention results"""
        
        report_lines = [
            "# SAE Intervention Evaluation Report\n",
            f"Model: {self.config.model_name}",
            f"Layers evaluated: {self.config.layers_to_evaluate}",
            f"Datasets: {self.config.test_datasets}",
            f"Intervention scales: {self.config.intervention_scales}\n",
            "## Results Summary:\n"
        ]
        
        # Create comparison table
        comparison_data = []
        baseline_data = []
        
        for dataset_name in self.results.keys():
            # Process baseline performance first
            if 'baseline' in self.results[dataset_name]:
                baseline_performance = self.results[dataset_name]['baseline'].get('performance', {})
                print(f"  Baseline performance for {dataset_name}: {baseline_performance}")
                

                if dataset_name == 'mmlu-pro':
                    for metric_type, acc_key in [('Reasoning', 'reason_accuracy'), ('Memory', 'memory_accuracy')]:
                        if acc_key in baseline_performance:
                            baseline_data.append({
                                'Dataset': dataset_name,
                                'Metric Type': metric_type, 
                                'Baseline Performance': f"{baseline_performance[acc_key]:.4f}"
                            })
                else:
                    accuracy = baseline_performance.get('accuracy') or baseline_performance.get('reason_accuracy')
                    if accuracy is not None:
                        baseline_data.append({
                            'Dataset': dataset_name,
                            'Metric Type': 'Overall',
                            'Baseline Performance': f"{accuracy:.4f}"
                        })
            
            # Process intervention results
            # Check if we have multi_layer results (new structure) or layer_X results (old structure)
            if 'multi_layer' in self.results[dataset_name]:
                # New multi-layer structure
                multi_layer_results = self.results[dataset_name]['multi_layer']
                
                for intervention_key, result_data in multi_layer_results.items():
                    # Parse: "projected_scale_0.1" -> ("projected", 0.1)
                    parts = intervention_key.rsplit('_scale_', 1)
                    if len(parts) == 2:
                        intervention_type = parts[0]
                        scale = float(parts[1])
                        
                        performance = result_data.get('performance', {})
                        num_layers = result_data.get('num_layers', 0)
                        
                        # Extract accuracy
                        accuracy = None
                        if dataset_name == 'mmlu-pro':
                            accuracy = performance.get('reason_accuracy')
                        else:
                            accuracy = performance.get('accuracy') or performance.get('reason_accuracy')
                        
                        if accuracy is not None:
                            comparison_data.append({
                                'Dataset': dataset_name,
                                'Layers': f"{num_layers} layers",
                                'Intervention Type': intervention_type,
                                'Performance': f"{accuracy:.4f}",
                                'Scale': scale
                            })
            else:
                # Old single-layer structure (for backwards compatibility)
                for layer_key in self.results[dataset_name].keys():
                    if layer_key == 'baseline':  # Skip baseline
                        continue
                        
                    layer = int(layer_key.split('_')[1])
                    
                    # Check both intervention types
                    for intervention_type in ['projected', 'raw', 'activation_patching']:
                        best_scale = 0.0
                        best_performance = 0.0
                        
                        # Find best scale for this intervention type
                        for scale in self.config.intervention_scales:
                            key = f'{intervention_type}_scale_{scale}'
                            if key in self.results[dataset_name][layer_key]:
                                result_data = self.results[dataset_name][layer_key][key]
                                performance = result_data.get('performance', {})
                                
                                # Extract accuracy based on dataset and performance structure
                                accuracy = None
                                if dataset_name == 'mmlu-pro':
                                    # For MMLU-Pro, prioritize reasoning accuracy
                                    if 'reason_accuracy' in performance:
                                        accuracy = performance['reason_accuracy']
                                    elif 'memory_accuracy' in performance:
                                        accuracy = performance['memory_accuracy']
                                else:
                                    # For other datasets (gsm8k, mgsm, popqa), look for accuracy
                                    if 'accuracy' in performance:
                                        accuracy = performance['accuracy']
                                    elif 'reason_accuracy' in performance:
                                        accuracy = performance['reason_accuracy']
                                    elif 'memory_accuracy' in performance:
                                        accuracy = performance['memory_accuracy']
                                # Skip if accuracy is None or not a number
                                if accuracy is not None and isinstance(accuracy, (int, float)) and accuracy > best_performance:
                                    best_performance = accuracy
                                    best_scale = scale
                        
                        if best_performance > 0:
                            comparison_data.append({
                                'Dataset': dataset_name,
                                'Layer': layer,
                                'Intervention Type': intervention_type,
                                'Best Performance': f"{best_performance:.3f}",
                                'Best Scale': best_scale
                            })
        
        # Create DataFrame and format as table
        # First, add baseline performance table
        if baseline_data:
            df_baseline = pd.DataFrame(baseline_data)
            report_lines.append("### Baseline Performance (No Intervention):")
            
            # Manual table creation for baseline
            headers = list(df_baseline.columns)
            rows = df_baseline.values.tolist()
            
            header_row = "| " + " | ".join(headers) + " |"
            separator_row = "|" + "|".join(["-" * (len(str(h)) + 2) for h in headers]) + "|"
            
            report_lines.append(header_row)
            report_lines.append(separator_row)
            
            for row in rows:
                data_row = "| " + " | ".join([str(cell) for cell in row]) + " |"
                report_lines.append(data_row)
            
            report_lines.append("")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            report_lines.append("### SAE Intervention Performance:")
            
            # Manual table creation (simple format)
            headers = list(df.columns)
            rows = df.values.tolist()
            
            # Create header row
            header_row = "| " + " | ".join(headers) + " |"
            separator_row = "|" + "|".join(["-" * (len(str(h)) + 2) for h in headers]) + "|"
            
            report_lines.append(header_row)
            report_lines.append(separator_row)
            
            # Create data rows
            for row in rows:
                data_row = "| " + " | ".join([str(cell) for cell in row]) + " |"
                report_lines.append(data_row)
            
            report_lines.append("")
            
            # Add best intervention per dataset
            report_lines.append("### Best Intervention per Dataset:")
            for dataset_name in self.config.test_datasets:
                if dataset_name in df['Dataset'].values:
                    dataset_data = df[df['Dataset'] == dataset_name]
                    
                    # Handle both old and new structure
                    if 'Performance' in df.columns:
                        # New multi-layer structure
                        performance_col = 'Performance'
                        best_row = dataset_data.loc[dataset_data[performance_col].astype(float).idxmax()]
                        layers_info = best_row.get('Layers', 'N/A')
                        report_lines.append(f"- **{dataset_name}**: {best_row['Intervention Type']} ({layers_info}, Performance: {best_row[performance_col]}, Scale: {best_row['Scale']})")
                    else:
                        # Old single-layer structure
                        performance_col = 'Best Performance'
                        best_row = dataset_data.loc[dataset_data[performance_col].astype(str).str.replace('[^0-9.]', '', regex=True).astype(float).idxmax()]
                        report_lines.append(f"- **{dataset_name}**: {best_row['Intervention Type']} (Layer {best_row['Layer']}, Performance: {best_row[performance_col]})")
            
            report_lines.append("")
        else:
            report_lines.append("No results to display.")
        
        # Save report
        report_path = os.path.join(self.config.output_dir, 'sae_intervention_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"SAE intervention report saved to: {report_path}")
        
        # Generate plots
        self.generate_plots()
    
    def generate_plots(self):
        """Generate comparison plots for SAE intervention results"""
        
        try:
            # Create plots directory
            plots_dir = os.path.join(self.config.output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot 1: SAE intervention comparison across layers
            fig, axes = plt.subplots(1, len(self.config.test_datasets), figsize=(5*len(self.config.test_datasets), 5))
            if len(self.config.test_datasets) == 1:
                axes = [axes]
            
            for idx, dataset_name in enumerate(self.config.test_datasets):
                ax = axes[idx]
                
                if dataset_name in self.results:
                    # Get baseline performance for reference line
                    baseline_performance = None
                    if 'baseline' in self.results[dataset_name]:
                        baseline_data = self.results[dataset_name]['baseline'].get('performance', {})
                        if dataset_name == 'mmlu-pro':
                            # Use reasoning accuracy for MMLU-Pro baseline
                            baseline_performance = baseline_data.get('reason_accuracy')
                        else:
                            baseline_performance = baseline_data.get('accuracy') or baseline_data.get('reason_accuracy')
                    
                    # Plot both intervention types
                    for intervention_type in ['projected', 'raw', 'activation_patching']:
                        layers = []
                        performances = []
                        
                        for layer_key in self.results[dataset_name].keys():
                            if layer_key == 'baseline':  # Skip baseline in layer iteration
                                continue
                            layer = int(layer_key.split('_')[1])
                            
                            # Find best performance for this intervention type
                            best_performance = 0.0
                            for scale in self.config.intervention_scales:
                                key = f'{intervention_type}_scale_{scale}'
                                if key in self.results[dataset_name][layer_key]:
                                    result_data = self.results[dataset_name][layer_key][key]
                                    performance = result_data.get('performance', {})
                                    
                                    # Extract accuracy based on dataset
                                    accuracy = None
                                    if isinstance(performance, dict):
                                        if dataset_name == 'mmlu-pro':
                                            # For MMLU-Pro, prioritize reasoning accuracy
                                            if 'reason_accuracy' in performance:
                                                accuracy = performance['reason_accuracy']
                                            elif 'memory_accuracy' in performance:
                                                accuracy = performance['memory_accuracy']
                                        else:
                                            # For other datasets
                                            if 'accuracy' in performance:
                                                accuracy = performance['accuracy']
                                            elif 'reason_accuracy' in performance:
                                                accuracy = performance['reason_accuracy']
                                            elif 'memory_accuracy' in performance:
                                                accuracy = performance['memory_accuracy']
                                    elif isinstance(performance, (int, float)):
                                        accuracy = performance
                                    
                                    if accuracy is not None and isinstance(accuracy, (int, float)):
                                        best_performance = max(best_performance, accuracy)
                            
                            if best_performance > 0:
                                layers.append(layer)
                                performances.append(best_performance)
                        
                        if layers:
                            ax.plot(layers, performances, marker='o', 
                                   label=f'{intervention_type} features', linewidth=2)
                    
                    # Add baseline reference line
                    if baseline_performance is not None and layers:
                        ax.axhline(y=baseline_performance, color='red', linestyle='--', 
                                  label='Baseline (no intervention)', linewidth=2, alpha=0.8)
                
                ax.set_xlabel('Layer')
                ax.set_ylabel('Best Performance')
                ax.set_title(f'{dataset_name.upper()} - SAE Intervention vs Baseline')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'sae_intervention_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plots saved to: {plots_dir}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")


def main():
    """Main evaluation function"""
    print("=== SAE vs Steering Evaluation ===")
    print("==========================================")
    
    parser = argparse.ArgumentParser(description="Evaluate SAE vs steering methods")
    
    parser.add_argument('--model_name', type=str, default="Llama-3.1-8B")
    parser.add_argument('--sae_models_path', type=str, 
                       default="/home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/outputs/final_results/models")
    parser.add_argument('--datasets', nargs='+', default=['mmlu-pro'], choices=['gsm8k', 'popqa', 'mmlu-pro', 'mgsm', 'gsm-symbolic', 'c-eval-h'])
    parser.add_argument('--layers', nargs='+', type=int, default=[15, 20])
    parser.add_argument('--scales', nargs='+', type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument('--config', type=str, default=None, help="Path to config file (optional)") #yaml
    
    args = parser.parse_args()
    
    # Create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = EvaluationConfig(**config_dict)
    else:
        config = EvaluationConfig()
        config.model_name = args.model_name
        config.sae_models_path = args.sae_models_path
        config.test_datasets = args.datasets
        config.layers_to_evaluate = args.layers
        config.intervention_scales = args.scales
    
    print("Evaluation Configuration:")
    print(f"Model: {config.model_name}")
    print(f"Datasets: {config.test_datasets}")
    print(f"Layers: {config.layers_to_evaluate}")
    print(f"Scales: {config.intervention_scales}")
    
    # Run evaluation
    evaluator = InterventionEvaluator(config)
    evaluator.run_comprehensive_evaluation()
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()