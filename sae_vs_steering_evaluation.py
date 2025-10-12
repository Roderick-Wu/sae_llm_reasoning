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

# Import utility functions 
from utils import (
    load_dataset, 
    load_prompt_template, 
    get_prediction, 
    compute_performance_on_reason_memory_subset,
    compute_performance_on_reason_subset,
    generate_questions,
    form_options,
    form_options_ceval
)

warnings.filterwarnings('ignore')


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
    test_datasets: List[str] = field(default_factory=lambda: ['gsm8k', 'popqa', 'mmlu-pro'])
    reasoning_threshold: float = 0.5
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
        top_indices = torch.topk(differential, k=top_k)[1]
        
        # Extract feature directions from decoder
        feature_directions = sae.decoder.weight[:, top_indices].T  # Transpose to flip to [top_k, hidden_dim]
        
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
                                 run_in_fewshot=True, run_in_cot=True, intervention_layer=None, 
                                 intervention_directions=None, intervention_type=None, 
                                 batch_size=4, scale=0.1, ds_name='MMLU-Pro'):
        """Custom evaluation function for SAE interventions (based on utils.evaluation_on_dataset)"""
        
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
                if intervention_layer is not None and intervention_directions is not None:
                    responses = self.generate_with_sae_intervention(
                        queries_batch, intervention_layer, intervention_directions, 
                        intervention_type, scale
                    )
                else:
                    # Baseline generation without intervention
                    responses = generate_questions(
                        model=self.model, tokenizer=self.tokenizer, 
                        questions=queries_batch, n_new_tokens=200
                    )
                    
                # Process responses and compute correctness (same as utils version)
                if ds_name == 'MMLU-Pro': 
                    for answer, entry in zip(responses, entry_batch):
                        entry['solution'] = answer
                        prediction = get_prediction(answer, ds_name)
                        entry['model_predict_correctness'] = (entry["answer"] == prediction)
                        
                        # Store intervention response if applicable
                        if intervention_layer is not None:
                            entry['pred_with_intervention'] = answer

                elif ds_name in ['GSM8k', 'GSM-symbolic']:  
                    for answer, entry in zip(responses, entry_batch):
                        entry['solution'] = answer
                        prediction = get_prediction(answer, ds_name)
                        entry['model_predict_correctness'] = (entry["final_answer"] == prediction)
                        
                        if intervention_layer is not None:
                            entry['pred_with_intervention'] = answer

                elif ds_name in ['MGSM']:  
                    for answer, entry in zip(responses, entry_batch):
                        prediction = get_prediction(answer, ds_name)
                        entry['model_predict_correctness'] = (float(entry['answer']) == prediction)
                        
                        if intervention_layer is not None:
                            entry['pred_with_intervention'] = answer
                
                elif ds_name in ['C-Eval-H']:  
                    for answer, entry in zip(responses, entry_batch):
                        prediction = get_prediction(answer, ds_name)
                        entry['model_predict_correctness'] = (entry['answer'] == prediction)
                        
                        if intervention_layer is not None:
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
                        
                        if intervention_layer is not None:
                            entry['pred_with_intervention'] = answer
                        
                queries_batch = []
                entry_batch = []
    
    def generate_with_sae_intervention(self, questions, intervention_layer, intervention_directions, 
                                     intervention_type, scale):
        """Generate text with SAE feature interventions"""
        
        inputs = self.tokenizer(questions, return_tensors="pt", padding="longest", 
                               return_token_type_ids=False).to(self.device)
        input_length = inputs.input_ids.size(1)

        # Create SAE intervention hooks
        hooks = self.set_sae_intervention_hooks(
            intervention_layer, intervention_directions, intervention_type, scale
        )
        
        # Generate with hooks
        gen_tokens = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        
        # Remove hooks
        self.remove_hooks(hooks)

        # Decode generated text
        gen_text = self.tokenizer.batch_decode(gen_tokens[:, input_length:], skip_special_tokens=True)
        
        return gen_text
    
    def set_sae_intervention_hooks(self, layer, directions, intervention_type, scale):
        """Set up hooks for SAE intervention at a specific layer"""
        
        hooks = []
        
        def create_sae_intervention_hook(directions, intervention_type, scale):
            def hook_fn(module, input, output):
                # Handle tuple output (common in transformers)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                # Apply intervention to all positions (not just last token like steering)
                if intervention_type == 'projected':
                    # Use decoder weights as directions (existing approach)
                    directions_tensor = directions.to(hidden_states.device).to(hidden_states.dtype)
                    
                    # Apply each feature direction
                    for direction in directions_tensor:
                        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                        projection = (hidden_states @ direction).unsqueeze(-1) * direction
                        hidden_states = hidden_states + scale * projection
                        
                elif intervention_type == 'raw':
                    # Add raw sparse feature activations directly
                    # These are scalar activations, so we need to broadcast them to the hidden dimension
                    activations = directions.to(hidden_states.device).to(hidden_states.dtype)
                    
                    # Raw activations are 1D (top_k features), expand to match hidden_states
                    # For simplicity, we'll create a simple broadcasting pattern
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    
                    # Create a simple activation pattern: repeat activation values across hidden dimensions
                    if len(activations.shape) == 1:  # [top_k]
                        # Repeat each activation across (hidden_dim // top_k) dimensions
                        top_k = activations.shape[0]
                        if hidden_dim % top_k == 0:
                            repeat_factor = hidden_dim // top_k
                            activation_pattern = activations.repeat_interleave(repeat_factor)
                        else:
                            # Handle case where hidden_dim doesn't divide evenly
                            activation_pattern = torch.zeros(hidden_dim, device=activations.device, dtype=activations.dtype)
                            step = hidden_dim // top_k
                            for i, act in enumerate(activations):
                                start_idx = i * step
                                end_idx = min(start_idx + step, hidden_dim)
                                activation_pattern[start_idx:end_idx] = act
                        
                        # Expand to batch and sequence dimensions [batch, seq, hidden]
                        intervention_tensor = activation_pattern.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                        
                        # Add scaled raw activations
                        hidden_states = hidden_states + scale * intervention_tensor
                
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
            create_sae_intervention_hook(directions, intervention_type, scale)
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
            
            # Load dataset and prompt template using utils functions
            ds_data = load_dataset(ds_name=utils_dataset_name, dataset_dir=self.config.dataset_dir, split='test')
            prompt_template, prompt_template_no_cot = load_prompt_template(ds_name=utils_dataset_name, dataset_dir=self.config.dataset_dir)
            
            # Sample data (similar to features_intervention.py approach)
            ds_data = random.sample(ds_data, 200)
            
            # Get model architecture info for utils functions
            layer_name, attn_name, mlp_name, model_layers_num = self._get_model_architecture_info()
            
            # Initialize results for this dataset
            self.results[dataset_name] = {}
            
            # Baseline evaluation (no intervention)
            print(f'****Running baseline evaluation on {dataset_name}')
            self.sae_evaluation_on_dataset(
                val_sampled_data=ds_data, 
                prompts_cot=prompt_template, 
                prompts_no_cot=prompt_template_no_cot,
                run_in_fewshot=True, 
                run_in_cot=True,
                intervention_layer=None,
                intervention_directions=None, 
                intervention_type=None,
                batch_size=self.config.batch_size, 
                ds_name=utils_dataset_name,
                scale=0.0
            )
            
            # Compute baseline performance
            if utils_dataset_name != 'MMLU-Pro':
                baseline_performance = compute_performance_on_reason_subset(
                    val_sampled_data=ds_data, 
                    intervention=False, 
                    ds_name=utils_dataset_name
                )
            else:
                reason_indices = [ix for ix, sample in enumerate(ds_data) if sample['memory_reason_score'] > 0.5]
                memory_indices = [ix for ix, sample in enumerate(ds_data) if sample['memory_reason_score'] <= 0.5]
                baseline_performance = compute_performance_on_reason_memory_subset(
                    val_sampled_data=ds_data, 
                    memory_indices=memory_indices,
                    reason_indices=reason_indices, 
                    intervention=False
                )
            
            # Intervention evaluation across layers
            for layer in self.config.layers_to_evaluate:
                if layer not in self.feature_extractor.sae_models:
                    print(f"Skipping layer {layer} for SAE (model not available)")
                    continue
                    
                if layer <= 2:
                    continue
                
                print(f'Doing SAE Intervention in Layer {layer}')
                
                # Extract SAE features for this layer
                sae_features = self.feature_extractor.extract_sae_features(layer, top_k=5)
                # Returns:
                # - 'feature_directions': SAE decoder weights for top reasoning features [top_k, hidden_dim]
                # - 'reasoning_activations': Mean sparse activations for reasoning samples [top_k]
                # - 'feature_indices': Indices of the top features in SAE dictionary
                # - 'differential_scores': Reasoning vs memory activation differences
                
                self.results[dataset_name][f'layer_{layer}'] = {}
                
                # Test different intervention methods and scales
                for intervention_type in ['projected', 'raw']:
                    for scale in self.config.intervention_scales:
                        print(f'  Testing {intervention_type} intervention with scale {scale}')
                        
                        # Prepare intervention directions based on method
                        if intervention_type == 'projected':
                            # Use SAE decoder directions: project activations onto learned feature directions
                            # Shape: [top_k, hidden_dim] - these are the decoder weight vectors
                            ablation_dir = sae_features['feature_directions']
                        else:  # raw
                            # Use raw sparse activations: directly add the activation magnitudes
                            # Shape: [top_k] - these are scalar activation values
                            ablation_dir = sae_features['reasoning_activations']
                        
                        # Create a copy of data for this intervention
                        ds_data_copy = copy.deepcopy(ds_data)
                        
                        # Run evaluation with SAE intervention
                        self.sae_evaluation_on_dataset(
                            val_sampled_data=ds_data_copy, 
                            prompts_cot=prompt_template, 
                            prompts_no_cot=prompt_template_no_cot,
                            run_in_fewshot=True, 
                            run_in_cot=True,
                            intervention_layer=layer,
                            intervention_directions=ablation_dir, 
                            intervention_type=intervention_type,
                            batch_size=self.config.batch_size, 
                            ds_name=utils_dataset_name,
                            scale=scale
                        )
                        
                        # Compute performance for this intervention
                        if utils_dataset_name != 'MMLU-Pro':
                            intervention_performance = compute_performance_on_reason_subset(
                                val_sampled_data=ds_data_copy, 
                                intervention=True, 
                                ds_name=utils_dataset_name, 
                                intervention_layer=layer
                            )
                        else:
                            reason_indices = [ix for ix, sample in enumerate(ds_data_copy) if sample['memory_reason_score'] > 0.5]
                            memory_indices = [ix for ix, sample in enumerate(ds_data_copy) if sample['memory_reason_score'] <= 0.5]
                            intervention_performance = compute_performance_on_reason_memory_subset(
                                val_sampled_data=ds_data_copy, 
                                memory_indices=memory_indices,
                                reason_indices=reason_indices, 
                                intervention=True, 
                                intervention_layer=layer
                            )
                        
                        # Store results
                        key = f'{intervention_type}_scale_{scale}'
                        self.results[dataset_name][f'layer_{layer}'][key] = {
                            'performance': intervention_performance,
                            'responses': [item.get('pred_with_intervention', '') for item in ds_data_copy]
                        }
        
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
        
        for dataset_name in self.results.keys():
            for layer_key in self.results[dataset_name].keys():
                layer = int(layer_key.split('_')[1])
                
                # Check both intervention types
                for intervention_type in ['projected', 'raw']:
                    best_scale = 0.0
                    best_performance = 0.0
                    
                    # Find best scale for this intervention type
                    for scale in self.config.intervention_scales:
                        key = f'{intervention_type}_scale_{scale}'
                        if key in self.results[dataset_name][layer_key]:
                            performance = self.results[dataset_name][layer_key][key]['performance']
                            # Extract accuracy from performance dict if available
                            if isinstance(performance, dict) and 'accuracy' in performance:
                                accuracy = performance['accuracy']
                            else:
                                accuracy = performance  # Assume it's a float
                            
                            if accuracy > best_performance:
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
                    best_row = dataset_data.loc[dataset_data['Best Performance'].astype(str).str.replace('[^0-9.]', '', regex=True).astype(float).idxmax()]
                    report_lines.append(f"- **{dataset_name}**: {best_row['Intervention Type']} (Layer {best_row['Layer']}, Performance: {best_row['Best Performance']})")
            
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
                    # Plot both intervention types
                    for intervention_type in ['projected', 'raw']:
                        layers = []
                        performances = []
                        
                        for layer_key in self.results[dataset_name].keys():
                            layer = int(layer_key.split('_')[1])
                            
                            # Find best performance for this intervention type
                            best_performance = 0.0
                            for scale in self.config.intervention_scales:
                                key = f'{intervention_type}_scale_{scale}'
                                if key in self.results[dataset_name][layer_key]:
                                    performance = self.results[dataset_name][layer_key][key]['performance']
                                    # Extract accuracy if it's a dict
                                    if isinstance(performance, dict) and 'accuracy' in performance:
                                        accuracy = performance['accuracy']
                                    else:
                                        accuracy = performance
                                    
                                    best_performance = max(best_performance, accuracy)
                            
                            if best_performance > 0:
                                layers.append(layer)
                                performances.append(best_performance)
                        
                        if layers:
                            ax.plot(layers, performances, marker='o', 
                                   label=f'{intervention_type} features', linewidth=2)
                
                ax.set_xlabel('Layer')
                ax.set_ylabel('Best Performance')
                ax.set_title(f'{dataset_name} - SAE Intervention')
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
    parser.add_argument('--datasets', nargs='+', default=['mmlu-pro'], choices=['gsm8k', 'popqa', 'mmlu-pro'])
    parser.add_argument('--layers', nargs='+', type=int, default=[15, 20])
    parser.add_argument('--scales', nargs='+', type=float, default=[0.1, 0.3, 0.5])
    
    args = parser.parse_args()
    
    # Create config
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