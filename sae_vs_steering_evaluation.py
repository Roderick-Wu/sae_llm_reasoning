#!/usr/bin/env python3
"""
SAE vs Steering Comparison Script

This script compares SAE-extracted reasoning features with the original
linear steering approach from the paper. It evaluates both approaches
on reasoning enhancement and provides comprehensive comparisons.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
import copy
import pandas as pd
from dataclasses import dataclass, field
from sae_reasoning_trainer import SparseAutoencoder, SAEConfig
import warnings
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
    max_new_tokens: int = 100
    batch_size: int = 8
    device: str = "cuda"


class FeatureExtractor:
    """Extract features using different methods (SAE, PCA, linear probe)"""
    
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
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Recreate SAE model
                sae_config = checkpoint['config']
                dict_size = sae_config.hidden_dim * sae_config.expansion_factor
                
                sae = SparseAutoencoder(
                    input_dim=sae_config.hidden_dim,
                    dict_size=dict_size,
                    sparsity_penalty=sae_config.sparsity_penalty
                )
                
                sae.load_state_dict(checkpoint['model_state_dict'])
                sae.to(self.device)
                sae.eval()
                
                self.sae_models[layer] = sae
                print(f"Loaded SAE for layer {layer}")
            else:
                print(f"Warning: SAE model not found for layer {layer}")
    
    def load_data(self):
        """Load hidden states and reasoning metadata"""
        print("Loading evaluation data...")
        
        # Load hidden states
        hs_cache_path = os.path.join(self.config.hs_cache_path, f'{self.config.model_name}-base_hs_cache_no_cot.pt')
        self.hs_cache = torch.load(hs_cache_path, map_location='cpu')
        
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
    
    def extract_pca_direction(self, layer: int) -> torch.Tensor:
        """Extract linear reasoning direction using PCA"""
        hidden_states = self.hs_cache[layer].squeeze(1)
        
        reasoning_mask = self.labels.bool()
        memory_mask = ~reasoning_mask
        
        reasoning_hs = hidden_states[reasoning_mask]
        memory_hs = hidden_states[memory_mask]
        
        # Compute mean difference (simple linear probe)
        reasoning_mean = torch.mean(reasoning_hs, dim=0)
        memory_mean = torch.mean(memory_hs, dim=0)
        linear_direction = reasoning_mean - memory_mean
        linear_direction = F.normalize(linear_direction, dim=0)
        
        return linear_direction
    
    def extract_logistic_direction(self, layer: int) -> torch.Tensor:
        """Extract reasoning direction using logistic regression"""
        hidden_states = self.hs_cache[layer].squeeze(1).numpy()
        labels = self.labels.numpy()
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, C=0.01)
        clf.fit(hidden_states, labels)
        
        # Extract weight vector as direction
        direction = torch.from_numpy(clf.coef_[0]).float()
        direction = F.normalize(direction, dim=0)
        
        return direction


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
    
    def create_intervention_hook(self, layer: int, directions: torch.Tensor, scale: float):
        """Create a hook for intervention during forward pass"""
        
        def hook(module, input, output):
            # output is typically a tuple (hidden_states, ...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Apply intervention to the last token
            if len(hidden_states.shape) == 3:  # [batch, seq_len, hidden_dim]
                last_token_hs = hidden_states[:, -1, :]  # [batch, hidden_dim]
                
                # Apply multiple directions if provided
                if len(directions.shape) == 2:  # Multiple directions
                    for direction in directions:
                        direction = direction.to(hidden_states.device)
                        projection = torch.sum(last_token_hs * direction.unsqueeze(0), dim=1, keepdim=True)
                        last_token_hs = last_token_hs + scale * projection * direction.unsqueeze(0)
                else:  # Single direction
                    direction = directions.to(hidden_states.device)
                    projection = torch.sum(last_token_hs * direction.unsqueeze(0), dim=1, keepdim=True)
                    last_token_hs = last_token_hs + scale * projection * direction.unsqueeze(0)
                
                # Update the hidden states
                hidden_states[:, -1, :] = last_token_hs
                
                if isinstance(output, tuple):
                    output = (hidden_states,) + output[1:]
                else:
                    output = hidden_states
            
            return output
        
        return hook
    
    def evaluate_intervention(self, questions: List[str], layer: int, 
                            directions: torch.Tensor, scale: float) -> List[str]:
        """Evaluate intervention on a set of questions"""
        
        # Get the target layer
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            target_layer = self.model.model.layers[layer]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            target_layer = self.model.transformer.h[layer]
        else:
            raise ValueError(f"Cannot find layer {layer} in model architecture")
        
        # Create and register hook
        hook = self.create_intervention_hook(layer, directions, scale)
        handle = target_layer.register_forward_hook(hook)
        
        try:
            # Generate responses
            responses = []
            
            for i in range(0, len(questions), self.config.batch_size):
                batch_questions = questions[i:i+self.config.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_questions,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode responses
                batch_responses = self.tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                
                responses.extend(batch_responses)
        
        finally:
            # Remove hook
            handle.remove()
        
        return responses
    
    def evaluate_method_on_dataset(self, method: str, dataset_name: str, layer: int) -> Dict:
        """Evaluate a specific method on a dataset"""
        
        print(f"Evaluating {method} on {dataset_name} (layer {layer})")
        
        # Load test questions
        test_questions = self.load_test_questions(dataset_name)
        if not test_questions:
            return {}
        
        # Extract features based on method
        if method == "sae":
            features = self.feature_extractor.extract_sae_features(layer, top_k=5)
            directions = features['feature_directions']
        elif method == "pca":
            direction = self.feature_extractor.extract_pca_direction(layer)
            directions = direction.unsqueeze(0)  # Make it 2D
        elif method == "logistic":
            direction = self.feature_extractor.extract_logistic_direction(layer)
            directions = direction.unsqueeze(0)  # Make it 2D
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results = {}
        
        # Evaluate baseline (no intervention)
        baseline_responses = self.evaluate_intervention(test_questions, layer, directions, 0.0)
        baseline_scores = self.score_responses(baseline_responses, dataset_name)
        results['baseline'] = {'responses': baseline_responses, 'scores': baseline_scores}
        
        # Evaluate different intervention scales
        for scale in self.config.intervention_scales:
            intervention_responses = self.evaluate_intervention(test_questions, layer, directions, scale)
            intervention_scores = self.score_responses(intervention_responses, dataset_name)
            
            results[f'scale_{scale}'] = {
                'responses': intervention_responses,
                'scores': intervention_scores
            }
            
            print(f"  Scale {scale}: {intervention_scores['accuracy']:.3f} accuracy")
        
        return results
    
    def load_test_questions(self, dataset_name: str, n_samples: int = 100) -> List[str]:
        """Load test questions from a specific dataset"""
        
        questions = []
        
        if dataset_name == 'gsm8k':
            # Load GSM8K dataset
            try:
                from datasets import load_from_disk
                dataset_path = os.path.join(self.config.dataset_dir, 'gsm8k', 'main')
                dataset = load_from_disk(dataset_path)
                test_data = list(dataset['test'])[:n_samples]
                
                questions = [f"Q: {item['question']}\nA:" for item in test_data]
                
            except Exception as e:
                print(f"Error loading GSM8K: {e}")
                return []
        
        elif dataset_name == 'mmlu-pro':
            # Load MMLU-Pro samples
            try:
                samples_path = os.path.join(self.config.dataset_dir, 'mmlu-pro-600samples.json')
                with open(samples_path, 'r') as f:
                    samples = json.load(f)[:n_samples]
                
                questions = [f"Q: {item['question']}\nA:" for item in samples]
                
            except Exception as e:
                print(f"Error loading MMLU-Pro: {e}")
                return []
        
        elif dataset_name == 'popqa':
            # Load PopQA dataset
            try:
                from datasets import load_from_disk
                dataset_path = os.path.join(self.config.dataset_dir, 'PopQA', 'test')
                dataset = load_from_disk(dataset_path)
                test_data = list(dataset)[:n_samples]
                
                questions = [f"Q: {item['question']}\nA:" for item in test_data]
                
            except Exception as e:
                print(f"Error loading PopQA: {e}")
                return []
        
        else:
            print(f"Unknown dataset: {dataset_name}")
            return []
        
        print(f"Loaded {len(questions)} questions from {dataset_name}")
        return questions
    
    def score_responses(self, responses: List[str], dataset_name: str) -> Dict:
        """Score responses based on dataset-specific metrics"""
        
        # This is a simplified scoring function
        # In practice, you'd want more sophisticated evaluation
        
        scores = {
            'accuracy': 0.0,
            'avg_length': np.mean([len(response) for response in responses]),
            'num_responses': len(responses)
        }
        
        # Simple heuristic: longer responses might indicate more reasoning
        # This would need to be replaced with proper answer evaluation
        reasoning_indicators = ['because', 'therefore', 'since', 'step', 'first', 'then', 'solution']
        reasoning_scores = []
        
        for response in responses:
            response_lower = response.lower()
            reasoning_score = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
            reasoning_scores.append(reasoning_score)
        
        scores['avg_reasoning_indicators'] = np.mean(reasoning_scores)
        scores['responses_with_reasoning'] = sum(1 for score in reasoning_scores if score > 0) / len(responses)
        
        return scores
    
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
        """Run comprehensive evaluation comparing all methods"""
        
        print("Starting comprehensive evaluation...")
        
        methods = ['sae', 'pca', 'logistic']
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Evaluating method: {method.upper()}")
            print(f"{'='*60}")
            
            self.results[method] = {}
            
            for dataset_name in self.config.test_datasets:
                self.results[method][dataset_name] = {}
                
                for layer in self.config.layers_to_evaluate:
                    if method == 'sae' and layer not in self.feature_extractor.sae_models:
                        continue
                    
                    try:
                        results = self.evaluate_method_on_dataset(method, dataset_name, layer)
                        self.results[method][dataset_name][f'layer_{layer}'] = results
                    
                    except Exception as e:
                        print(f"Error evaluating {method} on {dataset_name} layer {layer}: {e}")
                        continue
        
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
        """Generate a comprehensive comparison report"""
        
        report_lines = [
            "# SAE vs Steering Methods Comparison Report\n",
            f"Model: {self.config.model_name}",
            f"Layers evaluated: {self.config.layers_to_evaluate}",
            f"Datasets: {self.config.test_datasets}",
            f"Intervention scales: {self.config.intervention_scales}\n",
            "## Results Summary:\n"
        ]
        
        # Create comparison table
        comparison_data = []
        
        for method in self.results.keys():
            for dataset in self.results[method].keys():
                for layer_key in self.results[method][dataset].keys():
                    layer = int(layer_key.split('_')[1])
                    
                    if 'baseline' in self.results[method][dataset][layer_key]:
                        baseline_score = self.results[method][dataset][layer_key]['baseline']['scores']['avg_reasoning_indicators']
                        
                        # Find best intervention scale
                        best_scale = 0.0
                        best_score = baseline_score
                        
                        for scale in self.config.intervention_scales:
                            scale_key = f'scale_{scale}'
                            if scale_key in self.results[method][dataset][layer_key]:
                                score = self.results[method][dataset][layer_key][scale_key]['scores']['avg_reasoning_indicators']
                                if score > best_score:
                                    best_score = score
                                    best_scale = scale
                        
                        improvement = best_score - baseline_score
                        
                        comparison_data.append({
                            'Method': method.upper(),
                            'Dataset': dataset,
                            'Layer': layer,
                            'Baseline Score': f"{baseline_score:.3f}",
                            'Best Score': f"{best_score:.3f}",
                            'Best Scale': best_scale,
                            'Improvement': f"{improvement:.3f}"
                        })
        
        # Create DataFrame and format as table
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            report_lines.append("### Performance Comparison:")
            report_lines.append(df.to_markdown(index=False))
            report_lines.append("")
            
            # Add best methods per dataset
            report_lines.append("### Best Method per Dataset:")
            for dataset in self.config.test_datasets:
                dataset_data = df[df['Dataset'] == dataset]
                if not dataset_data.empty:
                    best_row = dataset_data.loc[dataset_data['Improvement'].astype(float).idxmax()]
                    report_lines.append(f"- **{dataset}**: {best_row['Method']} (Layer {best_row['Layer']}, Improvement: {best_row['Improvement']})")
            
            report_lines.append("")
        
        # Save report
        report_path = os.path.join(self.config.output_dir, 'comparison_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Comparison report saved to: {report_path}")
        
        # Generate plots
        self.generate_plots()
    
    def generate_plots(self):
        """Generate comparison plots"""
        
        try:
            # Create plots directory
            plots_dir = os.path.join(self.config.output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot 1: Method comparison across layers
            fig, axes = plt.subplots(1, len(self.config.test_datasets), figsize=(5*len(self.config.test_datasets), 5))
            if len(self.config.test_datasets) == 1:
                axes = [axes]
            
            for idx, dataset in enumerate(self.config.test_datasets):
                ax = axes[idx]
                
                for method in self.results.keys():
                    if dataset in self.results[method]:
                        layers = []
                        improvements = []
                        
                        for layer_key in self.results[method][dataset].keys():
                            layer = int(layer_key.split('_')[1])
                            
                            if 'baseline' in self.results[method][dataset][layer_key]:
                                baseline = self.results[method][dataset][layer_key]['baseline']['scores']['avg_reasoning_indicators']
                                
                                # Find best improvement
                                best_improvement = 0
                                for scale in self.config.intervention_scales:
                                    scale_key = f'scale_{scale}'
                                    if scale_key in self.results[method][dataset][layer_key]:
                                        score = self.results[method][dataset][layer_key][scale_key]['scores']['avg_reasoning_indicators']
                                        improvement = score - baseline
                                        best_improvement = max(best_improvement, improvement)
                                
                                layers.append(layer)
                                improvements.append(best_improvement)
                        
                        if layers:
                            ax.plot(layers, improvements, marker='o', label=method.upper(), linewidth=2)
                
                ax.set_xlabel('Layer')
                ax.set_ylabel('Best Improvement')
                ax.set_title(f'{dataset} - Reasoning Enhancement')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plots saved to: {plots_dir}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")


def main():
    """Main evaluation function"""
    
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