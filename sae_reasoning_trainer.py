#!/usr/bin/env python3
"""
Sparse Autoencoder (SAE) Training for Reasoning Feature Extraction

This script trains Sparse Autoencoders on model hidden states to extract
reasoning features that can be compared with the linear direction approach
from the original paper.

Key Features:
- Train SAEs on multiple layers simultaneously or individually
- Extract reasoning-specific features using sparsity constraints
- Compare SAE features with PCA/linear probe approaches
- Evaluate reasoning enhancement capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import copy
from typing import Dict, List, Tuple, Optional, Union
import wandb
from dataclasses import dataclass, field
import yaml


@dataclass
class SAEConfig:
    """Configuration for SAE training"""
    # Model parameters
    model_name: str = "Llama-3.1-8B"
    model_dir: str = "/home/wuroderi/projects/def-zhijing/wuroderi/models"
    
    # Data parameters
    dataset_dir: str = "/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/dataset"
    output_dir: str = "/home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/outputs"
    hs_cache_path: str = "/home/wuroderi/projects/def-zhijing/wuroderi/Linear_Reasoning_Features/outputs"
    
    # SAE Architecture
    hidden_dim: int = 4096  # Model hidden dimension
    expansion_factor: int = 4  # SAE dictionary size = hidden_dim * expansion_factor
    layers_to_train: List[int] = field(default_factory=lambda: [10, 15, 20, 25])  # Which layers to train SAEs on
    
    # Training parameters
    batch_size: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 100
    sparsity_penalty: float = 1e-3
    reconstruction_weight: float = 1.0
    device: str = "cuda"
    
    # Feature extraction
    top_k_features: int = 10  # Number of top features to analyze
    reasoning_threshold: float = 0.5  # Threshold for reasoning vs memory classification
    
    # Logging
    log_interval: int = 100
    save_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "sae-reasoning-features"


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for extracting reasoning features"""
    
    def __init__(self, input_dim: int, dict_size: int, sparsity_penalty: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity_penalty = sparsity_penalty
        
        # Encoder (input -> sparse representation)
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        
        # Decoder (sparse representation -> reconstruction)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        
        # Normalize decoder columns to unit norm (dictionary learning convention)
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation"""
        return F.relu(self.encoder(x))
    
    def decode(self, sparse_acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation to reconstruction"""
        return self.decoder(sparse_acts)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning sparse activations and reconstruction"""
        sparse_acts = self.encode(x)
        reconstruction = self.decode(sparse_acts)
        return sparse_acts, reconstruction
    
    def compute_loss(self, x: torch.Tensor, sparse_acts: torch.Tensor, reconstruction: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute SAE loss components"""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)
        
        # Sparsity loss (L1 penalty on activations)
        sparsity_loss = torch.mean(torch.abs(sparse_acts))
        
        # Total loss
        total_loss = recon_loss + self.sparsity_penalty * sparsity_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'sparsity_penalty': torch.tensor(self.sparsity_penalty),
            'activation_density': torch.mean((sparse_acts > 0).float())
        }


class ReasoningDataset(Dataset):
    """Dataset for reasoning vs memory hidden states"""
    
    def __init__(self, hidden_states: torch.Tensor, labels: torch.Tensor, reasoning_scores: torch.Tensor):
        self.hidden_states = hidden_states
        self.labels = labels  # 1 for reasoning, 0 for memory
        self.reasoning_scores = reasoning_scores
        
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        return {
            'hidden_states': self.hidden_states[idx],
            'labels': self.labels[idx],
            'reasoning_scores': self.reasoning_scores[idx]
        }


class SAETrainer:
    """Main trainer class for SAE reasoning feature extraction"""
    
    def __init__(self, config: SAEConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        if config.use_wandb:
            wandb.init(project=config.wandb_project, config=config.__dict__)
        
        # Load data
        self.load_data()
        
        # Initialize SAEs
        self.saes = {}
        self.optimizers = {}
        self.init_saes()
        
        # Results storage
        self.training_history = {}
        self.feature_analysis = {}
        
    def load_data(self):
        """Load hidden states and prepare datasets"""
        print("Loading hidden states...")
        
        # Load cached hidden states
        hs_cache_path = os.path.join(self.config.hs_cache_path, f'{self.config.model_name}-base_hs_cache_no_cot.pt')
        if not os.path.exists(hs_cache_path):
            raise FileNotFoundError(f"Hidden states cache not found: {hs_cache_path}")
        
        hs_cache = torch.load(hs_cache_path, map_location='cpu', weights_only=False)
        
        # Load reasoning scores
        with open(os.path.join(self.config.dataset_dir, 'mmlu-pro-3000samples.json'), 'r') as f:
            samples = json.load(f)
        
        reasoning_scores = torch.tensor([sample['memory_reason_score'] for sample in samples])
        labels = (reasoning_scores > self.config.reasoning_threshold).float()
        
        print(f"Loaded data: {len(samples)} samples")
        print(f"Reasoning samples: {labels.sum().item()}")
        print(f"Memory samples: {(1 - labels).sum().item()}")
        
        # Prepare datasets for each layer
        self.datasets = {}
        self.dataloaders = {}
        
        for layer in self.config.layers_to_train:
            if layer not in hs_cache:
                print(f"Warning: Layer {layer} not found in cache, skipping...")
                continue
                
            hidden_states = hs_cache[layer].squeeze(1)  # Remove token dimension if present
            
            dataset = ReasoningDataset(hidden_states, labels, reasoning_scores)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True,
                num_workers=2 if self.device.type == 'cuda' else 0
            )
            
            self.datasets[layer] = dataset
            self.dataloaders[layer] = dataloader
            
            print(f"Layer {layer}: {hidden_states.shape} hidden states")
            
    def init_saes(self):
        """Initialize SAE models and optimizers"""
        print("Initializing SAEs...")
        
        for layer in self.datasets.keys():
            dict_size = self.config.hidden_dim * self.config.expansion_factor
            
            sae = SparseAutoencoder(
                input_dim=self.config.hidden_dim,
                dict_size=dict_size,
                sparsity_penalty=self.config.sparsity_penalty
            ).to(self.device)
            
            optimizer = optim.AdamW(
                sae.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5
            )
            
            self.saes[layer] = sae
            self.optimizers[layer] = optimizer
            
            print(f"Layer {layer}: SAE with {dict_size} dictionary elements")
    
    def train_layer(self, layer: int, num_epochs: int) -> Dict[str, List[float]]:
        """Train SAE for a specific layer"""
        sae = self.saes[layer]
        optimizer = self.optimizers[layer]
        dataloader = self.dataloaders[layer]
        
        history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'sparsity_loss': [],
            'activation_density': []
        }
        
        sae.train()
        for epoch in range(num_epochs):
            epoch_losses = {key: [] for key in history.keys()}
            
            progress_bar = tqdm(dataloader, desc=f'Layer {layer} Epoch {epoch+1}/{num_epochs}')
            for batch_idx, batch in enumerate(progress_bar):
                hidden_states = batch['hidden_states'].to(self.device)
                
                # Forward pass
                sparse_acts, reconstruction = sae(hidden_states)
                
                # Compute loss
                loss_dict = sae.compute_loss(hidden_states, sparse_acts, reconstruction)
                
                # Backward pass
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                optimizer.step()
                
                # Normalize decoder weights
                with torch.no_grad():
                    sae.decoder.weight.div_(sae.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)
                
                # Log losses
                for key, value in loss_dict.items():
                    if key in epoch_losses:
                        epoch_losses[key].append(value.item())
                
                # Update progress bar
                if batch_idx % self.config.log_interval == 0:
                    progress_bar.set_postfix({
                        'Loss': f"{loss_dict['total_loss'].item():.4f}",
                        'Recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                        'Sparsity': f"{loss_dict['sparsity_loss'].item():.4f}",
                        'Density': f"{loss_dict['activation_density'].item():.3f}"
                    })
            
            # Store epoch averages
            for key in history.keys():
                if epoch_losses[key]:
                    history[key].append(np.mean(epoch_losses[key]))
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    f'layer_{layer}_total_loss': history['total_loss'][-1],
                    f'layer_{layer}_reconstruction_loss': history['reconstruction_loss'][-1],
                    f'layer_{layer}_sparsity_loss': history['sparsity_loss'][-1],
                    f'layer_{layer}_activation_density': history['activation_density'][-1],
                    'epoch': epoch
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(layer, epoch + 1)
        
        return history
    
    def train_all_layers(self):
        """Train SAEs for all specified layers"""
        print("Starting SAE training for all layers...")
        
        for layer in self.datasets.keys():
            print(f"\n{'='*60}")
            print(f"Training SAE for Layer {layer}")
            print(f"{'='*60}")
            
            history = self.train_layer(layer, self.config.num_epochs)
            self.training_history[layer] = history
            
            # Analyze features after training
            self.analyze_layer_features(layer)
        
        print("\nTraining completed for all layers!")
        self.save_final_results()
    
    def analyze_layer_features(self, layer: int):
        """Analyze learned features for a specific layer"""
        print(f"Analyzing features for layer {layer}...")
        
        sae = self.saes[layer]
        dataset = self.datasets[layer]

        all_sparse_acts = []
        all_labels = []
        all_hidden_states = []

        sae.eval()
        with torch.no_grad():
            # Get all activations
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
             
            for batch in tqdm(dataloader, desc="Extracting features"):
                hidden_states = batch['hidden_states'].to(self.device)
                labels = batch['labels']
                
                sparse_acts, _ = sae(hidden_states)
                
                all_sparse_acts.append(sparse_acts.cpu())
                all_labels.append(labels)
                all_hidden_states.append(hidden_states.cpu())
            
            all_sparse_acts = torch.cat(all_sparse_acts, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_hidden_states = torch.cat(all_hidden_states, dim=0)
        
        # Feature analysis
        analysis = self.perform_feature_analysis(
            all_sparse_acts, all_labels, all_hidden_states, layer
        )
        
        self.feature_analysis[layer] = analysis
        
        return analysis
    
    def perform_feature_analysis(self, sparse_acts: torch.Tensor, labels: torch.Tensor, 
                                 hidden_states: torch.Tensor, layer: int) -> Dict:
        """Perform comprehensive feature analysis"""
        
        # 1. Feature activation statistics
        feature_stats = self.compute_feature_statistics(sparse_acts, labels)
        
        # 2. Reasoning-specific features
        reasoning_features = self.identify_reasoning_features(sparse_acts, labels)
        
        # 3. Compare with PCA baseline
        pca_comparison = self.compare_with_pca(sparse_acts, hidden_states, labels)
        
        # 4. Feature interpretability metrics
        interpretability = self.compute_interpretability_metrics(sparse_acts, labels)
        
        analysis = {
            'feature_stats': feature_stats,
            'reasoning_features': reasoning_features,
            'pca_comparison': pca_comparison,
            'interpretability': interpretability,
            'layer': layer
        }
        
        # Save analysis results
        self.save_layer_analysis(layer, analysis)
        
        return analysis
    
    def compute_feature_statistics(self, sparse_acts: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Compute basic statistics about feature activations"""
        
        reasoning_mask = labels.bool()
        memory_mask = ~reasoning_mask
        
        reasoning_acts = sparse_acts[reasoning_mask]
        memory_acts = sparse_acts[memory_mask]
        
        stats = {
            'total_features': sparse_acts.shape[1],
            'avg_activation_density': torch.mean((sparse_acts > 0).float()).item(),
            'reasoning_density': torch.mean((reasoning_acts > 0).float()).item(),
            'memory_density': torch.mean((memory_acts > 0).float()).item(),
            'feature_activation_counts': torch.sum(sparse_acts > 0, dim=0),
            'feature_mean_activations': torch.mean(sparse_acts, dim=0),
            'feature_max_activations': torch.max(sparse_acts, dim=0)[0]
        }
        
        return stats
    
    def identify_reasoning_features(self, sparse_acts: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Identify features most associated with reasoning vs memory"""
        
        reasoning_mask = labels.bool()
        memory_mask = ~reasoning_mask
        
        reasoning_acts = sparse_acts[reasoning_mask]
        memory_acts = sparse_acts[memory_mask]
        
        # Compute differential activation
        reasoning_mean = torch.mean(reasoning_acts, dim=0)
        memory_mean = torch.mean(memory_acts, dim=0)
        
        differential = reasoning_mean - memory_mean
        
        # Get top reasoning features
        top_reasoning_indices = torch.topk(differential, k=self.config.top_k_features)[1]
        top_memory_indices = torch.topk(-differential, k=self.config.top_k_features)[1]
        
        # Compute feature discriminative power
        feature_correlations = []
        for i in range(sparse_acts.shape[1]):
            acts = sparse_acts[:, i]
            if torch.std(acts) > 1e-6:  # Avoid division by zero
                corr = torch.corrcoef(torch.stack([acts, labels.float()]))[0, 1].item()
                feature_correlations.append(abs(corr))
            else:
                feature_correlations.append(0.0)
        
        feature_correlations = torch.tensor(feature_correlations)
        top_discriminative = torch.topk(feature_correlations, k=self.config.top_k_features)[1]
        
        return {
            'top_reasoning_features': top_reasoning_indices.tolist(),
            'top_memory_features': top_memory_indices.tolist(),
            'top_discriminative_features': top_discriminative.tolist(),
            'reasoning_feature_activations': reasoning_mean[top_reasoning_indices].tolist(),
            'memory_feature_activations': memory_mean[top_memory_indices].tolist(),
            'feature_correlations': feature_correlations.tolist(),
            'differential_activations': differential.tolist()
        }
    
    def compare_with_pca(self, sparse_acts: torch.Tensor, hidden_states: torch.Tensor, 
                         labels: torch.Tensor) -> Dict:
        """Compare SAE features with PCA baseline"""
        
        # PCA on hidden states
        pca = PCA(n_components=min(50, hidden_states.shape[1]))  # Top 50 components
        pca_features = pca.fit_transform(hidden_states.numpy())
        pca_features = torch.from_numpy(pca_features)
        
        # Train linear classifiers
        sae_classifier = LogisticRegression(max_iter=1000)
        pca_classifier = LogisticRegression(max_iter=1000)
        
        # Use a subset for training to avoid memory issues
        n_samples = min(5000, len(labels))
        indices = torch.randperm(len(labels))[:n_samples]
        
        sae_acts_subset = sparse_acts[indices].numpy()
        pca_features_subset = pca_features[indices].numpy()
        labels_subset = labels[indices].numpy()
        
        sae_classifier.fit(sae_acts_subset, labels_subset)
        pca_classifier.fit(pca_features_subset, labels_subset)
        
        # Evaluate
        sae_acc = accuracy_score(labels_subset, sae_classifier.predict(sae_acts_subset))
        pca_acc = accuracy_score(labels_subset, pca_classifier.predict(pca_features_subset))
        
        return {
            'sae_classification_accuracy': sae_acc,
            'pca_classification_accuracy': pca_acc,
            'pca_explained_variance_ratio': pca.explained_variance_ratio_[:10].tolist(),
            'improvement_over_pca': sae_acc - pca_acc
        }
    
    def compute_interpretability_metrics(self, sparse_acts: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Compute metrics for feature interpretability"""
        
        # Sparsity metrics
        activation_frequencies = torch.mean((sparse_acts > 0).float(), dim=0)
        sparsity = 1 - activation_frequencies
        
        # Feature uniqueness (how often features activate together)
        coactivation_matrix = torch.corrcoef(sparse_acts.T)
        coactivation_matrix = torch.abs(coactivation_matrix)
        coactivation_matrix.fill_diagonal_(0)
        
        avg_coactivation = torch.mean(coactivation_matrix)
        max_coactivation = torch.max(coactivation_matrix)
        
        return {
            'mean_sparsity': torch.mean(sparsity).item(),
            'sparsity_std': torch.std(sparsity).item(),
            'avg_feature_coactivation': avg_coactivation.item(),
            'max_feature_coactivation': max_coactivation.item(),
            'features_with_high_sparsity': torch.sum(sparsity > 0.9).item(),
            'features_with_low_sparsity': torch.sum(sparsity < 0.5).item()
        }
    
    def save_checkpoint(self, layer: int, epoch: int):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.output_dir, 'checkpoints', f'layer_{layer}')
        os.makedirs(checkpoint_path, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.saes[layer].state_dict(),
            'optimizer_state_dict': self.optimizers[layer].state_dict(),
            'epoch': epoch,
            'config': self.config
        }, os.path.join(checkpoint_path, f'sae_epoch_{epoch}.pt'))
    
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
    
    def save_layer_analysis(self, layer: int, analysis: Dict):
        """Save analysis results for a layer"""
        analysis_path = os.path.join(self.config.output_dir, 'analysis')
        os.makedirs(analysis_path, exist_ok=True)
        
        with open(os.path.join(analysis_path, f'layer_{layer}_analysis.json'), 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_analysis = self._convert_tensors_to_json(analysis)
            json.dump(json_analysis, f, indent=2)
    
    def save_final_results(self):
        """Save final training results and analysis"""
        results_path = os.path.join(self.config.output_dir, 'final_results')
        os.makedirs(results_path, exist_ok=True)
        
        # Save training history
        with open(os.path.join(results_path, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save feature analysis
        with open(os.path.join(results_path, 'feature_analysis.json'), 'w') as f:
            json_feature_analysis = self._convert_tensors_to_json(self.feature_analysis)
            json.dump(json_feature_analysis, f, indent=2)
        
        # Save trained models
        models_path = os.path.join(results_path, 'models')
        os.makedirs(models_path, exist_ok=True)
        
        for layer, sae in self.saes.items():
            torch.save({
                'model_state_dict': sae.state_dict(),
                'config': self.config,
                'layer': layer
            }, os.path.join(models_path, f'sae_layer_{layer}.pt'))
        
        # Generate summary report
        self.generate_summary_report(results_path)
    
    def generate_summary_report(self, results_path: str):
        """Generate a summary report of the training and analysis"""
        
        report_lines = [
            "# SAE Training Summary Report\n",
            f"Model: {self.config.model_name}",
            f"Layers trained: {list(self.saes.keys())}",
            f"Training epochs: {self.config.num_epochs}",
            f"Sparsity penalty: {self.config.sparsity_penalty}",
            f"Dictionary expansion factor: {self.config.expansion_factor}\n",
            "## Layer-wise Results:\n"
        ]
        
        for layer in sorted(self.saes.keys()):
            if layer in self.feature_analysis:
                analysis = self.feature_analysis[layer]
                
                report_lines.extend([
                    f"### Layer {layer}:",
                    f"- Final reconstruction loss: {self.training_history[layer]['reconstruction_loss'][-1]:.4f}",
                    f"- Final sparsity loss: {self.training_history[layer]['sparsity_loss'][-1]:.4f}",
                    f"- Activation density: {analysis['feature_stats']['avg_activation_density']:.3f}",
                    f"- SAE vs PCA accuracy: {analysis['pca_comparison']['sae_classification_accuracy']:.3f} vs {analysis['pca_comparison']['pca_classification_accuracy']:.3f}",
                    f"- Mean feature sparsity: {analysis['interpretability']['mean_sparsity']:.3f}",
                    ""
                ])
        
        with open(os.path.join(results_path, 'summary_report.md'), 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to: {os.path.join(results_path, 'summary_report.md')}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train SAE for reasoning feature extraction")
    
    # Add command line arguments
    parser.add_argument('--config', type=str, help="Path to YAML config file")
    parser.add_argument('--model_name', type=str, default="Llama-3.1-8B")
    parser.add_argument('--layers', nargs='+', type=int, default=[10, 15, 20, 25])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--sparsity_penalty', type=float, default=1e-3)
    parser.add_argument('--expansion_factor', type=int, default=4)
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = SAEConfig(**config_dict)
    else:
        config = SAEConfig()
    
    # Override with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.layers:
        config.layers_to_train = args.layers
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.sparsity_penalty:
        config.sparsity_penalty = args.sparsity_penalty
    if args.expansion_factor:
        config.expansion_factor = args.expansion_factor
    if args.use_wandb:
        config.use_wandb = True
    
    print("Configuration:")
    print(f"Model: {config.model_name}")
    print(f"Layers: {config.layers_to_train}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Sparsity penalty: {config.sparsity_penalty}")
    print(f"Dictionary expansion: {config.expansion_factor}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = SAETrainer(config)
    
    # Start training
    trainer.train_all_layers()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()