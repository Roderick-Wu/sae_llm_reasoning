#!/usr/bin/env python3
"""
Standalone script to extract and analyze SAE evaluation results from evaluation_results.json
This script provides a quick way to view the performance data without running the full evaluation.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Tuple, Any

def load_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def extract_performance_data(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract performance data into a flat structure for analysis"""
    
    performance_data = []
    
    for dataset_name, dataset_results in results.items():
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Process baseline results first
        if 'baseline' in dataset_results:
            baseline_data = dataset_results['baseline']
            performance = baseline_data.get('performance', {})
            print(f"  Baseline: {performance}")
            
            if isinstance(performance, dict):
                if dataset_name == 'mmlu-pro':
                    # MMLU-Pro has both reason and memory accuracy
                    for metric_type, acc_key in [('reasoning', 'reason_accuracy'), ('memory', 'memory_accuracy')]:
                        if acc_key in performance:
                            performance_data.append({
                                'dataset': dataset_name,
                                'layer': 'baseline',
                                'intervention_type': 'none',
                                'scale': 0.0,
                                'metric_type': metric_type,
                                'accuracy': performance[acc_key],
                                'num_responses': len(baseline_data.get('responses', []))
                            })
                else:
                    # Other datasets
                    accuracy = performance.get('accuracy') or performance.get('reason_accuracy')
                    if accuracy is not None:
                        performance_data.append({
                            'dataset': dataset_name,
                            'layer': 'baseline',
                            'intervention_type': 'none',
                            'scale': 0.0,
                            'metric_type': 'overall',
                            'accuracy': accuracy,
                            'num_responses': len(baseline_data.get('responses', []))
                        })
            elif isinstance(performance, (int, float)):
                performance_data.append({
                    'dataset': dataset_name,
                    'layer': 'baseline',
                    'intervention_type': 'none',
                    'scale': 0.0,
                    'metric_type': 'overall',
                    'accuracy': performance,
                    'num_responses': len(baseline_data.get('responses', []))
                })
        
        # Process intervention results
        for layer_key, layer_results in dataset_results.items():
            if layer_key == 'baseline':  # Skip baseline, already processed
                continue
            layer = int(layer_key.split('_')[1])
            
            for result_key, result_data in layer_results.items():
                # Parse result key (e.g., "projected_scale_0.1", "raw_scale_0.3")
                parts = result_key.split('_')
                intervention_type = parts[0]
                scale = float(parts[2])
                
                # Extract performance metrics
                performance = result_data.get('performance', {})
                
                if isinstance(performance, dict):
                    # Handle different dataset structures
                    if dataset_name == 'mmlu-pro':
                        # MMLU-Pro has both reason and memory accuracy
                        reason_acc = performance.get('reason_accuracy', None)
                        memory_acc = performance.get('memory_accuracy', None)
                        
                        if reason_acc is not None:
                            performance_data.append({
                                'dataset': dataset_name,
                                'layer': layer,
                                'intervention_type': intervention_type,
                                'scale': scale,
                                'metric_type': 'reasoning',
                                'accuracy': reason_acc,
                                'num_responses': len(result_data.get('responses', []))
                            })
                        
                        if memory_acc is not None:
                            performance_data.append({
                                'dataset': dataset_name,
                                'layer': layer,
                                'intervention_type': intervention_type,
                                'scale': scale,
                                'metric_type': 'memory',
                                'accuracy': memory_acc,
                                'num_responses': len(result_data.get('responses', []))
                            })
                    else:
                        # Other datasets have a single accuracy metric
                        accuracy = performance.get('accuracy', None)
                        if accuracy is None and 'reason_accuracy' in performance:
                            accuracy = performance['reason_accuracy']
                        
                        if accuracy is not None:
                            performance_data.append({
                                'dataset': dataset_name,
                                'layer': layer,
                                'intervention_type': intervention_type,
                                'scale': scale,
                                'metric_type': 'overall',
                                'accuracy': accuracy,
                                'num_responses': len(result_data.get('responses', []))
                            })
                elif isinstance(performance, (int, float)):
                    # Direct accuracy value
                    performance_data.append({
                        'dataset': dataset_name,
                        'layer': layer,
                        'intervention_type': intervention_type,
                        'scale': scale,
                        'metric_type': 'overall',
                        'accuracy': performance,
                        'num_responses': len(result_data.get('responses', []))
                    })
                
                print(f"  Layer {layer}, {intervention_type}, scale {scale}: {performance}")
    
    return performance_data

def analyze_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance data and generate summary statistics"""
    
    analysis = {}
    
    # Overall statistics
    analysis['total_experiments'] = len(df)
    analysis['datasets'] = df['dataset'].unique().tolist()
    # Get layers excluding baseline
    intervention_layers = df[df['layer'] != 'baseline']['layer'].unique()
    analysis['layers'] = sorted([int(l) for l in intervention_layers if isinstance(l, int)])
    analysis['intervention_types'] = df['intervention_type'].unique().tolist()
    analysis['scales'] = sorted(df['scale'].unique())
    
    print(f"Analysis: {len(df)} experiments across datasets {analysis['datasets']}")
    
    # Best performance per dataset (excluding baseline)
    analysis['best_per_dataset'] = {}
    intervention_df = df[df['intervention_type'] != 'none']  # Exclude baseline
    for dataset in intervention_df['dataset'].unique():
        dataset_df = intervention_df[intervention_df['dataset'] == dataset]
        best_idx = dataset_df['accuracy'].idxmax()
        best_row = dataset_df.loc[best_idx]
        
        analysis['best_per_dataset'][dataset] = {
            'accuracy': best_row['accuracy'],
            'layer': best_row['layer'],
            'intervention_type': best_row['intervention_type'],
            'scale': best_row['scale'],
            'metric_type': best_row['metric_type']
        }
    
    # Best performance per layer (excluding baseline)
    analysis['best_per_layer'] = {}
    for layer in intervention_df['layer'].unique():
        layer_df = intervention_df[intervention_df['layer'] == layer]
        best_idx = layer_df['accuracy'].idxmax()
        best_row = layer_df.loc[best_idx]
        
        analysis['best_per_layer'][layer] = {
            'accuracy': best_row['accuracy'],
            'dataset': best_row['dataset'],
            'intervention_type': best_row['intervention_type'],
            'scale': best_row['scale']
        }
    
    # Intervention type comparison
    analysis['intervention_comparison'] = {}
    for intervention in df['intervention_type'].unique():
        intervention_df = df[df['intervention_type'] == intervention]
        analysis['intervention_comparison'][intervention] = {
            'mean_accuracy': intervention_df['accuracy'].mean(),
            'std_accuracy': intervention_df['accuracy'].std(),
            'max_accuracy': intervention_df['accuracy'].max(),
            'min_accuracy': intervention_df['accuracy'].min(),
            'count': len(intervention_df)
        }
    
    return analysis

def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations of the results"""
    
    plt.style.use('default')
    plots_dir = os.path.join(output_dir, 'analysis_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Performance by Layer and Intervention Type
    fig, axes = plt.subplots(1, len(df['dataset'].unique()), figsize=(5*len(df['dataset'].unique()), 6))
    if len(df['dataset'].unique()) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == dataset]
        
        for intervention_type in dataset_df['intervention_type'].unique():
            intervention_df = dataset_df[dataset_df['intervention_type'] == intervention_type]
            
            # Group by layer and take max accuracy
            layer_performance = intervention_df.groupby('layer')['accuracy'].max().reset_index()
            
            ax.plot(layer_performance['layer'], layer_performance['accuracy'], 
                   marker='o', label=f'{intervention_type}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Best Accuracy')
        ax.set_title(f'{dataset.upper()} - Performance by Layer')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_by_layer.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scale Analysis Heatmap
    fig, axes = plt.subplots(2, len(df['dataset'].unique()), figsize=(4*len(df['dataset'].unique()), 8))
    if len(df['dataset'].unique()) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        
        for row, intervention_type in enumerate(df['intervention_type'].unique()):
            ax = axes[row, idx]
            intervention_df = dataset_df[dataset_df['intervention_type'] == intervention_type]
            
            if len(intervention_df) > 0:
                # Create pivot table for heatmap
                pivot_df = intervention_df.pivot_table(
                    values='accuracy', 
                    index='layer', 
                    columns='scale', 
                    aggfunc='max'
                )
                
                sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', ax=ax)
                ax.set_title(f'{dataset.upper()} - {intervention_type.capitalize()} Intervention')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dataset.upper()} - {intervention_type.capitalize()} Intervention')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scale_analysis_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution Analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy distribution by intervention type
    axes[0,0].boxplot([df[df['intervention_type'] == t]['accuracy'].values 
                       for t in df['intervention_type'].unique()],
                      labels=df['intervention_type'].unique())
    axes[0,0].set_title('Accuracy Distribution by Intervention Type')
    axes[0,0].set_ylabel('Accuracy')
    
    # Accuracy distribution by dataset
    axes[0,1].boxplot([df[df['dataset'] == d]['accuracy'].values 
                       for d in df['dataset'].unique()],
                      labels=df['dataset'].unique())
    axes[0,1].set_title('Accuracy Distribution by Dataset')
    axes[0,1].set_ylabel('Accuracy')
    
    # Scale vs Accuracy scatter
    for intervention in df['intervention_type'].unique():
        intervention_df = df[df['intervention_type'] == intervention]
        axes[1,0].scatter(intervention_df['scale'], intervention_df['accuracy'], 
                         label=intervention, alpha=0.6)
    axes[1,0].set_xlabel('Scale')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_title('Scale vs Accuracy')
    axes[1,0].legend()
    
    # Layer vs Accuracy scatter
    for intervention in df['intervention_type'].unique():
        intervention_df = df[df['intervention_type'] == intervention]
        axes[1,1].scatter(intervention_df['layer'], intervention_df['accuracy'], 
                         label=intervention, alpha=0.6)
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_title('Layer vs Accuracy')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {plots_dir}")

def generate_summary_report(analysis: Dict[str, Any], df: pd.DataFrame, output_dir: str):
    """Generate a comprehensive summary report"""
    
    report_lines = [
        "# SAE Evaluation Results Analysis",
        f"Generated from evaluation_results.json\n",
        "## Overview",
        f"- Total experiments: {analysis['total_experiments']}",
        f"- Datasets evaluated: {', '.join(analysis['datasets'])}",
        f"- Layers tested: {analysis['layers']}",
        f"- Intervention types: {', '.join(analysis['intervention_types'])}",
        f"- Scales tested: {analysis['scales']}\n"
    ]
    
    # Add baseline performance section
    baseline_df = df[df['intervention_type'] == 'none']
    if len(baseline_df) > 0:
        report_lines.extend([
            "## Baseline Performance (No Intervention)",
            ""
        ])
        for dataset in baseline_df['dataset'].unique():
            dataset_baseline = baseline_df[baseline_df['dataset'] == dataset]
            for _, row in dataset_baseline.iterrows():
                report_lines.append(f"### {dataset.upper()}")
                report_lines.append(f"- **Baseline Accuracy:** {row['accuracy']:.4f}")
                report_lines.append(f"- **Metric Type:** {row['metric_type'].title()}")
                report_lines.append("")
    
    report_lines.extend([
        "## Best Performance per Dataset"
    ])
    
    intervention_df = df[df['intervention_type'] != 'none']  # Exclude baseline from best performance
    for dataset, best in analysis['best_per_dataset'].items():
        # Get baseline for comparison
        baseline_acc = None
        dataset_baseline = baseline_df[baseline_df['dataset'] == dataset]
        if len(dataset_baseline) > 0:
            if dataset == 'mmlu-pro':
                # Find baseline for same metric type
                matching_baseline = dataset_baseline[dataset_baseline['metric_type'] == best['metric_type']]
                if len(matching_baseline) > 0:
                    baseline_acc = matching_baseline.iloc[0]['accuracy']
            else:
                baseline_acc = dataset_baseline.iloc[0]['accuracy']
        
        report_lines.append(f"### {dataset.upper()}")
        report_lines.append(f"- **Best Accuracy:** {best['accuracy']:.4f}")
        if baseline_acc is not None:
            improvement = ((best['accuracy'] - baseline_acc) / baseline_acc) * 100
            report_lines.append(f"- **Baseline Accuracy:** {baseline_acc:.4f}")
            report_lines.append(f"- **Improvement:** {improvement:+.1f}%")
        report_lines.append(f"- **Layer:** {best['layer']}")
        report_lines.append(f"- **Intervention:** {best['intervention_type']}")
        report_lines.append(f"- **Scale:** {best['scale']}")
        report_lines.append(f"- **Metric Type:** {best['metric_type']}")
        report_lines.append("")
    
    report_lines.extend([
        "## Performance by Intervention Type"
    ])
    
    for intervention, stats in analysis['intervention_comparison'].items():
        report_lines.append(f"### {intervention.capitalize()} Intervention")
        report_lines.append(f"- **Mean Accuracy:** {stats['mean_accuracy']:.4f}")
        report_lines.append(f"- **Std Accuracy:** {stats['std_accuracy']:.4f}")
        report_lines.append(f"- **Max Accuracy:** {stats['max_accuracy']:.4f}")
        report_lines.append(f"- **Min Accuracy:** {stats['min_accuracy']:.4f}")
        report_lines.append(f"- **Experiments:** {stats['count']}")
        report_lines.append("")
    
    report_lines.extend([
        "## Best Performance by Layer"
    ])
    
    for layer in sorted(analysis['best_per_layer'].keys()):
        best = analysis['best_per_layer'][layer]
        report_lines.append(f"### Layer {layer}")
        report_lines.append(f"- **Best Accuracy:** {best['accuracy']:.4f}")
        report_lines.append(f"- **Dataset:** {best['dataset']}")
        report_lines.append(f"- **Intervention:** {best['intervention_type']}")
        report_lines.append(f"- **Scale:** {best['scale']}")
        report_lines.append("")
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_summary.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Analysis report saved to: {report_path}")

def export_to_csv(df: pd.DataFrame, output_dir: str):
    """Export results to CSV for further analysis"""
    
    print(f"Exporting DataFrame with shape {df.shape} and datasets: {df['dataset'].unique().tolist()}")
    
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results exported to CSV: {csv_path}")
    
    # Also create summary tables
    summary_tables = {}
    
    # Best performance per dataset/layer/intervention combination
    summary_tables['best_combinations'] = df.loc[df.groupby(['dataset', 'layer', 'intervention_type'])['accuracy'].idxmax()]
    
    # Mean performance by intervention type and scale
    summary_tables['mean_by_intervention_scale'] = df.groupby(['intervention_type', 'scale'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()
    
    # Best performance per dataset
    summary_tables['best_per_dataset'] = df.loc[df.groupby('dataset')['accuracy'].idxmax()]
    
    for name, table in summary_tables.items():
        table_path = os.path.join(output_dir, f'{name}.csv')
        table.to_csv(table_path, index=False)
        print(f"Summary table saved: {table_path}")

def main():
    """Main function to run the analysis"""
    
    # Configuration
    results_file = "/home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/evaluation_results/evaluation_results.json"
    output_dir = "/home/wuroderi/projects/def-zhijing/wuroderi/steering_vs_sae/evaluation_results"
    
    print("=== SAE Evaluation Results Analysis ===")
    print(f"Loading results from: {results_file}")
    
    try:
        # Load results
        results = load_results(results_file)
        print(f"Loaded results for {len(results)} datasets")
        
        # Extract performance data
        performance_data = extract_performance_data(results)
        print(f"Extracted {len(performance_data)} performance measurements")
        
        if len(performance_data) == 0:
            print("No performance data found!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(performance_data)
        print(f"Created DataFrame with {len(df)} rows")
        print("\nDataFrame info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        
        # Analyze performance
        analysis = analyze_performance(df)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        create_visualizations(df, output_dir)
        
        # Generate summary report
        print("Generating summary report...")
        generate_summary_report(analysis, df, output_dir)
        
        # Export to CSV
        print("Exporting to CSV...")
        export_to_csv(df, output_dir)
        
        # Print quick summary to console
        print("\n" + "="*60)
        print("QUICK SUMMARY")
        print("="*60)
        
        for dataset, best in analysis['best_per_dataset'].items():
            print(f"{dataset.upper()}: {best['accuracy']:.4f} (Layer {best['layer']}, {best['intervention_type']}, scale {best['scale']})")
        
        print(f"\nOverall best accuracy: {df['accuracy'].max():.4f}")
        best_idx = df['accuracy'].idxmax()
        best_overall = df.loc[best_idx]
        print(f"Best configuration: {best_overall['dataset']} - Layer {best_overall['layer']} - {best_overall['intervention_type']} - Scale {best_overall['scale']}")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()