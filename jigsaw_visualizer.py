import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import os

# Create output directory
os.makedirs("jigsaw_plots", exist_ok=True)

def plot_bias_metrics(baseline_bias, divdis_bias, metric='di_fpr', title=None):
    """
    Plot bias metrics comparison between baseline and DivDis models
    
    Args:
        baseline_bias: DataFrame with baseline bias metrics
        divdis_bias: DataFrame with DivDis bias metrics
        metric: Metric to plot ('di_fpr' or 'di_fnr')
        title: Plot title
    """
    # Prepare data
    plot_data = pd.merge(
        baseline_bias[['identity', metric]], 
        divdis_bias[['identity', metric]], 
        on='identity',
        suffixes=('_baseline', '_divdis')
    )
    
    # Sort by baseline metric
    plot_data = plot_data.sort_values(f"{metric}_baseline")
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Plot the data
    x = np.arange(len(plot_data))
    width = 0.35
    
    plt.bar(x - width/2, plot_data[f"{metric}_baseline"], width, label='Baseline')
    plt.bar(x + width/2, plot_data[f"{metric}_divdis"], width, label='DivDis')
    
    # Add a horizontal line at y=1 (perfect fairness)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='Perfect Fairness')
    
    # Add labels and title
    plt.xlabel('Identity Group')
    if metric == 'di_fpr':
        ylabel = 'Disparate Impact (FPR)'
    elif metric == 'di_fnr':
        ylabel = 'Disparate Impact (FNR)'
    else:
        ylabel = metric
    
    plt.ylabel(ylabel)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Disparate Impact Comparison: {ylabel}')
    
    plt.xticks(x, plot_data['identity'], rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"jigsaw_plots/{metric}_comparison.png")
    plt.show()

def plot_improvement(bias_comparison, metric='fpr_improvement', title=None):
    """
    Plot improvement in bias metrics from baseline to DivDis
    
    Args:
        bias_comparison: DataFrame with improvement metrics
        metric: Metric to plot ('fpr_improvement' or 'fnr_improvement')
        title: Plot title
    """
    # Sort by improvement
    plot_data = bias_comparison.sort_values(metric)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Plot the data
    bars = plt.barh(plot_data['identity'], plot_data[metric])
    
    # Color bars based on improvement (positive = blue, negative = red)
    for i, bar in enumerate(bars):
        if plot_data[metric].iloc[i] < 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    # Add a vertical line at x=0 (no improvement)
    plt.axvline(x=0.0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Improvement (positive = better)')
    plt.ylabel('Identity Group')
    
    if title:
        plt.title(title)
    else:
        if metric == 'fpr_improvement':
            plt.title('Improvement in False Positive Rate Disparate Impact')
        elif metric == 'fnr_improvement':
            plt.title('Improvement in False Negative Rate Disparate Impact')
        else:
            plt.title(f'Improvement in {metric}')
    
    plt.tight_layout()
    plt.savefig(f"jigsaw_plots/{metric}.png")
    plt.show()

def plot_head_specialization(head_analysis, metric='acc'):
    """
    Plot how different heads specialize on different identity groups
    
    Args:
        head_analysis: Dictionary with head analysis results
        metric: Metric to plot ('acc' or 'toxic_prob')
    """
    # Process data for plotting
    plot_data = []
    
    for group_name, results in head_analysis.items():
        for identity in results:
            for h, metrics in enumerate(identity['head_metrics']):
                plot_data.append({
                    'group': group_name,
                    'identity': identity['identity'],
                    'head': f'Head {h}',
                    'accuracy': metrics['acc'],
                    'toxic_prob': metrics['toxic_prob'],
                    'samples': identity['samples']
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create a pivot table for heatmap
    if metric == 'acc':
        pivot = plot_df.pivot_table(
            index='identity', 
            columns='head', 
            values='accuracy',
            aggfunc='mean'
        )
        title = 'Head Specialization by Accuracy'
        cmap = 'viridis'
    else:
        pivot = plot_df.pivot_table(
            index='identity', 
            columns='head', 
            values='toxic_prob',
            aggfunc='mean'
        )
        title = 'Head Specialization by Toxicity Probability'
        cmap = 'Reds'
    
    # Plot heatmap
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot, annot=True, cmap=cmap, fmt='.3f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"jigsaw_plots/head_specialization_{metric}.png")
    plt.show()
    
    # Plot bar chart grouped by identity group
    plt.figure(figsize=(15, 10))
    
    # Group by identity group
    for group in plot_df['group'].unique():
        group_df = plot_df[plot_df['group'] == group]
        
        # Create subplot for this group
        plt.figure(figsize=(12, 6))
        
        # Plot bars for each identity in this group
        identities = group_df['identity'].unique()
        x = np.arange(len(identities))
        width = 0.2
        
        for i, head in enumerate(['Head 0', 'Head 1', 'Head 2']):
            head_values = []
            for identity in identities:
                value = group_df[(group_df['identity'] == identity) & 
                                 (group_df['head'] == head)][metric].values
                head_values.append(value[0] if len(value) > 0 else 0)
            
            plt.bar(x + (i-1)*width, head_values, width, label=head)
        
        plt.xlabel('Identity')
        if metric == 'acc':
            plt.ylabel('Accuracy')
        else:
            plt.ylabel('Toxicity Probability')
        plt.title(f'{group} Group: Head Performance by {metric}')
        plt.xticks(x, identities, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"jigsaw_plots/head_performance_{group}_{metric}.png")
        plt.show()

def plot_confusion_matrices(model, data_dict, tokenizer, device, identity_cols=None):
    """
    Plot confusion matrices for different identity groups
    
    Args:
        model: Trained model (baseline or DivDis)
        data_dict: Dictionary with dataframes
        tokenizer: BERT tokenizer
        device: Device to run evaluation on
        identity_cols: List of identity columns to analyze (None for all)
    """
    from jigsaw_utils import JigsawDataset, JigsawDataLoader
    
    # Get test data
    test_data = data_dict['full_test']
    
    # If no identity columns specified, use all
    if identity_cols is None:
        identity_cols = [col for col in test_data.columns if col in [
            'male', 'female', 'transgender', 'other_gender',
            'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 
            'other_sexual_orientation', 'christian', 'jewish', 'muslim',
            'hindu', 'buddhist', 'atheist', 'other_religion', 'black',
            'white', 'asian', 'latino', 'other_race_or_ethnicity',
            'physical_disability', 'intellectual_or_learning_disability',
            'psychiatric_or_mental_illness', 'other_disability'
        ]]
    
    # Create figure for all confusion matrices
    n_cols = 4
    n_rows = (len(identity_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()
    
    # First, get overall confusion matrix
    overall_dataset = JigsawDataset(test_data, tokenizer)
    overall_loader = JigsawDataLoader(overall_dataset, batch_size=32)
    
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, identity_features, labels in overall_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            identity_features = identity_features.to(device)
            
            if hasattr(model, 'num_heads'):  # DivDis model
                logits = model(input_ids, attention_mask, identity_features)
                logits = logits.mean(dim=1)  # Average across heads
            else:  # Baseline model
                logits = model(input_ids, attention_mask, identity_features)
            
            _, predicted = torch.max(logits, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Plot overall confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title('Overall')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Plot confusion matrix for each identity group
    for i, col in enumerate(identity_cols):
        # Skip if index out of range
        if i+1 >= len(axes):
            break
            
        # Get data for this identity
        identity_data = test_data[test_data[col] == 1].copy()
        
        # Skip if too few samples
        if len(identity_data) < 10:
            axes[i+1].set_title(f'{col} (insufficient data)')
            continue
        
        # Create dataset and loader
        identity_dataset = JigsawDataset(identity_data, tokenizer)
        identity_loader = JigsawDataLoader(identity_dataset, batch_size=32)
        
        # Get predictions
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for input_ids, attention_mask, identity_features, labels in identity_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                identity_features = identity_features.to(device)
                
                if hasattr(model, 'num_heads'):  # DivDis model
                    logits = model(input_ids, attention_mask, identity_features)
                    logits = logits.mean(dim=1)  # Average across heads
                else:  # Baseline model
                    logits = model(input_ids, attention_mask, identity_features)
                
                _, predicted = torch.max(logits, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i+1], cmap='Blues')
        axes[i+1].set_title(col)
        axes[i+1].set_xlabel('Predicted')
        axes[i+1].set_ylabel('True')
    
    # Hide unused subplots
    for i in range(len(identity_cols) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    model_type = 'divdis' if hasattr(model, 'num_heads') else 'baseline'
    plt.savefig(f"jigsaw_plots/confusion_matrices_{model_type}.png")
    plt.show()

def plot_identity_attribution_differences(bias_df):
    """
    Plot differences in attributions across identity groups
    
    Args:
        bias_df: DataFrame with attribution differences
    """
    # Get all difference columns
    diff_cols = [col for col in bias_df.columns if '_vs_' in col and 'diff' in col]
    
    # Create a melted dataframe for easier plotting
    melted = pd.melt(
        bias_df, 
        id_vars=['text'], 
        value_vars=diff_cols,
        var_name='comparison', 
        value_name='difference'
    )
    
    # Extract head, group1, and group2 from comparison column
    melted[['head', 'groups', 'suffix']] = melted['comparison'].str.split('_', n=2, expand=True)
    melted[['group1', 'group2']] = melted['groups'].str.split('vs', expand=True)
    
    # Clean up group names
    melted['group1'] = melted['group1'].str.strip('_')
    melted['group2'] = melted['group2'].str.strip('_')
    
    # Plot differences by head and group comparison
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='head', y='difference', hue='groups', data=melted)
    plt.title('Attribution Differences Across Identity Groups')
    plt.xlabel('Model Head')
    plt.ylabel('Mean Absolute Difference in Attributions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("jigsaw_plots/attribution_differences.png")
    plt.show()
    
    # Plot differences by text example
    plt.figure(figsize=(15, 6))
    sns.boxplot(x='text', y='difference', data=melted)
    plt.title('Attribution Differences by Text Example')
    plt.xlabel('Text')
    plt.ylabel('Mean Absolute Difference in Attributions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("jigsaw_plots/attribution_differences_by_text.png")
    plt.show() 