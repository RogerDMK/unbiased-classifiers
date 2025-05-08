import torch
import argparse
from transformers import AutoTokenizer
from models import JigsawToxicityClassifier, JigsawDivDisClassifier
from train import train_jigsaw_baseline, train_jigsaw_divdis
from jigsaw_utils import get_jigsaw_dfs, JigsawDataset, JigsawDataLoader, evaluate_bias_metrics

# Parse command line arguments
parser = argparse.ArgumentParser(description='Jigsaw Toxicity Classification Pipeline')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'visualize'],
                    help='Pipeline mode: train, evaluate, or visualize')
parser.add_argument('--model_type', type=str, default='both', choices=['baseline', 'divdis', 'both'],
                    help='Model type to train/evaluate')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--train_samples', type=int, default=5000, help='Number of training samples')
parser.add_argument('--test_samples', type=int, default=1000, help='Number of test samples')
args = parser.parse_args()

# Use MPS if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Process Jigsaw dataset with specified sample sizes
data_dict = get_jigsaw_dfs(train_samples=args.train_samples, test_samples=args.test_samples)

# Create datasets with the BERT tokenizer
train_dataset = JigsawDataset(data_dict['full_train'], tokenizer)
test_dataset = JigsawDataset(data_dict['full_test'], tokenizer)
identity_train = JigsawDataset(data_dict['identity_train'], tokenizer)

# Create data loaders
train_loader = JigsawDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = JigsawDataLoader(test_dataset, batch_size=args.batch_size)
identity_loader = JigsawDataLoader(identity_train, batch_size=args.batch_size, shuffle=True)

# Get number of identity features
num_identity_features = len(train_dataset.identity_cols)

if args.mode == 'train':
    if args.model_type in ['baseline', 'both']:
        # Initialize and train baseline model
        print("Training baseline model...")
        baseline_model = JigsawToxicityClassifier(
            pretrain_model="bert-base-uncased",
            num_identity_features=num_identity_features,
            mlp_dropout=0.1
        )
        
        best_baseline, _ = train_jigsaw_baseline(
            baseline_model,
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=args.epochs,
            device=device
        )
        
        # Save baseline model
        torch.save(best_baseline.state_dict(), "jigsaw_baseline.pt")
        print("Baseline model saved to jigsaw_baseline.pt")
    
    if args.model_type in ['divdis', 'both']:
        # Initialize and train DivDis model
        print("Training DivDis model...")
        divdis_model = JigsawDivDisClassifier(
            pretrain_model="bert-base-uncased",
            num_identity_features=num_identity_features,
            num_heads=3,
            mlp_dropout=0.1,
            diversity_weight=1e-3
        )
        
        divdis_model = train_jigsaw_divdis(
            model=divdis_model,
            train_loader=train_loader,
            diverse_loader=identity_loader,
            val_loader=test_loader,
            num_epochs=args.epochs,
            device=device
        )
        
        # Save DivDis model
        torch.save(divdis_model.state_dict(), "jigsaw_divdis.pt")
        print("DivDis model saved to jigsaw_divdis.pt")

elif args.mode == 'evaluate':
    # Load models for evaluation
    if args.model_type in ['baseline', 'both']:
        baseline_model = JigsawToxicityClassifier(
            pretrain_model="bert-base-uncased",
            num_identity_features=num_identity_features,
            mlp_dropout=0.1
        )
        baseline_model.load_state_dict(torch.load("jigsaw_baseline.pt"))
        baseline_model.to(device)
        
        # Evaluate baseline model
        print("\nEvaluating baseline model on bias metrics...")
        baseline_metrics = evaluate_bias_metrics(baseline_model, data_dict, tokenizer, device=device)
        
        # Print summary metrics
        print("\nBaseline Summary Metrics:")
        for metric, value in baseline_metrics['summary'].items():
            print(f"{metric}: {value:.4f}")
    
    if args.model_type in ['divdis', 'both']:
        divdis_model = JigsawDivDisClassifier(
            pretrain_model="bert-base-uncased",
            num_identity_features=num_identity_features,
            num_heads=3,
            mlp_dropout=0.1,
            diversity_weight=1e-3
        )
        divdis_model.load_state_dict(torch.load("jigsaw_divdis.pt"))
        divdis_model.to(device)
        
        # Evaluate DivDis model
        print("\nEvaluating DivDis model on bias metrics...")
        divdis_metrics = evaluate_bias_metrics(divdis_model, data_dict, tokenizer, device=device)
        
        # Print summary metrics
        print("\nDivDis Summary Metrics:")
        for metric, value in divdis_metrics['summary'].items():
            print(f"{metric}: {value:.4f}")
    
    # Compare models if both are evaluated
    if args.model_type == 'both':
        print("\n=== Identity Subgroup Performance Comparison ===")
        print("Identity\tBaseline\tDivDis\tImprovement")
        
        for identity in baseline_metrics['identity_metrics'].keys():
            if identity in divdis_metrics['identity_metrics']:
                baseline_score = baseline_metrics['identity_metrics'][identity]['subgroup_auc']
                divdis_score = divdis_metrics['identity_metrics'][identity]['subgroup_auc']
                improvement = divdis_score - baseline_score
                print(f"{identity}\t{baseline_score:.4f}\t{divdis_score:.4f}\t{improvement:.4f}")
        
        # Save detailed results to file
        import json
        with open('bias_metrics_results.json', 'w') as f:
            json.dump({
                'baseline': baseline_metrics,
                'divdis': divdis_metrics
            }, f, indent=2)
        
        print("\nDetailed results saved to bias_metrics_results.json")

elif args.mode == 'visualize':
    # Import visualization modules only when needed
    from jigsaw_visualizer import (
        plot_bias_metrics, 
        plot_improvement, 
        plot_head_specialization, 
        plot_confusion_matrices,
        plot_identity_attribution_differences
    )
    import pandas as pd
    
    # Load saved metrics and generate visualizations
    try:
        import json
        with open('bias_metrics_results.json', 'r') as f:
            metrics = json.load(f)
        
        # Generate visualizations based on saved metrics
        print("Generating visualizations from saved metrics...")
        # Add visualization code here
    except:
        print("Error loading metrics file. Run evaluation first.")