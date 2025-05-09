import torch
import argparse
from transformers import AutoTokenizer
from models import JigsawToxicityClassifier, JigsawDivDisClassifier
from train import train_jigsaw_baseline, train_jigsaw_divdis
from jigsaw_utils import get_jigsaw_dfs, JigsawDataset, JigsawDataLoader, evaluate_bias_metrics, evaluate_bias_metrics_per_head, split_labeled_unlabeled

# Parse command line arguments
parser = argparse.ArgumentParser(description='Jigsaw Toxicity Classification Pipeline')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'visualize'],
                    help='Pipeline mode: train, evaluate, or visualize')
parser.add_argument('--model_type', type=str, default='both', choices=['baseline', 'divdis', 'both'],
                    help='Model type to train/evaluate')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--train_samples', type=int, default=25000, help='Number of training samples')
parser.add_argument('--test_samples', type=int, default=2000, help='Number of test samples')
args = parser.parse_args()

# Use MPS if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Process Jigsaw dataset with specified sample sizes
data_dict = get_jigsaw_dfs(train_samples=args.train_samples, test_samples=args.test_samples)

# Split training data into labeled and unlabeled sets
labeled_train, unlabeled_train = split_labeled_unlabeled(data_dict['full_train'], labeled_ratio=0.7)

# Create datasets with the BERT tokenizer
labeled_dataset = JigsawDataset(labeled_train, tokenizer)
unlabeled_dataset = JigsawDataset(unlabeled_train, tokenizer)
test_dataset = JigsawDataset(data_dict['full_test'], tokenizer)

# Create data loaders
labeled_loader = JigsawDataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
unlabeled_loader = JigsawDataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = JigsawDataLoader(test_dataset, batch_size=args.batch_size)

# Get number of identity features
num_identity_features = len(labeled_dataset.identity_cols)

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
            train_loader=labeled_loader,
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
            diversity_weight=5e-3
        )
        
        divdis_model = train_jigsaw_divdis(
            model=divdis_model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            val_loader=test_loader,
            num_epochs=args.epochs,
            device=device,
            validate_every_epoch=False  # Skip per-epoch validation
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
        
        # Evaluate DivDis model (ensemble and per-head)
        print("\nEvaluating DivDis model on bias metrics...")
        divdis_metrics = evaluate_bias_metrics(divdis_model, data_dict, tokenizer, device=device)
        
        # Evaluate each head separately
        print("\nEvaluating individual DivDis heads...")
        head_metrics = evaluate_bias_metrics_per_head(divdis_model, data_dict, tokenizer, device=device)
        
        # Print summary metrics for ensemble
        print("\nDivDis Ensemble Summary Metrics:")
        for metric, value in divdis_metrics['summary'].items():
            print(f"{metric}: {value:.4f}")
        
        # Print summary metrics for each head
        for head_name, metrics in head_metrics.items():
            if head_name.startswith('head_'):
                print(f"\n{head_name.replace('_', ' ').title()} Summary Metrics:")
                for metric, value in metrics['summary'].items():
                    print(f"{metric}: {value:.4f}")
        
        # Compare heads on bias metrics
        print("\n=== Head Bias Performance Comparison ===")
        print("Metric\tHead 0\tHead 1\tHead 2\tEnsemble")
        
        metrics_to_compare = ['overall_auc', 'subgroup_auc_mean', 'bpsn_auc_mean', 'bnsp_auc_mean', 'bias_score', 'final_score']
        
        for metric in metrics_to_compare:
            values = [
                head_metrics['head_0']['summary'][metric],
                head_metrics['head_1']['summary'][metric],
                head_metrics['head_2']['summary'][metric],
                divdis_metrics['summary'][metric]
            ]
            print(f"{metric}\t{values[0]:.4f}\t{values[1]:.4f}\t{values[2]:.4f}\t{values[3]:.4f}")
        
        # Find the best head for each identity group
        print("\n=== Best Head for Each Identity Group ===")
        print("Identity\tBest Head\tAUC\tImprovement over Ensemble")
        
        for identity in divdis_metrics['identity_metrics'].keys():
            ensemble_auc = divdis_metrics['identity_metrics'][identity]['subgroup_auc']
            head_aucs = []
            
            for h in range(3):
                head_name = f"head_{h}"
                if identity in head_metrics[head_name]['identity_metrics']:
                    head_auc = head_metrics[head_name]['identity_metrics'][identity]['subgroup_auc']
                    head_aucs.append((h, head_auc))
            
            if head_aucs:
                best_head, best_auc = max(head_aucs, key=lambda x: x[1])
                improvement = best_auc - ensemble_auc
                print(f"{identity}\tHead {best_head}\t{best_auc:.4f}\t{improvement:.4f}")
        
        # Find the best head vs baseline for each identity group
        if args.model_type == 'both':
            print("\n=== Best Head vs. Baseline for Each Identity Group ===")
            print("Identity\tBest Head\tHead AUC\tBaseline AUC\tImprovement")
            
            for identity in divdis_metrics['identity_metrics'].keys():
                if identity in baseline_metrics['identity_metrics']:
                    baseline_auc = baseline_metrics['identity_metrics'][identity]['subgroup_auc']
                    head_aucs = []
                    
                    for h in range(3):
                        head_name = f"head_{h}"
                        if identity in head_metrics[head_name]['identity_metrics']:
                            head_auc = head_metrics[head_name]['identity_metrics'][identity]['subgroup_auc']
                            head_aucs.append((h, head_auc))
                    
                    if head_aucs:
                        best_head, best_auc = max(head_aucs, key=lambda x: x[1])
                        improvement = best_auc - baseline_auc
                        print(f"{identity}\tHead {best_head}\t{best_auc:.4f}\t{baseline_auc:.4f}\t{improvement:.4f}")
    
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
                'divdis': divdis_metrics,
                'divdis_heads': head_metrics
            }, f, indent=2)
        
        print("\nDetailed results saved to bias_metrics_results.json")

elif args.mode == 'visualize':
    # Import visualization modules only when needed
    from jigsaw_visualizer import (
        plot_bias_metrics, 
        plot_improvement, 
        plot_head_specialization, 
        plot_confusion_matrices,
        plot_identity_attribution_differences,
        plot_head_bias_comparison,
        plot_best_head_vs_baseline
    )
    import pandas as pd
    
    # Load saved metrics and generate visualizations
    try:
        import json
        with open('bias_metrics_results.json', 'r') as f:
            metrics = json.load(f)
        
        # Generate visualizations based on saved metrics
        print("Generating visualizations from saved metrics...")
        plot_head_bias_comparison(metrics)
        plot_best_head_vs_baseline(metrics)
        print("Head bias comparison visualizations saved to head_bias_comparison.png and head_identity_specialization.png")
        print("Best head vs baseline comparison saved to best_head_vs_baseline.png")
    except Exception as e:
        print(f"Error generating visualizations: {e}")