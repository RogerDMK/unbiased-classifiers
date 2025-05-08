import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
import pandas as pd

class JigsawExplainer:
    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize explainer for Jigsaw models
        
        Args:
            model: Trained model (baseline or DivDis)
            tokenizer: BERT tokenizer
            device: Device to run explanations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize explainers
        if hasattr(model, 'num_heads'):  # DivDis model
            self.is_divdis = True
            self.explainers = []
            for h in range(model.num_heads):
                self.explainers.append(
                    LayerIntegratedGradients(
                        self._forward_func_divdis(h), 
                        model.bert.embeddings
                    )
                )
        else:  # Baseline model
            self.is_divdis = False
            self.explainer = LayerIntegratedGradients(
                self._forward_func, 
                model.bert.embeddings
            )
    
    def _forward_func(self, input_ids, attention_mask=None, identity_features=None):
        """Forward function for baseline model"""
        output = self.model(input_ids, attention_mask, identity_features)
        return output
    
    def _forward_func_divdis(self, head_idx):
        """Forward function for specific DivDis head"""
        def forward(input_ids, attention_mask=None, identity_features=None):
            output = self.model(input_ids, attention_mask, identity_features)
            return output[:, head_idx, :]
        return forward
    
    def explain_text(self, text, identity_features=None, target=1, head_idx=None, n_steps=50, visualize=True):
        """
        Explain model prediction for a given text
        
        Args:
            text: Text to explain
            identity_features: Identity features tensor (optional)
            target: Target class (0=non-toxic, 1=toxic)
            head_idx: Head index for DivDis model (None for all heads)
            n_steps: Number of steps for integrated gradients
            visualize: Whether to visualize the explanation
            
        Returns:
            attributions: Token attributions
            visualization: Visualization of attributions (if visualize=True)
        """
        # Tokenize input
        tokens = self.tokenizer.encode_plus(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        # Prepare identity features
        if identity_features is not None:
            identity_features = identity_features.to(self.device)
        
        # Get baseline (all padding tokens)
        baseline_input_ids = torch.zeros_like(input_ids).to(self.device)
        baseline_attention_mask = torch.zeros_like(attention_mask).to(self.device)
        
        # Get attributions
        if self.is_divdis:
            if head_idx is not None:  # Explain specific head
                attributions = self._explain_head(
                    head_idx, input_ids, attention_mask, identity_features,
                    baseline_input_ids, baseline_attention_mask, target, n_steps
                )
                head_attributions = {f"Head {head_idx}": attributions}
            else:  # Explain all heads
                head_attributions = {}
                for h in range(self.model.num_heads):
                    attributions = self._explain_head(
                        h, input_ids, attention_mask, identity_features,
                        baseline_input_ids, baseline_attention_mask, target, n_steps
                    )
                    head_attributions[f"Head {h}"] = attributions
        else:  # Baseline model
            attributions = self.explainer.attribute(
                inputs=(input_ids, attention_mask, identity_features),
                baselines=(baseline_input_ids, baseline_attention_mask, identity_features),
                target=target,
                n_steps=n_steps,
                additional_forward_args=None
            )
            head_attributions = {"Baseline": attributions[0]}
        
        # Decode tokens
        all_tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids[0]]
        
        # Visualize if requested
        if visualize:
            self._visualize_attributions(all_tokens, head_attributions, text)
        
        return head_attributions, all_tokens
    
    def _explain_head(self, head_idx, input_ids, attention_mask, identity_features,
                     baseline_input_ids, baseline_attention_mask, target, n_steps):
        """Explain a specific DivDis head"""
        attributions = self.explainers[head_idx].attribute(
            inputs=(input_ids, attention_mask, identity_features),
            baselines=(baseline_input_ids, baseline_attention_mask, identity_features),
            target=target,
            n_steps=n_steps,
            additional_forward_args=None
        )
        return attributions[0]  # Return only input_ids attributions
    
    def _visualize_attributions(self, tokens, head_attributions, text):
        """Visualize attributions for all heads"""
        num_heads = len(head_attributions)
        fig, axes = plt.subplots(num_heads, 1, figsize=(10, 3 * num_heads))
        
        if num_heads == 1:
            axes = [axes]
        
        for i, (head_name, attributions) in enumerate(head_attributions.items()):
            # Sum attributions across embedding dimension
            attr_sum = attributions.sum(dim=-1).squeeze(0)
            attr_sum = attr_sum.cpu().detach().numpy()
            
            # Filter out padding tokens
            valid_tokens = []
            valid_attr = []
            for token, attr in zip(tokens, attr_sum):
                if token not in ['[PAD]', '[CLS]', '[SEP]']:
                    valid_tokens.append(token)
                    valid_attr.append(attr)
            
            # Normalize attributions
            valid_attr = np.array(valid_attr)
            abs_attr = np.abs(valid_attr)
            norm_attr = valid_attr / (abs_attr.max() + 1e-10)
            
            # Create heatmap
            sns.heatmap(
                norm_attr.reshape(1, -1), 
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                center=0,
                annot=np.array(valid_tokens).reshape(1, -1),
                fmt='',
                cbar=True,
                ax=axes[i]
            )
            
            axes[i].set_title(f"{head_name} Attribution")
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
        
        plt.tight_layout()
        plt.savefig(f"explanation_{text[:20].replace(' ', '_')}.png")
        plt.show()
    
    def compare_identity_explanations(self, text, identity_groups, target=1, head_idx=None):
        """
        Compare explanations across different identity groups
        
        Args:
            text: Text to explain
            identity_groups: Dictionary mapping group names to identity feature tensors
            target: Target class (0=non-toxic, 1=toxic)
            head_idx: Head index for DivDis model (None for all heads)
        """
        all_attributions = {}
        tokens = None
        
        for group_name, identity_features in identity_groups.items():
            attributions, tokens = self.explain_text(
                text, identity_features, target, head_idx, visualize=False
            )
            all_attributions[group_name] = attributions
        
        # Visualize comparison
        self._visualize_identity_comparison(tokens, all_attributions, text)
    
    def _visualize_identity_comparison(self, tokens, all_attributions, text):
        """Visualize attribution comparison across identity groups"""
        num_groups = len(all_attributions)
        num_heads = len(next(iter(all_attributions.values())))
        
        fig, axes = plt.subplots(num_groups, num_heads, figsize=(5 * num_heads, 3 * num_groups))
        
        if num_groups == 1 and num_heads == 1:
            axes = [[axes]]
        elif num_groups == 1:
            axes = [axes]
        elif num_heads == 1:
            axes = [[ax] for ax in axes]
        
        for i, (group_name, attributions) in enumerate(all_attributions.items()):
            for j, (head_name, attr) in enumerate(attributions.items()):
                # Sum attributions across embedding dimension
                attr_sum = attr.sum(dim=-1).squeeze(0)
                attr_sum = attr_sum.cpu().detach().numpy()
                
                # Filter out padding tokens
                valid_tokens = []
                valid_attr = []
                for token, a in zip(tokens, attr_sum):
                    if token not in ['[PAD]', '[CLS]', '[SEP]']:
                        valid_tokens.append(token)
                        valid_attr.append(a)
                
                # Normalize attributions
                valid_attr = np.array(valid_attr)
                abs_attr = np.abs(valid_attr)
                norm_attr = valid_attr / (abs_attr.max() + 1e-10)
                
                # Create heatmap
                sns.heatmap(
                    norm_attr.reshape(1, -1), 
                    cmap='RdBu_r',
                    vmin=-1, vmax=1,
                    center=0,
                    annot=np.array(valid_tokens).reshape(1, -1),
                    fmt='',
                    cbar=True,
                    ax=axes[i][j]
                )
                
                axes[i][j].set_title(f"{group_name} - {head_name}")
                axes[i][j].set_xticklabels([])
                axes[i][j].set_yticklabels([])
        
        plt.tight_layout()
        plt.savefig(f"identity_comparison_{text[:20].replace(' ', '_')}.png")
        plt.show()
    
    def analyze_identity_bias(self, texts, identity_groups, target=1):
        """
        Analyze how attributions differ across identity groups for multiple texts
        
        Args:
            texts: List of texts to analyze
            identity_groups: Dictionary mapping group names to identity feature tensors
            target: Target class (0=non-toxic, 1=toxic)
            
        Returns:
            DataFrame with attribution differences
        """
        results = []
        
        for text in texts:
            text_results = {'text': text}
            all_attributions = {}
            
            for group_name, identity_features in identity_groups.items():
                attributions, tokens = self.explain_text(
                    text, identity_features, target, visualize=False
                )
                all_attributions[group_name] = attributions
            
            # Calculate attribution differences
            for head_name in next(iter(all_attributions.values())).keys():
                for group1 in identity_groups.keys():
                    for group2 in identity_groups.keys():
                        if group1 < group2:  # Avoid duplicates
                            attr1 = all_attributions[group1][head_name].sum(dim=-1).squeeze(0)
                            attr2 = all_attributions[group2][head_name].sum(dim=-1).squeeze(0)
                            
                            # Calculate difference metrics
                            diff = (attr1 - attr2).abs().mean().item()
                            text_results[f'{head_name}_{group1}_vs_{group2}_diff'] = diff
            
            results.append(text_results)
        
        return pd.DataFrame(results) 