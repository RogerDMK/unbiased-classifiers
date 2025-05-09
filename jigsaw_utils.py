import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score

def get_jigsaw_dfs(train_samples=5000, test_samples=1000):
    """
    Process Jigsaw dataset for toxicity classification with identity attributes
    
    Args:
        train_samples: Number of training samples to use
        test_samples: Number of test samples to use
        
    Returns:
        Dictionary containing dataframes for training and testing
    """
    # Load data
    df = pd.read_csv('jigsaw_toxicity_bias/all_data.csv')
    
    # Group identity columns by type
    gender_cols = ['male', 'female', 'transgender', 'other_gender']
    orientation_cols = ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation']
    religion_cols = ['christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion']
    race_cols = ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity']
    disability_cols = ['physical_disability', 'intellectual_or_learning_disability', 
                      'psychiatric_or_mental_illness', 'other_disability']
    
    # Fill NaN values in identity columns with 0
    identity_cols = (gender_cols + orientation_cols + religion_cols + 
                    race_cols + disability_cols)
    
    # Ensure all identity columns exist and are numeric
    for col in identity_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            # Convert to numeric, coercing errors to NaN, then fill NaNs with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Create binary toxicity label (threshold at 0.5)
    df['toxic'] = (df['toxicity'] >= 0.5).astype(int)
    
    # Select features for the model
    feature_cols = ['comment_text'] + identity_cols
    label_col = 'toxic'
    
    # Additional toxicity-related features
    toxicity_features = ['severe_toxicity', 'obscene', 'sexual_explicit', 
                        'identity_attack', 'insult', 'threat']
    feature_cols.extend(toxicity_features)
    
    # Create metadata columns for analysis
    df['has_identity'] = df[identity_cols].sum(axis=1) > 0
    df['identity_count'] = df[identity_cols].sum(axis=1)
    
    # Create aggregate identity categories
    df['has_gender'] = df[gender_cols].sum(axis=1) > 0
    df['has_orientation'] = df[orientation_cols].sum(axis=1) > 0
    df['has_religion'] = df[religion_cols].sum(axis=1) > 0
    df['has_race'] = df[race_cols].sum(axis=1) > 0
    df['has_disability'] = df[disability_cols].sum(axis=1) > 0
    
    # Split into train and test based on 'split' column
    df_train = df[df['split'] == 'train'].copy()
    df_test = df[df['split'] == 'test'].copy()
    
    # Reset indices
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    # Create subset with identity annotations for analysis
    df_train_identity = df_train[df_train['has_identity']].copy()
    df_test_identity = df_test[df_test['has_identity']].copy()
    
    # Basic statistics
    print(f"Total samples: {len(df)}")
    print(f"Training samples (before sampling): {len(df_train)}")
    print(f"Test samples (before sampling): {len(df_test)}")
    print(f"Training samples with identity: {len(df_train_identity)}")
    print(f"Test samples with identity: {len(df_test_identity)}")
    print(f"Toxic comments: {df['toxic'].mean()*100:.2f}%")
    
    # Identity group statistics
    print("\nIdentity group coverage:")
    for col in ['has_gender', 'has_orientation', 'has_religion', 'has_race', 'has_disability']:
        print(f"{col}: {df[col].mean()*100:.2f}%")
    
    # Use a smaller sample size for faster testing
    df_train = df_train.sample(train_samples, random_state=42)
    df_test = df_test.sample(test_samples, random_state=42)
    
    # Update identity subsets after sampling
    df_train_identity = df_train[df_train['has_identity']].copy()
    df_test_identity = df_test[df_test['has_identity']].copy()
    
    print(f"\nAfter sampling:")
    print(f"Training samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    print(f"Training samples with identity: {len(df_train_identity)}")
    print(f"Test samples with identity: {len(df_test_identity)}")
    
    return {
        'full_train': df_train,
        'full_test': df_test,
        'identity_train': df_train_identity,
        'identity_test': df_test_identity,
        'feature_columns': feature_cols,
        'identity_columns': identity_cols,
        'toxicity_columns': toxicity_features
    }

class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        """
        Initialize Jigsaw dataset
        
        Args:
            df: Pandas dataframe with processed data
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Get identity columns
        self.identity_cols = [col for col in df.columns if col in [
            'male', 'female', 'transgender', 'other_gender',
            'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 
            'other_sexual_orientation', 'christian', 'jewish', 'muslim',
            'hindu', 'buddhist', 'atheist', 'other_religion', 'black',
            'white', 'asian', 'latino', 'other_race_or_ethnicity',
            'physical_disability', 'intellectual_or_learning_disability',
            'psychiatric_or_mental_illness', 'other_disability'
        ]]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            row['comment_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Get identity features - ensure they're numeric
        identity_values = row[self.identity_cols].values
        # Convert to float and handle any non-numeric values
        identity_features = torch.tensor([float(val) if not pd.isna(val) else 0.0 for val in identity_values], dtype=torch.float32)
        
        # Get label
        label = torch.tensor(row['toxic'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'identity_features': identity_features,
            'labels': label
        }

def collate_jigsaw_batch(batch):
    """
    Custom collate function for Jigsaw dataset that handles dictionary outputs
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    identity_features = torch.stack([item['identity_features'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return input_ids, attention_mask, identity_features, labels

class JigsawDataLoader(DataLoader):
    """
    Custom DataLoader that uses the collate_jigsaw_batch function
    """
    def __init__(self, dataset, batch_size=32, shuffle=False, **kwargs):
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_jigsaw_batch,
            **kwargs
        )

def evaluate_by_identity(model, test_loader, identity_col, device="cpu"):
    """
    Evaluate model performance on a specific identity group
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        identity_col: Name of identity column to analyze
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for input_ids, attention_mask, identity_features, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            identity_features = identity_features.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            if hasattr(model, 'num_heads') and hasattr(model, 'model'):  # DivDis model
                logits = model(input_ids, attention_mask, identity_features)
                # Use average of all heads for evaluation
                logits = logits.mean(dim=1)
            else:  # Baseline model
                logits = model(input_ids, attention_mask, identity_features)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of toxic class
    
    # Calculate metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    # Calculate false positive and false negative rates
    fp = np.logical_and(np.array(all_preds) == 1, np.array(all_labels) == 0).sum()
    fn = np.logical_and(np.array(all_preds) == 0, np.array(all_labels) == 1).sum()
    tn = np.logical_and(np.array(all_preds) == 0, np.array(all_labels) == 0).sum()
    tp = np.logical_and(np.array(all_preds) == 1, np.array(all_labels) == 1).sum()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'fnr': fnr,
        'avg_toxic_prob': np.mean(all_probs)
    }

def perform_bias_analysis(model, data_dict, tokenizer, device="cpu", batch_size=32):
    """
    Analyze model bias across different identity groups
    
    Args:
        model: Trained model
        data_dict: Dictionary with dataframes
        tokenizer: Tokenizer for text processing
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    
    Returns:
        DataFrame with bias metrics
    """
    identity_cols = [
        'male', 'female', 'transgender', 'other_gender',
        'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 
        'other_sexual_orientation', 'christian', 'jewish', 'muslim',
        'hindu', 'buddhist', 'atheist', 'other_religion', 'black',
        'white', 'asian', 'latino', 'other_race_or_ethnicity',
        'physical_disability', 'intellectual_or_learning_disability',
        'psychiatric_or_mental_illness', 'other_disability'
    ]
    
    results = []
    
    # Get test data with identity annotations
    test_data = data_dict['identity_test']
    
    # Evaluate on each identity group
    for col in identity_cols:
        # Filter data for this identity
        identity_data = test_data[test_data[col] == 1].copy()
        if len(identity_data) < 10:  # Skip if too few samples
            continue
            
        # Create dataset and loader
        identity_dataset = JigsawDataset(identity_data, tokenizer)
        identity_loader = JigsawDataLoader(identity_dataset, batch_size=batch_size)
        
        # Evaluate model
        metrics = evaluate_by_identity(model, identity_loader, col, device)
        
        # Add to results
        results.append({
            'identity': col,
            'samples': len(identity_data),
            'toxic_rate': identity_data['toxic'].mean(),
            **metrics
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate disparate impact
    overall_metrics = evaluate_by_identity(
        model, 
        JigsawDataLoader(JigsawDataset(test_data, tokenizer), batch_size=batch_size),
        'overall',
        device
    )
    
    results_df['di_fpr'] = results_df['fpr'] / overall_metrics['fpr']
    results_df['di_fnr'] = results_df['fnr'] / overall_metrics['fnr']
    results_df['di_toxic_prob'] = results_df['avg_toxic_prob'] / overall_metrics['avg_toxic_prob']
    
    return results_df

def analyze_divdis_heads(model, data_dict, tokenizer, device="cpu", batch_size=32):
    """
    Analyze how different DivDis heads perform on different identity groups
    
    Args:
        model: Trained DivDis model
        data_dict: Dictionary with dataframes
        tokenizer: Tokenizer for text processing
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with head analysis
    """
    model.eval()
    
    # Identity groups to analyze
    identity_groups = {
        'gender': ['male', 'female', 'transgender'],
        'sexual_orientation': ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual'],
        'religion': ['christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist'],
        'race': ['black', 'white', 'asian', 'latino'],
        'disability': ['physical_disability', 'intellectual_or_learning_disability', 
                      'psychiatric_or_mental_illness']
    }
    
    results = {}
    
    # Get test data with identity annotations
    test_data = data_dict['identity_test']
    test_dataset = JigsawDataset(test_data, tokenizer)
    test_loader = JigsawDataLoader(test_dataset, batch_size=batch_size)
    
    # Evaluate each head on each identity group
    for group_name, cols in identity_groups.items():
        group_results = []
        
        for col in cols:
            # Filter data for this identity
            identity_data = test_data[test_data[col] == 1].copy()
            if len(identity_data) < 10:  # Skip if too few samples
                continue
                
            # Create dataset and loader
            identity_dataset = JigsawDataset(identity_data, tokenizer)
            identity_loader = JigsawDataLoader(identity_dataset, batch_size=batch_size)
            
            # Get predictions from each head
            head_metrics = []
            
            with torch.no_grad():
                for input_ids, attention_mask, identity_features, labels in identity_loader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    identity_features = identity_features.to(device)
                    labels = labels.to(device)
                    
                    # Get model predictions for all heads
                    logits = model(input_ids, attention_mask, identity_features)
                    
                    # Evaluate each head
                    for h in range(model.num_heads):
                        head_logits = logits[:, h, :]
                        head_probs = torch.softmax(head_logits, dim=1)
                        head_preds = torch.argmax(head_logits, dim=1)
                        
                        # Calculate metrics for this batch
                        acc = (head_preds == labels).float().mean().item()
                        toxic_prob = head_probs[:, 1].mean().item()
                        
                        # Add to head metrics
                        if len(head_metrics) <= h:
                            head_metrics.append({'acc': [], 'toxic_prob': []})
                        
                        head_metrics[h]['acc'].append(acc)
                        head_metrics[h]['toxic_prob'].append(toxic_prob)
            
            # Average metrics for each head
            for h in range(len(head_metrics)):
                head_metrics[h]['acc'] = np.mean(head_metrics[h]['acc'])
                head_metrics[h]['toxic_prob'] = np.mean(head_metrics[h]['toxic_prob'])
            
            # Add to group results
            group_results.append({
                'identity': col,
                'samples': len(identity_data),
                'toxic_rate': identity_data['toxic'].mean(),
                'head_metrics': head_metrics
            })
        
        results[group_name] = group_results
    
    return results

def evaluate_bias_metrics_per_head(model, data_dict, tokenizer, device="cpu", batch_size=32, p_value=-5):
    """
    Evaluate each head of the DivDis model separately on bias metrics
    """
    model.eval()
    
    # Get test data
    test_data = data_dict['full_test']
    
    # Identity columns to evaluate
    identity_cols = data_dict['identity_columns']
    
    # Create dataset and loader for full test set
    test_dataset = JigsawDataset(test_data, tokenizer)
    test_loader = JigsawDataLoader(test_dataset, batch_size=batch_size)
    
    # Check if model has multiple heads
    has_multiple_heads = hasattr(model, 'num_heads')
    num_heads = model.num_heads if has_multiple_heads else 1
    
    # Store predictions for each head and ensemble
    all_head_preds = [[] for _ in range(num_heads)] if has_multiple_heads else [[]]
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, identity_features, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            identity_features = identity_features.to(device)
            
            if has_multiple_heads:
                # For DivDis model, get predictions from each head
                logits = model(input_ids, attention_mask, identity_features)
                
                # Store predictions from each head
                for h in range(num_heads):
                    head_logits = logits[:, h, :]
                    head_probs = torch.softmax(head_logits, dim=1)[:, 1].cpu().numpy()
                    all_head_preds[h].extend(head_probs)
            else:
                # For baseline model, just get the predictions
                logits = model(input_ids, attention_mask, identity_features)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_head_preds[0].extend(probs)
            
            all_labels.extend(labels.cpu().numpy())
    
    # Create results for each head
    results = {}
    
    # Process each head (or just the baseline model)
    for h in range(num_heads):
        # Create DataFrame with predictions for this head
        results_df = pd.DataFrame({
            'toxic': all_labels,
            'pred': all_head_preds[h]
        })
        
        # Add identity columns
        test_data_reset = test_data.reset_index(drop=True)
        if len(results_df) == len(test_data_reset):
            for col in identity_cols:
                results_df[col] = test_data_reset[col].values
        
        # Calculate bias metrics for each identity
        bias_metrics = {}
        
        for identity in identity_cols:
            # Skip if too few examples
            if results_df[identity].sum() < 10:
                continue
            
            # Subgroup: examples mentioning the identity
            subgroup_df = results_df[results_df[identity] > 0]
            if len(subgroup_df) >= 10 and len(set(subgroup_df['toxic'])) > 1:
                subgroup_auc = roc_auc_score(subgroup_df['toxic'], subgroup_df['pred'])
            else:
                subgroup_auc = np.nan
            
            # BPSN: non-toxic examples mentioning the identity and toxic examples not mentioning it
            bpsn_df = pd.concat([
                results_df[(results_df[identity] > 0) & (results_df['toxic'] == 0)],  # non-toxic with identity
                results_df[(results_df[identity] == 0) & (results_df['toxic'] == 1)]  # toxic without identity
            ])
            if len(bpsn_df) >= 10 and len(set(bpsn_df['toxic'])) > 1:
                bpsn_auc = roc_auc_score(bpsn_df['toxic'], bpsn_df['pred'])
            else:
                bpsn_auc = np.nan
            
            # BNSP: toxic examples mentioning the identity and non-toxic examples not mentioning it
            bnsp_df = pd.concat([
                results_df[(results_df[identity] > 0) & (results_df['toxic'] == 1)],  # toxic with identity
                results_df[(results_df[identity] == 0) & (results_df['toxic'] == 0)]  # non-toxic without identity
            ])
            if len(bnsp_df) >= 10 and len(set(bnsp_df['toxic'])) > 1:
                bnsp_auc = roc_auc_score(bnsp_df['toxic'], bnsp_df['pred'])
            else:
                bnsp_auc = np.nan
            
            bias_metrics[identity] = {
                'subgroup_auc': subgroup_auc,
                'bpsn_auc': bpsn_auc,
                'bnsp_auc': bnsp_auc
            }
        
        # Calculate overall AUC
        overall_auc = roc_auc_score(results_df['toxic'], results_df['pred'])
        
        # Calculate generalized means for each submetric
        subgroup_values = [m['subgroup_auc'] for m in bias_metrics.values() if not np.isnan(m['subgroup_auc'])]
        bpsn_values = [m['bpsn_auc'] for m in bias_metrics.values() if not np.isnan(m['bpsn_auc'])]
        bnsp_values = [m['bnsp_auc'] for m in bias_metrics.values() if not np.isnan(m['bnsp_auc'])]
        
        # Power mean function
        def power_mean(values, p):
            if p == 0:
                return np.exp(np.mean(np.log(values)))
            else:
                return np.power(np.mean(np.power(values, p)), 1/p)
        
        # Calculate generalized means
        subgroup_mean = power_mean(subgroup_values, p_value) if subgroup_values else np.nan
        bpsn_mean = power_mean(bpsn_values, p_value) if bpsn_values else np.nan
        bnsp_mean = power_mean(bnsp_values, p_value) if bnsp_values else np.nan
        
        # Calculate final score
        bias_score = (subgroup_mean + bpsn_mean + bnsp_mean) / 3
        final_score = 0.25 * overall_auc + 0.25 * subgroup_mean + 0.25 * bpsn_mean + 0.25 * bnsp_mean
        
        # Prepare results
        summary = {
            'overall_auc': overall_auc,
            'subgroup_auc_mean': subgroup_mean,
            'bpsn_auc_mean': bpsn_mean,
            'bnsp_auc_mean': bnsp_mean,
            'bias_score': bias_score,
            'final_score': final_score
        }
        
        # Store results for this head
        head_name = f"head_{h}" if has_multiple_heads else "baseline"
        results[head_name] = {
            'identity_metrics': bias_metrics,
            'summary': summary
        }
    
    return results

def evaluate_bias_metrics(model, data_dict, tokenizer, device="cpu", batch_size=32, p_value=-5):
    """
    Evaluate model on bias metrics as defined in the Jigsaw competition
    """
    model.eval()
    
    # Get test data
    test_data = data_dict['full_test']
    
    # Identity columns to evaluate
    identity_cols = data_dict['identity_columns']
    
    # Create dataset and loader for full test set
    test_dataset = JigsawDataset(test_data, tokenizer)
    test_loader = JigsawDataLoader(test_dataset, batch_size=batch_size)
    
    # Get predictions for all test examples
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, identity_features, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            identity_features = identity_features.to(device)
            
            # For DivDis model, average predictions across heads
            if hasattr(model, 'num_heads'):
                logits = model(input_ids, attention_mask, identity_features)
                logits = logits.mean(dim=1)
            else:
                logits = model(input_ids, attention_mask, identity_features)
            
            # Get probabilities for toxic class
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    # Create DataFrame with predictions
    results_df = pd.DataFrame({
        'toxic': all_labels,
        'pred': all_preds
    })
    
    # Add identity columns
    test_data_reset = test_data.reset_index(drop=True)
    if len(results_df) == len(test_data_reset):
        for col in identity_cols:
            results_df[col] = test_data_reset[col].values
    
    # Calculate bias metrics for each identity
    bias_metrics = {}
    
    for identity in identity_cols:
        # Skip if too few examples
        if results_df[identity].sum() < 10:
            continue
        
        # Subgroup: examples mentioning the identity
        subgroup_df = results_df[results_df[identity] > 0]
        if len(subgroup_df) >= 10 and len(set(subgroup_df['toxic'])) > 1:
            subgroup_auc = roc_auc_score(subgroup_df['toxic'], subgroup_df['pred'])
        else:
            subgroup_auc = np.nan
        
        # BPSN: non-toxic examples mentioning the identity and toxic examples not mentioning it
        bpsn_df = pd.concat([
            results_df[(results_df[identity] > 0) & (results_df['toxic'] == 0)],  # non-toxic with identity
            results_df[(results_df[identity] == 0) & (results_df['toxic'] == 1)]  # toxic without identity
        ])
        if len(bpsn_df) >= 10 and len(set(bpsn_df['toxic'])) > 1:
            bpsn_auc = roc_auc_score(bpsn_df['toxic'], bpsn_df['pred'])
        else:
            bpsn_auc = np.nan
        
        # BNSP: toxic examples mentioning the identity and non-toxic examples not mentioning it
        bnsp_df = pd.concat([
            results_df[(results_df[identity] > 0) & (results_df['toxic'] == 1)],  # toxic with identity
            results_df[(results_df[identity] == 0) & (results_df['toxic'] == 0)]  # non-toxic without identity
        ])
        if len(bnsp_df) >= 10 and len(set(bnsp_df['toxic'])) > 1:
            bnsp_auc = roc_auc_score(bnsp_df['toxic'], bnsp_df['pred'])
        else:
            bnsp_auc = np.nan
        
        bias_metrics[identity] = {
            'subgroup_auc': subgroup_auc,
            'bpsn_auc': bpsn_auc,
            'bnsp_auc': bnsp_auc
        }
    
    # Calculate overall AUC
    overall_auc = roc_auc_score(results_df['toxic'], results_df['pred'])
    
    # Calculate generalized means for each submetric
    subgroup_values = [m['subgroup_auc'] for m in bias_metrics.values() if not np.isnan(m['subgroup_auc'])]
    bpsn_values = [m['bpsn_auc'] for m in bias_metrics.values() if not np.isnan(m['bpsn_auc'])]
    bnsp_values = [m['bnsp_auc'] for m in bias_metrics.values() if not np.isnan(m['bnsp_auc'])]
    
    # Power mean function
    def power_mean(values, p):
        if p == 0:
            return np.exp(np.mean(np.log(values)))
        else:
            return np.power(np.mean(np.power(values, p)), 1/p)
    
    # Calculate generalized means
    subgroup_mean = power_mean(subgroup_values, p_value) if subgroup_values else np.nan
    bpsn_mean = power_mean(bpsn_values, p_value) if bpsn_values else np.nan
    bnsp_mean = power_mean(bnsp_values, p_value) if bnsp_values else np.nan
    
    # Calculate final score
    bias_score = (subgroup_mean + bpsn_mean + bnsp_mean) / 3
    final_score = 0.25 * overall_auc + 0.25 * subgroup_mean + 0.25 * bpsn_mean + 0.25 * bnsp_mean
    
    # Prepare results
    summary = {
        'overall_auc': overall_auc,
        'subgroup_auc_mean': subgroup_mean,
        'bpsn_auc_mean': bpsn_mean,
        'bnsp_auc_mean': bnsp_mean,
        'bias_score': bias_score,
        'final_score': final_score
    }
    
    return {
        'identity_metrics': bias_metrics,
        'summary': summary
    }

def split_labeled_unlabeled(df, labeled_ratio=0.7, random_state=42):
    """
    Split dataframe into labeled and unlabeled sets
    
    Args:
        df: Pandas dataframe with data
        labeled_ratio: Ratio of data to use as labeled
        random_state: Random seed for reproducibility
        
    Returns:
        labeled_df, unlabeled_df: Split dataframes
    """
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df_shuffled) * labeled_ratio)
    
    # Split the data
    labeled_df = df_shuffled.iloc[:split_idx].copy()
    unlabeled_df = df_shuffled.iloc[split_idx:].copy()
    
    print(f"Split data into {len(labeled_df)} labeled and {len(unlabeled_df)} unlabeled examples")
    
    return labeled_df, unlabeled_df
