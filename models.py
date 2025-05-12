import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, RobertaForSequenceClassification

class BERT_models_baseline(nn.Module):
    def __init__(self, pretrain_model : str =  "bert-base-uncased",
        mlp_dropout: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(16, num_classes)
        )

    def forward(self, input_ids, attn_masks=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=token_type_ids, return_dict=True)
        pooler_outputs = outputs.pooler_output
        logits = self.classifier(pooler_outputs)
        return logits

class BERT_models_DivDis(nn.Module):
    def __init__(
        self, pretrain_model : str =  "bert-base-uncased", 
        num_heads: int = 3,
        mlp_dropout: float = 0.1,
        num_classes: int = 3,
        diversity_weight: float = 1e-3,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.divdis = DivDisClassifier(
            input_dim = hidden,
            dropout_rate=mlp_dropout,
            num_heads=num_heads,
            num_classes=num_classes,
            diversity_weight= diversity_weight,
        )
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.diversity_weight = diversity_weight

    def forward(self, input_ids, attn_masks=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=token_type_ids, return_dict=True)
        pooler_outputs = outputs.pooler_output
        logits = self.divdis(pooler_outputs)
        return logits

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection


class ResidualMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.1, num_classes=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.num_classes = num_classes
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        logits = self.head(x)
        return logits

class BaseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.0, num_classes=3):
        super(BaseClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        self.shared_layers = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.recid_head = nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x):
        features = self.shared_layers(x)
        logits = self.recid_head(features)
        # output = torch.softmax(logits, 1)
        return logits
    
class DivDisClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2, num_heads = 3, num_classes = 2, diversity_weight = 0.001):
        super().__init__()

        self.model = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)
            ) for _ in range(num_heads)])
        
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.diversity_weight = diversity_weight
    
    def forward(self, x):
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        for idx in range(self.num_heads):
            y = self.model[idx](x)
            pred[:,idx,:] = y
        return pred
    
class DivDisResidualClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.1, num_classes=3, num_heads=3, diversity_weight = 0.001):
        super().__init__()
        self.input_proj = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        ) for _ in range(num_heads)])
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ]) for _ in range(num_heads)])
        self.diversity_weight = diversity_weight
        self.final_layer = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_heads)])
    
    def forward(self, x):
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        for idx in range(self.num_heads):
            y = self.input_proj[idx](x)
            blockout = self.blocks[idx](y)
            pred[:,idx,:] = self.final_layer[idx](blockout)
        return pred
    
class ModelWrapper(nn.Module):
    def __init__(self, model, target_class):
        super().__init__()
        self.model = model
        self.target_class = target_class

    def forward(self, x):
        output = self.model(x)
        return output[:, self.target_class]
    
class HeadWrapper(nn.Module):
    def __init__(self, divdis_model, head_index, apply_softmax=False):
        super().__init__()
        self.divdis_model = divdis_model
        self.head_index = head_index
        self.input_proj = divdis_model.input_proj[head_index]
        self.block = divdis_model.blocks[head_index]
        self.final_layer = divdis_model.final_layer[head_index]
        self.apply_softmax = apply_softmax
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        inp = self.input_proj(x)
        blk = self.block(inp)
        out = self.final_layer(blk)
        if self.apply_softmax:
            out = self.softmax(out)
        return out

class JigsawToxicityClassifier(nn.Module):
    def __init__(self, pretrain_model="bert-base-uncased", num_identity_features=0, mlp_dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        
        # If we want to incorporate identity features
        self.use_identity = num_identity_features > 0
        if self.use_identity:
            self.classifier = nn.Sequential(
                nn.Linear(hidden + num_identity_features, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(64, 16),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(16, 2)  # Binary classification for toxicity
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(64, 16),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(16, 2)  # Binary classification for toxicity
            )
        
    def forward(self, input_ids, attention_mask, identity_features=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.pooler_output
        
        if self.use_identity and identity_features is not None:
            # Concatenate with identity features
            combined = torch.cat([pooled_output, identity_features], dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(pooled_output)
            
        return logits

class JigsawDivDisClassifier(nn.Module):
    def __init__(self, pretrain_model="bert-base-uncased", num_identity_features=0, 
                 num_heads=3, mlp_dropout=0.1, diversity_weight=1e-3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.num_heads = num_heads
        self.diversity_weight = diversity_weight
        
        # If we want to incorporate identity features
        self.use_identity = num_identity_features > 0
        if self.use_identity:
            input_dim = hidden + num_identity_features
        else:
            input_dim = hidden
            
        # Create multiple classification heads
        self.model = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(16, 2)  # Binary classification for toxicity
        ) for _ in range(num_heads)])
        
    def forward(self, input_ids, attention_mask, identity_features=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.pooler_output
        
        if self.use_identity and identity_features is not None:
            # Concatenate with identity features
            combined = torch.cat([pooled_output, identity_features], dim=1)
            input_features = combined
        else:
            input_features = pooled_output
        
        # Get predictions from each head
        pred = torch.zeros(input_features.shape[0], self.num_heads, 2).to(input_features.device)
        for idx in range(self.num_heads):
            y = self.model[idx](input_features)
            pred[:,idx,:] = y
                
        # Return shape: [batch_size, num_heads, num_classes]
        return pred