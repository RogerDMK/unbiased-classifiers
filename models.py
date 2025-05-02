import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class BERT_models_baseline(nn.Module):
    def __init__(self, pretrain_model : str =  "bert-base-uncased",
        mlp_dropout: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_model)
        hidden = self.bert.config.hidden_size
        self.classifier = BaseClassifier(
            input_dim = hidden,
            dropout_rate=mlp_dropout,
            num_classes=num_classes
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
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4, num_classes)
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
        self.head_model = divdis_model.model[head_index]
        self.apply_softmax = apply_softmax
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        out = self.head_model(x)
        if self.apply_softmax:
            out = self.softmax(out)
        return out