
import html, re, torch, torch.nn as nn, numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset             
from torch.utils.data import DataLoader
from utils import collate_fn, collate_fn_unlab, twitter_load, perform_bias_analysis
from functools import partial
from models import BERT_models_baseline, BERT_models_DivDis
from train import train_bert_baseline, train_bert_DivDis

ds = load_dataset("tdavidson/hate_speech_offensive")["train"]
shuffled_ds = ds.shuffle(seed = 42)

def prepare(example):
# (i) rename the target column ------------------------------------------------
    example["labels"] = int(example["class"])       # 0 hate, 1 offensive, 2 neither

    # (ii) light tweet cleaning ---------------------------------------------------
    text = html.unescape(example["tweet"])
    text = re.sub(r"http\S+", "URL", text)          # keep URL token
    text = re.sub(r"@\w+", "@USER", text)           # normalise @mentions
    example["text"] = text

    return example
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def get_softmax(model, texts, device):
    model.eval()
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)           
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1).cpu()
    return probs

ds = ds.map(prepare, remove_columns=["count",
                                    "hate_speech_count",
                                    "offensive_language_count",
                                    "neither_count",
                                    "class",
                                    "tweet"])
total = len(ds)
n_labeled   = int(0.4 * total)
n_unlabled   = int(0.8 * total)
train_data_labeled = ds.select(range(0, n_unlabled))
train_data_labeled_DivDis = ds.select(range(0, n_labeled))
train_data_unlabeled = ds.select(range(n_labeled, n_unlabled))
test_data  = ds.select(range(n_unlabled, total))

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

labelled_loader = DataLoader(
    train_data_labeled,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn, tokenizer=tok)
)

labelled_loader_DivDis = DataLoader(
    train_data_labeled_DivDis,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn, tokenizer=tok)
)
unlabeled_loader = DataLoader(
    train_data_unlabeled,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn_unlab, tokenizer=tok)
)

test_loader  = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    collate_fn=partial(collate_fn, tokenizer=tok)
)

# model = BERT_models_baseline(pretrain_model="bert-base-uncased",
#     mlp_dropout=0.3,
#     num_classes=3)
   
# best_model, hist = train_bert_baseline(
#     model,
#     train_loader=labelled_loader,
#     val_loader=test_loader,   # or a separate val_loader
#     num_epochs=3
# )

# aa_path = "AA_eval.csv"
# white_path = "White_eval.csv"
# aa_data, white_data = twitter_load(aa_path = "AA_eval.csv", white_path = "White_eval.csv", num_samples = 1000, seed = 42)

# aa_probs = get_softmax(best_model, aa_data, device)
# white_probs = get_softmax(best_model, white_data, device)


# aa_hate, aa_offensive = aa_probs[:,0].numpy(), aa_probs[:,1].numpy()
# white_hate, white_offensive = white_probs[:,0].numpy(), white_probs[:,1].numpy()
# perform_bias_analysis(aa_hate, white_hate, "Base: hate")
# perform_bias_analysis(aa_offensive, white_offensive, "Base: offensive")



div_model = BERT_models_DivDis(
    pretrain_model="bert-base-uncased",
    num_heads=3,
    num_classes=3,
    mlp_dropout=0.2,
    diversity_weight=1e-3
)

criterion = nn.CrossEntropyLoss()

full_loss  = train_bert_DivDis(
    model = div_model,
    train_loader = labelled_loader_DivDis,
    diverse_loader = unlabeled_loader,
    val_loader = test_loader,
    criterion = criterion,
    num_epochs = 3,
    learning_rate = 2e-5
)
aa_path = "AA_eval.csv"
white_path = "White_eval.csv"
aa_data, white_data = twitter_load(aa_path = "AA_eval.csv", white_path = "White_eval.csv", num_samples = 1000, seed = 42)

aa_probs = get_softmax(div_model, aa_data, device)
white_probs = get_softmax(div_model, white_data, device)

num_heads = aa_probs.shape[1]

for h in range(num_heads):
    aa_hate, aa_offensive = aa_probs[:,h,0].numpy(), aa_probs[:,h,1].numpy()
    white_hate, white_offensive = white_probs[:,h,0].numpy(), white_probs[:,h,1].numpy()
    perform_bias_analysis(aa_hate, white_hate, "Head {}: hate".format(h))
    perform_bias_analysis(aa_offensive, white_offensive, "Head {}: offensive".format(h))

