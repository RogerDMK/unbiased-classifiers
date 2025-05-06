
import torch, torch.nn as nn, numpy as np, pandas as pd
from contextlib import redirect_stdout
from transformers import AutoTokenizer     
from train import train_bert_baseline, train_bert_DivDis 
from models import BERT_models_baseline, BERT_models_DivDis
from utils import twitter_load, perform_bias_analysis, random_seed_arr, get_softmax, prepare, BERT_create_data_splits
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}, GPU count: {torch.cuda.device_count()}")

ds = pd.read_csv("hate_offensive_data.csv")
ds = ds.drop(columns=["count","hate_speech", "offensive_language","neither"])

ds = [ prepare(row) for row in ds.to_dict(orient="records") ]

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

seed_arr = random_seed_arr(seed = 12)

for seed in seed_arr:
    seed_path = f"seed_{seed}.txt"
    with open(seed_path, "w") as seed_file, redirect_stdout(seed_file):
        labelled_loader_DivDis, unlabeled_loader ,_, test_loader, labelled_loader= BERT_create_data_splits(ds, tok, seed=seed)


        #Base Model
        model = BERT_models_baseline(pretrain_model="bert-base-uncased",
            mlp_dropout=0.3,
            num_classes=3)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        best_model, _ = train_bert_baseline(
            model,
            train_loader=labelled_loader,
            val_loader=test_loader,
            num_epochs=3,
            device=device
        )

        aa_path = "AA_eval.csv"
        white_path = "White_eval.csv"
        aa_data, white_data = twitter_load(aa_path = "AA_eval.csv", white_path = "White_eval.csv", num_samples = 1000, seed = seed)

        aa_probs = get_softmax(best_model, aa_data,tok, device)
        white_probs = get_softmax(best_model, white_data,tok, device)


        aa_hate, aa_offensive = aa_probs[:,0].numpy(), aa_probs[:,1].numpy()
        white_hate, white_offensive = white_probs[:,0].numpy(), white_probs[:,1].numpy()
        perform_bias_analysis(aa_hate, white_hate, "Base: hate")
        perform_bias_analysis(aa_offensive, white_offensive, "Base: offensive")

        print("################(End of base model)################\n")
        #DivDis Model
        div_model = BERT_models_DivDis(
            pretrain_model="bert-base-uncased",
            num_heads=3,
            num_classes=3,
            mlp_dropout=0.2,
            diversity_weight=1e-3
        )

        criterion = nn.CrossEntropyLoss()
        if torch.cuda.device_count() > 1:
            div_model = nn.DataParallel(div_model)
        full_loss  = train_bert_DivDis(
            model = div_model,
            train_loader = labelled_loader_DivDis,
            diverse_loader = unlabeled_loader,
            val_loader = test_loader,
            criterion = criterion,
            num_epochs = 3,
            learning_rate = 2e-5,
            device=device
        )

        aa_path = "AA_eval.csv"
        white_path = "White_eval.csv"
        aa_data, white_data = twitter_load(aa_path = "AA_eval.csv", white_path = "White_eval.csv", num_samples = 1000, seed = seed)

        aa_probs = get_softmax(div_model, aa_data,tok, device)
        white_probs = get_softmax(div_model, white_data,tok, device)

        num_heads = aa_probs.shape[1]

        for h in range(num_heads):
            aa_hate, aa_offensive = aa_probs[:,h,0].numpy(), aa_probs[:,h,1].numpy()
            white_hate, white_offensive = white_probs[:,h,0].numpy(), white_probs[:,h,1].numpy()
            perform_bias_analysis(aa_hate, white_hate, "Head {}: hate".format(h))
            perform_bias_analysis(aa_offensive, white_offensive, "Head {}: offensive".format(h))
    print(f"output of seed:{seed} is saved to {seed_path}")