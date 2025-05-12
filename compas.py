from utils import get_base_dfs, COMPASDataset, create_data_splits, explain_model, class_weights, random_seed_arr
from models import DivDisClassifier, BaseClassifier, ModelWrapper, HeadWrapper, ResidualMLPClassifier, DivDisResidualClassifier
from train import train_model, trainDivDis
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn as nn
import torch
from explainer import Explainer
import numpy as np

non_violent, violent = get_base_dfs()
train_columns = [
    "juv_fel_count", "juv_misd_count", "juv_other_count",
    "priors_count", "african-american", "caucasian", "hispanic",
    "other", "asian", "native-american", "less25", "greater45",
    "25to45", "felony", "misdemeanor", "two_years_r"
]
dataDf = violent[train_columns]
train_columns.pop()
all_seeds = [35, 12, 42]
for seed in all_seeds:
    data = COMPASDataset(dataDf, "two_years_r")
    print("Start of seed:", seed)
    base_text = "base_classifier_seed_{}.txt".format(seed)
    divdis_text = "divdis_classifier_seed_{}.txt".format(seed)
    trainData, unlabelData, _, testData, rawTrain, rawUnlabel, _, testRaw= create_data_splits(dataset=data, seed=seed)
    dataiter = iter(trainData)
    batch = next(dataiter)
    inputs, targets = batch
    drop_out = 0.3
    num_blocks = 1
    hidden_dim = 64
    num_classes = 2
    input_dim = int(inputs.shape[1])
    baseModel = ResidualMLPClassifier(
        input_dim=input_dim,  # number of input features
        hidden_dim=hidden_dim,            # you can tune this
        num_blocks=num_blocks,
        dropout_rate=drop_out,
        num_classes=num_classes              # adjust for your task
    )
    fullTrain = ConcatDataset([rawTrain, rawUnlabel])
    fullLoader = DataLoader(fullTrain, 64, True)
    weights = class_weights(fullTrain)
    criterion = nn.CrossEntropyLoss(weight=weights)
    baseModel, history = train_model(baseModel, fullLoader, testData, criterion, 3, 0.001, write_file=base_text)

    explain_model(baseModel, baseModel.num_classes, testRaw, 'Base seed: {}'.format(seed), input_dim, train_columns)
    divDisModel = DivDisResidualClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=drop_out,
        num_classes=num_classes,
        num_heads=3,
        diversity_weight=0.01
    )
    full_loss = trainDivDis(divDisModel, 15, trainData, unlabelData, testData, criterion, write_file=divdis_text)
    for head in range(divDisModel.num_heads):
        model_name = "Seed {} DivDis model head {}".format(seed, head)
        model_head = HeadWrapper(divDisModel, head)
        model_head.eval()
        explain_model(model_head, divDisModel.num_classes, testRaw, model_name, input_dim, train_columns)
    print("Seed finished")

