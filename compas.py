from utils import get_base_dfs, COMPASDataset, create_data_splits, explain_model, class_weights
from models import DivDisClassifier, BaseClassifier, ModelWrapper, HeadWrapper, ResidualMLPClassifier
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
data = COMPASDataset(dataDf, "two_years_r")
trainData, unlabelData, _, testData, rawTrain, rawUnlabel, _, testRaw= create_data_splits(data)
dataiter = iter(trainData)
batch = next(dataiter)
train_columns.pop()
inputs, targets = batch
input_dim = int(inputs.shape[1])
baseModel = ResidualMLPClassifier(
    input_dim=input_dim,  # number of input features
    hidden_dim=64,            # you can tune this
    num_blocks=1,
    dropout_rate=0.3,
    num_classes=2              # adjust for your task
)
fullTrain = ConcatDataset([rawTrain, rawUnlabel])
fullLoader = DataLoader(fullTrain, 64, True)
weights = class_weights(fullTrain)
criterion = nn.CrossEntropyLoss(weight=weights)
baseModel, history = train_model(baseModel, fullLoader, testData, criterion, 3, 0.001)

explain_model(baseModel, baseModel.num_classes, testRaw, 'Base', input_dim, train_columns)
# divDisModel = DivDisClassifier(input_dim)
# full_loss = trainDivDis(divDisModel, 15, trainData, unlabelData, testData, criterion)
# for head in range(divDisModel.num_heads):
#     model_name = "DivDis model head_ {}".format(head)
#     model_head = HeadWrapper(divDisModel, head)
#     model_head.eval()
#     explain_model(model_head, divDisModel.num_classes, testRaw, model_name, input_dim, train_columns)

