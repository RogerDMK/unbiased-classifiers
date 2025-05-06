import numpy as np
import torch
import torch.optim as optim
from divdis import DivDisLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
    

def train_model(model, train_loader, val_loader, criterion, num_epochs=20, learning_rate=0.001):
    device = torch.device("cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    history = {
        'val_f1': [],
        'val_loss': [],
        'train_loss': []
    }
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            targets = targets.long()
            outputs = model(features)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                targets = targets.long()
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        
        val_f1 = f1_score(all_targets, all_predicted, average='macro')
        print(confusion_matrix(all_targets,all_predicted))
        scheduler.step(val_f1)
        precision = precision_score(all_targets, all_predicted, average='macro')
        recall = recall_score(all_targets, all_predicted, average='macro')
        
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}')
        print(f'Precision: {precision:.4f} | Recall: {recall:.4f}')
        
        torch.save(model.state_dict(), 'best_recidivism_model.pt')
        
    
    model.load_state_dict(torch.load('best_recidivism_model.pt'))
    return model, history

def trainDivDis( model, epochs, train_loader, diverse_loader, val_loader, criterion, start_offset=0, learning_rate=0.001):
    divCriterion = DivDisLoss(model.num_heads)
    device = torch.device("cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    assert(len(train_loader) == len(diverse_loader))
    
    for epoch in range(start_offset, start_offset+epochs):
        
        
        class_loss_tracker = np.zeros(model.num_heads)
        class_acc_tracker = np.zeros(model.num_heads)
        div_loss_tracker = 0
        total_loss_tracker = 0
        labelled = iter(train_loader)
        unlabelled = iter(diverse_loader)
        
        for batch in range(len(train_loader)):
            labelled_loss = 0
            x, y = next(labelled)
            x = x.to(device)
            y = y.to(device)
            y = y.long()
            pred_y = model(x)
            for idx in range(model.num_heads):
                class_loss = criterion(torch.squeeze(pred_y[ :,idx,:]), torch.squeeze(y))
                class_loss_tracker[idx] += class_loss.item()
                pred_class = torch.argmax(pred_y[:,idx,:],dim=1).detach()
                class_acc_tracker[idx] += (torch.sum(pred_class==y).item())/len(y)
                labelled_loss += class_loss
            
            labelled_loss /= model.num_heads
            un_x, _ = next(unlabelled)
            un_x = un_x.to(device)
            unlabelled_pred = model(un_x)
            batch_size = unlabelled_pred.shape[0]
            unlabelled_pred = torch.reshape(unlabelled_pred, (batch_size, model.num_heads * model.num_classes))
            div_loss = divCriterion(unlabelled_pred)     
            div_loss_tracker += div_loss.item()
            
            objective = labelled_loss + model.diversity_weight*div_loss
            total_loss_tracker += objective.item()
            
            objective.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print("Epoch {}".format(epoch))
        for idx in range(model.num_heads):
            print("head {:.4f}: labelled loss = {:.4f} labelled accuracy = {:.4f}".format(idx, class_loss_tracker[idx], class_acc_tracker[idx]))
        
        print("div loss = {}".format(div_loss_tracker/epochs))
        print("ensemble loss = {}".format(total_loss_tracker/epochs))
    all_targets = []
    all_predicted = [[] for _ in range(model.num_heads)]
    with torch.no_grad():
        val_loss = np.zeros(model.num_heads)
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            targets = targets.long()
            pred_y = model(features)
            for idx in range(model.num_heads):
                class_loss = criterion(torch.squeeze(pred_y[ :,idx,:]), torch.squeeze(targets))
                class_loss_tracker[idx] += class_loss.item()
                pred_class = torch.argmax(pred_y[:,idx,:],dim=1).detach()
                val_loss[idx] += (torch.sum(pred_class==targets).item())/len(targets)
                labelled_loss += class_loss
                all_predicted[idx].extend(pred_class.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        for idx in range(model.num_heads):
            print("Stats for head:", idx)
            print(confusion_matrix(all_targets,all_predicted[idx]))
            val_f1 = f1_score(all_targets, all_predicted[idx], average='macro')
            print('F1 score:', val_f1)
            precision = precision_score(all_targets, all_predicted[idx], average='macro')
            recall = recall_score(all_targets, all_predicted[idx], average='macro')
            print(f'Precision: {precision:.4f} | Recall: {recall:.4f}')
            print("__________________________________________________")
        
        for idx in range(model.num_heads):
            print("Head {}: accuracy of test {}".format(idx, val_loss[idx]))
    return total_loss_tracker/epochs

   
def train_bert_baseline(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 1e-3,
    device = None,
):
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    warmup_steps  = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = -1.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=ids, attn_masks=mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * labels.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss, all_targets, all_predicts = 0.0, [], []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                logits = model(input_ids=ids, attn_masks=mask)
                loss   = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                all_targets.extend(logits.argmax(1).cpu().numpy())
                all_predicts.extend(labels.cpu().numpy())


        val_loss = val_loss / len(val_loader.dataset)
        val_f1   = f1_score(all_targets, all_predicts, average="macro")
        prec     = precision_score(all_targets, all_predicts, average="macro")
        rec      = recall_score(all_targets, all_predicts, average="macro")

        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(confusion_matrix(all_targets, all_predicts))
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f}")
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            
    print("Best macro-F1:", best_f1)
    return model, history

def train_bert_DivDis(
    model,
    train_loader,
    diverse_loader,
    val_loader,
    criterion,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 1e-3,
    device = None,
):
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device)
    model.to(device)

    divCriterion = DivDisLoss(model.num_heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = num_epochs * len(train_loader)
    warmup_steps  = int(0.1 * total_steps)  # first 10% warm-up

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    assert len(train_loader) == len(diverse_loader) , "Loaders must be equal length"

    for epoch in range(num_epochs):
        model.train()
        class_loss_tracker = np.zeros(model.num_heads)
        class_acc_tracker  = np.zeros(model.num_heads)
        div_loss_tracker   = 0.0
        total_loss_tracker = 0.0

        labelled_iter   = iter(train_loader)
        unlabelled_iter = iter(diverse_loader)

        for _ in range(len(train_loader)):
            ids, mask, labels = next(labelled_iter)
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(input_ids=ids, attn_masks=mask)
            labelled_loss = 0.0    
            for h in range(model.num_heads):
                cls_loss   = criterion(logits[:, h, :], labels)
                class_loss_tracker[h] += cls_loss.item()
                labelled_loss        += cls_loss
                pred_class = logits[:, h, :].argmax(dim=1)
                class_acc_tracker[h] += (pred_class == labels).float().mean().item()

            labelled_loss /= model.num_heads

            un_ids, un_mask = next(unlabelled_iter)
            un_ids, un_mask = un_ids.to(device), un_mask.to(device)

            un_preds = model(input_ids=un_ids, attn_masks=un_mask)
            B = un_preds.size(0)
            un_preds = un_preds.reshape(B, -1) 

            div_loss = divCriterion(un_preds)
            div_loss_tracker += div_loss.item()

            total_loss = labelled_loss + model.diversity_weight * div_loss
            total_loss_tracker += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch}")
        for h in range(model.num_heads):
            acc = class_acc_tracker[h] / len(train_loader)
            print(f"head {h}: loss {class_loss_tracker[h]:.4f} | acc {acc:.4f}")
        print(f"div loss : {div_loss_tracker/len(train_loader):.4f}")
        print(f" ensemble loss : {total_loss_tracker/len(train_loader):.4f}")

    model.eval()
    all_targets    = []
    all_predicted  = [[] for _ in range(model.num_heads)]
    val_acc        = np.zeros(model.num_heads)
    with torch.no_grad():
        for ids, mask, labels in val_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(input_ids=ids, attn_masks=mask)
            for h in range(model.num_heads):
                pred_cls = logits[:, h, :].argmax(dim=1)
                all_predicted[h].extend(pred_cls.cpu().numpy())
                val_acc[h] += (pred_cls == labels).float().sum().item()
            all_targets.extend(labels.cpu().numpy())


    for h in range(model.num_heads):
        print("\nStats for head", h)
        print(confusion_matrix(all_targets, all_predicted[h]))
        f1  = f1_score(all_targets, all_predicted[h], average='macro')
        pre = precision_score(all_targets, all_predicted[h], average='macro')
        rec = recall_score(all_targets, all_predicted[h], average='macro')
        acc = val_acc[h] / len(val_loader.dataset)
        print(f"F1 {f1:.4f} | Precision {pre:.4f} | Recall {rec:.4f} | Acc {acc:.4f}")

    return total_loss_tracker / len(train_loader)
