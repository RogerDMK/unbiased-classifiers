import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from divdis import DivDisLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch.nn as nn
from tqdm import tqdm, trange
import time
import copy
import signal
import platform

def train_model(model, train_loader, val_loader, criterion, num_epochs=20, learning_rate=0.001, write_file='base_classifier_log.txt'):
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
        with open(write_file, 'a') as f:
            f.write("Epoch: {}\n".format(epoch))
            f.write("Train_loss: {}\n".format(train_loss))
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
        with open(write_file, 'a') as f:
            f.write(str(confusion_matrix(all_targets,all_predicted)))
            scheduler.step(val_f1)
            precision = precision_score(all_targets, all_predicted, average='macro')
            recall = recall_score(all_targets, all_predicted, average='macro')
            
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            
            f.write(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}\n')
            f.write(f'Precision: {precision:.4f} | Recall: {recall:.4f}\n')
        
        torch.save(model.state_dict(), 'best_recidivism_model.pt')
        
    
    model.load_state_dict(torch.load('best_recidivism_model.pt'))
    return model, history

def trainDivDis( model, epochs, train_loader, diverse_loader, val_loader, criterion, start_offset=0, learning_rate=0.001, write_file='divdis_classifier_log.txt'):
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
        with open(write_file, 'a') as f:
            f.write("Epoch {}\n".format(epoch))
        
            for idx in range(model.num_heads):
                f.write("head {:.4f}: labelled loss = {:.4f} labelled accuracy = {:.4f}\n".format(idx, class_loss_tracker[idx], class_acc_tracker[idx]))
        
            f.write("div loss = {}\n".format(div_loss_tracker/epochs))
            f.write("ensemble loss = {}\n".format(total_loss_tracker/epochs))
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
        with open(write_file, 'a') as f:
            for idx in range(model.num_heads):
                f.write("Stats for head: {} \n".format(idx))
                f.write(str(confusion_matrix(all_targets,all_predicted[idx])))
                val_f1 = f1_score(all_targets, all_predicted[idx], average='macro')
                f.write('F1 score: {}\n'.format(val_f1))
                precision = precision_score(all_targets, all_predicted[idx], average='macro')
                recall = recall_score(all_targets, all_predicted[idx], average='macro')
                f.write(f'Precision: {precision:.4f} | Recall: {recall:.4f}\n')
                f.write("__________________________________________________\n")
            
            for idx in range(model.num_heads):
                f.write("Head {}: accuracy of test {}".format(idx, val_loss[idx]))
    return total_loss_tracker/epochs

   
def train_bert_baseline(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 1e-3,
    device = "cpu",
):
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
    device = "cpu",
):
    device = torch.device(device)
    model.to(device)

    divCriterion = DivDisLoss(model.num_heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = num_epochs * len(train_loader)
    warmup_steps  = int(0.1 * total_steps)

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
        print(f"ensemble loss : {total_loss_tracker/len(train_loader):.4f}")

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

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Training timed out")

def train_jigsaw_baseline(
    model,
    train_loader,
    val_loader,
    num_epochs=3,
    learning_rate=2e-5,
    weight_decay=1e-3,
    device="cpu",
    timeout_minutes=30
):
    # Set timeout (only on Unix systems)
    if platform.system() != 'Windows':
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)  # Convert minutes to seconds
    
    try:
        device = torch.device(device)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_f1 = -1.0
        
        # Create epoch progress bar
        epoch_bar = trange(num_epochs, desc="Epochs")
        
        for epoch in epoch_bar:
            model.train()
            train_loss = 0.0
            
            # Create batch progress bar
            batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            
            for input_ids, attention_mask, identity_features, labels in batch_bar:
                # Add timing information
                start_time = time.time()
                
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                identity_features = identity_features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, identity_features)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batch_time = time.time() - start_time
                batch_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "batch_time": f"{batch_time:.2f}s"
                })
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            # Add a progress bar for validation
            val_bar = tqdm(val_loader, desc="Validation", leave=False)

            with torch.no_grad():
                for input_ids, attention_mask, identity_features, labels in val_bar:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    identity_features = identity_features.to(device)
                    labels = labels.to(device)
                    
                    logits = model(input_ids, attention_mask, identity_features)
                    
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    _, preds = torch.max(logits, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_bar.set_postfix({
                        "loss": f"{loss.item():.4f}"
                    })
            
            # Calculate validation metrics
            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(all_labels, all_preds)
            
            history["val_loss"].append(avg_val_loss)
            history["val_f1"].append(val_f1)
            
            # Update progress bar with metrics
            epoch_bar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
                "val_f1": f"{val_f1:.4f}"
            })
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = copy.deepcopy(model)
        
        return best_model, history
    
    except TimeoutException:
        print("WARNING: Training timed out. Returning current model state.")
        return model, {"train_loss": [], "val_loss": [], "val_f1": []}
    
    finally:
        # Disable the alarm (only on Unix systems)
        if platform.system() != 'Windows':
            signal.alarm(0)

def train_jigsaw_divdis(
    model,
    labeled_loader,
    unlabeled_loader,
    val_loader=None,  # Make validation optional
    num_epochs=3,
    learning_rate=2e-5,
    weight_decay=1e-3,
    device="cpu",
    timeout_minutes=30,
    validate_every_epoch=False  # Add flag to control validation
):
    # Set timeout (only on Unix systems)
    if platform.system() != 'Windows':
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)  # Convert minutes to seconds
    
    try:
        device = torch.device(device)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Create epoch progress bar
        epoch_bar = trange(num_epochs, desc="Epochs")
        
        for epoch in epoch_bar:
            model.train()
            train_loss = 0.0
            divdis_loss = 0.0
            
            # Phase 1: Train on labeled data with task loss
            labeled_bar = tqdm(labeled_loader, desc=f"Labeled Training", leave=False)
            
            for input_ids, attention_mask, identity_features, labels in labeled_bar:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                identity_features = identity_features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, identity_features)
                
                # Calculate task loss for each head
                task_loss = 0
                for h in range(model.num_heads):
                    head_logits = logits[:, h, :]
                    task_loss += criterion(head_logits, labels)
                
                # Average task loss across heads
                task_loss = task_loss / model.num_heads
                
                # Backward pass with task loss only
                task_loss.backward()
                optimizer.step()
                
                train_loss += task_loss.item()
                labeled_bar.set_postfix({
                    "task_loss": f"{task_loss.item():.4f}"
                })
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(labeled_loader)
            
            # Phase 2: Train on unlabeled data with diversity loss only
            unlabeled_bar = tqdm(unlabeled_loader, desc=f"Unlabeled Training", leave=False)
            
            for input_ids, attention_mask, identity_features, _ in unlabeled_bar:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                identity_features = identity_features.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, identity_features)
                
                # Calculate diversity loss
                diversity_loss = calculate_diversity_loss(logits)
                
                # Backward pass with negative diversity loss (we want to maximize diversity)
                loss = -model.diversity_weight * diversity_loss
                loss.backward()
                optimizer.step()
                
                divdis_loss += diversity_loss.item()
                unlabeled_bar.set_postfix({
                    "div_loss": f"{diversity_loss.item():.4f}"
                })
            
            # Calculate average DivDis loss
            avg_divdis_loss = divdis_loss / len(unlabeled_loader)
            
            # Update progress bar with training metrics only
            epoch_bar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}", 
                "divdis_loss": f"{avg_divdis_loss:.4f}"
            })
            
            # Only validate if requested and val_loader is provided
            if validate_every_epoch and val_loader is not None:
                # Validation code here (existing validation code)
                pass  # Add this placeholder since validation code is commented out

            # Optional: Do a final validation after all epochs
            if val_loader is not None and not validate_every_epoch:
                print("Performing final validation...")
                model.eval()
                val_loss = 0.0
                
                # Track metrics for each head
                head_losses = [0.0] * model.num_heads
                head_correct = [0] * model.num_heads
                all_predicted = [[] for _ in range(model.num_heads)]
                all_targets = []
                
                # Track ensemble metrics
                ensemble_correct = 0
                ensemble_total = 0
                ensemble_preds_list = []
                
                with torch.no_grad():
                    for input_ids, attention_mask, identity_features, labels in tqdm(val_loader, desc="Validation"):
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        identity_features = identity_features.to(device)
                        labels = labels.to(device)
                        
                        logits = model(input_ids, attention_mask, identity_features)
                        
                        # Evaluate each head separately
                        batch_size = labels.size(0)
                        for h in range(model.num_heads):
                            head_logits = logits[:, h, :]
                            head_loss = criterion(head_logits, labels)
                            head_losses[h] += head_loss.item()
                            
                            _, head_preds = torch.max(head_logits, 1)
                            head_correct[h] += (head_preds == labels).sum().item()
                            all_predicted[h].extend(head_preds.cpu().numpy())
                        
                        # Store targets for later metric calculation
                        all_targets.extend(labels.cpu().numpy())
                        
                        # Average predictions across heads for ensemble evaluation
                        avg_logits = logits.mean(dim=1)
                        ensemble_loss = criterion(avg_logits, labels)
                        val_loss += ensemble_loss.item()
                        
                        _, ensemble_preds = torch.max(avg_logits, 1)
                        ensemble_preds_list.extend(ensemble_preds.cpu().numpy())
                        ensemble_total += batch_size
                        ensemble_correct += (ensemble_preds == labels).sum().item()
                
                # Calculate metrics for ensemble
                avg_val_loss = val_loss / len(val_loader)
                ensemble_acc = ensemble_correct / ensemble_total
                ensemble_preds_array = np.array(ensemble_preds_list)
                ensemble_f1 = f1_score(all_targets, ensemble_preds_array, average='macro')
                
                print(f"Final validation - Loss: {avg_val_loss:.4f}, Acc: {ensemble_acc:.4f}, F1: {ensemble_f1:.4f}")
                
                # Print head metrics
                for h in range(model.num_heads):
                    head_acc = head_correct[h] / ensemble_total
                    head_avg_loss = head_losses[h] / len(val_loader)
                    head_f1 = f1_score(all_targets, all_predicted[h], average='macro')
                    print(f"Head {h} - Loss: {head_avg_loss:.4f}, Acc: {head_acc:.4f}, F1: {head_f1:.4f}")
            
        return model
    
    except TimeoutException:
        print("WARNING: Training timed out. Returning current model state.")
        return model
    
    finally:
        # Disable the alarm (only on Unix systems)
        if platform.system() != 'Windows':
            signal.alarm(0)

def calculate_diversity_loss(logits):
    """
    Calculate diversity loss for DivDis
    
    Args:
        logits: Model logits of shape [batch_size, num_heads, num_classes]
    
    Returns:
        diversity_loss: Diversity loss
    """
    batch_size, num_heads, num_classes = logits.size()
    
    # Calculate pairwise KL divergence between heads
    diversity_loss = 0.0
    count = 0
    
    for i in range(num_heads):
        for j in range(i+1, num_heads):
            # KL divergence: KL(p_i || p_j) + KL(p_j || p_i)
            kl_ij = F.kl_div(F.log_softmax(logits[:, i, :], dim=1), 
                             F.softmax(logits[:, j, :], dim=1), 
                             reduction='batchmean')
            kl_ji = F.kl_div(F.log_softmax(logits[:, j, :], dim=1), 
                             F.softmax(logits[:, i, :], dim=1), 
                             reduction='batchmean')
            
            diversity_loss += kl_ij + kl_ji
            count += 2
    
    # Average over all pairs
    if count > 0:
        diversity_loss = diversity_loss / count
    
    return diversity_loss
