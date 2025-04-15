import os
import random
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate_func(x):
    smiles_embed, protein_embed, smiles_mask, protein_mask, labels = zip(*x)

    protein_embeddings = torch.cat(protein_embed, dim=0)
    smiles_embeddings = torch.cat(smiles_embed, dim=0)
    protein_masks = torch.cat(protein_mask, dim=0)
    smiles_masks = torch.cat(smiles_mask, dim=0)

    return [smiles_embeddings, protein_embeddings], [smiles_masks, protein_masks], torch.tensor(labels)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
        

def evaluate(model, dev_loader, device):
    y_labels = []
    y_preds = []
    with torch.no_grad():
        model.eval()
        for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
            inputs, masks, labels = [item.to(device) for item in inputs], [item.to(device) for item in masks], labels.type(torch.LongTensor).to(device)

            # rationales -- (batch_size, seq_length, 2)
            cls_logits = model.train_one_step(inputs, masks)

            # soft_pred = F.softmax(cls_logits, -1)
            # _, pred = torch.max(soft_pred, dim=-1)
            pred = F.softmax(cls_logits, dim=1)[:, 1]
            
            y_labels += labels.cpu().numpy().tolist()
            y_preds += pred.cpu().numpy().tolist()

    # cls
    # auroc = roc_auc_score(np.array(y_labels, dtype=np.float32), np.array(y_preds, dtype=np.float32))
    auroc = roc_auc_score(y_labels, y_preds)
    auprc = average_precision_score(y_labels, y_preds)
    
    fpr, tpr, thresholds = roc_curve(y_labels, y_preds)
    prec, recall, _ = precision_recall_curve(y_labels, y_preds)
    precision = tpr / (tpr + fpr + 0.00001)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    y_preds_s = [1 if i else 0 for i in (y_preds >= thred_optim)]
    cm1 = confusion_matrix(y_labels, y_preds_s)
    accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
    precision = cm1[1, 1] / (cm1[1, 1] + cm1[0, 1])
    recall = cm1[1, 1] / (cm1[1, 1] + cm1[1, 0])
    sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    precision1 = precision_score(y_labels, y_preds_s)
    
    f1 = np.max(f1[5:])
    return auroc.item(), auprc.item(), precision1, recall, f1.item(), accuracy.item(), sensitivity.item(), specificity.item()