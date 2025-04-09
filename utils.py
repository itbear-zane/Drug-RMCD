import os
import random
import numpy as np
import torch
from validate_util import validate_dev_sentence


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
        
def evaluate(model, dev_loader):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    model.eval()
    device = "cuda:0"
    print("Validate")
    with torch.no_grad():
        for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
            # inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            inputs, masks, labels = [item.to(device) for item in inputs], [item.to(device) for item in masks], labels.to(device)
            _, logits = model(inputs, masks)
            # pdb.set_trace()
            logits = torch.softmax(logits, dim=-1)
            _, pred = torch.max(logits, axis=-1)
            # compute accuarcy
            # TP predict 和 label 同时为1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict 和 label 同时为0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()
        print(TP, TN, FN, FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (recall + precision)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        print("Validate Sentence")
        validate_dev_sentence(model, dev_loader, device,(writer,epoch))
        
        return precision, recall, f1_score, accuracy