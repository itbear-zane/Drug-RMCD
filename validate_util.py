import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from metric import compute_micro_stats
import torch.nn.functional as F


def validate_dev_sentence(model, dev_loader, device,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    y_labels = []
    y_preds = []
    writer,epoch=writer_epoch
    for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
        inputs, masks, labels = [item.to(device) for item in inputs], [item.to(device) for item in masks], labels.type(torch.LongTensor).to(device)

        # rationales -- (batch_size, seq_length, 2)
        cls_logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)
        
        y_labels += labels.cpu().numpy().tolist()
        y_preds += pred.cpu().numpy().tolist()

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    auroc = roc_auc_score(y_labels, y_preds)
    auprc = average_precision_score(y_labels, y_preds)
    cm1 = confusion_matrix(y_labels, y_preds)
    sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

    print("validate_dev_sentence dataset : auroc:{:.4f}, auprc:{:.4f}, recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f} sensitivity:{:.4f} specificity:{:.4f}"\
        .format(auroc, auprc, recall, precision, f1_score, accuracy, sensitivity, specificity))
    if writer is not None or epoch is not None:
        writer.add_scalar('sent_acc',accuracy,epoch)
    return f1_score,accuracy
