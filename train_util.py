import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import numpy as np
from tqdm import tqdm

from sklearn.metrics import  roc_auc_score, average_precision_score, roc_curve, confusion_matrix, \
    precision_recall_curve, precision_score 
from metric import get_sparsity_loss, get_continuity_loss, binary_cross_entropy, JS_DIV, cross_entropy_logits
from utils import save_checkpoint, load_latest_checkpoint


def train_one_epoch(model, opt_gen, opt_pred, opt_embedding, dataset, device, epoch, args):
    cls_l = 0
    spar_l = 0
    cont_l = 0
    js=0
    train_sp = []
    class_losses = []
    gen_losses = []
    for (_, (inputs, masks, labels)) in enumerate(tqdm(dataset, desc=f'Training epoch {epoch}')):
        opt_gen.zero_grad()
        opt_pred.zero_grad()
        opt_embedding.zero_grad()
        inputs, org_masks, labels = [item.to(device) for item in inputs], [item.to(device) for item in masks], labels.type(torch.float).to(device)

        #train classification
        drug_rationales, prot_rationales, drug_masks, prot_masks = model.get_rationale(inputs, org_masks)

        drug_sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            drug_rationales[:, :, 1], drug_masks, args.sparsity_percentage)
        prot_sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            prot_rationales[:, :, 1], prot_masks, args.sparsity_percentage)
        sparsity_loss = drug_sparsity_loss + prot_sparsity_loss

        train_sp.append((torch.sum(drug_rationales[:, :, 1]) / torch.sum(drug_masks)).cpu().item() + 
                        (torch.sum(prot_rationales[:, :, 1]) / torch.sum(prot_masks)).cpu().item())

        drug_continuity_loss = args.continuity_lambda * get_continuity_loss(drug_rationales[:, :, 1])
        prot_continuity_loss = args.continuity_lambda * get_continuity_loss(prot_rationales[:, :, 1])
        continuity_loss = drug_continuity_loss + prot_continuity_loss

        forward_logit = model.pred_forward_logit(inputs, org_masks, torch.detach(drug_rationales), torch.detach(prot_rationales))
        full_text_logits = model.train_one_step(inputs, org_masks)

        cls_loss = args.cls_lambda * cross_entropy_logits(forward_logit, labels)[1]
        full_text_cls_loss = args.cls_lambda * cross_entropy_logits(full_text_logits, labels)[1]

        classification_loss = cls_loss + full_text_cls_loss + sparsity_loss + continuity_loss
        class_losses.append(classification_loss.cpu().detach().numpy())
        classification_loss.backward()

        opt_pred.step()
        opt_pred.zero_grad()
        opt_gen.step()
        opt_gen.zero_grad()
        opt_embedding.step()
        opt_embedding.zero_grad()

        # train rationale with sparsity, continuity, js-div
        opt_gen.zero_grad()
        name1=[]
        name2=[]
        name3=[]
        for idx, p in model.embedding_layer.named_parameters():
            if p.requires_grad == True:
                name1.append(idx)
                p.requires_grad = False
        for idx,p in model.encoder.named_parameters():
            if p.requires_grad==True:
                name2.append(idx)
                p.requires_grad=False
        for idx,p in model.cls_fc.named_parameters():
            if p.requires_grad == True:
                name3.append(idx)
                p.requires_grad = False
        #print('freeze name1={},name2={},name3={},name4={},name5={},name6={}'.format(name1, name2, name3, name4, name5, name6))
        
        drug_rationales, prot_rationales, drug_masks, prot_masks = model.get_rationale(inputs, org_masks)
        #rationales, masks = model.get_rationale(inputs, org_masks)

        forward_logit = model.pred_forward_logit(inputs, org_masks, drug_rationales, prot_rationales)
        full_text_logits = model.train_one_step(inputs, org_masks)
        if args.div=='js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div=='kl':
            jsd_loss=nn.functional.kl_div(F.softmax(forward_logit,dim=-1).log(), 
                                          F.softmax(full_text_logits,dim=-1), reduction='batchmean')

        drug_sparsity_loss = args.sparsity_lambda * get_sparsity_loss(drug_rationales[:, :, 1], masks, args.sparsity_percentage)
        prot_sparsity_loss = args.sparsity_lambda * get_sparsity_loss(prot_rationales[:, :, 1], masks, args.sparsity_percentage)
        sparsity_loss = drug_sparsity_loss + prot_sparsity_loss
        
        train_sp.append((torch.sum(drug_rationales[:, :, 1]) / torch.sum(drug_masks)).cpu().item() + 
                        (torch.sum(prot_rationales[:, :, 1]) / torch.sum(prot_masks)).cpu().item())

        drug_continuity_loss = args.continuity_lambda * get_continuity_loss(drug_rationales[:, :, 1])
        prot_continuity_loss = args.continuity_lambda * get_continuity_loss(prot_rationales[:, :, 1])
        continuity_loss = drug_continuity_loss + prot_continuity_loss
    
        gen_loss = sparsity_loss + continuity_loss + jsd_loss
        gen_losses.append(gen_loss.cpu().detach().numpy())
        gen_loss.backward()
        
        opt_gen.step()
        opt_gen.zero_grad()

        n1=0
        n2=0
        n3=0
        #############recover the parameters#############
        for idx, p in model.embedding_layer.named_parameters():
            if idx in name1:
                p.requires_grad = True
                n1+=1
        for idx,p in model.encoder.named_parameters():
            if idx in name2:
                p.requires_grad=True
                n2+=1
        for idx,p in model.cls_fc.named_parameters():
            if idx in name3:
                p.requires_grad=True
                n3+=1
        #print('recover n1={},n2={},n3={},n4={},n5={},n6={}'.format(n1, n2, n3, n4, n5, n6))
        
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
        js+=jsd_loss.cpu().item()
       
        print(f'avg classification_loss: {np.array(class_losses).mean()}')
        print(f'avg gen_loss: {np.array(gen_losses).mean()}')
    class_loss_epoch = np.array(class_losses).mean()
    gen_loss_epoch = np.array(gen_losses).mean()
    # writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    # writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    # writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    # writer_epoch[0].add_scalar('js', js, writer_epoch[1])
    # writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return class_loss_epoch, gen_loss_epoch


def evaluate(model, dataloader, device, mode=None):
    test_losses = []
    y_label = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for (_, (inputs, masks, labels)) in enumerate(dataloader):
            # inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            inputs, masks, labels = [item.to(device) for item in inputs], [item.to(device) for item in masks], labels.to(device)
            _, logits = model(inputs, masks)
            # pdb.set_trace()
            n, loss = cross_entropy_logits(logits, labels)
            test_losses.append(loss.cpu().detach().numpy())

            y_label = y_label + labels.to("cpu").tolist()
            y_pred = y_pred + n.to("cpu").tolist()

        test_loss_epoch = np.array(test_losses).mean()
            
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)

        if mode == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            precision1 = precision_score(y_label, y_pred_s)

            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss_epoch, thred_optim, precision1
        else:
            return auroc, auprc, test_loss_epoch
        

def train(model, optimizer_gen, optimizer_pred, optimizer_embedding, train_loader, val_loader, logger, args):
    # Resume training
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.model_save_dir):
            print(f"Reumse training from {args.model_save_dir}")
            start_epoch, model, optimizer_gen, optimizer_pred = load_latest_checkpoint(args.model_save_dir, start_epoch, model, optimizer_gen, optimizer_pred)
        else:    
            raise ValueError(f"{args.model_save_dir} does not exist!")
    
    # Set up early stopping
    best_auroc = 0

    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        class_loss_epoch, gen_loss_epoch = train_one_epoch(model, optimizer_gen, optimizer_pred, optimizer_embedding, train_loader, args.device, epoch, args)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Class Loss: {class_loss_epoch:.4f}, Gen Loss: {gen_loss_epoch:.4f}")

        # Evaluate
        auroc, auprc, test_loss_epoch = evaluate(model, val_loader, args.device)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Val AUROC: {auroc:.4f}, Val AUPRC: {auprc:.4f}, Val Loss: {test_loss_epoch:.4f}")

        # Create the checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_gen_state_dict': optimizer_gen.state_dict(),
            'optimizer_pred_state_dict': optimizer_pred.state_dict(),
            'auroc': auroc,
            'train_loss': class_loss_epoch,
            'train_gen_loss': gen_loss_epoch,
            'val_loss': test_loss_epoch,
            'args': args
        }

        # Check if the current validation loss is the best
        is_best = auroc > best_auroc
        if is_best:
            best_auroc = auroc
            checkpoint['auroc'] = best_auroc

        # Save the checkpoint
        save_checkpoint(
            state=checkpoint,
            is_best=is_best,
            checkpoint_dir=args.model_save_dir,
        )
