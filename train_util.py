import torch
import torch.nn.functional as F
import torch.nn as nn

from metric import get_sparsity_loss, get_continuity_loss, computer_pre_rec
import numpy as np
import math
from tqdm import tqdm

class JS_DIV(nn.Module):
    def __init__(self):
        super(JS_DIV, self).__init__()
        self.kl_div=nn.KLDivLoss(reduction='batchmean',log_target=True)
    def forward(self,p,q):
        p_s=F.softmax(p,dim=-1)
        q_s=F.softmax(q,dim=-1)
        p_s, q_s = p_s.view(-1, p_s.size(-1)), q_s.view(-1, q_s.size(-1))
        m = (0.5 * (p_s + q_s)).log()
        return 0.5 * (self.kl_div(m, p_s.log()) + self.kl_div(m, q_s.log()))

def train_decouple_causal2(model, opt_gen,opt_pred, opt_embedding, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    js=0
    train_sp = []
    batch_len=len(dataset)
    class_losses = []
    gen_losses = []
    _, epoch = writer_epoch
    for (batch, (inputs, masks, labels)) in enumerate(tqdm(dataset, desc=f'Training epoch {epoch}')):
        opt_gen.zero_grad()
        opt_pred.zero_grad()
        inputs, org_masks, labels = [item.to(device) for item in inputs], [item.to(device) for item in masks], labels.type(torch.LongTensor).to(device)

        #train classification
        rationales, masks = model.get_rationale(inputs, org_masks)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        forward_logit = model.pred_forward_logit(inputs, org_masks, torch.detach(rationales))
        full_text_logits = model.train_one_step(inputs, org_masks)
        cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)
        full_text_cls_loss = args.cls_lambda * F.cross_entropy(full_text_logits, labels)
        classification_loss=cls_loss+full_text_cls_loss+sparsity_loss + continuity_loss
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
        for idx,p in model.cls.named_parameters():
            if p.requires_grad==True:
                name1.append(idx)
                p.requires_grad=False
        for idx,p in model.cls_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False
        if args.model_type!='sp':
            for idx,p in model.layernorm2.named_parameters():
                if p.requires_grad == True:
                    name3.append(idx)
                    p.requires_grad = False
        if not args.update_embedding_parameters_at_stage2:
            name4=[]
            for idx,p in model.embedding_layer.named_parameters():
                if p.requires_grad == True:
                    name4.append(idx)
                    p.requires_grad = False
        # print('name1={},name2={},name3={}'.format(len(name1),len(name2),len(name3)))

        rationales, masks = model.get_rationale(inputs, org_masks)

        forward_logit = model.pred_forward_logit(inputs, org_masks, rationales)

        full_text_logits = model.train_one_step(inputs, org_masks)
        if args.div=='js':
            jsd_func = JS_DIV()
            jsd_loss = jsd_func(forward_logit, full_text_logits)
        elif args.div=='kl':
            jsd_loss=nn.functional.kl_div(F.softmax(forward_logit,dim=-1).log(), F.softmax(full_text_logits,dim=-1), reduction='batchmean')

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])
        
        gen_loss = sparsity_loss + continuity_loss + jsd_loss
        gen_loss.backward()
        gen_losses.append(gen_loss.cpu().detach().numpy())
        opt_gen.step()
        opt_gen.zero_grad()
        n1=0
        n2=0
        n3=0
        for idx,p in model.cls.named_parameters():
            if idx in name1:
                p.requires_grad=True
                n1+=1
        for idx,p in model.cls_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1
        if args.model_type != 'sp':
            for idx,p in model.layernorm2.named_parameters():
                if idx in name3:
                    p.requires_grad = True
                    n3 += 1
        if not args.update_embedding_parameters_at_stage2:
            n4=0
            for idx,p in model.embedding_layer.named_parameters():
                if idx in name4:
                    p.requires_grad = True
                    n4 += 1
        # print('recover name1={},name2={},name3={}'.format(n1, n2, n3))


        cls_soft_logits = torch.softmax(forward_logit, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)
        # print(pred, labels)
        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
        js+=jsd_loss.cpu().item()
        # print(TP, TN, FN, FP)
        print(f'avg classification_loss: {np.array(class_losses).mean()}')
        print(f'avg gen_loss: {np.array(gen_losses).mean()}')
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('js', js, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    writer_epoch[0].add_scalar('train_class_loss', np.array(class_losses).mean(), writer_epoch[1])
    writer_epoch[0].add_scalar('train_gen_loss', np.array(gen_losses).mean(), writer_epoch[1])
    return precision, recall, f1_score, accuracy

