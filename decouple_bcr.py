import argparse
import os
import time
import torch
from embedding import get_embeddings,get_glove_embedding
from model import Sp_norm_model
from train_util import train_decouple_causal2
from validate_util import validate_dev_sentence
from tensorboardX import SummaryWriter
from dataloader import get_dataloader
from utils import evaluate

def parse():
    parser = argparse.ArgumentParser(
        description="")

    # dataset parameters
    
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--correlated',
                        type=int,
                        default=1,
                        help='The aspect number of beer review [0, 1]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 100]')


    # model parameters
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--if_biattn',
                        action='store_true',
                        help='if use biattention in embedding')
    parser.add_argument('--div',
                        type=str,
                        default='kl',
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')
    parser.add_argument('--lay',
                        type=bool,
                        default=True,
                        help='Number of predicted classes [default: 2]')
    parser.add_argument('--model_type',
                        type=str,
                        default='sp',
                        help='Number of predicted classes [default: 2]')

    # ckpt parameters
    parser.add_argument('--output_dir',
                        type=str,
                        default='./res',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--dis_lr',
                        type=int,
                        default=0,
                        help='Number of training epoch')
    parser.add_argument('--epochs',
                        type=int,
                        default=37,
                        help='Number of training epoch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=6.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=6.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.1,
        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument(
        '--writer',
        type=str,
        default='./noname',
        help='Regularizer to control highlight percentage [default: .2]')
    args = parser.parse_args()
    return args


#####################
# set random seed
#####################
# torch.manual_seed(args.seed)

#####################
# parse arguments
#####################
args = parse()
torch.manual_seed(args.seed)
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr, value))

######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(args.seed)

######################
# load dataset
######################
train_loader, dev_loader, test_loader = get_dataloader('biosnap', 'random', batch_size=args.batch_size, num_workers=5)

######################
# load model
######################
writer=SummaryWriter(args.writer)

model=Sp_norm_model(args)
model.to(device)

######################
# Training
#####################
# g_para=list(map(id, model.generator.parameters()))
# p_para=filter(lambda p: id(p) not in g_para and p.requires_grad==True, model.parameters())
g_para=[]
for p in model.generator.parameters():
    if p.requires_grad==True:
        g_para.append(p)
p_para=[]
for p in model.cls.parameters():
    if p.requires_grad==True:
        p_para.append(p)
for p in model.cls_fc.parameters():
    if p.requires_grad==True:
        p_para.append(p)


# print('g_para={}'.format(g_para))
# print('p_para={}'.format(p_para))
lr2=args.lr
lr1=args.lr

# para=[
#     {'params': model.generator.parameters(), 'lr':lr1},
#     {'params':p_para,'lr':lr2}
# ]
g_para=filter(lambda p: p.requires_grad==True, model.generator.parameters())
para_gen=[{'params': g_para, 'lr':lr1}]
para_pred=[{'params':p_para,'lr':lr2}]


optimizer_gen = torch.optim.Adam(para_gen)
optimizer_pred=torch.optim.Adam(para_pred)

######################
# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
grad=[]
grad_loss=[]
for epoch in range(args.epochs):

    start = time.time()
    model.train()
    precision, recall, f1_score, accuracy = train_decouple_causal2(model,optimizer_gen,optimizer_pred, train_loader, device, args,(writer,epoch))

    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    # print('gen_lr={}, pred_lr={}'.format(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
    print("traning epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision, f1_score,
                                                                                                   accuracy))
    writer.add_scalar('train_acc',accuracy,epoch)
    writer.add_scalar('time',time.time()-strat_time,epoch)
    
    precision, recall, f1_score, accuracy = evaluate(model, dev_loader)
    print("val epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision,
                                                                                                   f1_score, accuracy))
    writer.add_scalar('dev_acc',accuracy,epoch)
    if accuracy > acc_best_dev[-1]:
        acc_best_dev.append(accuracy)
        best_dev_epoch.append(epoch)
        
    # TP = 0
    # TN = 0
    # FN = 0
    # FP = 0
    # model.eval()
    # print("Validate")
    # with torch.no_grad():
    #     for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
    #         # inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
    #         inputs, masks, labels = [item.to(device) for item in inputs], [item.to(device) for item in masks], labels.to(device)
    #         _, logits = model(inputs, masks)
    #         # pdb.set_trace()
    #         logits = torch.softmax(logits, dim=-1)
    #         _, pred = torch.max(logits, axis=-1)
    #         # compute accuarcy
    #         # TP predict 和 label 同时为1
    #         TP += ((pred == 1) & (labels == 1)).cpu().sum()
    #         # TN predict 和 label 同时为0
    #         TN += ((pred == 0) & (labels == 0)).cpu().sum()
    #         # FN predict 0 label 1
    #         FN += ((pred == 0) & (labels == 1)).cpu().sum()
    #         # FP predict 1 label 0
    #         FP += ((pred == 1) & (labels == 0)).cpu().sum()
    #     print(TP, TN, FN, FP)
    #     precision = TP / (TP + FP)
    #     recall = TP / (TP + FN)
    #     f1_score = 2 * precision * recall / (recall + precision)
    #     accuracy = (TP + TN) / (TP + TN + FP + FN)
    #     print("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
    #                                                                                                precision,
    #                                                                                                f1_score, accuracy))

    #     writer.add_scalar('dev_acc',accuracy,epoch)
    #     print("Validate Sentence")
    #     validate_dev_sentence(model, dev_loader, device,(writer,epoch))
        
    #     if accuracy>acc_best_dev[-1]:
    #         acc_best_dev.append(accuracy)
    #         best_dev_epoch.append(epoch)

    # 检查是否需要保存模型
    if (epoch + 1) % save_interval == 0:
        save_path = os.path.join(args.output_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_gen_state_dict': optimizer_gen.state_dict(),
            'optimizer_pred_state_dict': optimizer_pred.state_dict()
        }, save_path)
        print(f"Model saved at epoch {epoch} to {save_path}")
        
print(acc_best_dev)
print(best_dev_epoch)
