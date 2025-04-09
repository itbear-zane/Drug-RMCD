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
    parser.add_argument('--test',
                        action='store_true',
                        help='only test')
    parser.add_argument('--resume',
                        action='store_true',
                        help='resume training states')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='',
                        help='model weight path')
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
    parser.add_argument('--save_interval',
                        type=int,
                        default=10,
                        help='save_interval [default: 10]')

    # model parameters
    parser.add_argument('--if_biattn',
                        action='store_true',
                        help='if use biattention in embedding')
    parser.add_argument('--div',
                        type=str,
                        default='kl',
                        help='kl loss')
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
# Test
#####################
if args.test:
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)  # 加载权重文件
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded from {args.checkpoint_path}")
        precision, recall, f1_score, accuracy = evaluate(model, test_loader, None, None)
        print("test results: recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}"\
            .format(recall, precision, f1_score, accuracy))
        raise ValueError("Testing stops... Don't worry!")
    else:
        raise ValueError(f"{args.checkpoint_path} does not exist!")

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

lr2=args.lr
lr1=args.lr

g_para=filter(lambda p: p.requires_grad==True, model.generator.parameters())
para_gen=[{'params': g_para, 'lr':lr1}]
para_pred=[{'params': p_para,'lr':lr2}]

optimizer_gen = torch.optim.Adam(para_gen)
optimizer_pred = torch.optim.Adam(para_pred)

if args.resume:
    if os.path.exists(args.checkpoint_path):
        print(f"Reumse training from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
        optimizer_pred.load_state_dict(checkpoint['optimizer_pred_state_dict'])
    else:
        raise ValueError(f"{args.checkpoint_path} does not exist!")

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
    
    precision, recall, f1_score, accuracy = evaluate(model, dev_loader, writer, epoch)
    print("val epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision,
                                                                                                   f1_score, accuracy))
    writer.add_scalar('dev_acc',accuracy,epoch)
    if accuracy > acc_best_dev[-1]:
        acc_best_dev.append(accuracy)
        best_dev_epoch.append(epoch)

    # 检查是否需要保存模型
    if (epoch + 1) % args.save_interval == 0:
        precision, recall, f1_score, accuracy = evaluate(model, test_loader, None, None)
        print("test epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                    precision,
                                                                                                    f1_score, accuracy))
        save_path = os.path.join(args.writer, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_gen_state_dict': optimizer_gen.state_dict(),
            'optimizer_pred_state_dict': optimizer_pred.state_dict()
        }, save_path)
        print(f"Model saved at epoch {epoch} to {save_path}")
        
print(acc_best_dev)
print(best_dev_epoch)
