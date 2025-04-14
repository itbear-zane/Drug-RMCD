import argparse
import os
import glob
import time
import torch
from model import Sp_norm_model
from train_util import train_one_epoch
from dataloader import get_dataloader
from utils import evaluate
import copy
import wandb
os.environ["WANDB_MODE"]="offline"

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
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='random seed')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 100]')
    parser.add_argument('--num_workers',
                        type=int,
                        default=5,
                        help='Num workers [default: 5]')
    parser.add_argument('--save_interval',
                        type=int,
                        default=10,
                        help='save_interval [default: 10]')

    # model parameters
    parser.add_argument('--if_biattn',
                        default=True,
                        type=bool,
                        help='if use biattention in embedding')
    parser.add_argument('--if_position',
                        default=True,
                        type=bool,
                        help='if use positional embedding in transformer')
    parser.add_argument('--div',
                        type=str,
                        default='kl',
                        help='kl loss')
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU, TransformerDecoder, TransformerEncoder [default: GRU]')
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
    parser.add_argument('--update_embedding_parameters_at_stage2',
                        action='store_true',
                        help='if update embedding parameters')

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
os.makedirs(args.writer, exist_ok=True)
# 打开一个文本文件以写入模式（'w'）
with open(args.writer + "/parameters.txt", "w") as f:
    # 写入标题
    f.write("Parameters:\n")
    print("\nParameters:")
    
    # 遍历 args 对象的属性并写入文件
    for attr, value in sorted(args.__dict__.items()):
        f.write("\t{}={}\n".format(attr, value))
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
train_loader, dev_loader, test_loader = get_dataloader('biosnap', 'random', batch_size=args.batch_size, num_workers=args.num_workers)

######################
# load model
######################
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
        auroc, auprc, precision, recall, f1_score, accuracy, sensitivity, specificity = evaluate(model, test_loader, device)
        print("test results: auroc:{:.4f}, auprc:{:.4f}, recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f} sensitivity:{:.4f} specificity:{:.4f}"\
            .format(auroc, auprc, recall, precision, f1_score, accuracy, sensitivity, specificity))
        raise ValueError("Testing stops... Don't worry!")
    else:
        raise ValueError(f"{args.checkpoint_path} does not exist!")

######################
# Training
#####################
# g_para=list(map(id, model.generator.parameters()))
# p_para=filter(lambda p: id(p) not in g_para and p.requires_grad==True, model.parameters())
e_para=[]
for p in model.embedding_layer.parameters():
    if p.requires_grad==True:
        e_para.append(p)
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
lr3=args.lr

g_para=filter(lambda p: p.requires_grad==True, model.generator.parameters())
para_gen=[{'params': g_para, 'lr':lr1}]
para_pred=[{'params': p_para,'lr':lr2}]
para_embedding=[{'params': e_para,'lr':lr3}]

optimizer_gen = torch.optim.Adam(para_gen)
optimizer_pred = torch.optim.Adam(para_pred)
optimizer_embedding = torch.optim.Adam(para_embedding)

start_epoch = 0
if args.resume:
    if os.path.exists(args.checkpoint_path):
        print(f"Reumse training from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
        optimizer_pred.load_state_dict(checkpoint['optimizer_pred_state_dict'])
        if 'optimizer_embedding_state_dict' in checkpoint.keys():
            optimizer_embedding.load_state_dict(checkpoint['optimizer_embedding_state_dict'])
    else:
        raise ValueError(f"{args.checkpoint_path} does not exist!")

######################
# Training
######################

wanbd_run = wandb.init(
    project="Drug-RMCD",    # Specify your project
    name="_".join(args.writer.split('/')[1:]),
    config={                         # Track hyperparameters and metadata
        "learning_rate": args.lr,
    },
)

strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_epoch = 0
best_auroc = 0
best_model = None
grad=[]
grad_loss=[]
for epoch in range(start_epoch, args.epochs):
    # TODO wandb
    start = time.time()
    model.train()
    precision, recall, f1_score, accuracy = train_one_epoch(model, optimizer_gen, optimizer_pred, optimizer_embedding, train_loader, device, args, wanbd_run, epoch)

    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    # print('gen_lr={}, pred_lr={}'.format(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
    print("traning epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision, f1_score,
                                                                                                   accuracy))
    wanbd_run.log({"train/accuracy": accuracy, "train/precision": precision, "train/recall": recall, "train/f1": f1_score, "epoch": epoch})

    auroc, auprc, precision, recall, f1_score, accuracy, sensitivity, specificity = evaluate(model, dev_loader, device)
    print("val results: auroc:{:.4f}, auprc:{:.4f}, recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f} sensitivity:{:.4f} specificity:{:.4f}"\
            .format(auroc, auprc, recall, precision, f1_score, accuracy, sensitivity, specificity))
    wanbd_run.log({"val/auroc": auroc, "val/auprc": auprc, "val/sensitivity": sensitivity, "val/specificity": specificity, \
        "val/accuracy": accuracy, "val/precision": precision, "val/recall": recall, "val/f1": f1_score, "epoch": epoch})

    if auroc >= best_auroc:
        print('Test at Best Model of Epoch ' + str(best_epoch) + " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy))
        best_model = [copy.deepcopy(model), copy.deepcopy(optimizer_gen), copy.deepcopy(optimizer_pred), copy.deepcopy(optimizer_embedding)]
        best_auroc = auroc
        best_epoch = epoch

    # 检查是否需要保存模型
    if (epoch + 1) % args.save_interval == 0:
        auroc, auprc, precision, recall, f1_score, accuracy, sensitivity, specificity = evaluate(model, test_loader, device)
        print("test results: auroc:{:.4f}, auprc:{:.4f}, recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f} sensitivity:{:.4f} specificity:{:.4f}"\
            .format(auroc, auprc, recall, precision, f1_score, accuracy, sensitivity, specificity))
        wanbd_run.log({"test/auroc": auroc, "test/auprc": auprc, "test/sensitivity": sensitivity, "test/specificity": specificity, \
        "test/accuracy": accuracy, "test/precision": precision, "test/recall": recall, "test/f1": f1_score, "epoch": epoch})

        save_path = os.path.join(args.writer, f'model_epoch_{best_epoch}.pth')
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model[0].state_dict(),
            'optimizer_gen_state_dict': best_model[1].state_dict(),
            'optimizer_pred_state_dict': best_model[2].state_dict(),
            'optimizer_embedding_state_dict': best_model[3].state_dict(),
        }, save_path)
        print(f"Model saved at best epoch {best_epoch} to {save_path}")
        
        # 获取当前目录中的所有权重文件
        checkpoint_files = sorted(
            glob.glob(os.path.join(args.writer, 'model_epoch_*.pth')),
            key=os.path.getmtime  # 按修改时间排序（最早的在前）
        )
        # 如果权重文件数量超过最大限制，则删除最老的一个
        if len(checkpoint_files) > 10:
            oldest_checkpoint = checkpoint_files[0]
            os.remove(oldest_checkpoint)
            print(f"Deleted oldest checkpoint: {oldest_checkpoint}")