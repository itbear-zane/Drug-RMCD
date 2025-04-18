import argparse
import torch
from model import DrugRMCD
from train_util import train, evaluate
from dataloader import get_dataloader
from utils import set_seed, construct_logger
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description="Rationalized Minimum Conditional Dependence for Drug-Target Interaction Prediction")
    parser.add_argument("--device", type=str, default="cuda:0", help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')

    # dataset parameters
    parser.add_argument("--logger_dir", type=Path, default="/home_data/home/yangjie2024/dti/Drug-RMCD/loggers/")
    parser.add_argument('--test', action='store_true', help='only test')
    parser.add_argument('--resume', action='store_true', help='resume training states')
    parser.add_argument("--model_save_dir", type=Path, default=f"/home_data/home/yangjie2024/dti/Drug-RMCD/checkpoints/")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size [default: 100]')
    parser.add_argument('--num_workers', type=int, default=5, help='Num workers [default: 5]')

    # model parameters
    parser.add_argument('--num_layers', type=int, default=2, help='Num layers of TransformerEncoder Layer')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dims [default: 128]')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dim of feedforward layer')
    parser.add_argument('--num_heads', type=int, default=8, help='Num attention heads of transformer layer')
    parser.add_argument('--mlp_in_dim', type=int, default=256, help='MLP input dims [default: 128]')
    parser.add_argument('--mlp_hidden_dim', type=int, default=512, help='MLP hidden dims [default: 512]')
    parser.add_argument('--mlp_out_dim', type=int, default=128, help='MLP output dims [default: 128]')
    parser.add_argument('--num_class', type=int, default=2, help='Num of classes [default: 1]')

    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate [default: 0.2]')
    parser.add_argument('--div', type=str, default='kl', help='kl loss')

    # learning parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epoch')
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument('--lr1', type=float, default=0.0001, help='Learning rate [default: 1e-3]')
    parser.add_argument('--lr2', type=float, default=0.0001, help='Learning rate [default: 1e-3]')
    parser.add_argument('--lr3', type=float, default=0.0001, help='Learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda', type=float, default=6., help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda', type=float, default=6., help='Continuity trade-off [default: 4.]')
    parser.add_argument('--sparsity_percentage', type=float, default=0.1, help='Sparsity percentage [default: .2]')
    parser.add_argument('--cls_lambda', type=float, default=1., help='lambda for classification loss')
 
    args = parser.parse_args()
    return args

# parse arguments
args = parse()
set_seed(args.seed)
logger = construct_logger(args.logger_dir)

# load dataset
train_loader, val_loader, test_loader = get_dataloader('biosnap', 'random', batch_size=args.batch_size, num_workers=args.num_workers)

# load model
model=DrugRMCD(args)
model.to(args.device)

# ######################
# # Test
# #####################
# if args.test:
#     if os.path.exists(args.checkpoint_path):
#         checkpoint = torch.load(args.checkpoint_path, map_location=device)  # 加载权重文件
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
#         print(f"Model loaded from {args.checkpoint_path}")
#         auroc, auprc, precision, recall, f1_score, accuracy, sensitivity, specificity = evaluate(model, test_loader, device, mode="test")
#         print("test results: auroc:{:.4f}, auprc:{:.4f}, recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f} sensitivity:{:.4f} specificity:{:.4f}"\
#             .format(auroc, auprc, recall, precision, f1_score, accuracy, sensitivity, specificity))
#         raise ValueError("Testing stops... Don't worry!")
#     else:
#         raise ValueError(f"{args.checkpoint_path} does not exist!")


# Set up optimizer
# Embedding
e_para=[]
for p in model.embedding_layer.parameters():
    if p.requires_grad==True:
        e_para.append(p)
# Generator
g_para=[]
for p in model.generator.parameters():
    if p.requires_grad==True:
        g_para.append(p)
# Predictor
p_para=[]
for p in model.encoder.parameters():
    if p.requires_grad==True:
        p_para.append(p)
for p in model.cls_fc.parameters():
    if p.requires_grad==True:
        p_para.append(p)

# g_para=filter(lambda p: p.requires_grad==True, model.generator.parameters())
para_gen=[{'params': g_para, 'lr':args.lr1}]
para_pred=[{'params': p_para,'lr':args.lr2}]
para_embedding=[{'params': e_para,'lr':args.lr3}]

optimizer_gen = torch.optim.Adam(para_gen)
optimizer_pred = torch.optim.Adam(para_pred)
optimizer_embedding = torch.optim.Adam(para_embedding)

######################
# Training
######################
train(model, optimizer_gen, optimizer_pred, optimizer_embedding, train_loader, val_loader, logger, args)

######################
# Testing
######################
model = DrugRMCD(args)
model.load_state_dict(torch.load(f'{args.model_save_dir}/model_best.pth')['model_state_dict'])
model.to(args.device)
auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = evaluate(model, test_loader, args.device, mode="test")
logger.info("Test results: Auroc:{:.4f}, Auprc:{:.4f}, F1:{:.4f}, Sensitivity:{:.4f}, Specificity:{:.4f}, Accuracy:{:.4f}, Thred_optim:{:.4f}, Precision:{:.4f}"\
            .format(auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim, precision))

