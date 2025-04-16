import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sparsity_loss(z, mask, level):
    """
    Exact sparsity loss in a batchwise sense.
    Inputs:
        z -- (batch_size, sequence_length)
        mask -- (batch_size, seq_length)
        level -- sparsity level
    """
    sparsity = torch.sum(z) / torch.sum(mask)
    # print('get_sparsity={}'.format(sparsity.cpu().item()))
    return torch.abs(sparsity - level)


def get_continuity_loss(z):
    """
    Compute the continuity loss.
    Inputs:
        z -- (batch_size, sequence_length)
    """
    return torch.mean(torch.abs(z[:, 1:] - z[:, :-1]))


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


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