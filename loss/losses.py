import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor
from torch import nn
def get_loss(args, weights, train_dataset, num_cls):
    if args.loss == 'ce':
        loss_fxn = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
    elif args.loss == 'focal':
        # loss_fxn = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss', alpha=weights, gamma=args.fl_gamma, reduction='mean')
        loss_fxn = FocalLoss(alpha=weights, gamma=args.fl_gamma, reduction='mean')
    elif args.loss == 'ldam':
        loss_fxn = LDAMLoss(cls_num_list=train_dataset.cls_num_list, weight=weights)
    elif args.loss == 'logit':
        loss_fxn = LogitAdjust(train_dataset.cls_num_list).cuda()
    elif args.loss == 'ordi_logit':
        loss_fxn = OrdinalLogitAdjust(num_cls).cuda()
    return loss_fxn

def get_loss_fxn(loss, weights, cls_num_list, fl_gamma=2.0, s=30, num_cls=5, tau=1):
    if loss == 'ce':
        loss_fxn = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
    elif loss == 'Stick_break_CE_loss':
        loss_fxn = Stick_break_CE_loss().cuda()
    elif loss == 'focal':
        # loss_fxn = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss', alpha=weights, gamma=args.fl_gamma, reduction='mean')
        loss_fxn = FocalLoss(alpha=weights, gamma=fl_gamma, reduction='mean')
    elif loss == 'ldam':
        loss_fxn = LDAMLoss(cls_num_list=cls_num_list, weight=weights, s=s)
    elif loss == 'wkappa':
        loss_fxn = WeightedKappaLoss(num_classes = num_cls, regression = False)
    elif loss == 'logit':
        loss_fxn = LogitAdjust(cls_num_list, tau = tau).cuda()
    elif loss == 'logit1':
        loss_fxn = LogitAdjust1(cls_num_list, tau = tau).cuda()
    elif loss == 'logit2':
        loss_fxn = LogitAdjust2(cls_num_list, tau = tau).cuda()
    elif loss == 'logit3':
        loss_fxn = LogitAdjust3(cls_num_list, tau = tau).cuda()
    elif loss == 'ordi_logit':
        loss_fxn = OrdinalLogitAdjust(num_cls, tau = tau).cuda()    
    elif loss == 'ordi_logit_0':
        loss_fxn = OrdinalLogitAdjust_0(num_cls, tau = tau, weight=weights).cuda()  
    elif loss == 'ordi_logit1':
        loss_fxn = OrdinalLogitAdjust_1(num_cls, tau = tau).cuda()  
    elif loss == 'ordi_logit20':
        loss_fxn = OrdinalLogitAdjust20(cls_num_list, tau = tau).cuda()
    elif loss == 'ordi_logit20_1':
        loss_fxn = OrdinalLogitAdjust20_1(cls_num_list, tau = tau).cuda()
    elif loss == 'ordi_logit20_2':
        loss_fxn = OrdinalLogitAdjust20_2(cls_num_list, tau = tau).cuda()
    elif loss == 'logit_part1':
        loss_fxn = Logit_part1(cls_num_list).cuda()
    elif loss == 'logit_part2':
        loss_fxn = Logit_part2(cls_num_list).cuda()
    elif loss == 'ordi_logit_part1':
        loss_fxn = OrdinalLogit_part1(num_cls).cuda()
    elif loss == 'ordi_logit_part2':
        loss_fxn = OrdinalLogit_part2(num_cls).cuda()
    return loss_fxn


class Stick_break_CE_loss(torch.nn.Module):
    def __init__(self):
        super(Stick_break_CE_loss, self).__init__()
        self.loss_type = 'Stick_break_CE_loss'

    """
    破棍代替softmax
    """
    def forward(self, logits, target):
        # 使用 stick-breaking 替代 softmax
        probabilities = self.neuron_Stick_breaking(logits, dim=1)
        
        # 计算 log 概率
        log_probabilities = torch.log(probabilities + 1e-12)  # 加上小偏置避免 log(0)
        
        # 负对数似然损失 (NLL loss)
        loss = F.nll_loss(log_probabilities, target)
        
        return loss

    def neuron_Stick_breaking(self, x, dim=-1):
        """
        Stick-breaking 实现，替代 softmax 进行分类概率计算。
        """
        p_sigmoid = torch.sigmoid(x)
        batch_size, class_count_minus_1 = x.size()
        
        # 初始化概率张量
        p = torch.zeros(batch_size, class_count_minus_1 + 1, device=x.device)
        
        # 第一个类别概率
        p[:, 0] = p_sigmoid[:, 0]
        
        # 累积乘积计算 1 - sigmoid(x) for efficient vectorization
        prod_cum = torch.cumprod(1 - p_sigmoid, dim=dim)
        
        # 中间类别概率
        p[:, 1:-1] = p_sigmoid[:, 1:] * prod_cum[:, :-1]
        
        # 最后一个类别概率
        p[:, -1] = prod_cum[:, -1]
        
        return p


def get_CB_weights(samples_per_cls, beta):
    effective_num = 1.0 - np.power(beta, samples_per_cls)

    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(samples_per_cls)

    return weights

## CREDIT TO https://github.com/kaidic/LDAM-DRW ##
class LDAMLoss(torch.nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

        print(self.weight)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s*output, target, weight=self.weight)

## credit to https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py ##

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl

# class New_LogitAdjust(nn.Module):
#     # 统一不平衡分类和序分类，原始的logit调整函数与输如的x无关，只与y有关，即sigma[y];
#     # 修改后的还与x相关，即为sigma[x,y];
#     # dist_matrix即为距离矩阵，用来调整间距
#     def __init__(self, tau=1, weight=None, dist_matrix=None):
#         super(LogitAdjust1, self).__init__()
#         self.m_list = m_list.view(1, -1)
#         self.weight = weight
#         self.dist_matrix = dist_matrix

#     def forward(self, x, target):
#         tar = torch.clone(target)
#         if len(target.shape) > 1:
#             tar = torch.argmax(tar, dim=1)
#         dist = torch.zeros(len(tar), self.num_cls).cuda()
#         for i in range(len(tar)):
#             dist[i] = self.dist_matrix[tar[i]]
#         x_m = x + dist
#         return F.cross_entropy(x_m, target, weight=self.weight)


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

class LogitAdjust1(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust1, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x - self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

class LogitAdjust2(nn.Module):
    # logit 绝对值
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust2, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.num_cls = len(cls_num_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        # self.dist_matrix = m_list - m_list.unsqueeze(1)
        self.dist_matrix = abs(m_list.unsqueeze(1) - m_list)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)

class LogitAdjust3(nn.Module):
    # 同logit，但使用了矩阵形式
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust3, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.num_cls = len(cls_num_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.dist_matrix = m_list - m_list.unsqueeze(1)
        # self.dist_matrix = abs(m_list.unsqueeze(1) - m_list)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)

class OrdinalLogitAdjust(nn.Module):
    def __init__(self, num_cls=5, tau=1, weight=None):
        super(OrdinalLogitAdjust, self).__init__()
        self.weight = weight
        self.num_cls = num_cls
        self.dist_matrix = abs(torch.arange(self.num_cls).unsqueeze(1) - torch.arange(self.num_cls))*2/self.num_cls

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)
class OrdinalLogitAdjust_0(nn.Module):
    def __init__(self, num_cls=5, tau=1, weight=None):
        super(OrdinalLogitAdjust_0, self).__init__()
        self.weight = weight
        self.num_cls = num_cls
        # Paper definition: d(y, y') = 2 * |y-y'| / C.
        # `tau` is the beta margin coefficient used in the manuscript.
        self.dist_matrix = tau * abs(torch.arange(self.num_cls).unsqueeze(1) - torch.arange(self.num_cls))*2/self.num_cls

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)
class OrdinalLogitAdjust_1(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(OrdinalLogitAdjust_1, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.num_cls = len(cls_num_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.dist_matrix = tau * ((abs(torch.arange(self.num_cls).unsqueeze(1) - torch.arange(self.num_cls))/(self.num_cls-1))**0.5*2).cuda()
        # self.dist_matrix = abs(m_list.unsqueeze(1) - m_list)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)
class OrdinalLogitAdjust_matrix(nn.Module):
    def __init__(self, dist_matrix, num_cls=5, tau=1, weight=None):
        super(OrdinalLogitAdjust, self).__init__()
        self.weight = weight
        self.num_cls = num_cls
        if dist_matrix is not None:
            self.dist_matrix = tau * dist_matrix
        # self.dist_matrix = abs(torch.arange(self.num_cls).unsqueeze(1) - torch.arange(self.num_cls))*2/self.num_cls

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)
class OrdinalLogitAdjust20(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(OrdinalLogitAdjust20, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.num_cls = len(cls_num_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        dist_matrix1 = (m_list - m_list.unsqueeze(1))
        dist_matrix2 = (abs(torch.arange(self.num_cls).unsqueeze(1) - torch.arange(self.num_cls))*2/(self.num_cls-1)).cuda()
        self.dist_matrix =  dist_matrix1*dist_matrix2*(dist_matrix1>0)+dist_matrix1.div(1e-6+dist_matrix2)*(dist_matrix1<0)

        # self.dist_matrix = abs(m_list.unsqueeze(1) - m_list)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)
class OrdinalLogitAdjust20_1(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(OrdinalLogitAdjust20_1, self).__init__()
        print(OrdinalLogitAdjust20_1)
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.num_cls = len(cls_num_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        dist_matrix1 = (m_list - m_list.unsqueeze(1))
        dist_matrix2 = ((abs(torch.arange(self.num_cls).unsqueeze(1) - torch.arange(self.num_cls))/(self.num_cls-1))**0.5*2).cuda()
        self.dist_matrix =  dist_matrix1*dist_matrix2*(dist_matrix1>0)+dist_matrix1.div(1e-6+dist_matrix2)*(dist_matrix1<0)

        # self.dist_matrix = abs(m_list.unsqueeze(1) - m_list)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)

class OrdinalLogitAdjust20_2(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(OrdinalLogitAdjust20_2, self).__init__()
        print(OrdinalLogitAdjust20_2)
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.num_cls = len(cls_num_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        dist_matrix1 = (m_list - m_list.unsqueeze(1))
        dist_matrix2 = ((abs(torch.arange(self.num_cls).unsqueeze(1) - torch.arange(self.num_cls))*2+1).log()).cuda()
        self.dist_matrix =  dist_matrix1*dist_matrix2*(dist_matrix1>0)+dist_matrix1.div(1e-6+dist_matrix2)*(dist_matrix1<0)

        # self.dist_matrix = abs(m_list.unsqueeze(1) - m_list)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)
class Logit_part1(nn.Module):
    # 下三角
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(Logit_part1, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.num_cls = len(cls_num_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.dist_matrix = m_list - m_list.unsqueeze(1)
        self.dist_matrix = self.dist_matrix * (self.dist_matrix>0)
        # self.dist_matrix = abs(m_list.unsqueeze(1) - m_list)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)
class Logit_part2(nn.Module):
    # 上三角
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(Logit_part2, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.num_cls = len(cls_num_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.dist_matrix = m_list - m_list.unsqueeze(1)
        self.dist_matrix = self.dist_matrix * (self.dist_matrix<0)
        # self.dist_matrix = abs(m_list.unsqueeze(1) - m_list)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)

class OrdinalLogit_part1(nn.Module):
    # 下三角
    def __init__(self, num_cls=5, tau=1, weight=None):
        super(OrdinalLogit_part1, self).__init__()
        self.weight = weight
        self.num_cls = num_cls
        self.dist_matrix = (torch.arange(self.num_cls).unsqueeze(1) - torch.arange(self.num_cls))*2/self.num_cls
        self.dist_matrix = abs(self.dist_matrix) * (self.dist_matrix>0)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)
class OrdinalLogit_part2(nn.Module):
    # 上三角
    def __init__(self, num_cls=5, tau=1, weight=None):
        super(OrdinalLogit_part2, self).__init__()
        self.weight = weight
        self.num_cls = num_cls
        self.dist_matrix = (torch.arange(self.num_cls).unsqueeze(1) - torch.arange(self.num_cls))*2/self.num_cls
        self.dist_matrix = abs(self.dist_matrix) * (self.dist_matrix<0)

    def forward(self, x, target):
        tar = torch.clone(target)
        if len(target.shape) > 1:
            tar = torch.argmax(tar, dim=1)
        dist = torch.zeros(len(tar), self.num_cls).cuda()
        for i in range(len(tar)):
            dist[i] = self.dist_matrix[tar[i]]
        x_m = x + dist
        return F.cross_entropy(x_m, target, weight=self.weight)

import torch
from torch.nn import Module, Softmax
from typing import Optional

# started from https://gist.github.com/SupreethRao99/2e85884dad433a6b381f966fea7a6658

class WeightedKappaLoss(Module):
    r"""
    Implements Quadratic Weighted Kappa Loss. Weighted Kappa Loss was introduced in the
    [Weighted kappa loss function for multi-class classification
      of ordinal data in deep learning]
      (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    Weighted Kappa is widely used in Ordinal Classification Problems. The loss
    value lies in $[-\infty, \log 2]$, where $\log 2$ means the random prediction
    Usage: loss_fn = WeightedKappaLoss(num_classes = NUM_CLASSES)
    """

    def __init__(
            self,
            num_classes: int,
            device : Optional[str]     = 'cpu',
            # mode: Optional[str]        = 'quadratic',
            name: Optional[str]        = 'cohen_kappa_loss',
            epsilon: Optional[float]   = 1e-10,
            regression: Optional[bool] = True
            ):
        r"""Creates a `WeightedKappaLoss` instance.
            Args:
              num_classes: Number of unique classes in your dataset.
              device: (Optional) Device on which computation will be performed.
              name: (Optional) String name of the metric instance.
              epsilon: (Optional) increment to avoid log zero,
                so the loss will be $ \log(1 - k + \epsilon) $, where $ k $ lies
                in $ [-1, 1] $. Defaults to 1e-10.
              regression: (Optional) if True (default) will calculate the Loss in 
                a regression setting $ y \in R^n $, where $ n $ is the number of samples. 
                Otherwise it will assume a classification setting in which $ y \in R^{n \times m} $,
                where $ m $ is the number of classes.
            """

        super(WeightedKappaLoss, self).__init__()
        self.num_classes = num_classes

        self.epsilon = epsilon

        # Creates weight matrix (which is constant)
        self.weights = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).to(device)
        self.weights = torch.square((self.weights - self.weights.T))

        # bricks for later histogram of values
        self.hist_bricks = torch.eye(num_classes).to(device)

        if not regression:
            self.softmax = Softmax(dim=1)
        self.regression = regression

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        bsize = y_true.size(0)
        
        # Numerator: 
        if not self.regression:
            c = self.weights[y_true].squeeze()
            O = torch.mul(y_pred, c).sum()
        else:
            O = (y_pred - y_true).square().sum()
            
        # Denominator: 
        hist_true = torch.sum(self.hist_bricks[y_true], 0)
        
        if not self.regression: 
            hist_pred = y_pred.sum(axis=0)
        else:
            y_pred = y_pred.clamp(0, self.num_classes-1)
            y_pred_floor = y_pred.floor().long()
            y_pred_ceil  = y_pred.ceil().long()
            y_pred_perc  = (y_pred % 1).transpose(0,1)

            floor_loss = torch.mm(1-y_pred_perc, self.hist_bricks[y_pred_floor].squeeze())
            ceil_loss  = torch.mm(y_pred_perc,   self.hist_bricks[y_pred_ceil].squeeze())
            hist_pred = floor_loss + ceil_loss
            
        expected_probs = torch.mm(
            torch.reshape(hist_true, [num_classes, 1]),
            torch.reshape(hist_pred, [1, num_classes]))

        E = torch.sum(self.weights * expected_probs / bsize)

        return O / (E + self.epsilon)

    def forward(self, y_pred, y_true, log=True):
        if not self.regression:
            y_pred = self.softmax(y_pred)
        y_true = y_true.long()
        
        loss = self.kappa_loss(y_pred, y_true)
        
        if log:
            loss = torch.log(loss)
        return loss
