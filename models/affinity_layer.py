import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math


class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = Parameter(Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        #M = torch.matmul(X, self.A)
        M = torch.matmul(X, (self.A + self.A.transpose(0, 1).contiguous()) / 2)
        M = torch.matmul(M, Y.transpose(1, 2).contiguous())
        return M

class Modified_Affinity(nn.Module):
    def __init__(self,d):
        super(Modified_Affinity,self).__init__()
        self.d=d
    def forward(self,X,Y):
        X_v=torch.norm(X,p=2,dim=-1,keepdim=True)
        Y_v=torch.norm(Y,p=2,dim=-1,keepdim=True)
        div=torch.bmm(X_v,Y_v.transpose(2,1))
        M=torch.bmm(X,Y.transpose(2,1))
        return M/div
        
