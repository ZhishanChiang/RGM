import torch 
from torch import nn
from torch.nn import functional as F
from models.transformer import attention as attn
class PointNet(nn.Module):
    def __init__(self,input_dim,emb_dim):
        super(PointNet,self).__init__()
        self.layers=nn.Sequential(nn.Conv1d(input_dim,512,1),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Conv1d(512,1024,1),
                                  nn.BatchNorm1d(1024),
                                  nn.ReLU(),
                                  nn.Conv1d(1024,emb_dim,1),
                                  nn.BatchNorm1d(emb_dim),
                                  nn.ReLU()
                                 )
    def forward(self,x):
        return self.layers(x)
    
class AdjacencyMtx(nn.Module):
    def __init__(self,input_dim,emb_dim):
        super(AdjacencyMtx,self).__init__()
        self.encoder=PointNet(input_dim,emb_dim)
    def forward(self,src,tar):
        src_emb,tar_emb=src.transpose(2,1),tar.transpose(2,1)
        src_emb,tar_emb=self.encoder(src_emb),self.encoder(tar_emb)
        src_emb,src_attn=attn(src_emb,src_emb,src_emb)       #node-node
        tar_emb,tar_attn=attn(tar_emb,tar_emb,tar_emb)
        return src_emb,tar_emb
        
    
        
        
        
        