import torch
from torch import nn

class QGAB(nn.Module):
    def __init__(self,emb_dim):
        super(QGAB,self).__init__()
        self.layer=nn.Sequential(nn.Conv1d(emb_dim,2*emb_dim,1),
                                nn.ReLU(),
                                nn.BatchNorm1d(2*emb_dim),
                                nn.Conv1d(2*emb_dim,4*emb_dim,1),
                                nn.ReLU(),
                                nn.BatchNorm1d(4*emb_dim),
                                )  
    def forward(self,query,key):   #query src(B,M,C) key tgt(B,N,C)   M<N
        _,_,repeats=key.shape
        q=query.transpose(2,1)  #(B,C,M)  less points
        k=key.transpose(2,1)    #(B,C,N)
        q=self.layer(q)
        k=self.layer(k)
        attn=torch.bmm(k.transpose(2,1),q)  #(B,N,M)
        attn_c=attn.clone()
        attn=torch.softmax(attn,dim=-2)
        attn=torch.max(attn,dim=-1,keepdim=True)[0]   #(B,N,1)
        attn=torch.repeat_interleave(attn,repeats=repeats,dim=-1) #(B,N,C)
        res=attn*key+key
        return res,attn_c

class defect_attention(nn.Module):
    def __init__(self,emb_dim):
        super(defect_attention,self).__init__()
        self.layer1=nn.Sequential(nn.Conv1d(emb_dim,2*emb_dim,1),
                                nn.ReLU(),
                                nn.BatchNorm1d(2*emb_dim)) #,
                                # nn.Conv1d(2*emb_dim,4*emb_dim,1),
                                # nn.ReLU(),
                                # nn.BatchNorm1d(4*emb_dim),
                                # )  
        self.layer2=nn.Sequential(nn.Conv1d(emb_dim,2*emb_dim,1),
                                nn.ReLU(),
                                nn.BatchNorm1d(2*emb_dim)) #,
                                # nn.Conv1d(2*emb_dim,4*emb_dim,1),
                                # nn.ReLU(),
                                # nn.BatchNorm1d(4*emb_dim),
                                # )  
    def forward(self,query,key):   #query src(B,M,C) key tgt(B,N,C)   M<N
        _,_,repeats=key.shape
        q=query.transpose(2,1)  #(B,C,M)  less points
        k=key.transpose(2,1)    #(B,C,N)
        q=self.layer1(q)
        k=self.layer2(k)
        attn=torch.bmm(k.transpose(2,1),q)  #(B,N,M)
        attn_c=attn.clone()
        attn=torch.softmax(attn,dim=-2)
        attn=torch.max(attn,dim=-1,keepdim=True)[0]   #(B,N,1)
        attn=torch.repeat_interleave(attn,repeats=repeats,dim=-1) #(B,N,C)
        res=attn*key+key
        return res,attn_c

class dens_attention(nn.Module):
    def __init__(self,emb_dim):
        super(dens_attention,self).__init__()
        self.emb_dim=emb_dim
        self.layer=nn.Sequential(nn.Conv1d(emb_dim,2*emb_dim,1),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(2*emb_dim)) #,
                                 # nn.Conv1d(2*emb_dim,4*emb_dim,1),
                                 # nn.ReLU(),
                                 # nn.BatchNorm1d(4*emb_dim))
    def forward(self,emb):
        key=emb.transpose(2,1)
        key=self.layer(key)
        global_emb=torch.max(key,dim=-2,keepdim=True)[0] #(B,1,N)
        global_emb=global_emb.transpose(2,1)
        global_emb=torch.softmax(global_emb,dim=-1)
        global_emb=torch.repeat_interleave(global_emb,emb.shape[-1],dim=-1)
        attn_res=emb*global_emb+emb
        return attn_res