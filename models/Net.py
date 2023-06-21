import torch
import torch.nn as nn
import math
from models.dgcnn import DGCNN_ori
from models.dgcnn import DGCNN_SDIM
from models.dgcnn import DGCNN_SDIM_no_BN
from models.gconv import Siamese_Gconv
from models.affinity_layer import Affinity
from models.affinity_layer import Modified_Affinity
from models.transformer import Transformer
import numpy as np 
from utils.config import cfg
import torch.nn.functional as F
from utils.Lepard_PE import VolumetricPositionEncoding
from models.attention import defect_attention,dens_attention
from models.adjmtx import AdjacencyMtx as edgegen
import argparse

def sinkhorn_rpm(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])   #non-correspondence right and bottom 

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


class RGM(nn.Module):
    def __init__(self):
        super(RGM, self).__init__()
        self.pointfeaturer = DGCNN_ori(cfg.PGM.FEATURES, 20, 512) #dgcnn
        self.gnn_layer = 2
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(512+ 512, 512)
            else:
                gnn_layer = Siamese_Gconv(512, 512)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(512))
            if cfg.PGM.USEATTEND == 'attentiontransformer':
                self.add_module('gmattend{}'.format(i), Transformer(2 * 512
                                                                    if i == 0 else 512))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(512 * 2, 512))
    # @profile
    def forward(self, P_src, P_tgt, A_src, A_tgt, ns_src, ns_tgt):
        # extract feature
        Node_src, Edge_src = self.pointfeaturer(P_src)
        Node_tgt, Edge_tgt = self.pointfeaturer(P_tgt)

        emb_src, emb_tgt = torch.cat((Node_src, Edge_src), dim=1).transpose(1, 2).contiguous(), \
                           torch.cat((Node_tgt, Edge_tgt), dim=1).transpose(1, 2).contiguous()
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            if True:
                gmattends_layer = getattr(self, 'gmattend{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tgt)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src1 = torch.softmax(scores_src, dim=-1)
                A_tgt1 = torch.softmax(scores_tgt, dim=-1)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            else:
                emb_src, emb_tgt = gnn_layer([A_src, emb_src], [A_tgt, emb_tgt])
            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb_src, emb_tgt)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb_src, torch.bmm(s, emb_tgt)), dim=-1))
                emb2_new = cross_graph(torch.cat((emb_tgt, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new
                emb_tgt = emb2_new

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, srcinlier_s, refinlier_s


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pointfeaturer = DGCNN(cfg.PGM.FEATURES, cfg.PGM.NEIGHBORSNUM, cfg.PGM.FEATURE_EDGE_CHANNEL)
        self.gnn_layer = cfg.PGM.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PGM.FEATURE_NODE_CHANNEL + cfg.PGM.FEATURE_EDGE_CHANNEL, cfg.PGM.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PGM.GNN_FEAT, cfg.PGM.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PGM.GNN_FEAT))
            #if cfg.PGM.USEATTEND == 'attentiontransformer':
            self.add_module('gmattend{}'.format(i), Transformer(2 * cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                    if i == 0 else cfg.PGM.GNN_FEAT))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PGM.GNN_FEAT * 2, cfg.PGM.GNN_FEAT))
    # @profile
    def forward(self, P_src, P_tgt, A_src, A_tgt, ns_src, ns_tgt):
        # extract feature
        Node_src, Edge_src = self.pointfeaturer(P_src)
        Node_tgt, Edge_tgt = self.pointfeaturer(P_tgt)

        emb_src, emb_tgt = torch.cat((Node_src, Edge_src), dim=1).transpose(1, 2).contiguous(), \
                           torch.cat((Node_tgt, Edge_tgt), dim=1).transpose(1, 2).contiguous()
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            if cfg.PGM.USEATTEND == 'attentiontransformer':
                gmattends_layer = getattr(self, 'gmattend{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tgt)   #self-attention
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src1 = torch.softmax(scores_src, dim=-1)
                A_tgt1 = torch.softmax(scores_tgt, dim=-1)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])  #(B,1024,512)
            else:
                emb_src, emb_tgt = gnn_layer([A_src, emb_src], [A_tgt, emb_tgt])   #self-gconv
            affinity = getattr(self, 'affinity_{}'.format(i))            #graph 
            s = affinity(emb_src, emb_tgt)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=cfg.PGM.SKADDCR)
            s = torch.exp(log_s)   #correspondence matrix

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))             #cross-conv
                emb1_new = cross_graph(torch.cat((emb_src, torch.bmm(s, emb_tgt)), dim=-1))
                emb2_new = cross_graph(torch.cat((emb_tgt, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tgt = emb2_new.clone()

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, srcinlier_s, refinlier_s
###############################################################################
class modified_Net0(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=3,
                 neighbor_sum=[20,10,5],
                 embed_dims=[[6,64,64,128,256],[512,128,64,32,32],[256,64,32,16,16]],   
                 transformer_dims=[512,256,128],
                 graph_embed_dims=[512,256,128]):
        super(modified_Net0,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
            dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
            self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))
            

    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        slct_src=[]
        slct_tar=[]
        for i in range(self.ds_layer):
            pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
            src_pfeat,src_efeat=pe_extrctr(emb_src)
            tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
            emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                            torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
                
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)

            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            affinity = getattr(self, 'affinity_{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            # InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            # s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            aff_mtx = s.clone()
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)                                              #correspondence matrix

            cross_conv=getattr(self,'cross_conv{}'.format(i))
            emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
            emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
            emb_src = emb1_new.clone()
            emb_tar = emb2_new.clone()

            if not i==self.ds_layer-1:
                temp_src = torch.softmax(emb1_new,dim=1)
                temp_src = torch.sum(temp_src,dim=-1)
                src_idx = torch.topk(temp_src,k=self.transformer_dims[i],dim=-1)[1]
                src_idx = torch.unsqueeze(src_idx,dim=-1)
                slct_src.append(src_idx)
                src_idx = torch.repeat_interleave(src_idx,emb_src.shape[-1],dim=-1)
                emb_src = torch.gather(emb_src,1,src_idx)

                temp_tar = torch.softmax(emb2_new,dim=1)
                temp_tar = torch.sum(temp_tar,dim=-1)
                tar_idx = torch.topk(temp_tar,k=self.transformer_dims[i],dim=-1)[1]
                tar_idx = torch.unsqueeze(tar_idx,dim=-1)
                slct_tar.append(tar_idx)
                tar_idx = torch.repeat_interleave(tar_idx,emb_tar.shape[-1],dim=-1)
                emb_tar = torch.gather(emb_tar,1,tar_idx)

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, srcinlier_s, refinlier_s, slct_src, slct_tar

###############################################################################
class modified_Net1(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=3,
                 neighbor_sum=[20,10,5],
                 embed_dims=[[6,64,64,128,256],[512,128,64,32,32],[256,64,32,16,16]],   
                 transformer_dims=[512,256,128],
                 graph_embed_dims=[512,256,128]):
        super(modified_Net1,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if True:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            # dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
            #                                                 self.embed_dims[i],ori)
            # self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if not i==0:
                # cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i]*2)
                # self.add_model('gconv0_{}'.format(i),cur_gnn_layer)
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))
            

    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        slct_src=[]
        slct_tar=[]
        for i in range(self.ds_layer):
            if True:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
                
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)

            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            affinity = getattr(self, 'affinity_{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            aff_mtx = s.clone()
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)                                              #correspondence matrix

            cross_conv=getattr(self,'cross_conv{}'.format(i))
            emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
            emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
            emb_src = emb1_new.clone()
            emb_tar = emb2_new.clone()

            if not i==self.ds_layer-1:
                s1 = torch.sum(s,dim=-1)
                s1 = torch.unsqueeze(s1,dim=-1)
                s1 = torch.repeat_interleave(s1,emb_src.shape[-1],dim=-1)
                emb1_new = s1 * emb1_new
                temp_src = torch.softmax(emb1_new,dim=1)
                temp_src = torch.sum(temp_src,dim=-1)
                src_idx = torch.topk(temp_src,k=self.transformer_dims[i],dim=-1)[1]
                src_idx = torch.unsqueeze(src_idx,dim=-1)
                slct_src.append(src_idx)
                src_idx = torch.repeat_interleave(src_idx,emb_src.shape[-1],dim=-1)
                emb_src = torch.gather(emb_src,1,src_idx)

                s2 = torch.sum(s,dim=-2)
                s2 = torch.unsqueeze(s2,dim=-1)
                s2 = torch.repeat_interleave(s2,emb_tar.shape[-1],dim=-1)
                emb2_new = s2 * emb2_new
                temp_tar = torch.softmax(emb2_new,dim=1)
                temp_tar = torch.sum(temp_tar,dim=-1)
                tar_idx = torch.topk(temp_tar,k=self.transformer_dims[i],dim=-1)[1]
                tar_idx = torch.unsqueeze(tar_idx,dim=-1)
                slct_tar.append(tar_idx)
                tar_idx = torch.repeat_interleave(tar_idx,emb_tar.shape[-1],dim=-1)
                emb_tar = torch.gather(emb_tar,1,tar_idx)

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, srcinlier_s, refinlier_s, slct_src, slct_tar

###############################################################################
class modified_Net2(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,10],
                 embed_dims=[[6,64,64,128,256]], #,[512,128,64,32,32],[256,64,32,16,16]],   
                 transformer_dims=[512,256],         # ,256,128],
                 graph_embed_dims=[512,256]):        #,256,128]):
        super(modified_Net2,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            # dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
            #                                                 self.embed_dims[i],ori)
            # self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if not i==0:
                # cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i]*2)
                # self.add_model('gconv0_{}'.format(i),cur_gnn_layer)
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            self.add_module('affinity_layer{}'.format(i), Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))
        # self.rpoints_rate=torch.Tensor([0.5,0.25,0.25])
            

    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        rpoints=N2-N1  #missing points num
        
        slct_src=[]
        slct_tar=[]
        mpoints=torch.zeros(0,rpoints,6).to(device)
        cur_tgt=P_tgt.clone()
        for i in range(self.ds_layer):
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
                
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)

            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)                                              #correspondence matrix

            cross_conv=getattr(self,'cross_conv{}'.format(i))
            emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
            emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
            emb_src = emb1_new.clone()
            emb_tar = emb2_new.clone()
            
            if not i:
                mpoints_idx_temp=torch.sum(s,dim=-2)     
                rpoints_idx_temp=torch.topk(mpoints_idx_temp,k=N2-rpoints,largest=True)[1]    #remained points 
                mpoints_idx_temp=torch.topk(mpoints_idx_temp,k=rpoints,largest=False)[1]      #missing points 
                rpoints_idx_temp=torch.unsqueeze(rpoints_idx_temp,dim=-1)
                mpoints_idx_temp=torch.unsqueeze(mpoints_idx_temp,dim=-1)
                rpoints_idx_temp1=torch.repeat_interleave(rpoints_idx_temp,cur_tgt.shape[-1],dim=-1)
                rpoints_idx_temp2=torch.repeat_interleave(rpoints_idx_temp,emb_tar.shape[-1],dim=-1)
                mpoints_idx_temp=torch.repeat_interleave(mpoints_idx_temp,cur_tgt.shape[-1],dim=-1)
                mpoints_temp=torch.gather(cur_tgt,1,mpoints_idx_temp)
                if not i:
                    mpoints=torch.cat([mpoints,mpoints_temp],dim=0)                       #missing points index
                else:
                    mpoints=torch.cat([mpoints,mpoints_temp],dim=1)  
                cur_tgt=torch.gather(cur_tgt,1,rpoints_idx_temp1)                   #delete missing points from input
            
            if not i==self.ds_layer-1:
                temp_src = torch.softmax(emb1_new,dim=1)
                temp_src = torch.sum(temp_src,dim=-1)
                src_idx = torch.topk(temp_src,k=self.transformer_dims[i],dim=-1)[1]
                src_idx = torch.unsqueeze(src_idx,dim=-1)
                slct_src.append(src_idx)
                src_idx = torch.repeat_interleave(src_idx,emb_src.shape[-1],dim=-1)
                emb_src = torch.gather(emb_src,1,src_idx)

                if not i:
                    emb2_new = torch.gather(emb2_new,1,rpoints_idx_temp2)                  #delete missing points from features
                temp_tar = torch.softmax(emb2_new,dim=1)
                temp_tar = torch.sum(temp_tar,dim=-1)
                tar_idx = torch.topk(temp_tar,k=self.transformer_dims[i],dim=-1)[1]
                tar_idx = torch.unsqueeze(tar_idx,dim=-1)
                slct_tar.append(tar_idx)
                tar_idx1 = torch.repeat_interleave(tar_idx,cur_tgt.shape[-1],dim=-1)
                tar_idx2 = torch.repeat_interleave(tar_idx,emb_tar.shape[-1],dim=-1)
                cur_tgt = torch.gather(cur_tgt,1,tar_idx1)
                emb_tar = torch.gather(emb_tar,1,tar_idx2)
            

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, srcinlier_s, refinlier_s, slct_src, slct_tar, mpoints

###############################################################################
#parital2full not reasonable
class modified_Net3(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,15],
                 embed_dims=[[6,64,64,128,256]], #,[512,128,64,32,32],[256,64,32,16,16]],   
                 transformer_dims=[512,256],         #,256,128],
                 graph_embed_dims=[512,256]):        #,256,128]):
        super(modified_Net3,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if not i==0:
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            if i==0:
                self.add_module('dattn_{}'.format(i),defect_attention(graph_embed_dims[i]))
            self.add_module('affinity_layer{}'.format(i), Modified_Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i==0:
                self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))


    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        rpoints=N2-N1  #missing points num
        
        slct_src=[]
        slct_tar=[]
        mpoints=torch.zeros(0,rpoints,6).to(device)
        cur_tgt=P_tgt.clone()
        for i in range(self.ds_layer):
            #DGCNN
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
            #Transformer
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)

            #GConv
            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix
            
            if i==0:
                dattn=getattr(self,'dattn_{}'.format(i))
                _,attn=dattn(query=emb_src,key=emb_tar)

            #AIS Module
            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            if i==0:
                s = s * attn.transpose(2,1)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)                                              #correspondence matrix

            #CrossConv
            if not i:
                cross_conv=getattr(self,'cross_conv{}'.format(i))
                emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
                emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tar = emb2_new.clone()
            
            #Non-overlapeed area recognition
            if not i:
                mpoints_idx_temp=torch.sum(s,dim=-2)     
                rpoints_idx_temp=torch.topk(mpoints_idx_temp,k=N2-rpoints,largest=True)[1]    #remained points 
                mpoints_idx_temp=torch.topk(mpoints_idx_temp,k=rpoints,largest=False)[1]      #missing points 
                rpoints_idx_temp=torch.unsqueeze(rpoints_idx_temp,dim=-1)
                mpoints_idx_temp=torch.unsqueeze(mpoints_idx_temp,dim=-1)
                rpoints_idx_temp1=torch.repeat_interleave(rpoints_idx_temp,cur_tgt.shape[-1],dim=-1)
                rpoints_idx_temp2=torch.repeat_interleave(rpoints_idx_temp,emb_tar.shape[-1],dim=-1)
                mpoints_idx_temp=torch.repeat_interleave(mpoints_idx_temp,cur_tgt.shape[-1],dim=-1)
                mpoints_temp=torch.gather(cur_tgt,1,mpoints_idx_temp)
                mpoints=torch.cat([mpoints,mpoints_temp],dim=0)
                cur_tgt=torch.gather(cur_tgt,1,rpoints_idx_temp1)           #delete missing points from input
            
            #Sampling
            if not i==self.ds_layer-1:
                temp_src = torch.softmax(emb1_new,dim=1)
                temp_src = torch.sum(temp_src,dim=-1)
                src_idx = torch.topk(temp_src,k=self.transformer_dims[i],dim=-1)[1]
                src_idx = torch.unsqueeze(src_idx,dim=-1)
                slct_src.append(src_idx)
                src_idx = torch.repeat_interleave(src_idx,emb_src.shape[-1],dim=-1)
                emb_src = torch.gather(emb_src,1,src_idx)

                if not i:
                    emb2_new = torch.gather(emb2_new,1,rpoints_idx_temp2) #delete missing points from features
                temp_tar = torch.softmax(emb2_new,dim=1)
                temp_tar = torch.sum(temp_tar,dim=-1)
                tar_idx = torch.topk(temp_tar,k=self.transformer_dims[i],dim=-1)[1]
                tar_idx = torch.unsqueeze(tar_idx,dim=-1)
                slct_tar.append(tar_idx)
                tar_idx1 = torch.repeat_interleave(tar_idx,cur_tgt.shape[-1],dim=-1)
                tar_idx2 = torch.repeat_interleave(tar_idx,emb_tar.shape[-1],dim=-1)
                cur_tgt = torch.gather(cur_tgt,1,tar_idx1)
                emb_tar = torch.gather(emb_tar,1,tar_idx2)
            

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, srcinlier_s, refinlier_s, slct_src, slct_tar, mpoints

###############################################################################
#partial2partial
class modified_Net4(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,15],
                 embed_dims=[[6,64,64,128,256]], 
                 transformer_dims=[512,256],        
                 graph_embed_dims=[512,256]):   
        super(modified_Net4,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if not i==0:
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            if not i==0:   #not i==0
                self.add_module('QGAB_{}'.format(i),QGAB(graph_embed_dims[i]))
            self.add_module('affinity_layer{}'.format(i), Modified_Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i==0:
                self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))


    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        
        for i in range(self.ds_layer):
            #DGCNN
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
            #Transformer
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)
                

            #GConv
            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            if not i==0:  #not i==0
                dattn=getattr(self,'QGAB_{}'.format(i))
                _,attn=dattn(query=emb_src,key=emb_tar)

            #########################################################
            #AIS Module    
            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            if not i==0:
                s = s * attn.transpose(2,1)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)                                              #correspondence matrix
            
            ########################################################
            if not i:
                cross_conv=getattr(self,'cross_conv{}'.format(i))
                emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
                emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tar = emb2_new.clone()
            #########################################################

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, srcinlier_s, refinlier_s

###############################################################################
#partial2partial
class modified_Net5(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,15],
                 embed_dims=[[6,64,64,128,256]], #,[512,128,64,32,32],[256,64,32,16,16]],   
                 transformer_dims=[512,256],         #,256,128],
                 graph_embed_dims=[512,256]):        #,256,128]):
        super(modified_Net5,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if not i==0:
                self.add_module('edgeGen_{}'.format(i),edgegen(graph_embed_dims[i]*2,512))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            if not i==0:   #not i==0
                self.add_module('dattn_{}'.format(i),defect_attention(graph_embed_dims[i]))
            self.add_module('affinity_layer{}'.format(i), Modified_Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i==0:
                self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))


    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        
        for i in range(self.ds_layer):
            #DGCNN
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
            #Transformer
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)
        
            #GConv
            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            if not i==0:  #not i==0
                dattn=getattr(self,'dattn_{}'.format(i))
                _,attn=dattn(query=emb_src,key=emb_tar)

            #AIS Module
            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            if not i==0:
                s = s * attn.transpose(2,1)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)                                              #correspondence matrix

            #CrossConv
            if not i:
                cross_conv=getattr(self,'cross_conv{}'.format(i))
                emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
                emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tar = emb2_new.clone()

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, srcinlier_s, refinlier_s

###############################################################################
#partial2partial
class modified_Net6(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,15],
                 embed_dims=[[6,64,64,128,256]], 
                 transformer_dims=[512,256],        
                 graph_embed_dims=[512,256]):   
        super(modified_Net6,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if i==0:   #not i==0
                self.add_module('dattn_{}'.format(i),defect_attention(graph_embed_dims[i]))
                self.add_module('sattn_{}'.format(i),dens_attention(graph_embed_dims[i]))
            if not i==0:
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            self.add_module('affinity_layer{}'.format(i), Modified_Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i==0:
                self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))


    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        
        for i in range(self.ds_layer):
            #DGCNN
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
            #Transformer
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)
                

            #GConv
            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            if i==0:  #not i==0
                dattn=getattr(self,'dattn_{}'.format(i))
                emb_tar,_=dattn(query=emb_src,key=emb_tar)
                sattn=getattr(self,'sattn_{}'.format(i))
                emb_src=sattn(emb_src)

            #AIS Module
            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)      
            if i==0:
                s1=s.clone()

            #CrossConv
            if not i:
                cross_conv=getattr(self,'cross_conv{}'.format(i))
                emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
                emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tar = emb2_new.clone()

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, s1, srcinlier_s, refinlier_s

###############################################################################
#partial2partial
class modified_Net7(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,15],
                 embed_dims=[[6,64,64,128,256]], 
                 transformer_dims=[512,256],        
                 graph_embed_dims=[512,256]):   
        super(modified_Net7,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if i==0:   #not i==0
                self.add_module('dattn_{}'.format(i),defect_attention(graph_embed_dims[i]))
                self.add_module('sattn_{}'.format(i),dens_attention(graph_embed_dims[i]))
            if not i==0:
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            self.add_module('affinity_layer{}'.format(i), Modified_Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i==0:
                self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))


    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        
        for i in range(self.ds_layer):
            #DGCNN
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
            #Transformer
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)
                

            #GConv
            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            if i==0:  #not i==0
                dattn=getattr(self,'dattn_{}'.format(i))
                emb_tar,_=dattn(query=emb_src,key=emb_tar)
                sattn=getattr(self,'sattn_{}'.format(i))
                emb_src=sattn(emb_src)

            #AIS Module
            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)      
            if i==0:
                s1=s.clone()

            #CrossConv
            if not i:
                cross_conv=getattr(self,'cross_conv{}'.format(i))
                emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
                emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tar = emb2_new.clone()

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, s.clone(), srcinlier_s, refinlier_s

###############################################################################
#partial2partial
class modified_Net8(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,15],
                 embed_dims=[[6,64,64,128,256]], 
                 transformer_dims=[512,256],        
                 graph_embed_dims=[512,256]):   
        super(modified_Net8,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if i==0:   #not i==0
                self.add_module('dattn_{}'.format(i),defect_attention(graph_embed_dims[i]))
                self.add_module('sattn_{}'.format(i),dens_attention(graph_embed_dims[i]))
            if not i==0:
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            self.add_module('affinity_layer{}'.format(i), Modified_Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i==0:
                self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))


    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        
        for i in range(self.ds_layer):
            #DGCNN
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
            #Transformer
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)
                

            #GConv
            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            if i==0:  #not i==0
                dattn=getattr(self,'dattn_{}'.format(i))
                emb_tar,_=dattn(query=emb_src,key=emb_tar)
                sattn=getattr(self,'sattn_{}'.format(i))
                emb_src=sattn(emb_src)

            #AIS Module
            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)      
            if i==0:
                s1=s.clone()

            #CrossConv
            if not i:
                cross_conv=getattr(self,'cross_conv{}'.format(i))
                emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
                emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tar = emb2_new.clone()

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, s1, srcinlier_s, refinlier_s
    
###############################################################################
#partial2partial
class modified_Net9(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,15],
                 embed_dims=[[6,64,64,128,256]], 
                 transformer_dims=[512,256],        
                 graph_embed_dims=[512,256]):   
        super(modified_Net9,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if not i==0:
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            self.add_module('affinity_layer{}'.format(i), Modified_Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i==0:
                self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))


    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        
        for i in range(self.ds_layer):
            #DGCNN
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
            #Transformer
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)
                

            #GConv
            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix


            #AIS Module
            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)      
            if i==0:
                s1=s.clone()

            #CrossConv
            if not i:
                cross_conv=getattr(self,'cross_conv{}'.format(i))
                emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
                emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tar = emb2_new.clone()

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, s1, srcinlier_s, refinlier_s
    
###############################################################################
#partial2partial
class modified_Net10(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,15],
                 embed_dims=[[6,64,64,128,256]], 
                 transformer_dims=[512,256],        
                 graph_embed_dims=[512,256]):   
        super(modified_Net10,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if not i==0:
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            self.add_module('affinity_layer{}'.format(i), Modified_Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i==0:
                self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))


    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        
        for i in range(self.ds_layer):
            #DGCNN
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
            #Transformer
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)
                

            #GConv
            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            #AIS Module
            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)  

            #CrossConv
            if not i:
                cross_conv=getattr(self,'cross_conv{}'.format(i))
                emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
                emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tar = emb2_new.clone()

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, None, srcinlier_s, refinlier_s
    
###############################################################################
#partial2partial
class modified_Net11(nn.Module):
    def __init__(self,
                 point_n=None,
                 DS=2,
                 neighbor_sum=[20,15],
                 embed_dims=[[6,64,64,128,256]], 
                 transformer_dims=[512,256],        
                 graph_embed_dims=[512,256]):   
        super(modified_Net11,self).__init__()
        self.ds_layer=DS   #downsample and clustering block
        self.point_n=point_n
        self.feat='gxyz'
        self.neighbor_sum=neighbor_sum
        self.embed_dims=np.array(embed_dims)
        self.graph_embed_dims=graph_embed_dims
        self.transformer_dims=transformer_dims
        for i in range(self.ds_layer):
            ori=False
            if i==0:
                ori=True
                dgcnn_sdim=DGCNN_SDIM(self.feat,self.neighbor_sum[i],self.graph_embed_dims[i],    #dgcnn with specific dimension
                                                            self.embed_dims[i],ori)
                self.add_module('dgcnn_{}'.format(i),dgcnn_sdim)
            if i==0:   #not i==0
                self.add_module('dattn_{}'.format(i),defect_attention(graph_embed_dims[i]))
                self.add_module('sattn_{}'.format(i),dens_attention(graph_embed_dims[i]))
            if not i==0:
                self.add_module('edgeGen_{}'.format(i),Transformer(self.transformer_dims[i]*2))
            cur_gnn_layer=Siamese_Gconv(self.transformer_dims[i]*2,self.transformer_dims[i])
            self.add_module('gconv_{}'.format(i),cur_gnn_layer)
            self.add_module('affinity_layer{}'.format(i), Modified_Affinity(self.graph_embed_dims[i]))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i==0:
                self.add_module('cross_conv{}'.format(i),nn.Linear(self.graph_embed_dims[i]*2,self.graph_embed_dims[i]))


    def forward(self,P_src, P_tgt, A_src, A_tgt, ns_src=None, ns_tgt=None):
        device=P_src.device
        emb_src=P_src.clone()
        emb_tar=P_tgt.clone()
        _,N1,_=P_src.shape
        _,N2,_=P_tgt.shape
        
        for i in range(self.ds_layer):
            #DGCNN
            if i==0:
                pe_extrctr=getattr(self,'dgcnn_{}'.format(i))
                src_pfeat,src_efeat=pe_extrctr(emb_src)
                tar_pfeat,tar_efeat=pe_extrctr(emb_tar)
                emb_src,emb_tar=torch.cat((src_pfeat,src_efeat),dim=1).transpose(1,2).contiguous(),\
                                torch.cat((tar_pfeat,tar_efeat),dim=1).transpose(1,2).contiguous() 
            #Transformer
            if not i==0:
                gmattends_layer = getattr(self, 'edgeGen_{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tar)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src = torch.softmax(scores_src, dim=-1)
                A_tgt = torch.softmax(scores_tgt, dim=-1)
                

            #GConv
            gnn_layer=getattr(self,'gconv_{}'.format(i))
            emb_src,emb_tar=gnn_layer([A_src,emb_src],[A_tgt,emb_tar])   #adjacency matrix

            if i==0:  #not i==0
                dattn=getattr(self,'dattn_{}'.format(i))
                emb_tar,_=dattn(query=emb_src,key=emb_tar)
                sattn=getattr(self,'sattn_{}'.format(i))
                emb_src=sattn(emb_src)

            #AIS Module
            affinity = getattr(self, 'affinity_layer{}'.format(i))            #affinity matrix between graphs   
            s = affinity(emb_src, emb_tar)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=True)
            s = torch.exp(log_s)  

            #CrossConv
            if not i:
                cross_conv=getattr(self,'cross_conv{}'.format(i))
                emb1_new = cross_conv(torch.cat((emb_src, torch.bmm(s, emb_tar)), dim=-1))
                emb2_new = cross_conv(torch.cat((emb_tar, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new.clone()
                emb_tar = emb2_new.clone()

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, None, srcinlier_s, refinlier_s
    
    
    
#####################################################################
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

from common.torch import to_numpy
from models.pointnet_util import square_distance, angle_difference
from models.feature_nets import FeatExtractionEarlyFusion
from models.feature_nets import ParameterPredictionNet
# from models.feature_nets import ParameterPredictionNetConstant as ParameterPredictionNet
from common.math_torch import se3

_logger = logging.getLogger(__name__)

_EPS = 1e-5  # To prevent division by zero


def match_features(feat_src, feat_ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
         in the src agrees with every point in the ref.
    """
    assert feat_src.shape[-1] == feat_ref.shape[-1]

    if metric == 'l2':
        dist_matrix = square_distance(feat_src, feat_ref)
    elif metric == 'angle':
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError

    return dist_matrix


def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform


class RPMNet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.add_slack = not args.no_slack
        self.num_sk_iter = args.num_sk_iter

    def compute_affinity(self, beta, feat_distance, alpha=0.5):
        """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
        if isinstance(alpha, float):
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
        else:
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
        return hybrid_affinity

    def forward(self, data, num_iter: int = 1):
        """Forward pass for RPMNet

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, 6)
                    'points_ref': Reference points (B, K, 6)
            num_iter (int): Number of iterations. Recommended to be 2 for training

        Returns:
            transform: Transform to apply to source points such that they align to reference
            src_transformed: Transformed source points
        """
        endpoints = {}

        xyz_ref, norm_ref = data['points_ref'][:, :, :3], data['points_ref'][:, :, 3:6]
        xyz_src, norm_src = data['points_src'][:, :, :3], data['points_src'][:, :, 3:6]
        xyz_src_t, norm_src_t = xyz_src, norm_src

        transforms = []
        all_gamma, all_perm_matrices, all_weighted_ref = [], [], []
        all_beta, all_alpha = [], []
        for i in range(num_iter):

            beta, alpha = self.weights_net([xyz_src_t, xyz_ref])
            feat_src = self.feat_extractor(xyz_src_t, norm_src_t)
            feat_ref = self.feat_extractor(xyz_ref, norm_ref)

            feat_distance = match_features(feat_src, feat_ref)
            affinity = self.compute_affinity(beta, feat_distance, alpha=alpha)

            # Compute weighted coordinates
            log_perm_matrix = sinkhorn(affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
            perm_matrix = torch.exp(log_perm_matrix)
            weighted_ref = perm_matrix @ xyz_ref / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)

            # Compute transform and transform points
            transform = compute_rigid_transform(xyz_src, weighted_ref, weights=torch.sum(perm_matrix, dim=2))
            xyz_src_t, norm_src_t = se3.transform(transform.detach(), xyz_src, norm_src)

            transforms.append(transform)
            all_gamma.append(torch.exp(affinity))
            all_perm_matrices.append(perm_matrix)
            all_weighted_ref.append(weighted_ref)
            all_beta.append(to_numpy(beta))
            all_alpha.append(to_numpy(alpha))

        endpoints['perm_matrices_init'] = all_gamma
        endpoints['perm_matrices'] = all_perm_matrices
        endpoints['weighted_ref'] = all_weighted_ref
        endpoints['beta'] = np.stack(all_beta, axis=0)
        endpoints['alpha'] = np.stack(all_alpha, axis=0)

        return transforms, endpoints


class RPMNetEarlyFusion(RPMNet):
    """Early fusion implementation of RPMNet, as described in the paper"""
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.weights_net = ParameterPredictionNet(weights_dim=[0])
        self.feat_extractor = FeatExtractionEarlyFusion(
            features=args.features, feature_dim=args.feat_dim,
            radius=args.radius, num_neighbors=args.num_neighbors)


def get_model() -> RPMNet:
    args=argparse.Namespace(logdir='../logs', 
               dev=False, 
               name=None, 
               debug=False, 
               dataset_path='../datasets/modelnet40_ply_hdf5_2048', 
               dataset_type='modelnet_hdf', 
               num_points=1024, 
               noise_type='crop', 
               rot_mag=45.0, 
               trans_mag=0.5, 
               partial=[0.7], 
               method='rpmnet', 
               radius=0.3, 
               num_neighbors=64, 
               features=['ppf', 'dxyz', 'xyz'], 
               feat_dim=96, 
               no_slack=False, 
               num_sk_iter=5, 
               num_reg_iter=5, 
               loss_type='mae', 
               wt_inliers=0.01, 
               train_batch_size=8, 
               val_batch_size=16, 
               resume=None, 
               gpu=0, 
               train_categoryfile='./data_loader/modelnet40_half1.txt', 
               val_categoryfile='./data_loader/modelnet40_half1.txt', 
               lr=0.0001, 
               epochs=1000, 
               summary_every=200, 
               validate_every=-4, 
               num_workers=0, 
               num_train_reg_iter=2)
    return RPMNetEarlyFusion(args)
#####################################################################