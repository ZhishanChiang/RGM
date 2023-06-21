import torch
import time
from pathlib import Path
import numpy as np

from data.data_loader import get_dataloader, get_datasets
from utils.config import cfg
from utils.model_sl import load_model
from utils.hungarian import hungarian
from utils.loss_func import PermLoss
from utils.evaluation_metric import matching_accuracy, calcorrespondpc, summarize_metrics, print_metrics, compute_metrics, compute_transform
from parallel import DataParallel
from models.correspondSlover import SVDslover, RANSACSVDslover
from models.Net import get_model
from collections import defaultdict
from torch.nn import BCELoss as BCE
from datetime import datetime
import os
from pytorch3d.loss import chamfer_distance
from utils.chamfer_distance.chamfer_distance import ChamferDistance as chamfer_dist
from utils.ShapeMeasure.distance import EMDLoss,ChamferLoss 
from torch.nn import Softmax
from torch.nn import functional as F
# from lab_test import refine_registration
import open3d
from scipy.spatial.transform import Rotation as Rot
import copy
import sys
import math as m
import open3d as o3d
from torch import nn
from tqdm import tqdm
from utils.se3 import transform
from common import torch as torch1
from models.Net import modified_Net8
from utils.chamfer_distance.chamfer_distance import ChamferDistance as chamfer_dist0

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return -np.array((alpha, beta, gamma))
def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)
##################
#rotate/shift/shuffle point cloud
def rotate_point_cloud_by_angle(pc, alpha, beta, gamma):
    cos_a=np.cos(alpha)
    sin_a=np.sin(alpha)
    cos_b=np.cos(beta)
    sin_b=np.sin(beta)
    cos_g=np.cos(gamma)
    sin_g=np.sin(gamma)
    rotation_matrix = np.array([[cos_a*cos_b, cos_a*sin_b*sin_g-sin_a*sin_g, cos_a*sin_b*cos_g+sin_a*sin_g],
                                [sin_a*cos_b, sin_a*sin_b*sin_g+cos_a*cos_g, sin_a*sin_b*cos_g-cos_a*sin_g],
                                [-sin_b, cos_b*sin_g, cos_b*cos_g]])
    pc = np.matmul(pc, rotation_matrix)
    return {"pc":pc,"R":rotation_matrix}

def shift_point_cloud(pc,shift_x,shift_y,shift_z):
    pc=torch.Tensor(pc)
    pc[:,0]=pc[:,0]+shift_x
    pc[:,1]=pc[:,1]+shift_x
    pc[:,2]=pc[:,2]+shift_x
    return pc

def shuffle_points(pc):
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    idx1 = np.expand_dims(np.arange(pc.shape[0]),axis=-1)
    idx2 = np.expand_dims(idx,axis=-1)
    scatter = np.concatenate([idx1,idx2],axis=-1)
    perm = np.zeros(shape=(pc.shape[0],pc.shape[0]))
    perm[tuple(scatter.T)]=1
    return pc[idx,:],perm

def pca_compute(data):
    [center, covariance] = data.compute_mean_and_covariance()
    # SVD奇异值分解，得到covariance矩阵的特征值和特征向量
    eigenvectors, _, _ = np.linalg.svd(covariance)
    return eigenvectors, center

def pca_registration(P, X):

    P.paint_uniform_color([1, 0, 0])  # 给原始点云赋色
    X.paint_uniform_color([0, 1, 0])

    error = []   # 定义误差集合
    matrax = []  # 定义变换矩阵集合
    Up, Cp = pca_compute(P)  # PCA方法得到P对应的特征向量、点云质心
    Ux, Cx = pca_compute(X)  # PCA方法得到X对应的特征向量、点云质心
    # 主轴对应可能出现的情况
    Upcopy = Up
    sign1 = [1, -1, 1, 1, -1, -1, 1, -1]
    sign2 = [1, 1, -1, 1, -1, 1, -1, -1]
    sign3 = [1, 1, 1, -1, 1, -1, -1, -1]
    for nn in range(len(sign3)):
        Up[0] = sign1[nn]*Upcopy[0]
        Up[1] = sign2[nn]*Upcopy[1]
        Up[2] = sign3[nn]*Upcopy[2]
        R0 = np.dot(Ux, np.linalg.inv(Up))
        T0 = Cx-np.dot(R0, Cp)
        T = np.eye(4)
        T[:3, :3] = R0
        T[:3, 3] = T0
        T[3, 3] = 1
# 计算配准误差，误差最小时对应的变换矩阵即为最终完成配准的变换
        trans = copy.deepcopy(P).transform(T)
        dists = trans.compute_point_cloud_distance(X)
        dists = np.asarray(dists)  # 欧氏距离（单位是：米）
        mse = np.average(dists)
        error.append(mse)
        matrax.append(T)
    ind = error.index(min(error))  # 获取误差最小时对应的索引
    final_T = matrax[ind]  # 根据索引获取变换矩阵
    pcaregisted = copy.deepcopy(P).transform(final_T)
    return pcaregisted,final_T  # 返回配准后的点云

def Euler2R(euler=None):
    theta=euler[0]*np.pi/180.0
    alpha=euler[1]*np.pi/180.0
    beta=euler[2]*np.pi/180.0
    r = Rot.from_rotvec(np.array([theta,alpha,beta]))
    return r.as_matrix()

def pca(src,tar,init_R,init_T):
    pcd_src=open3d.geometry.PointCloud()
    pcd_src.points=open3d.utility.Vector3dVector(src)
    init_R=Euler2R(init_R)
    pcd_src=pcd_src.rotate(init_R)
    pcd_src=pcd_src.translate(init_T)
    pcd_tar=open3d.geometry.PointCloud()
    pcd_tar.points=open3d.utility.Vector3dVector(tar)
    transformed_src,T=pca_registration(pcd_src,pcd_tar)
    tol = sys.float_info.epsilon * 10
    pred_T=np.array([T[0,2],T[1,2],T[2,2]])   
    T=np.matmul(T[:3,:3],init_R)
    #eul1 phi  eul2 theta  eul3 psi
    eul1 = m.atan2(T[1, 0], T[0, 0])
    if eul1<-np.pi/2:
        eul1=np.pi+eul1
    sp = m.sin(eul1)
    cp = m.cos(eul1)
    eul2 = m.atan2(-T[2, 0], cp * T[0, 0] + sp * T[1, 0])
    eul3 = m.atan2(sp * T[0, 2] - cp * T[1, 2], cp * T[1, 1] - sp * T[0, 1])
    pred_euler=np.array([eul3,eul2,eul1])*180.0/np.pi 
    return -pred_euler,pred_T

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def prepare_dataset(voxel_size,o3d_src,o3d_tar):
    source = o3d_src
    target = o3d_tar
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def caliters_perm(model, P1_gt_copy, P2_gt_copy, A1_gt, A2_gt, n1_gt, n2_gt, estimate_iters): #, mask):
    lap_solver1 = hungarian
    s_perm_indexs = []
    for estimate_iter in range(estimate_iters):
        #####full
        # s_prem_i, Inlier_src_pre, Inlier_ref_pre, slct_src, slct_tar, mpoints_pre  = model(P1_gt_copy, P2_gt_copy,
        #                                                  A1_gt, A2_gt, n1_gt, n2_gt)
        s_prem_i, Inlier_src_pre, Inlier_ref_pre= model(P1_gt_copy, P2_gt_copy,
                                                          A1_gt, A2_gt, n1_gt, n2_gt)
        if cfg.PGM.USEINLIERRATE:
            s_prem_i = Inlier_src_pre * s_prem_i * Inlier_ref_pre.transpose(2, 1).contiguous()
        s_perm_i_mat = lap_solver1(s_prem_i, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
        P2_gt_copy1, s_perm_i_mat_index = calcorrespondpc(s_perm_i_mat, P2_gt_copy)
        s_perm_indexs.append(s_perm_i_mat_index)
        if cfg.EXPERIMENT.USERANSAC:
            R_pre, T_pre, s_perm_i_mat = RANSACSVDslover(P1_gt_copy[:,:,:3], P2_gt_copy1[:,:,:3], s_perm_i_mat)
        else:
            R_pre, T_pre = SVDslover(P1_gt_copy[:,:,:3], P2_gt_copy1[:,:,:3], s_perm_i_mat)
        P1_gt_copy[:,:,:3] = torch.bmm(P1_gt_copy[:,:,:3], R_pre.transpose(2, 1).contiguous()) + T_pre[:, None, :]
        P1_gt_copy[:,:,3:6] = P1_gt_copy[:,:,3:6] @ R_pre.transpose(-1, -2)
    return s_perm_i_mat


def eval_model(model, dataloader, model_path=None, eval_epoch=None, metric_is_save=False, estimate_iters=1,
               viz=None, usepgm=True, userefine=False, save_filetime='time'):
    print('-----------------Start evaluation-----------------')
    lap_solver = hungarian
    permLoss = PermLoss()
    emdloss=EMDLoss()
    softmax=Softmax(dim=-1)
    sigmoid=torch.nn.Sigmoid()
    since = time.time()
    all_val_metrics_np = defaultdict(list)
    iter_num = 0

    dataset_size = len(dataloader.dataset)

    was_training = model.training
    model.load_state_dict(torch.load("./output/RGM_DGCNN_ModelNet40Seen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/rgm_params_0085.pt"))
    # model.load_state_dict(torch.load("./output/RGM_DGCNN_privateSeen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/rgm_params_0033.pt"))
    model.cuda()
    model.eval()
    running_since = time.time()
    iter_=0
    iter_model=0
    
    model2 = get_model()
    # model2.load_state_dict(torch.load("../rpmnet/logs/230319_163926/ckpt/model-81792.pth"),strict=False)
    saver = torch1.CheckPointManager(os.path.join("../rpmnet/eval_results", 'ckpt', 'models'))
    saver.load("../rpmnet/logs/230319_163926/ckpt/model-15336.pth", model2)   #model-122688.pth
    model2.cuda()
    model2.eval()
    
    model3 = modified_Net8()
    model3.load_state_dict(torch.load("./output/RGM_DGCNN_ModelNet40Seen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/newzz9_params_0128.pt"))
    # model3.load_state_dict(torch.load("./output/RGM_DGCNN_privateSeen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/newzz8_params_0005_2.45.pt"))
    model3.cuda()
    model3.eval()
    
    own_r_mae,own_r_rmse,own_t_mae,own_t_rmse,own_ccd,own_cd=0,0,0,0,0,0
    rgm_r_mae,rgm_r_rmse,rgm_t_mae,rgm_t_rmse,rgm_ccd,rgm_cd=0,0,0,0,0,0
    rpm_r_mae,rpm_r_rmse,rpm_t_mae,rpm_t_rmse,rpm_ccd,rpm_cd=0,0,0,0,0,0
    icp_r_mae,icp_r_rmse,icp_t_mae,icp_t_rmse,icp_ccd,icp_cd=0,0,0,0,0,0
    pca_r_mae,pca_r_rmse,pca_t_mae,pca_t_rmse,pca_ccd,pca_cd=0,0,0,0,0,0
    fgr_r_mae,fgr_r_rmse,fgr_t_mae,fgr_t_rmse,fgr_ccd,fgr_cd=0,0,0,0,0,0
    
    cnt=0
    for inputs in tqdm(dataloader):
        iter_model+=1
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
        A1_gt, A2_gt = [_.cuda() for _ in inputs['As']]
        perm_mat = inputs['gt_perm_mat'].cuda()
        T1_gt, T2_gt = [_.cuda() for _ in inputs['Ts']]
        Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in inputs['Ins']]
        gt_T_ori=np.array(inputs['gt_T']) #(4,3,4)
        gt_euler=np.array(inputs['gt_euler'])
        gt_defect = inputs['mpoints_idx'].cuda()
        mpoints_gt = inputs['mpoints_tgt2src'].cuda()
        ori_src,ori_ref=inputs['ori_src'].cuda(),inputs['ori_ref'].cuda()
        input_ref=P2_gt.clone()
        P1_gt_copy,P2_gt_copy=P1_gt.clone(),P2_gt.clone()
        #########################################################################################
        #own method
        thresh=0.7
        s_pred, s1, Inlier_src_pre, Inlier_ref_pre = model3(P1_gt_copy,P2_gt_copy, A1_gt, A2_gt, n1_gt, n2_gt)
        s_pred = Inlier_src_pre * s_pred * Inlier_ref_pre.transpose(2, 1).contiguous()
        perm_mat = torch.repeat_interleave(torch.unsqueeze(gt_defect,dim=-1),perm_mat.shape[-2],dim=-1).transpose(2,1) * perm_mat
        permloss = permLoss(s_pred, perm_mat, n1_gt, n2_gt)
        s_tgt = torch.sum(s_pred,dim=-2)
        s_tgt = (s_tgt<=thresh)
        pre_mrate=torch.squeeze(torch.count_nonzero(s_tgt,dim=-1)/(s_tgt.shape[-1]))
        # mrate_mae=torch.mean(1-gt_res_rate)
        # acc_mrate_mae+=mrate_mae

        s_tgt=torch.unsqueeze(s_tgt,dim=-1)
        s_tgt=torch.repeat_interleave(s_tgt,input_ref.shape[-1],dim=-1)
        mpoints_pre=input_ref*s_tgt
        dist1, dist2, idx1, idx2 = chamfer_dist0(mpoints_pre[:,:,:3],mpoints_gt[:,:,:3])
        d3 = (torch.mean(dist1)) + (torch.mean(dist2))
        loss = permloss + 1e-1 * d3
        #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        s_perm_mat = lap_solver(s_pred, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
        match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
        perform_metrics,abs_err,pre_R,gt_R,pre_T,gt_T = compute_metrics(s_perm_mat, P1_gt_copy[:,:,:3], P2_gt_copy[:,:,:3], T1_gt[:, :3, :3], T1_gt[:, :3, 3])
        pre_T,gt_T=np.array(pre_T.cpu()),np.array(gt_T.cpu())
        pre_R,gt_R=pre_R.astype(float),gt_R.astype(float)
        pre_T,gt_T=pre_T.astype(float),gt_T.astype(float)
        rmse_t=m.sqrt(np.mean(np.square(pre_T-gt_T)))
        rmse_r=m.sqrt(np.mean(np.square(pre_R-gt_R)))
        # print("*****************************************")
        # print("loss:",loss) #,"chamfer:",d3)
        # print("r_mse:",rmse_r,"r_mae:",np.mean(perform_metrics['r_mae']))
        # print("t_mse:",rmse_t,"t_mae:",np.mean(perform_metrics['t_mae']))
        # print("chamfer_dist:",np.mean(perform_metrics['chamfer_dist']))
        # print("clipped_chamfer_dist:",np.mean(perform_metrics['clip_chamfer_dist']))
        own_r_mae+=np.mean(perform_metrics['r_mae'])
        own_r_rmse+=np.square(pre_R-gt_R)
        own_t_mae+=np.mean(perform_metrics['t_mae'])
        own_t_rmse+=np.square(pre_T-gt_T)
        own_cd+=perform_metrics['chamfer_dist']
        own_ccd+=perform_metrics['clip_chamfer_dist']

        cur_P1_t=np.asarray(perform_metrics['P1_t'].cpu())
        np.savetxt(f"./comp_vis/{iter_model}_own.txt",cur_P1_t[0])
        #########################################################################################
        #rgm 
        P1_gt_copy,P2_gt_copy=P1_gt.clone(),P2_gt.clone()
        s_perm_mat = caliters_perm(model, P1_gt_copy, P2_gt_copy, A1_gt, A2_gt, n1_gt, n2_gt, estimate_iters)
    
        
        s_pred, Inlier_src_pre, Inlier_ref_pre = model(P1_gt, P2_gt, A1_gt, A2_gt, n1_gt, n2_gt)
        s_pred = Inlier_src_pre * s_pred * Inlier_ref_pre.transpose(2, 1).contiguous()
        permloss = permLoss(s_pred, perm_mat, n1_gt, n2_gt)
        loss = permloss
        s_perm_mat = lap_solver(s_pred, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
        match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
        perform_metrics,abs_err,pre_R,gt_R,pre_T,gt_T = compute_metrics(s_perm_mat, P1_gt[:,:,:3], P2_gt[:,:,:3], T1_gt[:, :3, :3], T1_gt[:, :3, 3])
        pre_T,gt_T=np.array(pre_T.cpu()),np.array(gt_T.cpu())
        pre_R,gt_R=pre_R.astype(float),gt_R.astype(float)
        pre_T,gt_T=pre_T.astype(float),gt_T.astype(float)
        rmse_t=m.sqrt(np.mean(np.square(pre_T-gt_T)))
        rmse_r=m.sqrt(np.mean(np.square(pre_R-gt_R)))
        rgm_r_mae+=np.mean(perform_metrics['r_mae'])
        rgm_r_rmse+=np.square(pre_R-gt_R)
        rgm_t_mae+=np.mean(perform_metrics['t_mae'])
        rgm_t_rmse+=np.square(pre_T-gt_T)
        rgm_cd+=perform_metrics['chamfer_dist']
        rgm_ccd+=perform_metrics['clip_chamfer_dist']
        cur_P1_t=np.asarray(perform_metrics['P1_t'].cpu())
        np.savetxt(f"./comp_vis/{iter_model}_rgm.txt",cur_P1_t[0])
        # print("*****************************************")
        # print("loss:",loss) #,"chamfer:",d3)
        # print("r_mse:",rmse_r,"r_mae:",np.mean(perform_metrics['r_mae']))
        # print("t_mse:",rmse_t,"t_mae:",np.mean(perform_metrics['t_mae']))
        # print("chamfer_dist:",np.mean(perform_metrics['chamfer_dist']))
        # print("clipped_chamfer_dist:",np.mean(perform_metrics['clip_chamfer_dist']))
        # print(f"coarse rotation error {np.mean(np.mean(abs(pre_R-gt_R),axis=-2))}\ncoarse translation error {np.mean(np.mean(abs(pre_T-gt_T),axis=-2))}")
        
        #########################################################################################
        #rpm-net
        input_data={'points_src':P1_gt.clone(),
                    'points_ref':P2_gt.clone(),
                    'transform_gt':gt_T_ori}
        pred_test_transforms, endpoints = model2(input_data,2)
        pred_test_transforms = pred_test_transforms[0]
        pred_test_transforms = pred_test_transforms.cpu().detach().numpy()
        P1_gt_copy,P2_gt_copy=P1_gt.clone().cpu().numpy(),P2_gt.clone().cpu().numpy()
        for j in range(P1_gt_copy.shape[0]):
            cur_pred_transforms=pred_test_transforms[j]
            cur_pred_euler=rot2eul(cur_pred_transforms[:3,:3])*180/np.pi
            cur_pred_translate=cur_pred_transforms[:3,3]
            # print("r_mse: ",np.sqrt(np.mean((cur_pred_euler-gt_euler[j])*(cur_pred_euler-gt_euler[j]))),"r_mae: ",np.mean(abs(cur_pred_euler-gt_euler[j])))
            # print("t_mse: ",np.sqrt(np.mean((gt_T_ori[j,:3,3]-cur_pred_translate)*(gt_T_ori[j,:3,3]-cur_pred_translate))),"t_mae:",np.mean(abs(gt_T_ori[j,:3,3]-cur_pred_translate)))
                                                           
            P1_t=transform(np.concatenate([cur_pred_transforms[:3,:3],np.expand_dims(cur_pred_translate,axis=1)],axis=-1),P1_gt_copy[j,:,:3])
            temp_P1,temp_P2=torch.Tensor(np.expand_dims(P1_t,axis=0)),torch.Tensor(np.expand_dims(P2_gt_copy[j,:,:3],axis=0))
            dist_src = torch.min(square_distance(temp_P1,temp_P2), dim=-1)[0]
            dist_ref = torch.min(square_distance(temp_P2,temp_P1), dim=-1)[0]
            chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
            clip_val = torch.Tensor([0.1])
            dist_src = torch.min(torch.min(torch.sqrt(square_distance(temp_P1,temp_P2)), dim=-1)[0], clip_val)
            dist_ref = torch.min(torch.min(torch.sqrt(square_distance(temp_P2,temp_P1)), dim=-1)[0], clip_val)
            clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
            
            
            rpm_r_mae+=np.mean(abs(cur_pred_euler-gt_euler[j]))
            rpm_r_rmse+=(cur_pred_euler-gt_euler[j])*(cur_pred_euler-gt_euler[j])
            rpm_t_mae+=np.mean(abs(gt_T_ori[j,:3,3]-cur_pred_translate))
            rpm_t_rmse+=(gt_T_ori[j,:3,3]-cur_pred_translate)*(gt_T_ori[j,:3,3]-cur_pred_translate)
            rpm_cd+=chamfer_dist.item()
            rpm_ccd+=clip_chamfer_dist.item()
            
            if not j:
                cur_P1_t=np.asarray(P1_t)
                np.savetxt(f"./comp_vis/{iter_model}_rpmnet.txt",cur_P1_t)
        
        #########################################################################################
        infer_time = time.time()
        P1_gt,P2_gt=P1_gt_copy[:,:,:3],P2_gt_copy[:,:,:3]
        init_R,init_T=torch.Tensor([0,0,0]),torch.Tensor([0,0,0])
        trans_init=np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        P1_gt,P2_gt,init_R,init_T=np.asarray(P1_gt),np.asarray(P2_gt),np.asarray(init_R),np.asarray(init_T)
        gt_T=gt_T_ori
        #########################################################################################
        for j in range(P1_gt.shape[0]):
            #pca
            pred_euler,pred_T=pca(P1_gt[j,:,:3],P2_gt[j,:,:3],init_R,init_T)
            for k in range(3):
                cur_euler=pred_euler[k]
                if abs(cur_euler)>90:
                    if cur_euler>90:
                        pred_euler[k]=pred_euler[k]-90
                    else:
                        pred_euler[k]=pred_euler[k]+90
            # print("mae_t: ",np.mean(abs(pred_T-gt_T[j,:3,3])),"rmse_t: ",np.sqrt((pred_T-gt_T[j,:3,3])*(pred_T-gt_T[j,:3,3])/3))
            # print("mae_r: ",np.mean(abs(gt_euler[j]-pred_euler)),"rmse_r:",np.sqrt((gt_euler[j]-pred_euler)*(gt_euler[j]-pred_euler)/3))
            P1_t=transform(np.concatenate([Euler2R(pred_euler),np.expand_dims(pred_T,axis=1)],axis=-1),P1_gt[j,:,:3])
            temp_P1,temp_P2=torch.Tensor(np.expand_dims(P1_t,axis=0)),torch.Tensor(np.expand_dims(P2_gt[j,:,:3],axis=0))
            dist_src = torch.min(square_distance(temp_P1,temp_P2), dim=-1)[0]
            dist_ref = torch.min(square_distance(temp_P2,temp_P1), dim=-1)[0]
            chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
            clip_val = torch.Tensor([0.1])
            dist_src = torch.min(torch.min(torch.sqrt(square_distance(temp_P1,temp_P2)), dim=-1)[0], clip_val)
            dist_ref = torch.min(torch.min(torch.sqrt(square_distance(temp_P2,temp_P1)), dim=-1)[0], clip_val)
            clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
            
            pca_r_mae+=np.mean(abs(pred_euler-gt_euler[j]))
            pca_r_rmse+=(pred_euler-gt_euler[j])*(pred_euler-gt_euler[j])
            pca_t_mae+=np.mean(abs(gt_T[j,:3,3]-pred_T))
            pca_t_rmse+=(gt_T[j,:3,3]-pred_T)*(gt_T[j,:3,3]-pred_T)
            pca_cd+=chamfer_dist.item()
            pca_ccd+=clip_chamfer_dist.item()
            
            if not j:
                cur_P1_t=np.asarray(P1_t)
                np.savetxt(f"./comp_vis/{iter_model}_pca.txt",cur_P1_t)
            
            #########################################################################################
            #########################################################################################
            #icp
            #
            o3d_src = open3d.geometry.PointCloud()
            o3d_src.points = open3d.utility.Vector3dVector(P1_gt[j,:,:3])
            o3d_tar = open3d.geometry.PointCloud()
            o3d_tar.points = open3d.utility.Vector3dVector(P2_gt[j,:,:3])

            reg_p2p = o3d.pipelines.registration.registration_icp(o3d_src, o3d_tar, 0.1, trans_init,
                                                                  o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                  o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            icp_euler=rot2eul(reg_p2p.transformation[:3,:3])
            icp_translate=reg_p2p.transformation[:3,3]
            icp_euler=icp_euler*180/np.pi
            # print("mae_r: ",np.mean(abs(icp_euler-gt_euler[j])),"rmse_r: ",np.sqrt(np.mean((icp_euler-gt_euler[j])*(icp_euler-gt_euler[j]))))
            # print("mae_t:",np.mean(abs(gt_T[j,:3,3]-icp_translate)),"rmse_t: ",np.sqrt(np.mean((gt_T[j,:3,3]-icp_translate)*(gt_T[j,:3,3]-icp_translate))))
                                                           
            P1_t=transform(np.concatenate([reg_p2p.transformation[:3,:3],np.expand_dims(icp_translate,axis=1)],axis=-1),P1_gt[j,:,:3])
            temp_P1,temp_P2=torch.Tensor(np.expand_dims(P1_t,axis=0)),torch.Tensor(np.expand_dims(P2_gt[j,:,:3],axis=0))
            dist_src = torch.min(square_distance(temp_P1,temp_P2), dim=-1)[0]
            dist_ref = torch.min(square_distance(temp_P2,temp_P1), dim=-1)[0]
            chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
            clip_val = torch.Tensor([0.1])
            dist_src = torch.min(torch.min(torch.sqrt(square_distance(temp_P1,temp_P2)), dim=-1)[0], clip_val)
            dist_ref = torch.min(torch.min(torch.sqrt(square_distance(temp_P2,temp_P1)), dim=-1)[0], clip_val)
            clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
            
            
            icp_r_mae+=np.mean(abs(icp_euler-gt_euler[j]))
            icp_r_rmse+=(icp_euler-gt_euler[j])*(icp_euler-gt_euler[j])
            icp_t_mae+=np.mean(abs(gt_T[j,:3,3]-icp_translate))
            icp_t_rmse+=(gt_T[j,:3,3]-icp_translate)*(gt_T[j,:3,3]-icp_translate)
            icp_cd+=chamfer_dist.item()
            icp_ccd+=clip_chamfer_dist.item()
            # print(f"icp cd,ccd: {chamfer_dist}   {clip_chamfer_dist}")
            #
            
            if not j:
                cur_P1_t=np.asarray(P1_t)
                np.savetxt(f"./comp_vis/{iter_model}_icp.txt",cur_P1_t)
            #########################################################################################
            #########################################################################################
            #fgr
            o3d_src = open3d.geometry.PointCloud()
            o3d_src.points = open3d.utility.Vector3dVector(P1_gt[j,:,:3])
            o3d_tar = open3d.geometry.PointCloud()
            o3d_tar.points = open3d.utility.Vector3dVector(P2_gt[j,:,:3])
            voxel_size = 0.01  # means 5cm for this dataset
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size,o3d_src,o3d_tar)
            result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
            pca_pred_euler=rot2eul(np.array(result_ransac.transformation)[:3,:3])*180/np.pi
            pca_pred_translate=np.array(result_ransac.transformation[:3,3])
            for k in range(3):
                cur_euler=pca_pred_euler[k]
                if abs(cur_euler)>90:
                    if cur_euler<0:
                        pca_pred_euler[k]=pca_pred_euler[k]+90
                    else:
                        pca_pred_euler[k]=pca_pred_euler[k]-90
            P1_t=transform(np.concatenate([(np.array(result_ransac.transformation)[:3,:3]),np.expand_dims(pca_pred_translate,axis=1)],axis=-1),P1_gt[j,:,:3])
            temp_P1,temp_P2=torch.Tensor(np.expand_dims(P1_t,axis=0)),torch.Tensor(np.expand_dims(P2_gt[j,:,:3],axis=0))
            dist_src = torch.min(square_distance(temp_P1,temp_P2), dim=-1)[0]
            dist_ref = torch.min(square_distance(temp_P2,temp_P1), dim=-1)[0]
            chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
            clip_val = torch.Tensor([0.1])
            dist_src = torch.min(torch.min(torch.sqrt(square_distance(temp_P1,temp_P2)), dim=-1)[0], clip_val)
            dist_ref = torch.min(torch.min(torch.sqrt(square_distance(temp_P2,temp_P1)), dim=-1)[0], clip_val)
            clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
            fgr_r_mae+=np.mean(abs(pca_pred_euler-gt_euler[j]))
            fgr_r_rmse+=(pca_pred_euler-gt_euler[j])*(pca_pred_euler-gt_euler[j])
            fgr_t_mae+=np.mean(abs(gt_T[j,:3,3]-pca_pred_translate))
            fgr_t_rmse+=(gt_T[j,:3,3]-pca_pred_translate)*(gt_T[j,:3,3]-pca_pred_translate)
            fgr_cd+=chamfer_dist.item()
            fgr_ccd+=clip_chamfer_dist.item()
            if not j:
                cur_P1_t=np.asarray(P1_t)
                np.savetxt(f"./comp_vis/{iter_model}_fgr.txt",cur_P1_t)
                np.savetxt(f"./comp_vis/{iter_model}_realdata.txt",P2_gt[0])
            # print(f"fgr cd,ccd: {chamfer_dist}   {clip_chamfer_dist}")
            iter_+=1

    print("********************Final Result*********************")
    print("own result:")
    print(f"mae_t: {np.mean(own_t_mae/iter_model)}")
    print(f"mae_r: {np.mean(own_r_mae/iter_model)}")
    print(f"rmse_t: {m.sqrt(np.mean(own_t_rmse/iter_model))}")
    print(f"rmse_r: {m.sqrt(np.mean(own_r_rmse/iter_model))}")
    print(f"ccd: {np.mean(own_ccd/iter_model)}")
    print(f"cd: {np.mean(own_cd/iter_model)}")
    print("rgm result:")
    print(f"mae_t: {np.mean(rgm_t_mae/iter_model)}")
    print(f"mae_r: {np.mean(rgm_r_mae/iter_model)}")
    print(f"rmse_t: {m.sqrt(np.mean(rgm_t_rmse/iter_model))}")
    print(f"rmse_r: {m.sqrt(np.mean(rgm_r_rmse/iter_model))}")
    print(f"ccd: {np.mean(rgm_ccd/iter_model)}")
    print(f"cd: {np.mean(rgm_cd/iter_model)}")
    print("rpmnet result:")
    print(f"mae_t: {np.mean(rpm_t_mae/iter_)}")
    print(f"mae_r: {np.mean(rpm_r_mae/iter_)}")
    print(f"rmse_t: {m.sqrt(np.mean(rpm_t_rmse/iter_))}")
    print(f"rmse_r: {m.sqrt(np.mean(rpm_r_rmse/iter_))}")
    print(f"ccd: {np.mean(rpm_ccd/iter_)}")
    print(f"cd: {np.mean(rpm_cd/iter_)}")
    print("pca result:")
    print(f"mae_t: {np.mean(pca_t_mae/iter_)}")
    print(f"mae_r: {np.mean(pca_r_mae/iter_)}")
    print(f"rmse_t: {m.sqrt(np.mean(pca_t_rmse/iter_))}")
    print(f"rmse_r: {m.sqrt(np.mean(pca_r_rmse/iter_))}")
    print(f"ccd: {np.mean(pca_ccd/iter_)}")
    print(f"cd: {np.mean(pca_cd/iter_)}")
    print("icp result:")
    print(f"mae_t: {np.mean(icp_t_mae/iter_)}")
    print(f"mae_r: {np.mean(icp_r_mae/iter_)}")
    print(f"rmse_t: {m.sqrt(np.mean(icp_t_rmse/iter_))}")
    print(f"rmse_r: {m.sqrt(np.mean(icp_r_rmse/iter_))}")
    print(f"ccd: {np.mean(icp_ccd/iter_)}")
    print(f"cd: {np.mean(icp_cd/iter_)}")
    print("fgr result:")
    print(f"mae_t: {np.mean(fgr_t_mae/iter_)}")
    print(f"mae_r: {np.mean(fgr_r_mae/iter_)}")
    print(f"rmse_t: {m.sqrt(np.mean(fgr_t_rmse/iter_))}")
    print(f"rmse_r: {m.sqrt(np.mean(fgr_r_rmse/iter_))}")
    print(f"ccd: {np.mean(fgr_ccd/iter_)}")
    print(f"cd: {np.mean(fgr_cd/iter_)}")
    return None


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_argspc import parse_args
    from utils.print_easydict import print_easydict
    from utils.visdomshow import VisdomViz
    from models.Net import RGM

    import socket

    import scipy
    if np.__version__=='1.19.2' and scipy.__version__=='1.5.0':
        print('It is the same as the paper result')
    else:
        print('May not be the same as the results of the paper')

    args = parse_args('Point could registration of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    if cfg.VISDOM.OPEN:
        hostname = socket.gethostname()
        Visdomins = VisdomViz(env_name=hostname+'RGM_Eval', server=cfg.VISDOM.SERVER, port=cfg.VISDOM.PORT)
        Visdomins.viz.close()
    else:
        Visdomins = None

    torch.manual_seed(cfg.RANDOM_SEED)

    pc_dataset = get_datasets(partition = 'test',
                              num_points = cfg.DATASET.POINT_NUM,
                              unseen = cfg.DATASET.UNSEEN,
                              noise_type = cfg.DATASET.NOISE_TYPE,
                              rot_mag = cfg.DATASET.ROT_MAG,
                              trans_mag = cfg.DATASET.TRANS_MAG,
                              partial_p_keep = cfg.DATASET.PARTIAL_P_KEEP)

    dataloader = get_dataloader(pc_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = RGM()
    # model = model.to(device)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        metrics = eval_model(model, dataloader,
                             eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                             metric_is_save=True,
                             estimate_iters=cfg.EVAL.ITERATION_NUM,
                             viz=Visdomins,
                             usepgm=cfg.EXPERIMENT.USEPGM, userefine=cfg.EXPERIMENT.USEREFINE,
                             save_filetime=now_time)
