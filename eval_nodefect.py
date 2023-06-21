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

def refine_registration(src,tar,init_R,init_T,gt_R,gt_T):
    src,tar=np.array(src.cpu())[0,:,:],np.array(tar.cpu())[0,:,:]
    pcd_src=open3d.geometry.PointCloud()
    pcd_src.points=open3d.utility.Vector3dVector(src)
    init_R,init_T=init_R[0],init_T[0]
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
    return pred_euler,pred_T

def eval_model(model, dataloader, model_path=None, eval_epoch=None, metric_is_save=False, estimate_iters=1,
               viz=None, usepgm=True, userefine=False, save_filetime='time'):
    print('-----------------Start evaluation-----------------')
    lap_solver = hungarian
    permLoss = PermLoss()
    emdloss=EMDLoss()
    bceloss=BCE(weight=torch.Tensor([0.2]).cuda())
    softmax=Softmax(dim=-1)
    sigmoid=torch.nn.Sigmoid()
    since = time.time()
    all_val_metrics_np = defaultdict(list)
    iter_num = 0

    dataset_size = len(dataloader.dataset)
    print('train datasize: {}'.format(dataset_size))
    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    model_path="./output/RGM_DGCNN_ModelNet40Seen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/"
    # model_path="./output/RGM_DGCNN_privateSeen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/"
    model_path=os.path.join(model_path,"newzz11_params_0080.pt")  #8 3.87 21 3.11
    load_model(model,model_path)

    was_training = model.training
    model.eval()
    running_since = time.time()
    
    acc_err_deg=0
    acc_err_t=0
    acc_err_r=0
    acc_err_cd=0
    acc_err_ccd=0
    acc_rmse_t=0
    acc_rmse_r=0
    iter_=0 
    acc_permloss=0
    acc_chamfer=0
    thresh=0.7
    acc_mrate_mae=0

    for inputs in dataloader:
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
        A1_gt, A2_gt = [_.cuda() for _ in inputs['As']]
        perm_mat = inputs['gt_perm_mat'].cuda()
        T1_gt, T2_gt = [_.cuda() for _ in inputs['Ts']]
        Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in inputs['Ins']]
        gt_defect = inputs['mpoints_idx'].cuda()
        input_ref=P2_gt.clone()
        mpoints_gt = inputs['mpoints_tgt2src'].cuda()
        ori_src,ori_ref=inputs['ori_src'].cuda(),inputs['ori_ref'].cuda()

        batch_cur_size = perm_mat.size(0)
        iter_num = iter_num + 1
        infer_time = time.time()

        with torch.set_grad_enabled(False):
            print(P1_gt.device, P2_gt.device, A1_gt.device, A2_gt.device, n1_gt.device, n2_gt.device)
            s_pred, s1, Inlier_src_pre, Inlier_ref_pre = model(P1_gt, P2_gt, A1_gt, A2_gt, n1_gt, n2_gt)

            # for i in range(len(slct_src)):
            #     cur_slct_src=slct_src[i]
            #     cur_slct_tar=slct_tar[i]
            #     cur_slct_src=torch.repeat_interleave(cur_slct_src,perm_mat.shape[-1],dim=-1)
            #     perm_mat=torch.gather(perm_mat,1,cur_slct_src)
            #     cur_slct_tar=torch.repeat_interleave(cur_slct_tar,perm_mat.shape[-2],dim=-1)
            #     perm_mat=torch.gather(perm_mat,2,cur_slct_tar.permute(0,2,1))
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            # mpoints_gt=torch.zeros(mpoints_pre.shape)
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            # d=emdloss(mpoints_pre[:,:,:3],mpoints_gt[:,:,:3])
            # d=torch.sum(d)
            # dist1, dist2, idx1, idx2 = chamfer_dist(mpoints_pre[:,:,:3],mpoints_gt[:,:,:3])
            # d3 = (torch.mean(dist1)) + (torch.mean(dist2))
            s_pred = Inlier_src_pre * s_pred * Inlier_ref_pre.transpose(2, 1).contiguous()
            # perm_mat = torch.repeat_interleave(torch.unsqueeze(gt_defect,dim=-1),perm_mat.shape[-2],dim=-1).transpose(2,1) * perm_mat
            permloss = permLoss(s_pred, perm_mat, n1_gt, n2_gt)
            # s_tgt = torch.sum(s_pred,dim=-2)
            # s_tgt = (s_tgt<=thresh)
            # pre_mrate=torch.count_nonzero(s_tgt)/(s_tgt.shape[-1]*s_tgt.shape[0])
            # s_tgt=torch.unsqueeze(s_tgt,dim=-1)
            # s_tgt=torch.repeat_interleave(s_tgt,input_ref.shape[-1],dim=-1)
            # mpoints_pre=input_ref*s_tgt
            # dist1, dist2, idx1, idx2 = chamfer_dist(mpoints_pre[:,:,:3],mpoints_gt[:,:,:3])
            # d3 = (torch.mean(dist1)) + (torch.mean(dist2))
            loss = permloss #+ 1e-1 * d3
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            s_perm_mat = lap_solver(s_pred, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
            match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
            perform_metrics,abs_err,pre_R,gt_R,pre_T,gt_T = compute_metrics(s_perm_mat, P1_gt[:,:,:3], P2_gt[:,:,:3], T1_gt[:, :3, :3], T1_gt[:, :3, 3])
            pre_T,gt_T=np.array(pre_T.cpu()),np.array(gt_T.cpu())
            pre_R,gt_R=pre_R.astype(float),gt_R.astype(float)
            pre_T,gt_T=pre_T.astype(float),gt_T.astype(float)
            rmse_t=m.sqrt(np.mean(np.square(pre_T-gt_T)))
            rmse_r=m.sqrt(np.mean(np.square(pre_R-gt_R)))
            print("*****************************************")
            print("loss:",loss) #,"chamfer:",d3)
            print("r_mse:",rmse_r,"r_mae:",np.mean(perform_metrics['r_mae']))
            print("t_mse:",rmse_t,"t_mae:",np.mean(perform_metrics['t_mae']))
            print("chamfer_dist:",np.mean(perform_metrics['chamfer_dist']))
            print("clipped_chamfer_dist:",np.mean(perform_metrics['clip_chamfer_dist']))
            print(f"coarse rotation error {np.mean(np.mean(abs(pre_R-gt_R),axis=-2))}\ncoarse translation error {np.mean(np.mean(abs(pre_T-gt_T),axis=-2))}")
            acc_permloss+=permloss
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            # acc_chamfer+=d3
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            acc_err_t+=abs(pre_T-gt_T)
            acc_err_r+=abs(pre_R-gt_R) 
            acc_rmse_r+=np.mean(abs(pre_R-gt_R)*abs(pre_R-gt_R))
            acc_rmse_t+=np.mean(abs(pre_T-gt_T)*abs(pre_T-gt_T))
            acc_err_cd+=np.mean(perform_metrics['chamfer_dist'])
            acc_err_ccd+=np.mean(perform_metrics['clip_chamfer_dist'])
            iter_+=1
            np.savetxt(f"./output_vis/src/src_{iter_}.txt",np.array(ori_src[0,:,:3].cpu()))
            np.savetxt(f"./output_vis/tar/tar_{iter_}.txt",np.array(ori_ref[0,:,:3].cpu()))
            # np.savetxt(f"./output_vis/mpoints/mpoints_{iter_}.txt",np.array(mpoints_pre[0,:,:3].cpu()))
            # np.savetxt(f"./output_vis/mpoints_gt/mpoints_gt{iter_}.txt",np.array(mpoints_gt[0,:,:3].cpu()))
            #####################
            # refine_R,refine_T=refine_registration(P1_gt[:,:,:3],P2_gt[:,:,:3],pre_R,pre_T,gt_R,gt_T)
            # print(f"refined rotation error: {np.mean(np.mean(abs(refine_R-gt_R),axis=-2))}\nrefined translation error {np.mean(np.mean(abs(refine_T+pre_T-gt_T),axis=-2))}")
            #####################

        if iter_num % cfg.STATISTIC_STEP == 0 and metric_is_save:
            running_speed = cfg.STATISTIC_STEP * batch_cur_size / (time.time() - running_since)
            print('Iteration {:<4} {:>4.2f}sample/s'.format(iter_num, running_speed))
            running_since = time.time()
    print("********************Final Result*********************")
    print(f"permloss: {acc_permloss/iter_}")
    print(f"mae_t: {np.mean(acc_err_t/iter_)}")
    print(f"mae_r: {np.mean(acc_err_r/iter_)}")
    print(f"rmse_t: {m.sqrt(np.mean(acc_rmse_t/iter_))}")
    print(f"rmse_r: {m.sqrt(np.mean(acc_rmse_r/iter_))}")
    print(f"cd: {np.mean(acc_err_cd/iter_)}")
    print(f"ccd: {np.mean(acc_err_ccd/iter_)}")
    # print(f"chamfer: {acc_chamfer/iter_}")
    return summary_metrics


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


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_argspc import parse_args
    from utils.print_easydict import print_easydict
    from utils.visdomshow import VisdomViz
    from models.Net import modified_Net0
    from models.Net import modified_Net4
    from models.Net import modified_Net5
    from models.Net import modified_Net6
    from models.Net import modified_Net7
    from models.Net import modified_Net8
    from models.Net import modified_Net9
    from models.Net import modified_Net10
    from models.Net import modified_Net11

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

    model = modified_Net11()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

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
