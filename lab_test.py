import numpy as np 
import torch
import open3d as o3d
import os
import sys
sys.path.append("..")
from utils.build_graphs import fully_connect
from models.Net import modified_Net0
from models.Net import modified_Net1
from models.Net import modified_Net6
from models.Net import modified_Net8
from models.Net import Net
from utils.hungarian import hungarian
from utils.loss_func import PermLoss
import time
from datetime import datetime
from collections import defaultdict
import open3d
from utils.evaluation_metric import matching_accuracy, calcorrespondpc, summarize_metrics, print_metrics, compute_metrics, compute_transform
import data.data_transform_syndata as Transforms
from data.data_loader import get_transforms
import torchvision
import math
from utils.build_graphs import build_graphs
from utils import pc_processing 
import time
from scipy.spatial.transform import Rotation as Rot
import copy
import math as m
import open3d


all_val_metrics_np = defaultdict(list)
###################
#others
lap_solver = hungarian
permevalLoss = PermLoss()
iter_num = 0

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
    print(init_T)
    pcd_src=pcd_src.translate(init_T.cpu())
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
    

def test_model(idx,acc_err_deg,acc_err_t,acc_err_abs):
    transform = get_transforms(partition="test", num_points=1024 , noise_type="crop",
                                rot_mag = 45.0, trans_mag = 0.5, partial_p_keep = [0.7])
    transform = torchvision.transforms.Compose(transform)

    ##################
    #input data
    filename=f"../lab_data/phase1_20/points_double{idx}.txt"
    pcd=o3d.io.read_point_cloud(filename,format='xyz')
    points=np.array(pcd.points)
    pcd = pcd.uniform_down_sample(every_k_points=100)

    # P1_gt=pc_processing.private_data_preprocessing(pcd)
    points=np.array(pcd.points)
    P1_gt = np.asarray(pcd.points)   #pc
    P1_gt = P1_gt / max(abs(P1_gt.min()),P1_gt.max())

    src_o3 = open3d.geometry.PointCloud()
    src_o3.points = open3d.utility.Vector3dVector(P1_gt)
    # surface_curvature = pc_processing.caculate_surface_curvature(src_o3, radius=0.0012)
    # print(surface_curvature.shape,np.array(src_o3.points).shape)
    src_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.0012, max_nn=30))
    P1_gt = np.concatenate([P1_gt,src_o3.normals],axis=-1)
    idx=np.random.randint(low=0,high=math.pow(2,32)-1,size=(1,))
    sample={'points':P1_gt,'idx':idx[0]}
    sample=transform(sample)
    T_ab = sample['transform_gt']
    T_ba = np.concatenate((T_ab[:,:3].T, np.expand_dims(-(T_ab[:,:3].T).dot(T_ab[:,3]), axis=1)), axis=-1)
    n1_gt, n2_gt = sample['perm_mat'].shape
    A1_gt, e1_gt = build_graphs(sample['points_src'], sample['src_inlier'], n1_gt, stg="fc")
    A2_gt, e2_gt = build_graphs(sample['points_ref'], sample['ref_inlier'], n2_gt, stg="fc")
    src_o3 = open3d.geometry.PointCloud()
    ref_o3 = open3d.geometry.PointCloud()
    src_o3.points = open3d.utility.Vector3dVector(sample['points_src'][:, :3])
    ref_o3.points = open3d.utility.Vector3dVector(sample['points_ref'][:, :3])
    src_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    ref_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    sample['points_src'][:, 3:6] = src_o3.normals
    sample['points_ref'][:, 3:6] = ref_o3.normals
    inputs = {'Ps': [torch.Tensor(x) for x in [sample['points_src'], sample['points_ref']]],
                        'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]],
                        'es': [torch.tensor(x) for x in [e1_gt, e2_gt]],
                        'gt_perm_mat': torch.tensor(sample['perm_mat'].astype('float32')),
                        'As': [torch.Tensor(x) for x in [A1_gt, A2_gt]],
                        'Ts': [torch.Tensor(x) for x in [T_ab.astype('float32'), T_ba.astype('float32')]],
                        'Ins': [torch.Tensor(x) for x in [sample['src_inlier'], sample['ref_inlier']]],
                        'raw': torch.Tensor(sample['points_raw']),
                        }
    P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
    n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
    A1_gt, A2_gt = [_.cuda() for _ in inputs['As']]
    perm_mat = inputs['gt_perm_mat'].cuda()
    T1_gt, T2_gt = [_.cuda() for _ in inputs['Ts']]
    Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in inputs['Ins']]
    # ###################
    # #model
    model=modified_Net8()
    model=model.cuda()
    model_path="./output//RGM_DGCNN_ModelNet40Seen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/"   #clean
    # model_path="./output/RGM_DGCNN_ModelNet40Seen_NoPreW['xyz', 'gxyz']_attentiontransformer_clean/params/"   #CLEAN
    model_path=os.path.join(model_path,"newzz9_params_0100.pt")  #2isbest   exp2_params_0200.pt
    model.load_state_dict(torch.load(model_path))
    model.eval()
    n1,n2=P1_gt.shape[0],P2_gt.shape[0]
    P1_gt,P2_gt,A1_gt,A2_gt,n1_gt,n2_gt,T1_gt,T2_gt=torch.unsqueeze(P1_gt,dim=0),torch.unsqueeze(P2_gt,dim=0),torch.unsqueeze(A1_gt,dim=0),torch.unsqueeze(A2_gt,dim=0),torch.Tensor([n1]),torch.Tensor([n2]),torch.unsqueeze(T1_gt,dim=0),torch.unsqueeze(T2_gt,dim=0)
    perm_mat=torch.unsqueeze(perm_mat,dim=0)
    s_pred, s1, Inlier_src_pre, Inlier_ref_pre = model(P1_gt, P2_gt, A1_gt, A2_gt, n1_gt, n2_gt)

    # for i in range(len(slct_src)):
    #     cur_slct_src=slct_src[i]
    #     cur_slct_tar=slct_tar[i]
    #     cur_slct_src=torch.repeat_interleave(cur_slct_src,perm_mat.shape[-1],dim=-1)
    #     perm_mat=torch.gather(perm_mat,1,cur_slct_src)
    #     cur_slct_tar=torch.repeat_interleave(cur_slct_tar,perm_mat.shape[-2],dim=-1)
    #     perm_mat=torch.gather(perm_mat,2,cur_slct_tar.permute(0,2,1))

    n1_gt,n2_gt=n1_gt.to(int),n2_gt.to(int)
    permloss = permevalLoss(s_pred, perm_mat, n1_gt, n2_gt)
    loss = permloss

    s_perm_mat = lap_solver(s_pred, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
    match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
    perform_metrics,abs_err,pre_R,gt_R,pre_T,gt_T = compute_metrics(s_perm_mat, P1_gt[:,:,:3], P2_gt[:,:,:3], T1_gt[:, :3, :3], T1_gt[:, :3, 3])

    print("******************************************")
    print(f"gt_R {gt_R}")
    print("err_deg:",perform_metrics['err_r_deg'],"err_t:",perform_metrics['err_t'])
    refine_R,refine_T=refine_registration(P1_gt[:,:,:3],P2_gt[:,:,:3],pre_R,pre_T,gt_R,gt_T)
    print(f"coarse rotation error {pre_R-gt_R}\ncoarse translation error {pre_T.cpu()-gt_T.cpu()}")
    print(f"refined rotation error {refine_R-gt_R}\nrefined translation error {pre_T.cpu()+refine_T-gt_T.cpu()}")
    print(f"refine R {refine_R} gt_R {gt_R}")
    # acc_err_deg+=perform_metrics['err_r_deg']
    acc_err_deg+=np.abs(np.mean(abs(refine_R-gt_R)))
    acc_err_t+=abs(pre_T.cpu()+refine_T-gt_T.cpu())
    acc_err_abs+=abs(refine_R-gt_R)

    return acc_err_deg,acc_err_t,acc_err_abs

if __name__ == '__main__':
    acc_err_deg=0.0
    acc_err_t=0.0
    acc_err_abs=np.array([[0.0,0.0,0.0]])
    iter_num=100
    for i in range(iter_num):
        idx=np.random.randint(1,20,size=1)[0]
        print(f"{i}:  number {idx} point cloud:")
        st=time.time()
        acc_err_deg,acc_err_t,acc_err_abs=test_model(idx,acc_err_deg,acc_err_t,acc_err_abs)
        ed=time.time()
        print(acc_err_deg," time consumption: ",{ed-st})

    print("avg_err_deg:",acc_err_deg/iter_num)
    print("avg_err_t:",acc_err_t/iter_num)
    print("avg_err_abs:",acc_err_abs/iter_num)
