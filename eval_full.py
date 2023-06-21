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
from utils.ShapeMeasure.distance import EMDLoss,ChamferLoss 
from datetime import datetime
import os


def eval_model(model, dataloader, model_path=None, eval_epoch=None, metric_is_save=False, estimate_iters=1,
               viz=None, usepgm=True, userefine=False, save_filetime='time'):
    print('-----------------Start evaluation-----------------')
    lap_solver = hungarian
    permevalLoss = PermLoss()
    emdLoss=EMDLoss()
    since = time.time()
    all_val_metrics_np = defaultdict(list)
    iter_num = 0

    dataset_size = len(dataloader.dataset)
    print('train datasize: {}'.format(dataset_size))
    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    model_path="./output//RGM_DGCNN_ModelNet40Seen_NoPreW['xyz', 'gxyz']_attentiontransformer_clean/params/"
    model_path=os.path.join(model_path,"params_0030.pt")
    load_model(model,model_path)

    was_training = model.training
    model.eval()
    running_since = time.time()
    
    acc_err_deg=0
    acc_err_t=0
    acc_err_abs=0
    iter_=0

    for inputs in dataloader:
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
        A1_gt, A2_gt = [_.cuda() for _ in inputs['As']]
        perm_mat = inputs['gt_perm_mat'].cuda()
        T1_gt, T2_gt = [_.cuda() for _ in inputs['Ts']]
        Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in inputs['Ins']]
        #####full
        # mask = inputs['mask'].cuda()
        B,_,C=P1_gt.shape

        batch_cur_size = perm_mat.size(0)
        iter_num = iter_num + 1
        infer_time = time.time()

        with torch.set_grad_enabled(False):
            s_pred, Inlier_src_pre, Inlier_ref_pre, slct_src, slct_tar = model(P1_gt, P2_gt, A1_gt, A2_gt, n1_gt, n2_gt)

            for i in range(len(slct_src)):
                cur_slct_src=slct_src[i]
                cur_slct_tar=slct_tar[i]
                cur_slct_src=torch.repeat_interleave(cur_slct_src,perm_mat.shape[-1],dim=-1)
                perm_mat=torch.gather(perm_mat,1,cur_slct_src)
                cur_slct_tar=torch.repeat_interleave(cur_slct_tar,perm_mat.shape[-2],dim=-1)
                perm_mat=torch.gather(perm_mat,2,cur_slct_tar.permute(0,2,1))
            permloss = permevalLoss(s_pred, perm_mat, n1_gt, n2_gt)
            loss = permloss

            s_perm_mat = lap_solver(s_pred, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
            match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
            perform_metrics,abs_err,pre_R,gt_R,pre_T,gt_T = compute_metrics(s_perm_mat, P1_gt[:,:,:3], P2_gt[:,:,:3], T1_gt[:, :3, :3], T1_gt[:, :3, 3],slct_src=slct_src,slct_tar=slct_tar)
            print("******************************************")
            # print("err_deg:",perform_metrics['err_r_deg'],"err_t:",perform_metrics['err_t'])
            # pre_R,gt_R=pre_R.cpu(),gt_R.cpu()
            pre_T,gt_T=np.array(pre_T.cpu()),np.array(gt_T.cpu())
            pre_R,gt_R=pre_R.astype(float),gt_R.astype(float)
            pre_T,gt_T=pre_T.astype(float),gt_T.astype(float)
            print(f"coarse rotation error {np.mean(abs(pre_R-gt_R),axis=-2)}\ncoarse translation error {np.mean(abs(pre_T-gt_T),axis=-2)}")
            acc_err_t+=abs(pre_T-gt_T)
            acc_err_abs+=abs(pre_R-gt_R)
            iter_+=1

        if iter_num % cfg.STATISTIC_STEP == 0 and metric_is_save:
            running_speed = cfg.STATISTIC_STEP * batch_cur_size / (time.time() - running_since)
            print('Iteration {:<4} {:>4.2f}sample/s'.format(iter_num, running_speed))
            running_since = time.time()
    print("********************Final Result*********************")
    # print(f"err_deg: {acc_err_deg/iter_}")
    print(f"err_t: {acc_err_t/iter_}")
    print(f"err_abs: {acc_err_abs/iter_}")
    return summary_metrics


def caliters_perm(model, P1_gt_copy, P2_gt_copy, A1_gt, A2_gt, n1_gt, n2_gt, estimate_iters): #, mask):
    lap_solver1 = hungarian
    s_perm_indexs = []
    for estimate_iter in range(estimate_iters):
        #####full
        # s_prem_i, Inlier_src_pre, Inlier_ref_pre, slct_src, slct_tar, mpoints_pre  = model(P1_gt_copy, P2_gt_copy,
        #                                                  A1_gt, A2_gt, n1_gt, n2_gt)
        s_prem_i, Inlier_src_pre, Inlier_ref_pre, slct_src, slct_tar = model(P1_gt_copy, P2_gt_copy,
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
    from models.Net import modified_Net1

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

    model = modified_Net1()
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
