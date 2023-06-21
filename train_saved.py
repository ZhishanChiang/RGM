import os
import torch
import torch.optim as optim
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import math

from data.data_loader import get_datasets, get_dataloader
from utils.loss_func import PermLoss
from utils.evaluation_metric import matching_accuracy, compute_metrics, summarize_metrics, print_metrics
from utils.model_sl import load_model, save_model
from utils.hungarian import hungarian
from utils.visdomshow import VisdomViz
from utils.config import cfg
from parallel import DataParallel
from eval import eval_model
from models.Net import modified_Net1
from models.Net import modified_Net2
from models.Net import modified_Net3
from models.Net import modified_Net4
from models.Net import modified_Net5
from models.Net import modified_Net6
from models.Net import modified_Net7
from models.Net import modified_Net8
from models.Net import modified_Net9
from models.Net import modified_Net10
from models.Net import modified_Net11
from utils.ShapeMeasure.distance import EMDLoss,ChamferLoss 
from pytorch3d.loss import chamfer_distance
from utils.chamfer_distance.chamfer_distance import ChamferDistance as chamfer_dist
from utils.visualization import save_pts
from torch.nn import CrossEntropyLoss as CE
from torch.nn import BCELoss as BCE
from torch.nn import Softmax
from torch.nn import functional as F
from utils.awl import AutomaticWeightedLoss as awl
inf=10
def train_eval_model(model,
                     permLoss,
                     optimizer,
                     dataloader,
                     num_epochs=25,
                     resume=False,
                     start_epoch=0,
                     viz=None,
                     savefiletime='time'):
    print('**************************************')
    print('Start training...')
    dataset_size = len(dataloader['train'].dataset)
    print('train datasize: {}'.format(dataset_size))

    since = time.time()
    lap_solver = hungarian
    optimal_acc = 0.0
    optimal_rot = np.inf
    device = next(model.parameters()).device
    emdloss = EMDLoss()
    celoss=CE()
    bceloss=BCE(weight=torch.Tensor([0.2]).cuda())
    sigmoid=torch.nn.Sigmoid()
    awloss=awl(2)
    thresh=0.7
    softmax=torch.nn.Softmax(dim=-1)

    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        assert start_epoch != 0
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)
    #########################
    ##with warmup
    # warm_up_iter=10
    # lambda_lr = lambda cur_iter: (cur_iter+1) / (warm_up_iter+1e-8) if  cur_iter < warm_up_iter else 1
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    

    cnt=0
    last_d3=0.001
    for epoch in range(start_epoch, num_epochs):
#         model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
#         print('Loading model parameters from {}'.format(model_path))
        
#         # Eval in each epoch
#         val_metrics = eval_model(model, dataloader['val'], model_path)
#         if viz is not None:
#             viz.update('val_acc', epoch, {'acc': val_metrics['acc_gt']})
#             viz.update('val_metric', epoch, {'r_mae': val_metrics['r_mae'],
#                                              't_mae': val_metrics['t_mae']})
#         if optimal_acc < val_metrics['acc_gt']:
#             optimal_acc = val_metrics['acc_gt']
#             print('Current best acc model is {}'.format(epoch + 1))
#         if optimal_rot > val_metrics['r_mae']:
#             optimal_rot = val_metrics['r_mae']
#             print('Current best rotation model is {}'.format(epoch + 1))

#         # Test in each epochgi
#         test_metrics = eval_model(model, dataloader['test'])
#         if viz is not None:
#             viz.update('test_acc', epoch, {'acc': test_metrics['acc_gt']})
#             viz.update('test_metric', epoch, {'r_mae': test_metrics['r_mae'],
#                                               't_mae': test_metrics['t_mae']})

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        iter_num = 0
        running_since = time.time()
        all_train_metrics_np = defaultdict(list)

        # Iterate over data3d.
        for inputs in dataloader['train']:
            P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]     #keypoints coordinate   (B,N,6)
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]     #keypoints number       (B) 
            A1_gt, A2_gt = [_.cuda() for _ in inputs['As']]     #edge connect matrix    (B,N,N)
            perm_mat = inputs['gt_perm_mat'].cuda()             #permute matrix
            T1_gt, T2_gt = [_.cuda() for _ in inputs['Ts']]
            Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in inputs['Ins']]
            gt_defect = inputs['mpoints_idx'].cuda()
            mpoints_gt = inputs['mpoints_tgt2src'].cuda()

            batch_cur_size = perm_mat.size(0)
            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                B,_,C=P1_gt.shape
                input_ref=P2_gt.clone()
                s_pred,s1, Inlier_src_pre, Inlier_ref_pre = model(P1_gt, P2_gt, A1_gt, A2_gt, n1_gt, n2_gt)

                # for i in range(len(slct_src)):
                #     cur_slct_src=slct_src[i]
                #     cur_slct_tar=slct_tar[i]
                #     cur_slct_src=torch.repeat_interleave(cur_slct_src,perm_mat.shape[-1],dim=-1)
                #     perm_mat=torch.gather(perm_mat,1,cur_slct_src)
                #     cur_slct_tar=torch.repeat_interleave(cur_slct_tar,perm_mat.shape[-2],dim=-1)
                #     perm_mat=torch.gather(perm_mat,2,cur_slct_tar.permute(0,2,1))   ###################
                if cfg.DATASET.NOISE_TYPE == 'clean':
                    s_tgt = torch.sum(s_pred,dim=-2)
                    s_tgt[s_tgt>=thresh]=0
                    s_tgt[s_tgt<thresh]=1
                    s_tgt=torch.unsqueeze(s_tgt,dim=-1)
                    s_tgt=torch.repeat_interleave(s_tgt,input_ref.shape[-1],dim=-1)
                    mpoints_pre=input_ref*s_tgt
                    permloss = permLoss(s_pred, perm_mat, n1_gt, n2_gt)
                    dist1, dist2, idx1, idx2 = chamfer_dist(mpoints_pre[:,:,:3],mpoints_gt[:,:,:3])
                    d3 = (torch.mean(dist1)) + (torch.mean(dist2))
                    loss = permloss  #+ 10 * d3
                else:
                    if cfg.PGM.USEINLIERRATE:
                        s_pred = Inlier_src_pre * s_pred * Inlier_ref_pre.transpose(2, 1).contiguous()
                    s_tgt = torch.sum(s_pred,dim=-2)
                    s_tgt = (s_tgt<=thresh)
                    s_tgt = s_tgt.to(int)
                    pre_mrate=torch.count_nonzero(s_tgt)/(s_tgt.shape[-1]*s_tgt.shape[0])
                    # s_pred = torch.repeat_interleave(torch.unsqueeze(torch.logical_not(s_tgt),dim=-1),s_pred.shape[-2],dim=-1).transpose(2,1) * s_pred
                    # s_pred = s_pred.to(float)
                    perm_mat = torch.repeat_interleave(torch.unsqueeze(gt_defect,dim=-1),perm_mat.shape[-2],dim=-1).transpose(2,1) * perm_mat
                    permloss = permLoss(s_pred, perm_mat, n1_gt, n2_gt)
                    s_tgt=torch.unsqueeze(s_tgt,dim=-1)
                    s_tgt=torch.repeat_interleave(s_tgt,input_ref.shape[-1],dim=-1)
                    mpoints_pre=input_ref*s_tgt
                    dist1, dist2, idx1, idx2 = chamfer_dist(mpoints_pre[:,:,:3],mpoints_gt[:,:,:3])
                    d3 = (torch.mean(dist1)) + (torch.mean(dist2))
                    loss = permloss + abs(last_d3-d3)/last_d3 * 1e-3 * d3
                    last_d3=0.4*last_d3+0.6*abs(last_d3-d3)

                loss.backward()
                optimizer.step()

                s_perm_mat = lap_solver(s_pred, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
                match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)  #matching_accuracy(pmat_pred, pmat_gt,ns)
                perform_metrics = compute_metrics(s_perm_mat, P1_gt[:,:,:3], P2_gt[:,:,:3], T1_gt[:, :3, :3], T1_gt[:, :3, 3])

                for k in match_metrics:
                    all_train_metrics_np[k].append(match_metrics[k])
                for k in perform_metrics:
                    all_train_metrics_np[k].append(perform_metrics[k])
                all_train_metrics_np['loss'].append(np.repeat(loss.item(), 4))
                all_train_metrics_np['permloss'].append(np.repeat(permloss.item(),4))
                all_train_metrics_np['chamferloss'].append(np.repeat(d3.item(),4))
                all_train_metrics_np['mrate'].append(np.repeat(pre_mrate.item(),4))


                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * batch_cur_size / (time.time() - running_since)
                    # globalstep = epoch * dataset_size + iter_num * batch_cur_size
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f} PermLoss={:<8.4f} chamfer={:<8.4f} mrate={:<8.4f} GT-Acc:{:.4f} Pred-Acc:{:.4f}'
                          .format(epoch, iter_num, running_speed,
                                  np.mean(np.concatenate(all_train_metrics_np['loss'])[-cfg.STATISTIC_STEP*batch_cur_size:]),
                                  np.mean(np.concatenate(all_train_metrics_np['permloss'])[-cfg.STATISTIC_STEP*batch_cur_size:]),
                                  np.mean(np.concatenate(all_train_metrics_np['chamferloss'])[-cfg.STATISTIC_STEP*batch_cur_size:]),
                                  np.mean(np.concatenate(all_train_metrics_np['mrate'])[-cfg.STATISTIC_STEP*batch_cur_size:]),
                                  np.mean(np.concatenate(all_train_metrics_np['acc_gt'])[-cfg.STATISTIC_STEP*batch_cur_size:]),
                                  np.mean(np.concatenate(all_train_metrics_np['acc_pred'])[-cfg.STATISTIC_STEP*batch_cur_size:])))
                    running_since = time.time()

        all_train_metrics_np = {k: np.concatenate(all_train_metrics_np[k]) for k in all_train_metrics_np}
        summary_metrics = summarize_metrics(all_train_metrics_np)
        print('Epoch {:<4} Mean-Loss: {:.4f} Mean-PermLoss: {:.4f} Mean-chamfer: {:.4f} GT-Acc:{:.4f} Pred-Acc:{:.4f}'.
              format(epoch, summary_metrics['loss'], summary_metrics['permloss'], summary_metrics['chamferloss'], summary_metrics['acc_gt'], summary_metrics['acc_pred']) )
        print_metrics(summary_metrics)
    
        model_path = str(checkpoint_path / 'newzz8(9)_params_{:04}.pt'.format(epoch + 1))
        save_model(model, model_path)
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        # to save values during training
        metric_is_save= False
        if metric_is_save:
            np.save(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + savefiletime + '_metric')),
                    all_train_metrics_np)

        if viz is not None:
            viz.update('train_loss', epoch, {'loss': summary_metrics['loss']})
            viz.update('train_acc', epoch, {'acc': summary_metrics['acc_gt']})
            viz.update('train_metric', epoch, {'r_mae': summary_metrics['r_mae'],
                                                    't_mae': summary_metrics['t_mae']})


#         # Eval in each epoch
#         val_metrics = eval_model(model, dataloader['val'], model_path)
#         if viz is not None:
#             viz.update('val_acc', epoch, {'acc': val_metrics['acc_gt']})
#             viz.update('val_metric', epoch, {'r_mae': val_metrics['r_mae'],
#                                              't_mae': val_metrics['t_mae']})
#         if optimal_acc < val_metrics['acc_gt']:
#             optimal_acc = val_metrics['acc_gt']
#             print('Current best acc model is {}'.format(epoch + 1))
#         if optimal_rot > val_metrics['r_mae']:
#             optimal_rot = val_metrics['r_mae']
#             print('Current best rotation model is {}'.format(epoch + 1))

#         # Test in each epochgi
#         test_metrics = eval_model(model, dataloader['test'])
#         if viz is not None:
#             viz.update('test_acc', epoch, {'acc': test_metrics['acc_gt']})
#             viz.update('test_metric', epoch, {'r_mae': test_metrics['r_mae'],
#                                               't_mae': test_metrics['t_mae']})

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_argspc import parse_args
    from utils.print_easydict import print_easydict
    import socket
    
    
    #Reproducibility
    # torch.use_deterministic_algorithms(True)
    # torch.cuda.manual_seed(3407)  #GPU
    # torch.backends.cudnn.benchmark = False
    # torch.manual_seed(3407) #CPU

    args = parse_args('Point could registration of graph matching training & evaluation code.')

    if cfg.VISDOM.OPEN:
        hostname = socket.gethostname()
        Visdomins = VisdomViz(env_name=hostname+'RGM_Train', server=cfg.VISDOM.SERVER, port=cfg.VISDOM.PORT)
        Visdomins.viz.close()
    else:
        Visdomins = None

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    pc_dataset = {x: get_datasets(partition = x,
                                  num_points = cfg.DATASET.POINT_NUM,
                                  unseen = cfg.DATASET.UNSEEN,
                                  noise_type = cfg.DATASET.NOISE_TYPE,
                                  rot_mag = cfg.DATASET.ROT_MAG,
                                  trans_mag = cfg.DATASET.TRANS_MAG,
                                  partial_p_keep = cfg.DATASET.PARTIAL_P_KEEP,
                                  crossval = (x == 'train'),
                                  train_part = (x == 'train')) for x in ('train', 'test')}
    pc_dataset['val'] = get_datasets(partition = 'train',
                                     num_points = cfg.DATASET.POINT_NUM,
                                     unseen = cfg.DATASET.UNSEEN,
                                     noise_type = cfg.DATASET.NOISE_TYPE,
                                     rot_mag = cfg.DATASET.ROT_MAG,
                                     trans_mag = cfg.DATASET.TRANS_MAG,
                                     partial_p_keep = cfg.DATASET.PARTIAL_P_KEEP,
                                     crossval = True,
                                     train_part = False)

    dataloader = {x: get_dataloader(pc_dataset[x], shuffle=(x == 'train')) for x in ('train', 'val', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = modified_Net9()
    # model.load_state_dict(torch.load("./output/RGM_DGCNN_ModelNet40Seen_NoPreW['xyz', 'gxyz']_attentiontransformer_clean/params/params_0030_pre.pt"),strict=False)
    # model.load_state_dict(torch.load("./output/RGM_DGCNN_ModelNet40Seen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/new88_to_be_continued.pt"),strict=False)
    # model.load_state_dict(torch.load("./output/RGM_DGCNN_privateSeen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/newzz_params_0044_2.57.pt"),strict=True)
    model = model.cuda()
    permLoss = PermLoss()

    if cfg.TRAIN.OPTIM == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True, weight_decay=1e-6)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0)

    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, permLoss, optimizer, dataloader,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 resume=cfg.TRAIN.START_EPOCH != 0,
                                 start_epoch=cfg.TRAIN.START_EPOCH,
                                 viz = Visdomins,savefiletime=now_time)
