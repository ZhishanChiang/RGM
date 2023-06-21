import open3d 
import numpy as np
import os

def save_pts(pc,save_path,cnt):
    for i in range(pc.shape[0]):
        cur_pc=pc[i,:,:3].cpu()
        cur_path=os.path.join(save_path,f"{cnt}_{i}.txt")
        np.savetxt(cur_path, cur_pc.numpy())