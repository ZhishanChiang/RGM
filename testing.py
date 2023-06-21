import open3d as o3d
import pytorch3d
import torch
import numpy as np
from open3d import JVisualizer
def read_off(file_path):
    file = open(file_path, "r")
    str=file.readline().strip()
    if len(str)>3:
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in str[3:].strip().split(' ')])
    else:
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    verts = torch.Tensor(verts)
    #faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts
pts_s = read_off("../dataset/ModelNet40/airplane/train/airplane_0001.off")
pts_s=np.vstack((pts_s[:,0],pts_s[:,1],pts_s[:,2])).transpose()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_s)
o3d.visualization.draw_geometries([pcd])