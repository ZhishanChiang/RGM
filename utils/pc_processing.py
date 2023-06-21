import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

class Kmeans(object):
    # 构造函数
    def __init__(self, k=2, max_iterations=1500, tolerance=0.00001):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    # 随机选取k个聚类中心点
    def init_random_centroids(self, X):
        # save the shape of X
        n_samples, n_features = np.shape(X)
        # make a zero matrix to store values
        centroids = np.zeros((self.k, n_features))
        # 因为有k个中心点，所以执行k次循环
        for i in range(self.k):
            # 随机选取范围内的值
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 查找距离样本点最近的中心

    def closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        # np.argmin 返回距离最小值的下标
        closest_i = np.argmin(distances)
        return closest_i

    # 确定聚类
    def create_clusters(self, centroids, X):
        # 这是为了构造用于存储集群的嵌套列表
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self.closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 基于均值算法更新质心
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 获取标签

    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 预测标签
    def predict(self, X):
        # 随机选取中心点
        centroids = self.init_random_centroids(X)

        for _ in range(self.max_iterations):
            # 对所有点进行聚类
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids
            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)
            # 判断是否满足收敛
            diff = centroids - former_centroids
            if diff.any() < self.tolerance:
                break

        return self.get_cluster_labels(clusters, X)


def pca_compute(data, sort=True):
    """
     SVD分解计算点云的特征值与特征向量
    :param data: 输入数据
    :param sort: 是否将特征值特征向量进行排序
    :return: 特征值与特征向量
    """
    average_data = np.mean(data, axis=0)  # 求均值
    decentration_matrix = data - average_data  # 去中心化
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵 H
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # SVD求解特征值、特征向量

    if sort:
        sort = eigenvalues.argsort()[::-1]  # 降序排列
        eigenvalues = eigenvalues[sort]  # 索引

    return eigenvalues


def caculate_surface_curvature(cloud, radius=0.003):
    """
    计算点云的表面曲率
    :param cloud: 输入点云
    :param radius: k近邻搜索的半径，默认值为：0.003m
    :return: 点云中每个点的表面曲率
    """
    points = np.asarray(cloud.points)
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    num_points = len(cloud.points)
    curvature = []  # 储存表面曲率
    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)

        neighbors = points[idx, :]
        w = pca_compute(neighbors)  # w为特征值
        delt = np.divide(w[2], np.sum(w), out=np.zeros_like(w[2]), where=np.sum(w) != 0)
        curvature.append(delt)
    curvature = np.array(curvature, dtype=np.float64)
    return curvature

def private_data_preprocessing(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40,
                                        std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40,
                                        std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    pc = np.asarray(pcd.points)   #pc
    pc = pc / max(abs(pc.min()),pc.max())
    return pc

