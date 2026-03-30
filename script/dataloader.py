import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset

# ==========================================================
# 1. 内存映射数据集 (支持分区切片)
# ==========================================================
class MmapDataset(Dataset):
    def __init__(self, processed_dir, split, partition_id=None, node_indices=None, device='cpu'):
        """
        args:
            processed_dir: 预处理数据存放目录
            split: 'train', 'val', 'test'
            partition_id: (int) 当前主分区ID。如果为 None，则返回所有主分区标签。
            node_indices: (list/array) 该分区对应的子节点列索引。如果为 None，则返回所有节点数据。
        """
        self.device = device
        self.partition_id = partition_id
        self.node_indices = node_indices
        
        # 1. 构建路径
        x_path = os.path.join(processed_dir, f"{split}_x.npy")
        y_sub_path = os.path.join(processed_dir, f"{split}_y_sub.npy")
        y_main_path = os.path.join(processed_dir, f"{split}_y_main.npy")
        
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"数据文件未找到: {x_path}")

        # 2. 建立内存映射 (mmap_mode='r' 表示只读，不占用内存)
        # X shape: (Samples, Feat, Time, All_Nodes)
        self.X = np.load(x_path, mmap_mode='r')
        # Y_sub shape: (Samples, All_Nodes)
        self.Y_sub = np.load(y_sub_path, mmap_mode='r')
        # Y_main shape: (Samples, N_Main_Partitions)
        self.Y_main = np.load(y_main_path, mmap_mode='r')
        
        self.total_len = len(self.X)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # --- 1. 处理输入特征 X ---
        # 原始维度: (Feat, Time, All_Nodes)
        full_x = self.X[idx] 
        
        if self.node_indices is not None:
            # [切片操作] 只取属于当前分区的节点列
            # 假设节点维在最后: ..., node_indices
            x_data = full_x[..., self.node_indices]
        else:
            x_data = full_x

        # --- 2. 处理子分区标签 Y_sub (可选) ---
        full_y_sub = self.Y_sub[idx]
        if self.node_indices is not None:
            # 只取对应子节点的标签
            y_sub_data = full_y_sub[self.node_indices]
        else:
            y_sub_data = full_y_sub

        # --- 3. 处理主分区标签 Y_main (目标) ---
        full_y_main = self.Y_main[idx]
        if self.partition_id is not None:
            # [切片操作] 只取当前这一类主分区的标签 (Scalar)
            y_main_data = full_y_main[self.partition_id]
        else:
            y_main_data = full_y_main

        # --- 4. 转换为 Tensor ---
        # 注意: 这里才真正把数据从硬盘拷贝到内存
        x = torch.from_numpy(np.array(x_data)).float()
        y_sub = torch.from_numpy(np.array(y_sub_data)).long()
        y_main = torch.from_numpy(np.array(y_main_data)).long() # 确保是 Long 类型用于 CrossEntropy

        # 返回 (Input, Sub_Label, Main_Label)
        # Main_Label 如果切片了，就是一个标量；没切片就是一个向量
        return x, y_sub, y_main

# ==========================================================
# 2. 图结构加载器
# ==========================================================
def load_graph_data(dataset_path):
    """
    加载图结构，增加了对 grouping_matrix 的强校验
    """
    print(f"Loading graph data from: {dataset_path}")
    
    # 加载邻接矩阵
    try:
        adj_sub = sp.load_npz(os.path.join(dataset_path, 'adj_sub.npz')).tocsc()
        adj_main = sp.load_npz(os.path.join(dataset_path, 'adj_main.npz')).tocsc()
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing adj .npz files in {dataset_path}")

    # 加载分组矩阵 (必须有，用于划分节点)
    grouping_path_npz = os.path.join(dataset_path, 'grouping_matrix.npz')
    grouping_path_csv = os.path.join(dataset_path, 'grouping_matrix.csv')
    
    if os.path.exists(grouping_path_npz):
        grouping_matrix = sp.load_npz(grouping_path_npz).toarray()
    elif os.path.exists(grouping_path_csv):
        grouping_matrix = pd.read_csv(grouping_path_csv, header=None).values
    else:
        raise FileNotFoundError("Missing grouping_matrix (.npz or .csv)")

    n_vertex = adj_sub.shape[0]
    n_main = adj_main.shape[0]
    
    # 转换为 Tensor 方便后续处理
    grouping_matrix = torch.tensor(grouping_matrix, dtype=torch.float32)
    
    return adj_sub, adj_main, grouping_matrix, n_vertex, n_main

# ==========================================================
# 3. 辅助工厂函数 (兼容旧代码调用)
# ==========================================================
def create_classification_dataset(dataset_path, split, n_his, data_ratio=1.0, device='cpu', 
                                  partition_id=None, node_indices=None):
    """
    创建 Dataset 的入口函数
    """
    processed_dir = os.path.join(dataset_path, f"processed_window_{n_his}")
    
    dataset = MmapDataset(
        processed_dir=processed_dir, 
        split=split, 
        partition_id=partition_id,     # 传入分区ID
        node_indices=node_indices,     # 传入节点索引
        device=device
    )
    
    # 如果需要处理 data_ratio (截断数据量)，可以在这里通过 Subset 实现
    # 但为了保持 MmapDataset 逻辑简单，建议在 Dataset 内部实现 len 截断，或者这里不做处理
    
    return dataset