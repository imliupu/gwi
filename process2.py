import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn import preprocessing
import gc

# =========================================================
# 1. 特征工程（保持不变）
# =========================================================
def add_drainage_features(original_data):
    df = pd.DataFrame(original_data)
    feat_raw = original_data
    feat_diff = np.diff(df.values, axis=0, prepend=df.values[0:1, :])
    feat_ma = df.rolling(window=6, min_periods=1, center=False).mean().fillna(0).values
    feat_std = df.rolling(window=6, min_periods=1, center=False).std().fillna(0).values
    data_aug = np.stack([feat_raw, feat_diff, feat_ma, feat_std], axis=2)
    return data_aug

# =========================================================
# 2. 辅助函数：加载分区映射（保持不变）
# =========================================================
def load_partition_mapping(dataset_path):
    matrix_path = os.path.join(dataset_path, 'grouping_matrix.csv')
    if not os.path.exists(matrix_path):
        print(f"⚠️ 警告: 未找到 {matrix_path}，将无法处理主分区(Main Label)异常！")
        return None
    
    df = pd.read_csv(matrix_path, header=None)
    matrix = df.values
    mapping = {}
    num_partitions = matrix.shape[1]
    
    for pid in range(num_partitions):
        node_indices = np.where(matrix[:, pid] == 1)[0]
        mapping[pid] = node_indices
        
    print(f"✅ 已加载分区映射: {num_partitions} 个主分区 -> {matrix.shape[0]} 个子节点")
    return mapping

# =========================================================
# 3. 注入函数（保留定义，但不调用）
# =========================================================
def inject_hybrid_zigzag(window_data, anomaly_indices, n_his, target_node_idx, current_window_start_idx):
    """
    保留定义以便其他文件导入，但主流程中不再调用
    """
    return window_data  # 直接返回原始数据

# =========================================================
# 4. 主流程（移除所有注入调用）
# =========================================================
def generate_data(args):
    dataset_path = os.path.join(args.data_path, args.dataset)
    save_dir = os.path.join(dataset_path, f"processed_window_{args.n_his}")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    print(f"--- 纯净版: 无注入，仅处理原始数据 (Window={args.n_his}) ---")

    # 1. 读取数据
    df = pd.read_csv(os.path.join(dataset_path, 'features.csv'), header=None)
    data = df.values.astype(np.float32)
    n_vertex = data.shape[1]
    
    labels_sub = pd.read_csv(os.path.join(dataset_path, 'labels_sub.csv'), header=None).values.astype('int64')
    labels_main = pd.read_csv(os.path.join(dataset_path, 'labels_main.csv'), header=None).values.astype('int64')
    n_main = labels_main.shape[1]

    # 2. 加载映射
    partition_map = load_partition_mapping(dataset_path)

    # 3. 静态特征
    static_path = os.path.join(dataset_path, 'static_features.csv')
    if not os.path.exists(static_path): static_path = os.path.join(dataset_path, 'static_features_llm.csv')
    has_static = False
    args.n_static_features = 0
    try:
        if os.path.exists(static_path):
            static_df = pd.read_csv(static_path, header=None).apply(pd.to_numeric, errors='coerce').fillna(0)
            if static_df.shape[1] == n_vertex:
                args.n_static_features = static_df.shape[0]
                static_norm = preprocessing.StandardScaler().fit_transform(static_df.values).astype(np.float32)
                static_norm = np.expand_dims(static_norm.T, 1) 
                has_static = True
    except: pass

    # 4. 归一化与特征计算（无注入）
    print("3. 归一化与特征计算（无注入）...")
    scaler = preprocessing.StandardScaler()
    data_norm = scaler.fit_transform(data).astype(np.float32)
    data_enhanced = add_drainage_features(data_norm) 
    n_dyn_feats = data_enhanced.shape[2]

    # 5. 划分数据集
    print("4. 划分数据集...")
    total_len = len(data)
    len_val = int(total_len * 0.15)
    len_test = int(total_len * 0.15)
    len_train = total_len - len_val - len_test

    def process_and_save(split_name, start_idx, end_idx):
        length = end_idx - start_idx
        num_samples = length - args.n_his + 1
        if num_samples <= 0: return

        total_feat = n_dyn_feats + (args.n_static_features if has_static else 0)
        
        x_path = os.path.join(save_dir, f"{split_name}_x.npy")
        y_sub_path = os.path.join(save_dir, f"{split_name}_y_sub.npy")
        y_main_path = os.path.join(save_dir, f"{split_name}_y_main.npy")

        print(f"处理 {split_name}: {num_samples} 样本...")
        
        X_mem = np.lib.format.open_memmap(x_path, mode='w+', dtype='float32', shape=(num_samples, total_feat, args.n_his, n_vertex))
        Y_sub_mem = np.lib.format.open_memmap(y_sub_path, mode='w+', dtype='int64', shape=(num_samples, n_vertex)) 
        Y_main_mem = np.lib.format.open_memmap(y_main_path, mode='w+', dtype='int64', shape=(num_samples, n_main))

        curr_data = data_enhanced[start_idx:end_idx]
        curr_sub = labels_sub[start_idx:end_idx]
        curr_main = labels_main[start_idx:end_idx]

        for i in tqdm(range(num_samples)):
            dyn_window = curr_data[i : i+args.n_his]  # 直接取原始窗口，不注入
            
            # ==================== 移除所有注入调用 ====================
            # 通道A（子分区）: 已移除
            # 通道B（主分区）: 已移除
            # =========================================================
            
            # 仅保存原始数据
            dyn_window_t = np.transpose(dyn_window, (2, 0, 1))
            
            if has_static:
                stat_window = np.broadcast_to(static_norm, (args.n_static_features, args.n_his, n_vertex))
                sample = np.concatenate([dyn_window_t, stat_window], axis=0)
            else:
                sample = dyn_window_t
            
            target_idx = i + args.n_his - 1
            X_mem[i] = sample
            Y_sub_mem[i] = curr_sub[target_idx]
            Y_main_mem[i] = curr_main[target_idx]

        del X_mem, Y_sub_mem, Y_main_mem
        gc.collect()

    process_and_save('train', 0, len_train)
    process_and_save('val', len_train, len_train + len_val)
    process_and_save('test', len_train + len_val, total_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='my_drainage_system')
    parser.add_argument('--n_his', type=int, default=32) 
    args = parser.parse_args()
    
    generate_data(args)