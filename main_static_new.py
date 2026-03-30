import os
import sys
import argparse
import json
import subprocess
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

# 引入评估指标
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from model.DCRNN_Class_Model import DCRNN_Class_Model
from script.dataloader import load_graph_data, MmapDataset

# =========================================================
# 基础工具函数
# =========================================================
def set_env(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = False

def get_partition_structure(adj_sub, grouping_matrix, pid, device):
    if torch.is_tensor(grouping_matrix):
        gm_np = grouping_matrix.cpu().numpy()
    else:
        gm_np = grouping_matrix
    node_indices = np.where(gm_np[:, pid] == 1)[0]
    if len(node_indices) == 0: raise ValueError(f"Partition {pid} empty!")
    adj_part = adj_sub[node_indices, :][:, node_indices]
    adj_coo = adj_part.tocoo()
    row = torch.from_numpy(adj_coo.row).long()
    col = torch.from_numpy(adj_coo.col).long()
    edge_index = torch.stack([row, col], dim=0).to(device)
    grouping_part = torch.ones((len(node_indices), 1), dtype=torch.float32).to(device)
    return node_indices, edge_index, grouping_part

def align_dimensions(x, n_vertex, n_his):
    """
    确保输出维度为 (Batch, Time, Nodes, Feat)
    """
    B, D1, D2, D3 = x.shape
    if D1 == n_his:
        if D2 == n_vertex: return x 
        if D3 == n_vertex: return x.permute(0, 1, 3, 2)
    elif D2 == n_his:
        if D1 == n_vertex: return x.permute(0, 2, 1, 3) 
        if D3 == n_vertex: return x.permute(0, 2, 3, 1) 
    elif D3 == n_his:
        if D2 == n_vertex: return x.permute(0, 3, 2, 1) 
        if D1 == n_vertex: return x.permute(0, 3, 1, 2) 
    return x 

# =========================================================
# [数据增强] 合成入渗生成器
# =========================================================
def inject_synthetic_anomaly(x, y_label, chance=0.5):
    """
    在线随机注入入渗波形。
    注意：输入 x 必须是 (Batch, Time, Nodes, Feat) 格式
    """
    if random.random() > chance:
        return x, y_label

    batch_size, n_his, n_nodes, n_feat = x.shape
    device = x.device
    x_aug = x.clone()
    y_aug = y_label.clone()
    
    if y_aug.dim() > 1:
        y_check = y_aug.mean(dim=1)
    else:
        y_check = y_aug

    for b in range(batch_size):
        if y_check[b] > 0: # 如果已经是异常，跳过
            continue
            
        num_targets = random.randint(1, min(3, n_nodes))
        target_nodes = np.random.choice(range(n_nodes), size=num_targets, replace=False)
        magnitude = random.uniform(2.0, 6.0)
        
        duration = random.randint(4, n_his) 
        start_t = random.randint(0, n_his - duration)
        
        t_range = torch.linspace(0, np.pi, duration).to(device)
        wave = torch.sin(t_range) * magnitude
        
        for node in target_nodes:
            x_aug[b, start_t : start_t+duration, node, 0] += wave
            
        if y_aug.dim() == 1:
            y_aug[b] = 1.0
        else:
            y_aug[b, :] = 1.0
        
    return x_aug, y_aug

# =========================================================
# 辅助打印函数
# =========================================================
def print_confusion_matrix(targets, preds, title="Confusion Matrix"):
    cm = confusion_matrix(targets, preds)
    if cm.shape == (1, 1):
        if targets[0] == 0:
            tn, fp, fn, tp = cm[0,0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0,0]
    else:
        tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "-"*45)
    print(f"      {title}")
    print("-"*45)
    print(f"{'':>15} {'Pred:0':>10} {'Pred:1':>10}")
    print(f"{'True:0':>15} {tn:>10} {fp:>10}")
    print(f"{'True:1':>15} {fn:>10} {tp:>10}")
    print("-" * 45)
    print(f"TN: {tn:<6} | FP: {fp:<6} | FN: {fn:<6} | TP: {tp:<6}")
    print("="*45 + "\n")
    return cm


def compute_cls_metrics(targets, preds, probs=None):
    targets = np.asarray(targets).astype(int)
    preds = np.asarray(preds).astype(int)

    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    uar = (recall_pos + recall_neg) / 2.0
    gmean = float(np.sqrt(max(recall_pos * recall_neg, 0.0)))
    mcc = float(matthews_corrcoef(targets, preds)) if len(np.unique(targets)) > 1 else 0.0

    pr_auc = None
    roc_auc = None
    if probs is not None and len(np.unique(targets)) > 1:
        try:
            pr_auc = float(average_precision_score(targets, probs))
        except ValueError:
            pr_auc = None
        try:
            roc_auc = float(roc_auc_score(targets, probs))
        except ValueError:
            roc_auc = None

    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "specificity": float(recall_neg),
        "uar": float(uar),
        "gmean": gmean,
        "mcc": mcc,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

# =========================================================
# 训练主逻辑
# =========================================================
def train_class_worker(args):
    pid = args.pid
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    
    log_dir = os.path.join(args.save_dir, "logs")
    if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter(f'[PID {pid}] %(asctime)s - %(message)s')
    fh = logging.FileHandler(os.path.join(log_dir, f"partition_{pid}_Class.log"), mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    strategy = "Original Distribution (No Sampler)" if args.no_sampler else f"Balanced Sampler (Target Ratio={args.target_ratio})"
    logging.info(f"Start Training PID {pid}. Strategy: {strategy}")
    
    if args.no_anomaly:
        logging.info("NOTICE: Data Augmentation (Synthetic Anomaly) is DISABLED.")
        
    set_env(args.seed)

    # 1. 加载数据
    full_dataset_path = os.path.join(args.data_path, args.dataset)
    try:
        adj_sub, _, grouping_matrix, _, _ = load_graph_data(full_dataset_path)
    except Exception as e:
        logging.error(f"Load graph failed: {e}")
        return

    node_indices, edge_index, grouping_part = get_partition_structure(adj_sub, grouping_matrix, pid, device)
    current_nodes = len(node_indices)
    
    processed_dir = os.path.join(full_dataset_path, f"processed_window_{args.n_his}")
    train_set = MmapDataset(processed_dir, 'train', partition_id=pid, node_indices=node_indices, device=device)
    val_set = MmapDataset(processed_dir, 'val', partition_id=pid, node_indices=node_indices, device=device)

    # =================================================================
    # 2. [关键修复] 先处理数据截断 (Data Ratio)，再决定是否采样
    # =================================================================
    all_train_labels = train_set.Y_main[:, pid]
    
    # 将截断逻辑移到最外层，保证无论是否使用 sampler 都会生效
    if args.data_ratio < 1.0:
        limit = int(len(train_set) * args.data_ratio)
        train_indices = list(range(limit))
        train_set = Subset(train_set, train_indices)
        
        # 注意：Subset之后，用于计算权重的标签也要切片
        train_labels = all_train_labels[train_indices]
        
        # 验证集同样截断
        val_limit = int(len(val_set) * args.data_ratio)
        val_set = Subset(val_set, range(val_limit))
        logging.info(f"Data Ratio applied: {args.data_ratio} (Train Size: {len(train_set)})")
    else:
        train_labels = all_train_labels

    # 统计当前（截断后）的正负样本分布
    num_pos = np.sum(train_labels == 1)
    num_neg = np.sum(train_labels == 0)
    logging.info(f"Current Data Distribution: Pos={num_pos}, Neg={num_neg}")

    # =================================================================
    # 3. 准备数据加载器 (Sampler 逻辑)
    # =================================================================
    if args.no_sampler:
        # 【模式 A】不使用采样器 (保留原始分布)
        logging.info(">>> Sampler is DISABLED by user request. Using Shuffle=True.")
        sampler = None
        
        if num_pos > 0:
            suggested_weight = num_neg / num_pos
            logging.info(f"Suggest using --pos_weight {suggested_weight:.2f} to handle imbalance.")
    else:
        # 【模式 B】使用平衡采样器
        if num_pos == 0 or num_neg == 0:
            logging.warning("Warning: One class has 0 samples, skipping sampler.")
            sampler = None
        else:
            logging.info(f"Calculating sampler weights for Target Pos Ratio = {args.target_ratio}...")
            target_pos_ratio = args.target_ratio
            target_neg_ratio = 1.0 - target_pos_ratio
            
            weight_pos = target_pos_ratio / num_pos
            weight_neg = target_neg_ratio / num_neg
            
            samples_weights = np.zeros(len(train_labels))
            samples_weights[train_labels == 0] = weight_neg
            samples_weights[train_labels == 1] = weight_pos
            
            samples_weights = torch.from_numpy(samples_weights).double()
            sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
            logging.info(f">>> Sampler created.")

    # 构建 Loader
    if sampler is not None:
        # 使用采样器时，shuffle必须为False
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=2)
    else:
        # 没有采样器时，必须开启 Shuffle
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 修正静态特征维度获取 (兼容 Subset)
    args.n_static_features = train_set.dataset.X.shape[1] - 1 if isinstance(train_set, Subset) else train_set.X.shape[1] - 1

    # 4. 模型与优化器
    model = DCRNN_Class_Model(args, current_nodes, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # [修改] 打印 Loss 权重信息
    if args.pos_weight > 0:
        pos_weight_val = args.pos_weight
    else:
        pos_weight_val = 1.0
    
    logging.info(f"Loss Function: BCEWithLogitsLoss (Pos Weight = {pos_weight_val})")
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_uar = -1.0
    best_f1 = -1.0
    patience_count = 0
    
    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"PID {pid} | Ep {epoch} [Train]", leave=False)
        
        train_preds_list = []
        train_targets_list = []
        train_probs_list = []
        
        for x, _, y_label in loop:
            x = x.to(device)
            y_label = y_label.to(device).float()
            
            x = align_dimensions(x, current_nodes, args.n_his)
            
            # 合成注入依然生效（这是数据增强，不是采样）
            if not args.no_anomaly:
                x, y_label = inject_synthetic_anomaly(x, y_label, chance=0.2)
            
            optimizer.zero_grad()
            logits = model(x, edge_index, grouping_part)
            
            if y_label.dim() == 1: y_label = y_label.unsqueeze(1)
            if y_label.shape[1] == 1: y_label = y_label.expand(-1, current_nodes)
            
            loss = criterion(logits, y_label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                train_preds_list.extend(preds.cpu().numpy().flatten())
                train_targets_list.extend(y_label.cpu().numpy().flatten())
                train_probs_list.extend(probs.cpu().numpy().flatten())
            
            loop.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)

        train_metrics = compute_cls_metrics(train_targets_list, train_preds_list, probs=train_probs_list)
        
        print("\n" + "#"*60)
        print(f" >>> Epoch {epoch} Summary")
        print("#"*60)
        print_confusion_matrix(train_targets_list, train_preds_list, title=f"TRAIN Set (PID {pid} Ep {epoch})")
        logging.info(
            "[TRAIN] Ep %d: Loss=%.4f | UAR=%.4f, F1=%.4f, Rec=%.4f, Prec=%.4f, Acc=%.4f, MCC=%.4f, PR-AUC=%s",
            epoch,
            avg_train_loss,
            train_metrics["uar"],
            train_metrics["f1"],
            train_metrics["recall"],
            train_metrics["precision"],
            train_metrics["accuracy"],
            train_metrics["mcc"],
            f"{train_metrics['pr_auc']:.4f}" if train_metrics["pr_auc"] is not None else "NA",
        )

        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []
        val_probs_list = []
        val_loss = 0
        
        with torch.no_grad():
            for x, _, y_label in val_loader:
                x = x.to(device)
                y_label = y_label.to(device).float()
                x = align_dimensions(x, current_nodes, args.n_his)
                logits = model(x, edge_index, grouping_part)
                
                if y_label.dim() == 1: y_label = y_label.unsqueeze(1)
                if y_label.shape[1] == 1: y_label = y_label.expand(-1, current_nodes)
                
                v_loss = criterion(logits, y_label)
                val_loss += v_loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                val_preds_list.extend(preds.cpu().numpy().flatten())
                val_targets_list.extend(y_label.cpu().numpy().flatten())
                val_probs_list.extend(probs.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = compute_cls_metrics(val_targets_list, val_preds_list, probs=val_probs_list)
        
        val_cm = print_confusion_matrix(val_targets_list, val_preds_list, title=f"VALIDATION Set (PID {pid} Ep {epoch})")
        logging.info(
            "[VAL]   Ep %d: Loss=%.4f | UAR=%.4f, F1=%.4f, Rec=%.4f, Prec=%.4f, Acc=%.4f, MCC=%.4f, PR-AUC=%s, ROC-AUC=%s",
            epoch,
            avg_val_loss,
            val_metrics["uar"],
            val_metrics["f1"],
            val_metrics["recall"],
            val_metrics["precision"],
            val_metrics["accuracy"],
            val_metrics["mcc"],
            f"{val_metrics['pr_auc']:.4f}" if val_metrics["pr_auc"] is not None else "NA",
            f"{val_metrics['roc_auc']:.4f}" if val_metrics["roc_auc"] is not None else "NA",
        )

        is_better = (val_metrics["uar"] > best_uar) or (
            np.isclose(val_metrics["uar"], best_uar) and val_metrics["f1"] > best_f1
        )
        if is_better and HAS_PLOT:
            try:
                plt.figure(figsize=(6, 5))
                sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Infiltration'], yticklabels=['Normal', 'Infiltration'])
                plt.title(f"VAL CM (PID {pid}, Ep {epoch}, UAR={val_metrics['uar']:.3f}, F1={val_metrics['f1']:.3f})")
                cm_path = os.path.join(args.save_dir, f"cm_val_pid{pid}_best.png")
                plt.savefig(cm_path)
                plt.close()
            except: pass

        if is_better:
            best_uar = val_metrics["uar"]
            best_f1 = val_metrics["f1"]
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_class_{pid}_best.pth"))
            report_path = os.path.join(args.save_dir, f"class_report_pid{pid}_best.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({"pid": pid, "epoch": epoch, "metrics": val_metrics}, f, ensure_ascii=False, indent=2)
            logging.info(f" >>> New Best Model Saved! Val UAR={val_metrics['uar']:.4f}, F1={val_metrics['f1']:.4f}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

# =========================================================
# 调度器
# =========================================================
def run_dispatcher(args):
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    processes = []
    gpu_count = torch.cuda.device_count()
    gpu_ids = list(range(gpu_count)) if gpu_count > 0 else [-1]
    
    print(f"--- 启动训练 ---")
    if args.no_sampler:
        print(">>> 模式: 原始分布 (无采样器, 使用 Shuffle)")
    else:
        print(f">>> 模式: 平衡采样 (Target Ratio={args.target_ratio})")
    
    for pid in range(args.n_partitions):
        target_device = gpu_ids[pid % len(gpu_ids)]
        device_arg = str(target_device) if target_device != -1 else 'cpu'
        
        cmd = [sys.executable, 'main_static_new.py', 
               '--worker_mode', 
               '--pid', str(pid), 
               '--device_id', device_arg,
               '--data_path', args.data_path, 
               '--dataset', args.dataset, 
               '--n_his', str(args.n_his),
               '--epochs', str(args.epochs), 
               '--batch_size', str(args.batch_size), 
               '--save_dir', args.save_dir,
               '--lr', str(args.lr), 
               '--patience', str(args.patience), 
               '--data_ratio', str(args.data_ratio),
               '--pos_weight', str(args.pos_weight),
               '--target_ratio', str(args.target_ratio)]
        
        if args.no_anomaly: cmd.append('--no_anomaly')
        if args.no_sampler: cmd.append('--no_sampler')
                
        processes.append(subprocess.Popen(cmd))
        print(f"-> 提交 PID {pid} 到 GPU {target_device}")
        time.sleep(1)
        
    for p in processes: p.wait() 
    print("--- 所有训练结束 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_mode', default=True, action='store_true')
    parser.add_argument('--data_path', type=str, default='./GWI_data/data')
    parser.add_argument('--dataset', type=str, default='my_drainage_system') 
    parser.add_argument('--save_dir', type=str, default='./models_output_class')
    
    parser.add_argument('--n_partitions', type=int, default=7)
    parser.add_argument('--n_static_features', type=int, default=0)
    
    parser.add_argument('--n_his', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--data_ratio', type=float, default=0.01)
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--n_classes', type=int, default=1)
    
    # 权重建议值: 如果不采样，建议设为 (负样本数/正样本数)，通常在 5~20 之间
    parser.add_argument('--pos_weight', type=float, default=1.0)
    
    parser.add_argument('--no_anomaly', action='store_true', help='Disable synthetic anomaly injection')
    parser.add_argument('--target_ratio', type=float, default=0.3)
    
    # [新增] 禁用采样器开关
    parser.add_argument('--no_sampler', action='store_true', help='Disable WeightedRandomSampler and use original data distribution')

    args = parser.parse_args()

    if args.worker_mode:
        train_class_worker(args)
    else:
        run_dispatcher(args)
