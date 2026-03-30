import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.DCRNN_AE_Model import DCRNN_AE_Model
from model.STGCN_AE_Model import STGCN_AE_Model
from script.dataloader import MmapDataset, load_graph_data
from script.localized_anomaly_scorer import LocalizedAnomalyScorer


def set_env(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def align_dimensions(x, n_vertex, n_his):
    _, d1, d2, d3 = x.shape
    if d1 == n_his:
        if d2 == n_vertex:
            return x
        if d3 == n_vertex:
            return x.permute(0, 1, 3, 2)
    elif d2 == n_his:
        if d1 == n_vertex:
            return x.permute(0, 2, 1, 3)
        if d3 == n_vertex:
            return x.permute(0, 2, 3, 1)
    elif d3 == n_his:
        if d2 == n_vertex:
            return x.permute(0, 3, 2, 1)
        if d1 == n_vertex:
            return x.permute(0, 3, 1, 2)
    return x


def get_partition_structure(adj_sub, grouping_matrix, pid, device):
    gm_np = grouping_matrix.cpu().numpy() if torch.is_tensor(grouping_matrix) else grouping_matrix
    node_indices = np.where(gm_np[:, pid] == 1)[0]
    if len(node_indices) == 0:
        raise ValueError(f"Partition {pid} is empty.")

    adj_part = adj_sub[node_indices, :][:, node_indices]
    adj_coo = adj_part.tocoo()
    row = torch.from_numpy(adj_coo.row).long()
    col = torch.from_numpy(adj_coo.col).long()
    edge_index = torch.stack([row, col], dim=0).to(device)
    return node_indices, edge_index


def build_logger(save_dir: str, pid: int):
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"ae_v2_pid_{pid}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(f"[AE-V2 PID {pid}] %(asctime)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"partition_{pid}_ae_v2.log"), mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def build_ratio_indices(total_len: int, ratio: float):
    if ratio >= 1.0:
        return np.arange(total_len)
    return np.arange(max(int(total_len * ratio), 1))


def select_normal_indices(dataset: MmapDataset, pid: int, candidate_indices: np.ndarray):
    labels = dataset.Y_main[candidate_indices, pid]
    return candidate_indices[labels == 0]


def labels_to_binary(y: np.ndarray):
    if y.ndim > 1:
        return (y.mean(axis=1) > 0).astype(np.int64)
    return (y > 0).astype(np.int64)


def build_model(args, input_dim, n_nodes, device):
    if args.model_type == "stgcn":
        return STGCN_AE_Model(
            input_dim=input_dim,
            n_vertex=n_nodes,
            device=device,
            hidden_dim=args.hidden_dim,
            k_order=args.k_order,
            k_t=args.k_t,
            dropout=args.dropout,
        ).to(device)
    return DCRNN_AE_Model(
        input_dim=input_dim,
        n_vertex=n_nodes,
        device=device,
        hidden_dim=args.hidden_dim,
        k_order=args.k_order,
    ).to(device)


def train_worker(args):
    pid = args.pid
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.device_id != "cpu" else "cpu")
    logger = build_logger(args.save_dir, pid)
    set_env(args.seed)

    full_dataset_path = os.path.join(args.data_path, args.dataset)
    adj_sub, _, grouping_matrix, _, _ = load_graph_data(full_dataset_path)
    node_indices, edge_index = get_partition_structure(adj_sub, grouping_matrix, pid, device)
    current_nodes = len(node_indices)

    processed_dir = os.path.join(full_dataset_path, f"processed_window_{args.n_his}")
    train_base = MmapDataset(processed_dir, "train", partition_id=pid, node_indices=node_indices, device=device)
    val_base = MmapDataset(processed_dir, "val", partition_id=pid, node_indices=node_indices, device=device)
    test_base = MmapDataset(processed_dir, "test", partition_id=pid, node_indices=node_indices, device=device)

    train_indices = build_ratio_indices(len(train_base), args.data_ratio)
    val_indices = build_ratio_indices(len(val_base), args.data_ratio)
    test_indices = build_ratio_indices(len(test_base), args.data_ratio)

    final_train_indices = train_indices if args.train_on_all else select_normal_indices(train_base, pid, train_indices)
    if len(final_train_indices) == 0:
        raise ValueError(f"PID {pid}: no normal train samples after filtering.")

    train_set = Subset(train_base, final_train_indices.tolist())
    val_set = Subset(val_base, val_indices.tolist())
    test_set = Subset(test_base, test_indices.tolist())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args, train_base.X.shape[1], current_nodes, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss() if args.loss == "smoothl1" else nn.MSELoss()

    scorer = LocalizedAnomalyScorer(
        node_topk_ratio=args.node_topk_ratio,
        time_topk_ratio=args.time_topk_ratio,
        mix_weights=(args.w_global, args.w_node, args.w_time, args.w_peak),
    )

    best_val = float("inf")
    patience_count = 0
    best_path = os.path.join(args.save_dir, f"model_ae_v2_{pid}_best.pth")

    logger.info(
        "Start pid=%d | model=%s | train=%d val=%d test=%d | scorer(topk_node=%.2f, topk_time=%.2f)",
        pid,
        args.model_type,
        len(train_set),
        len(val_set),
        len(test_set),
        args.node_topk_ratio,
        args.time_topk_ratio,
    )

    def collect_outputs(loader):
        x_all, xhat_all, labels_all, losses = [], [], [], []
        with torch.no_grad():
            for x, _, y_main in loader:
                x = align_dimensions(x.to(device), current_nodes, args.n_his)
                x_hat = model(x, edge_index)
                losses.append(float(criterion(x_hat, x).item()))
                x_all.append(x.cpu().numpy())
                xhat_all.append(x_hat.cpu().numpy())
                labels_all.append(y_main.cpu().numpy())

        x_all = np.concatenate(x_all, axis=0) if x_all else np.empty((0, args.n_his, current_nodes, train_base.X.shape[1]))
        xhat_all = np.concatenate(xhat_all, axis=0) if xhat_all else np.empty_like(x_all)
        labels_all = np.concatenate(labels_all, axis=0) if labels_all else np.empty((0,))
        avg_loss = float(np.mean(losses)) if losses else np.nan
        return x_all, xhat_all, labels_all, avg_loss

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"AE-V2 PID {pid} | Ep {epoch} [Train]", leave=False)

        for x, _, _ in loop:
            x = align_dimensions(x.to(device), current_nodes, args.n_his)
            optimizer.zero_grad()
            x_hat = model(x, edge_index)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        train_x, train_xhat, _, _ = collect_outputs(train_loader)
        val_x, val_xhat, val_labels, avg_val = collect_outputs(val_loader)

        train_scores = scorer.score(train_xhat, train_x)["fused"]
        val_scores = scorer.score(val_xhat, val_x)["fused"]
        val_gt = labels_to_binary(val_labels)
        threshold, val_metrics = scorer.fit_threshold_by_validation(val_scores, val_gt, n_steps=args.threshold_steps)

        logger.info(
            "Ep %d | train=%.6f val=%.6f thr=%.6f | UAR=%.4f F1=%.4f Rec=%.4f Prec=%.4f",
            epoch,
            float(np.mean(train_losses)) if train_losses else np.nan,
            avg_val,
            threshold,
            val_metrics["uar"],
            val_metrics["f1"],
            val_metrics["recall"],
            val_metrics["precision"],
        )

        if avg_val < best_val:
            best_val = avg_val
            patience_count = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.info("Early stop at epoch=%d", epoch)
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    train_x, train_xhat, _, _ = collect_outputs(train_loader)
    val_x, val_xhat, val_labels, _ = collect_outputs(val_loader)
    test_x, test_xhat, test_labels, _ = collect_outputs(test_loader)

    train_comp = scorer.score(train_xhat, train_x)
    val_comp = scorer.score(val_xhat, val_x)
    test_comp = scorer.score(test_xhat, test_x)

    val_gt = labels_to_binary(val_labels)
    threshold, val_metrics = scorer.fit_threshold_by_validation(val_comp["fused"], val_gt, n_steps=args.threshold_steps)

    test_gt = labels_to_binary(test_labels)
    test_pred = (test_comp["fused"] >= threshold).astype(np.int64)
    test_metrics = scorer._metrics(test_gt, test_pred)

    report = {
        "pid": pid,
        "model_type": args.model_type,
        "threshold": float(threshold),
        "scorer": {
            "node_topk_ratio": args.node_topk_ratio,
            "time_topk_ratio": args.time_topk_ratio,
            "mix_weights": [args.w_global, args.w_node, args.w_time, args.w_peak],
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    report_path = os.path.join(args.save_dir, f"ae_v2_report_pid{pid}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(args.save_dir, f"ae_v2_test_fused_score_pid{pid}.npy"), test_comp["fused"])
    np.save(os.path.join(args.save_dir, f"ae_v2_test_node_err_pid{pid}.npy"), test_comp["node_err"])
    np.save(os.path.join(args.save_dir, f"ae_v2_test_time_err_pid{pid}.npy"), test_comp["time_err"])

    logger.info("Saved report to %s", report_path)
    logger.info(
        "Final test: UAR=%.4f F1=%.4f Rec=%.4f Prec=%.4f",
        test_metrics["uar"],
        test_metrics["f1"],
        test_metrics["recall"],
        test_metrics["precision"],
    )


def run_dispatcher(args):
    os.makedirs(args.save_dir, exist_ok=True)
    processes = []
    gpu_count = torch.cuda.device_count()
    gpu_ids = list(range(gpu_count)) if gpu_count > 0 else [-1]

    print("--- Launch AE-V2 ---")
    for pid in range(args.n_partitions):
        target_device = gpu_ids[pid % len(gpu_ids)]
        device_arg = str(target_device) if target_device != -1 else "cpu"

        cmd = [
            sys.executable,
            "main_autoencoder_v2.py",
            "--worker_mode",
            "--pid",
            str(pid),
            "--device_id",
            device_arg,
            "--data_path",
            args.data_path,
            "--dataset",
            args.dataset,
            "--save_dir",
            args.save_dir,
            "--n_his",
            str(args.n_his),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
            "--data_ratio",
            str(args.data_ratio),
            "--hidden_dim",
            str(args.hidden_dim),
            "--k_order",
            str(args.k_order),
            "--k_t",
            str(args.k_t),
            "--dropout",
            str(args.dropout),
            "--threshold_steps",
            str(args.threshold_steps),
            "--loss",
            str(args.loss),
            "--model_type",
            args.model_type,
            "--node_topk_ratio",
            str(args.node_topk_ratio),
            "--time_topk_ratio",
            str(args.time_topk_ratio),
            "--w_global",
            str(args.w_global),
            "--w_node",
            str(args.w_node),
            "--w_time",
            str(args.w_time),
            "--w_peak",
            str(args.w_peak),
        ]
        if args.train_on_all:
            cmd.append("--train_on_all")

        processes.append(subprocess.Popen(cmd))
        print(f"-> submit pid={pid} to device={target_device}")
        time.sleep(1)

    for p in processes:
        p.wait()
    print("--- AE-V2 all done ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_mode", default=True, action="store_true")

    parser.add_argument("--data_path", type=str, default="./GWI_data/data")
    parser.add_argument("--dataset", type=str, default="my_drainage_system")
    parser.add_argument("--save_dir", type=str, default="./models_output_ae_v2")

    parser.add_argument("--n_partitions", type=int, default=7)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--device_id", type=str, default="0")

    parser.add_argument("--n_his", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_ratio", type=float, default=1.0)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_order", type=int, default=2)
    parser.add_argument("--k_t", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold_steps", type=int, default=250)
    parser.add_argument("--loss", type=str, choices=["mse", "smoothl1"], default="mse")

    parser.add_argument("--model_type", type=str, choices=["dcrnn", "stgcn"], default="dcrnn")
    parser.add_argument("--node_topk_ratio", type=float, default=0.2)
    parser.add_argument("--time_topk_ratio", type=float, default=0.2)
    parser.add_argument("--w_global", type=float, default=0.2)
    parser.add_argument("--w_node", type=float, default=0.35)
    parser.add_argument("--w_time", type=float, default=0.35)
    parser.add_argument("--w_peak", type=float, default=0.1)

    parser.add_argument("--train_on_all", action="store_true")

    args = parser.parse_args()

    if args.worker_mode:
        train_worker(args)
    else:
        run_dispatcher(args)
