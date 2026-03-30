import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.DCRNN_AE_Model import DCRNN_AE_Model
from script.dataloader import MmapDataset, load_graph_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_env(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def align_dimensions(x, n_vertex, n_his):
    """Ensure x shape is (B, T, N, F)."""
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

    logger = logging.getLogger(f"ae_pid_{pid}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(f"[AE PID {pid}] %(asctime)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"partition_{pid}_ae.log"), mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def build_ratio_indices(total_len: int, ratio: float):
    if ratio >= 1.0:
        return np.arange(total_len)
    limit = int(total_len * ratio)
    return np.arange(max(limit, 1))


def select_normal_indices(dataset: MmapDataset, pid: int, candidate_indices: np.ndarray):
    labels = dataset.Y_main[candidate_indices, pid]
    return candidate_indices[labels == 0]


def labels_to_binary(y: np.ndarray):
    if y.ndim > 1:
        return (y.mean(axis=1) > 0).astype(np.int64)
    return (y > 0).astype(np.int64)


def compute_cls_metrics(gt: np.ndarray, pred: np.ndarray, scores: np.ndarray = None):
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity / recall+
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity / recall-
    uar = (tpr + tnr) / 2.0  # balanced accuracy / UAR
    gmean = float(np.sqrt(max(tpr * tnr, 0.0)))
    mcc = float(matthews_corrcoef(gt, pred)) if len(np.unique(gt)) > 1 else 0.0

    pr_auc = None
    roc_auc = None
    if scores is not None and len(np.unique(gt)) > 1:
        try:
            pr_auc = float(average_precision_score(gt, scores))
        except ValueError:
            pr_auc = None
        try:
            roc_auc = float(roc_auc_score(gt, scores))
        except ValueError:
            roc_auc = None

    return {
        "accuracy": float(accuracy_score(gt, pred)),
        "precision": float(precision_score(gt, pred, zero_division=0)),
        "recall": float(recall_score(gt, pred, zero_division=0)),
        "f1": float(f1_score(gt, pred, zero_division=0)),
        "specificity": float(tnr),
        "uar": float(uar),
        "gmean": gmean,
        "mcc": mcc,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def enumerate_thresholds(train_errors: np.ndarray, val_errors: np.ndarray, n_steps: int):
    train_errors = np.asarray(train_errors, dtype=np.float64)
    val_errors = np.asarray(val_errors, dtype=np.float64)
    all_errors = np.concatenate([train_errors, val_errors]) if len(train_errors) and len(val_errors) else (train_errors if len(train_errors) else val_errors)
    if len(all_errors) == 0:
        return np.array([0.0], dtype=np.float64)

    lo = float(np.min(all_errors))
    hi = float(np.max(all_errors))
    if np.isclose(lo, hi):
        return np.array([lo], dtype=np.float64)

    n_steps = max(int(n_steps), 2)
    return np.linspace(lo, hi, num=n_steps, dtype=np.float64)


def find_best_threshold(train_errors: np.ndarray, val_errors: np.ndarray, val_gt: np.ndarray, n_steps: int):
    thresholds = enumerate_thresholds(train_errors, val_errors, n_steps)
    best_threshold = float(thresholds[0])
    best_metrics = None

    for thr in thresholds:
        pred = (val_errors >= thr).astype(np.int64)
        metrics = compute_cls_metrics(val_gt, pred, scores=val_errors)
        if best_metrics is None:
            best_threshold = float(thr)
            best_metrics = metrics
            continue
        if metrics["uar"] > best_metrics["uar"] or (
            np.isclose(metrics["uar"], best_metrics["uar"]) and metrics["f1"] > best_metrics["f1"]
        ) or (
            np.isclose(metrics["uar"], best_metrics["uar"])
            and np.isclose(metrics["f1"], best_metrics["f1"])
            and metrics["recall"] > best_metrics["recall"]
        ):
            best_threshold = float(thr)
            best_metrics = metrics

    return best_threshold, best_metrics


def plot_val_error_distribution(
    val_errors: np.ndarray,
    val_gt: np.ndarray,
    threshold: float,
    epoch: int,
    pid: int,
    save_dir: str,
):
    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    normal_err = val_errors[val_gt == 0]
    anomaly_err = val_errors[val_gt == 1]

    plt.figure(figsize=(8, 5))
    if len(normal_err) > 0:
        plt.hist(normal_err, bins=40, alpha=0.6, density=True, label=f"normal (n={len(normal_err)})")
    if len(anomaly_err) > 0:
        plt.hist(anomaly_err, bins=40, alpha=0.6, density=True, label=f"anomaly (n={len(anomaly_err)})")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"threshold={threshold:.6f}")
    plt.title(f"PID {pid} Epoch {epoch} Validation Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f"pid{pid}_epoch{epoch:04d}_val_error_dist.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()


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

    if args.train_on_all:
        final_train_indices = train_indices
    else:
        final_train_indices = select_normal_indices(train_base, pid, train_indices)
        if len(final_train_indices) == 0:
            raise ValueError(f"PID {pid}: no normal samples in train split after filtering.")

    train_set = Subset(train_base, final_train_indices.tolist())
    # Validation keeps both normal and anomaly samples for epoch-wise detection metrics.
    val_set = Subset(val_base, val_indices.tolist())
    test_set = Subset(test_base, test_indices.tolist())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    input_dim = train_base.X.shape[1]

    model = DCRNN_AE_Model(
        input_dim=input_dim,
        n_vertex=current_nodes,
        device=device,
        hidden_dim=args.hidden_dim,
        k_order=args.k_order,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss() if args.loss == "smoothl1" else nn.MSELoss()

    best_val = float("inf")
    patience_count = 0
    best_path = os.path.join(args.save_dir, f"model_ae_{pid}_best.pth")

    logger.info(f"Start AE training pid={pid}, nodes={current_nodes}, input_dim={input_dim}")
    logger.info(f"Loss={args.loss}, threshold_steps={args.threshold_steps}")
    logger.info(
        "Split sizes -> train:%d val:%d test:%d | train_on_all=%s",
        len(train_set),
        len(val_set),
        len(test_set),
        args.train_on_all,
    )

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"AE PID {pid} | Ep {epoch} [Train]", leave=False)

        for x, _, _ in loop:
            x = x.to(device)
            x = align_dimensions(x, current_nodes, args.n_his)

            optimizer.zero_grad()
            x_hat = model(x, edge_index)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_losses = []
        train_errors = []
        with torch.no_grad():
            # threshold source: train reconstruction errors
            for x, _, _ in train_loader:
                x = x.to(device)
                x = align_dimensions(x, current_nodes, args.n_his)
                x_hat = model(x, edge_index)
                sample_err, _ = model.reconstruction_error(x_hat, x)
                train_errors.extend(sample_err.cpu().numpy().tolist())

            val_errors = []
            val_labels = []
            for x, _, y_main in val_loader:
                x = x.to(device)
                x = align_dimensions(x, current_nodes, args.n_his)
                x_hat = model(x, edge_index)
                vloss = criterion(x_hat, x)
                val_losses.append(vloss.item())
                sample_err, _ = model.reconstruction_error(x_hat, x)
                val_errors.extend(sample_err.cpu().numpy().tolist())
                val_labels.extend(y_main.cpu().numpy().tolist())

        avg_train = float(np.mean(train_losses)) if train_losses else np.nan
        avg_val = float(np.mean(val_losses)) if val_losses else np.nan
        val_errors = np.array(val_errors, dtype=np.float64)
        val_gt = labels_to_binary(np.array(val_labels))
        epoch_threshold, epoch_metrics = find_best_threshold(
            train_errors=np.array(train_errors, dtype=np.float64),
            val_errors=val_errors,
            val_gt=val_gt,
            n_steps=args.threshold_steps,
        )
        plot_val_error_distribution(
            val_errors=val_errors,
            val_gt=val_gt,
            threshold=epoch_threshold,
            epoch=epoch,
            pid=pid,
            save_dir=args.save_dir,
        )
        logger.info(
            "Epoch %d: train=%.6f val=%.6f thr=%.6f | uar=%.4f f1=%.4f rec=%.4f prec=%.4f acc=%.4f mcc=%.4f pr_auc=%s",
            epoch,
            avg_train,
            avg_val,
            epoch_threshold,
            epoch_metrics["uar"],
            epoch_metrics["f1"],
            epoch_metrics["recall"],
            epoch_metrics["precision"],
            epoch_metrics["accuracy"],
            epoch_metrics["mcc"],
            f"{epoch_metrics['pr_auc']:.4f}" if epoch_metrics["pr_auc"] is not None else "NA",
        )

        if avg_val < best_val:
            best_val = avg_val
            patience_count = 0
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved: val={best_val:.6f}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.info(f"Early stop at epoch={epoch}")
                break

    # Evaluation: load best and compute thresholds/scores
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    def collect_errors(loader):
        all_sample_err = []
        all_node_err = []
        all_labels = []
        with torch.no_grad():
            for x, _, y_main in loader:
                x = x.to(device)
                x = align_dimensions(x, current_nodes, args.n_his)
                x_hat = model(x, edge_index)
                sample_err, node_err = model.reconstruction_error(x_hat, x)
                all_sample_err.extend(sample_err.cpu().numpy().tolist())
                all_node_err.extend(node_err.cpu().numpy().tolist())
                all_labels.extend(y_main.cpu().numpy().tolist())
        return np.array(all_sample_err), np.array(all_node_err), np.array(all_labels)

    val_sample_err, _, val_labels = collect_errors(val_loader)
    test_sample_err, test_node_err, test_labels = collect_errors(test_loader)

    # Enumerate thresholds and choose the best one by validation UAR/F1.
    train_sample_err, _, _ = collect_errors(train_loader)
    val_gt = labels_to_binary(val_labels)
    threshold, val_metrics = find_best_threshold(
        train_errors=train_sample_err,
        val_errors=val_sample_err,
        val_gt=val_gt,
        n_steps=args.threshold_steps,
    )
    test_pred = (test_sample_err >= threshold).astype(np.int64)

    gt = labels_to_binary(test_labels)
    test_metrics = compute_cls_metrics(gt, test_pred, scores=test_sample_err)

    report = {
        "pid": pid,
        "threshold": threshold,
        "val_error_mean": float(np.mean(val_sample_err)) if len(val_sample_err) else None,
        "test_error_mean": float(np.mean(test_sample_err)) if len(test_sample_err) else None,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    report_path = os.path.join(args.save_dir, f"ae_report_pid{pid}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(args.save_dir, f"ae_test_sample_err_pid{pid}.npy"), test_sample_err)
    np.save(os.path.join(args.save_dir, f"ae_test_node_err_pid{pid}.npy"), test_node_err)

    logger.info(f"Final report saved: {report_path}")
    logger.info(
        "Final test metrics: uar=%.4f f1=%.4f rec=%.4f prec=%.4f acc=%.4f mcc=%.4f pr_auc=%s roc_auc=%s",
        test_metrics["uar"],
        test_metrics["f1"],
        test_metrics["recall"],
        test_metrics["precision"],
        test_metrics["accuracy"],
        test_metrics["mcc"],
        f"{test_metrics['pr_auc']:.4f}" if test_metrics["pr_auc"] is not None else "NA",
        f"{test_metrics['roc_auc']:.4f}" if test_metrics["roc_auc"] is not None else "NA",
    )


def run_dispatcher(args):
    os.makedirs(args.save_dir, exist_ok=True)
    processes = []
    gpu_count = torch.cuda.device_count()
    gpu_ids = list(range(gpu_count)) if gpu_count > 0 else [-1]

    print("--- Launch AE training ---")
    for pid in range(args.n_partitions):
        target_device = gpu_ids[pid % len(gpu_ids)]
        device_arg = str(target_device) if target_device != -1 else "cpu"

        cmd = [
            sys.executable,
            "main_autoencoder.py",
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
            "--threshold_steps",
            str(args.threshold_steps),
            "--loss",
            str(args.loss),
        ]
        if args.train_on_all:
            cmd.append("--train_on_all")

        processes.append(subprocess.Popen(cmd))
        print(f"-> submit pid={pid} to device={target_device}")
        time.sleep(1)

    for p in processes:
        p.wait()
    print("--- AE all done ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_mode", default=True, action="store_true")

    parser.add_argument("--data_path", type=str, default="./GWI_data/data")
    parser.add_argument("--dataset", type=str, default="my_drainage_system")
    parser.add_argument("--save_dir", type=str, default="./models_output_ae")

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
    parser.add_argument(
        "--threshold_steps",
        type=int,
        default=200,
        help="Number of candidate thresholds to enumerate each epoch and at final evaluation.",
    )
    parser.add_argument("--loss", type=str, choices=["mse", "smoothl1"], default="mse")
    parser.add_argument("--train_on_all", action="store_true", help="Use all samples (including anomalies) for AE training.")

    args = parser.parse_args()

    if args.worker_mode:
        train_worker(args)
    else:
        run_dispatcher(args)
