import argparse
import copy
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

    logger = logging.getLogger(f"ae_opt_pid_{pid}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(f"[AE-OPT PID {pid}] %(asctime)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"partition_{pid}_ae_opt.log"), mode="w")
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
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    uar = (tpr + tnr) / 2.0
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
    if len(train_errors) and len(val_errors):
        all_errors = np.concatenate([train_errors, val_errors])
    else:
        all_errors = train_errors if len(train_errors) else val_errors
    if len(all_errors) == 0:
        return np.array([0.0], dtype=np.float64)

    lo = float(np.min(all_errors))
    hi = float(np.max(all_errors))
    if np.isclose(lo, hi):
        return np.array([lo], dtype=np.float64)

    return np.linspace(lo, hi, num=max(int(n_steps), 2), dtype=np.float64)


def find_best_threshold(train_errors: np.ndarray, val_errors: np.ndarray, val_gt: np.ndarray, n_steps: int):
    thresholds = enumerate_thresholds(train_errors, val_errors, n_steps)
    best_thr = float(thresholds[0])
    best_metrics = None

    for thr in thresholds:
        pred = (val_errors >= thr).astype(np.int64)
        metrics = compute_cls_metrics(val_gt, pred, scores=val_errors)
        if best_metrics is None:
            best_thr, best_metrics = float(thr), metrics
            continue
        if metrics["uar"] > best_metrics["uar"] or (
            np.isclose(metrics["uar"], best_metrics["uar"]) and metrics["f1"] > best_metrics["f1"]
        ) or (
            np.isclose(metrics["uar"], best_metrics["uar"])
            and np.isclose(metrics["f1"], best_metrics["f1"])
            and metrics["recall"] > best_metrics["recall"]
        ):
            best_thr, best_metrics = float(thr), metrics

    return best_thr, best_metrics


def plot_val_error_distribution(val_errors, val_gt, threshold, epoch, pid, save_dir):
    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    normal_err = val_errors[val_gt == 0]
    anomaly_err = val_errors[val_gt == 1]

    plt.figure(figsize=(8, 5))
    if len(normal_err) > 0:
        plt.hist(normal_err, bins=50, alpha=0.6, density=True, label=f"normal (n={len(normal_err)})")
    if len(anomaly_err) > 0:
        plt.hist(anomaly_err, bins=50, alpha=0.6, density=True, label=f"anomaly (n={len(anomaly_err)})")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"thr={threshold:.6f}")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title(f"AE-OPT PID {pid} Epoch {epoch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"pid{pid}_epoch{epoch:04d}_val_error_dist.png"), dpi=150)
    plt.close()


class EMAHelper:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            current = model.state_dict()
            for k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(current[k].detach(), alpha=(1.0 - self.decay))

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)



def apply_denoise_augmentation(x: torch.Tensor, noise_std: float, mask_ratio: float):
    x_in = x
    if noise_std > 0:
        x_in = x_in + torch.randn_like(x_in) * noise_std
    if mask_ratio > 0:
        mask = (torch.rand_like(x_in) > mask_ratio).float()
        x_in = x_in * mask
    return x_in


def weighted_recon_loss(x_hat, x_true, hard_mining_q=0.9, hard_weight=1.5, base_loss="smoothl1"):
    if base_loss == "mse":
        elem = (x_hat - x_true) ** 2
    else:
        elem = torch.nn.functional.smooth_l1_loss(x_hat, x_true, reduction="none")

    sample_err = elem.mean(dim=(1, 2, 3))
    if hard_weight <= 1.0:
        return sample_err.mean(), sample_err

    with torch.no_grad():
        q = torch.quantile(sample_err.detach(), hard_mining_q)
        sample_w = torch.where(sample_err >= q, torch.tensor(hard_weight, device=sample_err.device), torch.tensor(1.0, device=sample_err.device))

    loss = (sample_err * sample_w).mean()
    return loss, sample_err


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
        raise ValueError(f"PID {pid}: no train samples after filtering.")

    train_set = Subset(train_base, final_train_indices.tolist())
    val_set = Subset(val_base, val_indices.tolist())
    test_set = Subset(test_base, test_indices.tolist())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = DCRNN_AE_Model(
        input_dim=train_base.X.shape[1],
        n_vertex=current_nodes,
        device=device,
        hidden_dim=args.hidden_dim,
        k_order=args.k_order,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    ema = EMAHelper(model, decay=args.ema_decay)

    best_score = -1.0
    patience_count = 0
    best_path = os.path.join(args.save_dir, f"model_ae_opt_{pid}_best.pth")

    logger.info(
        "Start pid=%d | train=%d val=%d test=%d | denoise(std=%.4f, mask=%.3f) | hard_mining(q=%.2f, w=%.2f)",
        pid,
        len(train_set),
        len(val_set),
        len(test_set),
        args.noise_std,
        args.mask_ratio,
        args.hard_mining_q,
        args.hard_weight,
    )

    def eval_with_model(eval_model, loader):
        all_err, all_labels, all_loss = [], [], []
        with torch.no_grad():
            for x, _, y_main in loader:
                x = align_dimensions(x.to(device), current_nodes, args.n_his)
                x_hat = eval_model(x, edge_index)
                batch_loss, sample_err = weighted_recon_loss(
                    x_hat,
                    x,
                    hard_mining_q=args.hard_mining_q,
                    hard_weight=1.0,
                    base_loss=args.loss,
                )
                all_loss.append(float(batch_loss.item()))
                all_err.extend(sample_err.cpu().numpy().tolist())
                all_labels.extend(y_main.cpu().numpy().tolist())
        return np.array(all_err), np.array(all_labels), (float(np.mean(all_loss)) if all_loss else np.nan)

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"AE-OPT PID {pid} | Ep {epoch} [Train]", leave=False)
        for x, _, _ in loop:
            x = align_dimensions(x.to(device), current_nodes, args.n_his)
            x_input = apply_denoise_augmentation(x, args.noise_std, args.mask_ratio)

            optimizer.zero_grad()
            x_hat = model(x_input, edge_index)
            loss, _ = weighted_recon_loss(
                x_hat,
                x,
                hard_mining_q=args.hard_mining_q,
                hard_weight=args.hard_weight,
                base_loss=args.loss,
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)

            train_losses.append(loss.item())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # evaluate with EMA weights for stability
        eval_model = copy.deepcopy(model)
        ema.apply_to(eval_model)
        eval_model.eval()

        train_err, _, _ = eval_with_model(eval_model, train_loader)
        val_err, val_labels, val_loss = eval_with_model(eval_model, val_loader)
        val_gt = labels_to_binary(val_labels)
        epoch_thr, val_metrics = find_best_threshold(train_err, val_err, val_gt, args.threshold_steps)
        plot_val_error_distribution(val_err, val_gt, epoch_thr, epoch, pid, args.save_dir)

        avg_train = float(np.mean(train_losses)) if train_losses else np.nan
        logger.info(
            "Ep %d | train=%.6f val=%.6f lr=%.6g thr=%.6f | UAR=%.4f F1=%.4f Rec=%.4f Prec=%.4f MCC=%.4f PR-AUC=%s",
            epoch,
            avg_train,
            val_loss,
            scheduler.get_last_lr()[0],
            epoch_thr,
            val_metrics["uar"],
            val_metrics["f1"],
            val_metrics["recall"],
            val_metrics["precision"],
            val_metrics["mcc"],
            f"{val_metrics['pr_auc']:.4f}" if val_metrics["pr_auc"] is not None else "NA",
        )

        score = val_metrics["uar"] + 0.1 * val_metrics["f1"]
        if score > best_score:
            best_score = score
            patience_count = 0
            torch.save({"model": eval_model.state_dict(), "epoch": epoch}, best_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.info("Early stop at epoch=%d", epoch)
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def collect_errors(loader):
        sample_errs, node_errs, labels = [], [], []
        with torch.no_grad():
            for x, _, y_main in loader:
                x = align_dimensions(x.to(device), current_nodes, args.n_his)
                x_hat = model(x, edge_index)
                sample_err, node_err = model.reconstruction_error(x_hat, x)
                sample_errs.extend(sample_err.cpu().numpy().tolist())
                node_errs.extend(node_err.cpu().numpy().tolist())
                labels.extend(y_main.cpu().numpy().tolist())
        return np.array(sample_errs), np.array(node_errs), np.array(labels)

    train_sample_err, _, _ = collect_errors(train_loader)
    val_sample_err, _, val_labels = collect_errors(val_loader)
    test_sample_err, test_node_err, test_labels = collect_errors(test_loader)

    val_gt = labels_to_binary(val_labels)
    threshold, val_metrics = find_best_threshold(train_sample_err, val_sample_err, val_gt, args.threshold_steps)

    test_gt = labels_to_binary(test_labels)
    test_pred = (test_sample_err >= threshold).astype(np.int64)
    test_metrics = compute_cls_metrics(test_gt, test_pred, scores=test_sample_err)

    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, f"ae_opt_test_sample_err_pid{pid}.npy"), test_sample_err)
    np.save(os.path.join(args.save_dir, f"ae_opt_test_node_err_pid{pid}.npy"), test_node_err)

    report = {
        "pid": pid,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "threshold": float(threshold),
        "settings": {
            "denoise_noise_std": args.noise_std,
            "denoise_mask_ratio": args.mask_ratio,
            "hard_mining_q": args.hard_mining_q,
            "hard_weight": args.hard_weight,
            "ema_decay": args.ema_decay,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    report_path = os.path.join(args.save_dir, f"ae_opt_report_pid{pid}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "Final test | UAR=%.4f F1=%.4f Rec=%.4f Prec=%.4f MCC=%.4f PR-AUC=%s ROC-AUC=%s",
        test_metrics["uar"],
        test_metrics["f1"],
        test_metrics["recall"],
        test_metrics["precision"],
        test_metrics["mcc"],
        f"{test_metrics['pr_auc']:.4f}" if test_metrics["pr_auc"] is not None else "NA",
        f"{test_metrics['roc_auc']:.4f}" if test_metrics["roc_auc"] is not None else "NA",
    )
    logger.info("Saved report to %s", report_path)


def run_dispatcher(args):
    os.makedirs(args.save_dir, exist_ok=True)
    processes = []
    gpu_count = torch.cuda.device_count()
    gpu_ids = list(range(gpu_count)) if gpu_count > 0 else [-1]

    print("--- Launch AE-OPT training ---")
    for pid in range(args.n_partitions):
        target_device = gpu_ids[pid % len(gpu_ids)]
        device_arg = str(target_device) if target_device != -1 else "cpu"

        cmd = [
            sys.executable,
            "main_autoencoder_optimized.py",
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
            "--min_lr",
            str(args.min_lr),
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
            "--noise_std",
            str(args.noise_std),
            "--mask_ratio",
            str(args.mask_ratio),
            "--hard_mining_q",
            str(args.hard_mining_q),
            "--hard_weight",
            str(args.hard_weight),
            "--ema_decay",
            str(args.ema_decay),
            "--grad_clip",
            str(args.grad_clip),
        ]
        if args.train_on_all:
            cmd.append("--train_on_all")

        processes.append(subprocess.Popen(cmd))
        print(f"-> submit pid={pid} to device={target_device}")
        time.sleep(1)

    for p in processes:
        p.wait()
    print("--- AE-OPT all done ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_mode", default=True, action="store_true")
    parser.add_argument("--data_path", type=str, default="./GWI_data/data")
    parser.add_argument("--dataset", type=str, default="my_drainage_system")
    parser.add_argument("--save_dir", type=str, default="./models_output_ae_opt")

    parser.add_argument("--n_partitions", type=int, default=7)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--device_id", type=str, default="0")

    parser.add_argument("--n_his", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_ratio", type=float, default=1.0)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_order", type=int, default=2)
    parser.add_argument("--threshold_steps", type=int, default=250)
    parser.add_argument("--loss", type=str, choices=["mse", "smoothl1"], default="smoothl1")

    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--mask_ratio", type=float, default=0.05)
    parser.add_argument("--hard_mining_q", type=float, default=0.9)
    parser.add_argument("--hard_weight", type=float, default=1.5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--train_on_all", action="store_true")

    args = parser.parse_args()

    if args.worker_mode:
        train_worker(args)
    else:
        run_dispatcher(args)
