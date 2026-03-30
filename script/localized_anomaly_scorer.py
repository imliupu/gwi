import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


class LocalizedAnomalyScorer:
    """
    Build anomaly scores that are robust to *localized* anomalies.

    Why: global mean error can dilute anomalies when only a few nodes/timestamps/features are abnormal.
    This scorer keeps localized signals via top-k and peak pooling.
    """

    def __init__(
        self,
        node_topk_ratio: float = 0.2,
        time_topk_ratio: float = 0.2,
        mix_weights=(0.2, 0.35, 0.35, 0.10),
    ):
        self.node_topk_ratio = float(node_topk_ratio)
        self.time_topk_ratio = float(time_topk_ratio)
        self.mix_weights = np.asarray(mix_weights, dtype=np.float64)
        if self.mix_weights.shape[0] != 4:
            raise ValueError("mix_weights must have 4 components: [global, node_topk, time_topk, peak].")
        s = self.mix_weights.sum()
        if s <= 0:
            raise ValueError("mix_weights sum must be > 0.")
        self.mix_weights = self.mix_weights / s

    @staticmethod
    def _safe_topk_mean(arr: np.ndarray, ratio: float, axis: int):
        ratio = min(max(ratio, 1e-6), 1.0)
        k = max(int(np.ceil(arr.shape[axis] * ratio)), 1)
        # partition is O(n), faster than full sort for large arrays
        part = np.partition(arr, kth=arr.shape[axis] - k, axis=axis)
        topk = np.take(part, indices=range(arr.shape[axis] - k, arr.shape[axis]), axis=axis)
        return topk.mean(axis=axis)

    def score_components(self, x_hat: np.ndarray, x: np.ndarray):
        """
        Args:
            x_hat, x: shape (B, T, N, F)

        Returns:
            dict of score components (all shape (B,)).
        """
        diff = np.abs(x_hat - x)

        # 1) global average error (baseline)
        s_global = diff.mean(axis=(1, 2, 3))

        # 2) node-localized: first mean over (T,F)->(B,N), then top-k nodes mean
        node_err = diff.mean(axis=(1, 3))  # (B,N)
        s_node_topk = self._safe_topk_mean(node_err, self.node_topk_ratio, axis=1)

        # 3) time-localized: first mean over (N,F)->(B,T), then top-k timesteps mean
        time_err = diff.mean(axis=(2, 3))  # (B,T)
        s_time_topk = self._safe_topk_mean(time_err, self.time_topk_ratio, axis=1)

        # 4) peak score: maximum local deviation (captures sharp spikes)
        s_peak = diff.max(axis=(1, 2, 3))

        return {
            "global": s_global,
            "node_topk": s_node_topk,
            "time_topk": s_time_topk,
            "peak": s_peak,
            "node_err": node_err,
            "time_err": time_err,
        }

    def score(self, x_hat: np.ndarray, x: np.ndarray):
        c = self.score_components(x_hat, x)
        # normalize each component by robust scale (median + MAD) to avoid scale dominance
        fused_terms = []
        for key in ["global", "node_topk", "time_topk", "peak"]:
            v = c[key].astype(np.float64)
            med = np.median(v)
            mad = np.median(np.abs(v - med)) + 1e-12
            fused_terms.append((v - med) / mad)
        fused = np.zeros_like(fused_terms[0])
        for w, t in zip(self.mix_weights, fused_terms):
            fused += w * t
        c["fused"] = fused
        return c

    @staticmethod
    def _labels_to_binary(y: np.ndarray):
        if y.ndim > 1:
            return (y.mean(axis=1) > 0).astype(np.int64)
        return (y > 0).astype(np.int64)

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        uar = (recall_pos + recall_neg) / 2.0
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "uar": float(uar),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    def fit_threshold_by_quantile(self, normal_scores: np.ndarray, q: float = 0.995):
        q = min(max(float(q), 0.0), 1.0)
        return float(np.quantile(normal_scores, q))

    def fit_threshold_by_validation(self, val_scores: np.ndarray, val_labels: np.ndarray, n_steps: int = 300):
        y = self._labels_to_binary(val_labels)
        lo = float(np.min(val_scores))
        hi = float(np.max(val_scores))
        if np.isclose(lo, hi):
            pred = (val_scores >= lo).astype(np.int64)
            return lo, self._metrics(y, pred)

        best_t = lo
        best_m = None
        grid = np.linspace(lo, hi, num=max(int(n_steps), 2))
        for t in grid:
            pred = (val_scores >= t).astype(np.int64)
            m = self._metrics(y, pred)
            if best_m is None:
                best_t, best_m = float(t), m
                continue
            if m["uar"] > best_m["uar"] or (np.isclose(m["uar"], best_m["uar"]) and m["f1"] > best_m["f1"]):
                best_t, best_m = float(t), m
        return best_t, best_m


__all__ = ["LocalizedAnomalyScorer"]
