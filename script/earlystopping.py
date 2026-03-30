import math
import torch
import numpy as np # 导入 numpy 来处理无穷大

class EarlyStopping:
    """
    [已修改] 
    Early stops the training if a monitored metric doesn't improve after a given patience.
    可以工作在 'min' (e.g., loss) 或 'max' (e.g., F1-score) 模式。
    """
    def __init__(self, delta: float = 0.0, patience: int = 7, verbose: bool = True, 
                 path: str = 'checkpoint.pt', mode: str = 'min'):
        """
        Args:
            patience (int): ... (同上)
            verbose (bool): ... (同上)
            delta (float): ... (同上)
            path (str): ... (同上)
            mode (str): [新参数] 'min' 或 'max'. 
                        'min': 监控的指标越低越好 (e.g., loss)。
                        'max': 监控的指标越高越好 (e.g., f1-score, accuracy)。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.delta = delta
        self.mode = mode

        if self.mode not in ['min', 'max']:
            raise ValueError("mode 必须是 'min' 或 'max'")
        
        # 根据模式，初始化最佳分数
        if self.mode == 'min':
            self.best_score = np.inf
            self.val_metric_best = np.inf # 用于打印
        else:
            self.best_score = -np.inf
            self.val_metric_best = -np.inf # 用于打印

    def __call__(self, metric_value, model):
        
        improved = False
        
        if self.mode == 'min':
            # 模式: 越低越好
            # 检查 metric_value 是否比 (best_score - delta) 更好（更小）
            if metric_value < self.best_score - self.delta:
                improved = True
        else:
            # 模式: 越高越好
            # 检查 metric_value 是否比 (best_score + delta) 更好（更大）
            if metric_value > self.best_score + self.delta:
                improved = True

        # --- 根据是否改进来执行操作 ---
        if improved:
            # 指标已改进
            self.best_score = metric_value
            self.save_checkpoint(metric_value, model) # 保存新模型
            self.counter = 0
        else:
            # 指标未改进
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, metric_value, model):
        """保存模型（当指标改进时）"""
        if self.verbose:
            # 打印信息（注意：self.val_metric_best 是上一次的最佳值）
            print(f'Validation metric improved ({self.val_metric_best:.4f} --> {metric_value:.4f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_best = metric_value # 更新用于打印的最佳值