# AE 优化版（main_autoencoder_optimized.py）

该文件是相对于 `main_autoencoder.py` 的“增强版新入口”，不改原文件。

## 主要增强

1. **去噪训练**：输入端可加高斯噪声与随机 mask（`--noise_std`、`--mask_ratio`），提升泛化。  
2. **Hard Mining 重建损失**：对高误差样本加权训练（`--hard_mining_q`、`--hard_weight`），强化异常边界。  
3. **EMA 评估权重**：训练期维护参数 EMA（`--ema_decay`），验证更稳定。  
4. **Cosine 学习率 + 梯度裁剪**：更平稳收敛（`--min_lr`、`--grad_clip`）。  
5. **阈值自动搜索**：每个 epoch 枚举阈值，按 UAR/F1/Recall 选最优阈值。

## 单分区运行示例

```bash
python main_autoencoder_optimized.py \
  --worker_mode \
  --pid 0 \
  --device_id 0 \
  --data_path ./GWI_data/data \
  --dataset my_drainage_system \
  --save_dir ./models_output_ae_opt \
  --threshold_steps 250 \
  --noise_std 0.01 \
  --mask_ratio 0.05 \
  --hard_mining_q 0.9 \
  --hard_weight 1.5
```

## 产物

- `model_ae_opt_<pid>_best.pth`
- `ae_opt_report_pid<pid>.json`
- `ae_opt_test_sample_err_pid<pid>.npy`
- `ae_opt_test_node_err_pid<pid>.npy`
- `figures/pid<pid>_epochXXXX_val_error_dist.png`
- `logs/partition_<pid>_ae_opt.log`
