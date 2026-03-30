# Autoencoder 异常检测改造版

本改造新增了基于图时序自编码重建的异常检测流程，适用于极度类别不平衡场景。

## 新增文件

- `main_autoencoder.py`：训练/评估/调度入口。
- `model/DCRNN_AE_Model.py`：图时序自编码器模型。

## 训练（单分区）

```bash
python main_autoencoder.py \
  --worker_mode \
  --pid 0 \
  --device_id 0 \
  --data_path ./GWI_data/data \
  --dataset my_drainage_system \
  --save_dir ./models_output_ae \
  --n_his 32 \
  --epochs 100 \
  --batch_size 128 \
  --threshold_steps 200
```

## 多分区并行

```bash
python main_autoencoder.py \
  --data_path ./GWI_data/data \
  --dataset my_drainage_system \
  --save_dir ./models_output_ae \
  --n_partitions 7
```

## 输出

每个分区会输出：

- `model_ae_<pid>_best.pth`：最佳模型。
- `ae_report_pid<pid>.json`：阈值与指标（UAR、specificity、gmean、MCC、PR-AUC、ROC-AUC、acc/precision/recall/f1 + confusion matrix）。
- `ae_test_sample_err_pid<pid>.npy`：样本级重建误差。
- `ae_test_node_err_pid<pid>.npy`：节点级重建误差。
- `logs/partition_<pid>_ae.log`：训练日志。
- `figures/pid<pid>_epochXXXX_val_error_dist.png`：每个 epoch 的验证集误差分布图（正负样本）。

## 关键思路

1. 使用正常模式重建学习（loss: MSE 或 SmoothL1）。
2. 每个 epoch 枚举多个候选阈值（`--threshold_steps`），优先基于验证集 UAR（平衡准确率）挑选最优阈值，并结合 F1/Recall 做平局决策。
3. 测试集误差超过阈值判定为异常。

> 默认仅在训练阶段使用标签为 0 的正常样本来学习重建（更符合异常检测设定）。  
> 若你希望回退为“全部样本训练”，可追加参数 `--train_on_all`。
> 当前实现会在每个 epoch 保留完整验证集（正常+异常）并输出 `UAR/F1/Recall/Precision/ACC/MCC/PR-AUC`，用于观察检测效果。
