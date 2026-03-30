# main_autoencoder_v2.py 使用说明

`main_autoencoder_v2.py` 是在 `main_autoencoder.py` 基础上新增的入口（不改原文件），支持：
- `dcrnn` / `stgcn` 两种 AE 模型切换
- 局部异常鲁棒打分（global + node-topk + time-topk + peak 融合）

## 1) 单分区训练（默认 DCRNN）

```bash
python main_autoencoder_v2.py \
  --worker_mode \
  --pid 0 \
  --device_id 0 \
  --data_path ./GWI_data/data \
  --dataset my_drainage_system \
  --save_dir ./models_output_ae_v2 \
  --model_type dcrnn \
  --threshold_steps 250
```

## 2) 使用 STGCN AE

```bash
python main_autoencoder_v2.py \
  --worker_mode \
  --pid 0 \
  --device_id 0 \
  --model_type stgcn \
  --k_t 3 \
  --dropout 0.1
```

## 3) 多分区并行

```bash
python main_autoencoder_v2.py \
  --data_path ./GWI_data/data \
  --dataset my_drainage_system \
  --save_dir ./models_output_ae_v2 \
  --n_partitions 7 \
  --model_type stgcn
```

## 4) 局部异常打分参数

- `--node_topk_ratio`：节点 top-k 比例（默认 0.2）
- `--time_topk_ratio`：时间 top-k 比例（默认 0.2）
- `--w_global --w_node --w_time --w_peak`：融合权重

## 5) 输出文件

- `model_ae_v2_<pid>_best.pth`
- `ae_v2_report_pid<pid>.json`
- `ae_v2_test_fused_score_pid<pid>.npy`
- `ae_v2_test_node_err_pid<pid>.npy`
- `ae_v2_test_time_err_pid<pid>.npy`
- `logs/partition_<pid>_ae_v2.log`
