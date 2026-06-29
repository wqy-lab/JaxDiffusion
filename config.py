"""
JAX/Flax DDPM 配置
直接从 torch/config.py 迁移，结构和参数保持一致
"""

import jax

# ---- 数据集选择 ----
dataset = "cifar10"  # 'mnist' 或 'cifar10'

# ---- 扩散模型超参数 ----
T = 200               # 扩散步数
beta_start = 1e-4
beta_end = 0.02

# ---- 训练超参数 ----
batch_size = 128
lr = 1e-3
epochs = 200
early_stopping_patience = 200
early_stopping_min_delta = 1e-5

# ---- UNet 超参数 ----
base_channels = 64
channel_mults = [1, 2, 4, 8]
time_dim = 256
emb_size = 32
num_heads = 8
num_groups = 8
dropout = 0.2

# ---- 设备 ----
# jax.devices() 返回可用设备列表，cuda / tpu / cpu
_available = jax.devices()
if any("cuda" in str(d) for d in _available):
    device = "cuda"
elif any("tpu" in str(d) for d in _available):
    device = "tpu"
else:
    device = "cpu"

# ---- 数据集预设 ----
DATASET_PRESETS = {
    "mnist":   {"image_size": 32, "in_channels": 1, "out_channels": 1},
    "cifar10": {"image_size": 32, "in_channels": 3, "out_channels": 3},
}

_preset = DATASET_PRESETS[dataset]
image_size   = _preset["image_size"]
in_channels  = _preset["in_channels"]
out_channels = _preset["out_channels"]

# ---- Checkpoint 路径 ----
checkpoint_base = f"checkpoints/{dataset}"
init_checkpoint = None   # None = 随机初始化

# ---- 采样（sample.py） ----
sample_run   = "latest"
sample_ckpt  = "best"
sample_scale = 8          # 导出放大倍数：32->256
sample_padding = 4        # 拼图间距（像素）
sample_w     = 4          # CFG classifier-free guidance weight
