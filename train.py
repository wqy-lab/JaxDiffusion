"""
JAX/Flax DDPM 训练循环
从 torch/train.py 迁移，核心逻辑保持一致：
  1. 数据加载（MNIST / CIFAR10）
  2. 前向扩散 q_sample（加噪）
  3. 预测噪声的 MSE loss
  4. optax 梯度更新
  5. Orbax checkpointing
"""

import os
import argparse
import time

import numpy as np
from tqdm import tqdm

import torch                          # ← 加这个
from torch.utils.data import DataLoader  # ← 加这个
from torchvision import datasets, transforms  # ← 加这个

import jax
import jax.numpy as jnp
import optax
from flax.training import orbax_utils
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import orbax.checkpoint as ocp

from model.unet_classify import UNet
import config


# ============================================================
# 1. Diffusion 前向过程（纯函数）
# ============================================================
def q_sample(x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray, alpha_bar: jnp.ndarray):
    """
    前向扩散：对原始图像 x0 在时间步 t 加噪声得到 noisy image。
    x0:     (B, H, W, C)  范围 [-1, 1]
    t:      (B,) int32     时间步索引
    noise:  (B, H, W, C)   高斯噪声
    alpha_bar: (T,)        累计 alpha
    返回:   noisy image
    """
    t = jnp.asarray(t, dtype=jnp.int32)
    alpha_bar_t = alpha_bar[t]
    co1 = jnp.sqrt(alpha_bar_t).reshape(-1, 1, 1, 1)
    co2 = jnp.sqrt(1 - alpha_bar_t).reshape(-1, 1, 1, 1)
    return co1 * x0 + co2 * noise


def setup_diffusion(T, beta_start, beta_end):
    """创建扩散过程参数（torch 版的 Diffusion 类纯函数化）"""
    beta = jnp.linspace(beta_start, beta_end, T)
    alpha = 1.0 - beta
    alpha_bar = jnp.cumprod(alpha)
    return beta, alpha_bar


# ============================================================
# 2. 训练 step（JIT 编译）
# ============================================================
def create_train_step(model, optimizer, alpha_bar):
    """返回编译后的 train_step 函数"""
    @jax.jit
    def train_step(params, opt_state, batch, t, labels, rng):
        """
        params:     模型参数 Pytree
        opt_state:  optax 优化器状态
        batch:      (B, H, W, C) 原始图像
        t:          (B,) int32    时间步
        labels:     (B,) int32    类别标签
        rng:        JAX RNG key
        返回:       (params, opt_state, loss, new_rng)
        """
        # 生成噪声
        noise = jax.random.normal(rng, batch.shape)

        # 加噪
        noisy_images = q_sample(batch, t, noise, alpha_bar)

        # 预测噪声 MSE Loss
        def loss_fn(p):
            pred = model.apply(p, noisy_images, t, labels)
            return jnp.mean((pred - noise) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    return train_step


# ============================================================
# 3. 数据加载（复用 PyTorch DataLoader，转为 JAX 数组）
# ============================================================
def numpy_collate(batch):
    """PyTorch DataLoader collate_fn：将 Tensors 转为 JAX 数组"""
    images = jnp.array(np.stack([x[0].numpy() for x in batch]))
    labels = jnp.array(np.stack([x[1].numpy() for x in batch]))
    # PyTorch: (B, C, H, W) -> JAX: (B, H, W, C)
    images = images.transpose(0, 2, 3, 1).astype(jnp.float32)
    return images, labels.astype(jnp.int32)


def get_data_loader():
    """构建 PyTorch DataLoader"""
    if config.dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])
        dataset = datasets.MNIST("./data", train=True, transform=transform, download=True)
    elif config.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(config.image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10("./data", train=True, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=numpy_collate,
    )


# ============================================================
# 4. Checkpoint 工具（Orbax）
# ============================================================
def get_latest_run_dir():
    latest_link = os.path.join(config.checkpoint_base, "latest")
    if os.path.islink(latest_link):
        return os.readlink(latest_link)
    return None


def resolve_init_checkpoint(init_checkpoint):
    if init_checkpoint is None:
        return None
    if init_checkpoint == "latest":
        run_dir = get_latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError(f"No previous run found under {config.checkpoint_base}")
        return os.path.join(run_dir, "best")
    return init_checkpoint


def save_checkpoint_jax(ckpt_dir, params, opt_state, step, loss, timestamp, run_dir):
    """保存 Orbax checkpoint"""
    ckpt = {
        "params": params,
        "opt_state": opt_state,
        "epoch": step,
        "loss": float(loss),
        "timestamp": timestamp,
        "run_dir": run_dir,
    }
    save_checkpoint(ckpt_dir, target=ckpt, step=step, keep=1)
    meta = {
        "epoch": step,
        "loss": float(loss),
        "timestamp": timestamp,
        "run_dir": run_dir,
    }
    os.makedirs(ckpt_dir, exist_ok=True)
    np.save(os.path.join(ckpt_dir, "meta.npy"), meta)


def load_init_weights(params, init_checkpoint):
    """从 checkpoint 恢复 params"""
    ckpt_path = resolve_init_checkpoint(init_checkpoint)
    if ckpt_path is None:
        print("Initializing model with random weights")
        return params, None
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Initial checkpoint not found: {ckpt_path}")
    ckpt = restore_checkpoint(ckpt_path, target=None)
    print(f"Loaded from {ckpt_path} (epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?')})")
    return ckpt["params"], ckpt.get("opt_state")


# ============================================================
# 5. 主训练流程
# ============================================================
def train(init_checkpoint=None):
    run_dir = os.path.join(config.checkpoint_base, time.strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # latest 软链接
    latest_link = os.path.join(config.checkpoint_base, "latest")
    if os.path.islink(latest_link):
        os.remove(latest_link)
    try:
        os.symlink(run_dir, latest_link)
    except OSError:
        pass

    print(f"Device: {config.device}")
    print(f"Dataset: {config.dataset}, image_size={config.image_size}, channels={config.in_channels}")
    print(f"Run directory: {run_dir}")

    train_loader = get_data_loader()

    # ---- 模型初始化 ----
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        time_dim=config.time_dim,
        emb_size=config.emb_size,
        num_classes=10,
        num_groups=config.num_groups,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )

    rng = jax.random.PRNGKey(0)
    rng, init_key = jax.random.split(rng)
    dummy_x = jnp.ones((1, config.image_size, config.image_size, config.in_channels))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_labels = jnp.zeros((1,), dtype=jnp.int32)

    params = model.init(init_key, dummy_x, dummy_t, dummy_labels)
    params, loaded_opt_state = load_init_weights(params, init_checkpoint)

    # ---- 优化器 ----
    optimizer = optax.adam(config.lr)
    opt_state = optimizer.init(params)

    # ---- 扩散参数 ----
    _, alpha_bar = setup_diffusion(config.T, config.beta_start, config.beta_end)

    # ---- 编译后的训练 step ----
    train_step = create_train_step(model, optimizer, alpha_bar)

    best_loss = float("inf")
    epochs_without_improvement = 0
    final_epoch = 0
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    rng = jax.random.PRNGKey(42)

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for step, (images, labels) in enumerate(pbar):
            rng, t_key = jax.random.split(rng)
            t = jax.random.randint(t_key, shape=(images.shape[0],), minval=0, maxval=config.T)

            params, opt_state, loss = train_step(params, opt_state, images, t, labels, t_key)

            epoch_loss += float(loss)
            pbar.set_postfix(loss=float(loss))

        avg_loss = epoch_loss / (step + 1)
        final_epoch = epoch + 1
        print(f"[Epoch {final_epoch}/{config.epochs}] avg_loss={avg_loss:.6f}")

        improved = avg_loss < best_loss - config.early_stopping_min_delta

        if (epoch + 1) % 10 == 0:
            ckpt_dir = os.path.join(run_dir, f"model_epoch_{epoch+1}")
            save_checkpoint_jax(ckpt_dir, params, opt_state, epoch + 1, avg_loss, timestamp, run_dir)
            print(f"Checkpoint saved: {ckpt_dir}")

        if improved:
            best_loss = avg_loss
            epochs_without_improvement = 0
            best_dir = os.path.join(run_dir, "best")
            save_checkpoint_jax(best_dir, params, opt_state, epoch + 1, best_loss, timestamp, run_dir)
            print(f"Best model updated: loss={best_loss:.6f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{config.early_stopping_patience}")
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}, best_loss={best_loss:.6f}")
                break

    last_dir = os.path.join(run_dir, "last")
    save_checkpoint_jax(last_dir, params, opt_state, final_epoch, avg_loss, timestamp, run_dir)
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default=None, help='"latest" or checkpoint path')
    args = parser.parse_args()
    init_checkpoint = args.init if args.init is not None else config.init_checkpoint
    train(init_checkpoint=init_checkpoint)
