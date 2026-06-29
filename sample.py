"""
JAX/Flax DDPM 采样（逆向扩散过程）
从 torch/sample.py 迁移，核心逻辑保持一致：
  1. 逆向扩散 p_sample
  2. CFG (Classifier-Free Guidance) 条件生成
  3. 加载 Orbax checkpoint
  4. 保存图片
"""

import os
import argparse
import math

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from flax.training.checkpoints import restore_checkpoint

from model.unet_classify import UNet
import config


# ============================================================
# 1. 逆向扩散 p_sample（纯函数）
# ============================================================
def p_sample(
    xt: jnp.ndarray,
    t: int,
    pred_noise: jnp.ndarray,
    beta: jnp.ndarray,
    alpha: jnp.ndarray,
    alpha_bar: jnp.ndarray,
) -> jnp.ndarray:
    """
    逆向扩散单步：从 x_t 恢复 x_{t-1}
    xt:        (B, H, W, C) 当前噪声图像
    t:         int  时间步（从 T-1 到 0）
    pred_noise: (B, H, W, C) 模型预测的噪声
    返回:      x_{t-1}
    """
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    beta_t = beta[t]

    # 均值
    mean = (1.0 / jnp.sqrt(alpha_t)) * (
        xt - (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t) * pred_noise
    )

    if t == 0:
        return mean

    # 方差
    alpha_bar_prev = alpha_bar[t - 1] if t > 0 else jnp.array(1.0)
    variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
    std = jnp.sqrt(variance)

    # 加噪得到 x_{t-1}
    noise = jax.random.normal(jax.random.PRNGKey(t), xt.shape)
    return mean + std * noise


def setup_diffusion(T, beta_start, beta_end):
    beta = jnp.linspace(beta_start, beta_end, T)
    alpha = 1.0 - beta
    alpha_bar = jnp.cumprod(alpha)
    return beta, alpha, alpha_bar


# ============================================================
# 2. CFG 采样循环
# ============================================================
@jax.jit
def generate_images(params, batch_size, image_size, channels, labels, T, beta, alpha, alpha_bar, cfg_w):
    """
    完整逆向扩散采样（已 JIT 编译）
    params:     模型权重 Pytree
    batch_size: int
    image_size: int  (32)
    channels:   int  (3)
    labels:     (B,) int32  类别标签
    cfg_w:      float  CFG 权重
    返回:       (B, H, W, C)  范围 [-1, 1]
    """
    # 起始纯噪声
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, image_size, image_size, channels))

    def body_fn(t, x):
        # 条件预测（有标签）
        pred_cond = model_apply(params, x, t, labels)
        # 无条件预测（标签=None）
        pred_uncond = model_apply(params, x, t, None)
        # CFG 合并
        pred_noise = pred_uncond + cfg_w * (pred_cond - pred_uncond)
        # 逆向一步
        return p_sample(x, t, pred_noise, beta, alpha, alpha_bar)

    # 逐步逆向：从 T-1 降到 0
    x = jax.lax.fori_loop(T - 1, -1, -1, body_fn, x)
    return x


def model_apply(params, x, t, labels):
    """模型前向"""
    return model.apply(params, x, t, labels)


# ============================================================
# 3. 加载 checkpoint
# ============================================================
def load_model(ckpt_path, device=None):
    """加载模型和 diffusion 参数"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = restore_checkpoint(ckpt_path, target=None)
    print(f"Loaded epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?')}")

    model_local = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        time_dim=config.time_dim,
        emb_size=config.emb_size,
        num_classes=10,
        num_groups=config.num_groups,
        num_heads=config.num_heads,
        dropout=0.0,  # 推理不用 dropout
    )
    return model_local, ckpt["params"], ckpt.get("opt_state")


def get_latest_run_dir():
    latest_link = os.path.join(config.checkpoint_base, "latest")
    if os.path.islink(latest_link):
        return os.readlink(latest_link)
    return None


def resolve_checkpoint(run_name=None, ckpt=None):
    """解析 checkpoint 路径"""
    if run_name is None:
        run_name = config.sample_run
    if ckpt is None:
        ckpt = config.sample_ckpt

    if run_name == "latest":
        run_dir = get_latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError("No latest run found")
    else:
        run_dir = os.path.join(config.checkpoint_base, run_name)

    if ckpt in ("best", "last"):
        ckpt_path = os.path.join(run_dir, ckpt)
    elif ckpt.isdigit():
        ckpt_path = os.path.join(run_dir, f"model_epoch_{ckpt}")
    else:
        ckpt_path = ckpt

    return ckpt_path, run_dir


# ============================================================
# 4. 图片保存
# ============================================================
def save_images(images, output_path, scale=8, padding=4, nrow=None):
    """
    images: (B, H, W, C) 范围 [-1, 1]
    保存为拼图
    """
    # [-1, 1] -> [0, 1]
    images = (images + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)

    B, H, W, C = images.shape
    C = int(C)
    scale = int(scale)
    padding = int(padding)

    # 放大
    if scale > 1:
        images = np.repeat(np.repeat(images, scale, axis=1), scale, axis=2)

    H_s, W_s = images.shape[1], images.shape[2]
    nrow = nrow or max(1, int(math.sqrt(B)))
    ncol = math.ceil(B / nrow)

    grid_h = ncol * H_s + (ncol + 1) * padding
    grid_w = nrow * W_s + (nrow + 1) * padding
    grid = np.ones((grid_h, grid_w, C))

    for idx in range(B):
        row = idx // nrow
        col = idx % nrow
        y = row * H_s + (row + 1) * padding
        x = col * W_s + (col + 1) * padding
        grid[y:y+H_s, x:x+W_s] = images[idx]

    # BHWC -> HWC (如果是灰度图会退化成 HWC)
    if C == 1:
        grid = grid[:, :, 0]

    img = Image.fromarray((grid * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path)
    print(f"Saved to {output_path}")


# ============================================================
# 5. 主采样流程
# ============================================================
def sample(
    num_images=16,
    run_name=None,
    ckpt=None,
    output_path=None,
    scale=None,
    padding=None,
    labels=None,
):
    cfg_w = config.sample_w
    scale = scale if scale is not None else config.sample_scale
    padding = padding if padding is not None else config.sample_padding

    ckpt_path, run_dir = resolve_checkpoint(run_name=run_name, ckpt=ckpt)
    if output_path is None:
        output_path = os.path.join(run_dir, "samples.png")

    print(f"Dataset: {config.dataset}")
    print(f"Run: {run_dir}")
    print(f"Checkpoint: {ckpt_path}")

    model_local, params, _ = load_model(ckpt_path)
    beta, alpha, alpha_bar = setup_diffusion(config.T, config.beta_start, config.beta_end)

    print(f"Generating {num_images} images with CFG weight={cfg_w}...")

    if labels is not None:
        if len(labels) != num_images:
            raise ValueError(f"labels count ({len(labels)}) must match num_images ({num_images})")
        labels = jnp.array(labels, dtype=jnp.int32)
    else:
        labels = jnp.arange(0, 10, dtype=jnp.int32)
        # 如果 num_images > 10，循环
        if num_images <= 10:
            labels = labels[:num_images]
        else:
            labels = jnp.tile(labels, (num_images // 10) + 1)[:num_images]

    print(f"Labels: {np.array(labels).tolist()}")

    # JIT 编译采样
    from functools import partial

    def do_sample(lbls):
        return _sample_loop(params, model_local, lbls, config.T, beta, alpha, alpha_bar, cfg_w,
                           config.image_size, config.in_channels)

    # 分批生成（如果 num_images 很大）
    batch_size = num_images
    images = do_sample(labels[:batch_size])

    # 恢复到 [-1, 1] 并 clamp
    images = jnp.clip(images, -1.0, 1.0)

    save_images(
        np.array(images),
        output_path,
        scale=scale,
        padding=padding,
    )
    export_size = config.image_size * scale
    print(f"Done ({export_size}x{export_size} per image)")


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def _sample_loop(params, model_local, labels, T, beta, alpha, alpha_bar, cfg_w, image_size, channels):
    batch_size = labels.shape[0]
    x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, image_size, image_size, channels))

    def body_fn(t, x):
        t_arr = jnp.full((batch_size,), t, dtype=jnp.int32)
        pred_cond = model_local.apply(params, x, t_arr, labels)
        pred_uncond = model_local.apply(params, x, t_arr, None)
        pred_noise = pred_uncond + cfg_w * (pred_cond - pred_uncond)
        return p_sample(x, t, pred_noise, beta, alpha, alpha_bar)

    x = jax.lax.fori_loop(T - 1, -1, -1, body_fn, x)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-images", type=int, default=16)
    parser.add_argument("-r", "--run", type=str, default=None, help="run folder name or latest")
    parser.add_argument("-c", "--ckpt", type=str, default=None, help="best, last, or epoch number")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-s", "--scale", type=int, default=None)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("-l", "--labels", type=int, nargs="+", default=None)
    args = parser.parse_args()

    labels = jnp.array(args.labels, dtype=jnp.int32) if args.labels is not None else None
    sample(
        num_images=args.num_images,
        run_name=args.run,
        ckpt=args.ckpt,
        output_path=args.output,
        scale=args.scale,
        padding=args.padding,
        labels=labels,
    )
