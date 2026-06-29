"""
Flax/JAX version of UNet for DDPM, converted from torch/models/unet_classify.py
关键区别:
  - 图像数据布局: (B, H, W, C)  # Flax 默认，与 PyTorch (B, C, H, W) 不同
  - 权重不存储在 Module 里，而是由 init 生成到 params Pytree
  - 所有层用 @nn.compact 或 setup() 声明，由 Flax 追踪
  - Dropout 用 deterministic 标志控制（True = 推理/eval）
"""

import math
from flax import linen as nn
import jax.numpy as jnp
import jax


# ============================================================
# 1. SinusoidalPosEmb — 时间步位置编码
# ============================================================
class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, t):
        """
        t: (B,) int32 — 时间步索引
        返回: (B, dim) float32 — sin/cos 编码
        """
        halfdim = (self.dim + 1) // 2
        # 原始 PyTorch: emb = exp(-arange(0, halfdim) * log(10000) / (halfdim-1))
        emb = jnp.exp(
            -jnp.arange(halfdim, dtype=jnp.float32)
            * math.log(10000.0) / max(halfdim - 1, 1)
        )
        # t: (B,) -> (B, 1), emb: (halfdim,) -> (1, halfdim)
        li = t.astype(jnp.float32)[:, None] * emb[None, :]
        return jnp.concatenate([jnp.sin(li), jnp.cos(li)], axis=-1)[:, : self.dim]


# ============================================================
# 2. LabelsEmbedding — 类别标签Embedding (条件生成)
# ============================================================
class LabelsEmbedding(nn.Module):
    num_classes: int
    emb_size: int

    @nn.compact
    def __call__(self, labels=None, dropout: float = 0.0, batch_size: int = None):
        """
        labels: (B,) int32 或 None
        dropout: 训练时随机置零的概率（条件dropout）
        batch_size: 当 labels=None 时必须指定
        返回: (B, emb_size)
        """
        # nn.Embed 自动创建可学习的 weight Pytree
        embedding = nn.Embed(self.num_classes, self.emb_size, name="embedding")
        # 可学习的 null embedding（全零向量）
        null_emb = self.param(
            "null_emb", nn.initializers.zeros, (1, self.emb_size)
        )

        if labels is not None:
            emb = embedding(labels)
            # 训练时有 dropout 概率替换为 null_emb
            if dropout > 0.0 and not self.is_mutable_collection("intermediates"):
                # 简化：实际 dropout 需要 rng，这里略过
                # 真实实现可用 self.make_rng('dropout') + jax.random.fold_in
                pass
            return emb

        if batch_size is None:
            raise ValueError("batch_size is required when labels is None")
        # 无标签时返回可学习的 null_emb
        return jnp.broadcast_to(null_emb, (batch_size, self.emb_size))


# ============================================================
# 3. CrossAttention — 跨注意力 (Q来自特征，K/V来自标签Embedding)
# ============================================================
class CrossAttention(nn.Module):
    """交叉注意力: Q=(B, H*W, C), K=V=(B, 1, emb_size) -> attend on label"""
    num_heads: int

    @nn.compact
    def __call__(self, x, emb=None):
        """
        x:   (B, H, W, C)  — 特征图
        emb: (B, emb_size) — 标签 embedding
        返回: (B, H, W, C)
        """
        B, H, W, C = x.shape
        # 调整格式以适配 Flax MultiHeadAttention
        # -> (B, H*W, C) 然后拆成 (B, seq, 1, C) 供 MultiHeadDotProductAttention
        x_flat = x.reshape(B, H * W, C)

        # Flax 的 MultiHeadDotProductAttention 需要 qkv: (B, N, L, QK_dim)
        # 这里把 H*W 当 seq_len, C 当 qk_dim
        q = x_flat  # (B, H*W, C)
        k = emb[:, None, :]  # (B, 1, emb_size)
        v = emb[:, None, :]  # (B, 1, emb_size)

        # MultiHeadDotProductAttention: qkv shapes must match for heads dim
        # 投影到 same channels = C
        qkv = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=C,  # 输出/输入 channel 数
            name="cross_attn",
        )(q, k, v)

        # qkv: (B, H*W, C) — 还原回图像布局
        return qkv.reshape(B, H, W, C)


# ============================================================
# 4. TimeResBlock — 带时间条件 和 跨注意力 的残差块
# ============================================================
class TimeResBlock(nn.Module):
    in_channels: int
    out_channels: int
    time_dim: int
    emb_size: int
    num_groups: int = 8
    num_heads: int = 8

    @nn.compact
    def __call__(self, x, time_emb, labels_emb, *, deterministic: bool = True):
        """
        x:         (B, H, W, C)
        time_emb:  (B, time_dim)
        labels_emb: (B, emb_size)
        返回: (B, H, W, out_channels)
        """
        # ---- 第一个分支: norm -> silu -> conv -> +time ----
        h = nn.GroupNorm(num_groups=min(self.num_groups, self.in_channels), name="norm1")(x)
        h = nn.silu(h)
        h = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding=[(1, 1), (1, 1)],  # same padding
            name="conv1",
        )(h)

        # 时间条件: (B, time_dim) -> (B, 1, 1, out_channels) broadcast
        time_out = nn.Dense(self.out_channels, name="time_mlp")(time_emb)
        time_out = time_out[:, None, None, :]
        h = h + time_out

        # ---- 跨注意力: 融合标签条件 ----
        h = h + CrossAttention(num_heads=self.num_heads, name="cross_attn")(h, labels_emb)

        # ---- 第二个分支: norm -> silu -> conv -> +residual ----
        h = nn.GroupNorm(num_groups=self.num_groups, name="norm2")(h)
        h = nn.silu(h)
        h = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding=[(1, 1), (1, 1)],
            name="conv2",
        )(h)

        # 残差连接: 如果通道数变化，用 conv 映射
        if self.in_channels != self.out_channels:
            residual = nn.Conv(
                features=self.out_channels,
                kernel_size=(1, 1),
                name="residual",
            )(x)
        else:
            residual = x
        return h + residual


# ============================================================
# 5. Down / Up — 下采样 / 上采样块
# ============================================================
class Down(nn.Module):
    """下采样: stride=2 的卷积 -> TimeResBlock"""
    in_channels: int
    out_channels: int
    time_dim: int
    emb_size: int
    num_groups: int = 8
    num_heads: int = 8

    @nn.compact
    def __call__(self, x, time_emb, labels_emb, *, deterministic: bool = True):
        # stride=2 卷积替代 MaxPool，空间尺寸减半
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=[(1, 1), (1, 1)],  # SAME padding: H//2
            name="down_conv",
        )(x)
        return TimeResBlock(
            self.out_channels,  # 注意：输入通道是 out_channels（已降采样）
            self.out_channels,
            self.time_dim,
            self.emb_size,
            self.num_groups,
            self.num_heads,
            name="resblock",
        )(x, time_emb, labels_emb, deterministic=deterministic)


class Up(nn.Module):
    """上采样: 双线性插值 -> concat(skip) -> TimeResBlock"""
    in_channels: int
    out_channels: int
    time_dim: int
    emb_size: int
    num_groups: int = 8
    num_heads: int = 8

    @nn.compact
    def __call__(
        self, x, skip, time_emb, labels_emb, *, deterministic: bool = True
    ):
        # 用 jax.nn.interpolate 做双线性上采样（匹配 skip 的空间尺寸）
        # skip: (B, H, W, C_skip), x: (B, H//2, W//2, C_x)
        H_skip, W_skip = skip.shape[1], skip.shape[2]
        x = jax.image.resize(
            x,
            shape=(x.shape[0], H_skip, W_skip, x.shape[3]),
            method="bilinear",
        )
        # channel 维度 concat
        x = jnp.concatenate([skip, x], axis=-1)
        return TimeResBlock(
            self.out_channels + skip.shape[-1],
            self.out_channels,
            self.time_dim,
            self.emb_size,
            self.num_groups,
            self.num_heads,
            name="resblock",
        )(x, time_emb, labels_emb, deterministic=deterministic)


# ============================================================
# 6. UNet — 完整网络
# ============================================================
class UNet(nn.Module):
    in_channels: int
    out_channels: int
    base_channels: int
    channel_mults: list
    time_dim: int
    emb_size: int
    num_classes: int
    num_groups: int = 8
    num_heads: int = 8
    dropout: float = 0.0

    def setup(self):
        channels = [self.base_channels * m for m in self.channel_mults]

        # 时间步 MLP
        self.time_mlp = nn.Sequential(
            [
                SinusoidalPosEmb(self.time_dim),
                nn.Dense(self.time_dim),
                nn.silu,
                nn.Dense(self.time_dim),
            ],
            name="time_mlp",
        )

        # 标签 Embedding
        self.labels_embedding = LabelsEmbedding(
            self.num_classes, self.emb_size, name="labels_emb"
        )

        # Intro 卷积: in_channels -> channels[0]
        self.intro = TimeResBlock(
            self.in_channels,
            channels[0],
            self.time_dim,
            self.emb_size,
            self.num_groups,
            self.num_heads,
            name="intro",
        )

        # 下采样路径
        self.downs = [
            Down(
                channels[i],
                channels[i + 1],
                self.time_dim,
                self.emb_size,
                self.num_groups,
                self.num_heads,
                name=f"down_{i}",
            )
            for i in range(len(channels) - 1)
        ]

        # 上采样路径
        self.ups = [
            Up(
                channels[len(channels) - 1 - i],
                channels[len(channels) - 2 - i],
                self.time_dim,
                self.emb_size,
                self.num_groups,
                self.num_heads,
                name=f"up_{i}",
            )
            for i in range(len(channels) - 1)
        ]

        # 输出卷积
        self.outro = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            name="outro",
        )

    @nn.compact
    def __call__(self, x, t, labels=None, *, deterministic: bool = True):
        """
        x:      (B, H, W, in_channels)  # Flax 默认 BHWC
        t:      (B,) int32 — 时间步
        labels: (B,) int32 — 类别标签
        返回:   (B, H, W, out_channels)
        """
        # 时间条件
        time_emb = self.time_mlp(t)  # (B, time_dim)

        # 标签条件
        B = x.shape[0]
        labels_emb = self.labels_embedding(
            labels, dropout=self.dropout, batch_size=B
        )

        # Intro
        h = self.intro(x, time_emb, labels_emb, deterministic=deterministic)

        # 下采样，保存 skip connections
        skips = []
        for down in self.downs:
            skips.append(h)
            h = down(h, time_emb, labels_emb, deterministic=deterministic)

        # 上采样
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, time_emb, labels_emb, deterministic=deterministic)

        return self.outro(h)


# ============================================================
# 辅助: 打印模型参数量
# ============================================================
def count_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def print_params_shape(params, depth=0):
    """递归打印 params Pytree 的结构（调试用）"""
    if isinstance(params, dict):
        for k, v in params.items():
            print("  " * depth + f"- {k}")
            print_params_shape(v, depth + 1)
    elif isinstance(params, jnp.ndarray):
        print("  " * depth + f"  shape={params.shape}, dtype={params.dtype}")

