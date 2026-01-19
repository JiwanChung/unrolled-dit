"""
Unrolled SiT: One-step generation via layer-wise trajectory matching.

Each transformer block is assigned a fixed timestep and supervised to output
the corresponding point on the rectified flow trajectory.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class UnrolledBlock(nn.Module):
    """
    A transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Identical to SiTBlock but conceptually "owns" a fixed timestep.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final layer: LayerNorm + Linear projection to pixels."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class UnrolledSiT(nn.Module):
    """
    Unrolled SiT for one-step generation.

    Key differences from standard SiT:
    1. Each block has a FIXED timestep (not shared)
    2. Block outputs are supervised against teacher trajectory
    3. Single forward pass generates final image

    Args:
        input_size: Image size (32 for CIFAR-10)
        patch_size: Patch size for patchification
        in_channels: Input channels (3 for RGB)
        hidden_size: Transformer hidden dimension
        depth: Number of transformer blocks (= number of timesteps)
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        num_classes: Number of classes for conditioning
        class_dropout_prob: Dropout for classifier-free guidance
    """
    def __init__(
        self,
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=10,
        class_dropout_prob=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.hidden_size = hidden_size

        # Patch embedding
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches

        # Positional embedding (fixed sin-cos)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Class embedding
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Timestep embedder (used to precompute fixed embeddings)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Fixed timesteps for each block: t goes from 0 (noise) to 1 (clean)
        # Block i handles the transition from t_i to t_{i+1}
        timesteps = torch.linspace(0, 1, depth + 1)  # [0, 1/12, 2/12, ..., 1]
        self.register_buffer('timesteps', timesteps)

        # Transformer blocks (independent weights per block)
        self.blocks = nn.ModuleList([
            UnrolledBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer for output projection
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        # Precompute and cache fixed time embeddings
        self._precompute_time_embeddings()

    def _precompute_time_embeddings(self):
        """Precompute time embeddings for each block's fixed timestep."""
        with torch.no_grad():
            # Each block i uses timestep t_i (the "input" time for that block)
            t_embs = []
            for i in range(self.depth):
                t = self.timesteps[i:i+1]  # Shape (1,)
                t_emb = self.t_embedder(t)  # Shape (1, hidden_size)
                t_embs.append(t_emb)
            # Stack and register as buffer
            self.register_buffer('fixed_t_embs', torch.cat(t_embs, dim=0))  # (depth, hidden_size)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed with sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, num_patches, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, num_patches, patch_size**2 * C)
        """
        p = self.patch_size
        c = self.in_channels
        h = w = imgs.shape[2] // p

        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p * p * c))
        return x

    def forward(self, x, y, return_intermediates=False):
        """
        Forward pass for one-step generation.

        Args:
            x: (N, C, H, W) - input noise (t=0 state)
            y: (N,) - class labels
            return_intermediates: if True, return intermediate outputs for supervision

        Returns:
            out: (N, C, H, W) - generated image
            intermediates: list of (N, C, H, W) - intermediate states (if requested)
        """
        # Patchify and add positional embedding
        x = self.x_embedder(x) + self.pos_embed  # (N, num_patches, hidden_size)

        # Get class embedding
        y_emb = self.y_embedder(y, self.training)  # (N, hidden_size)

        intermediates = []

        # Process through each block with its fixed timestep
        for i, block in enumerate(self.blocks):
            # Get precomputed time embedding for this block
            t_emb = self.fixed_t_embs[i:i+1].expand(x.shape[0], -1)  # (N, hidden_size)

            # Conditioning = time + class
            c = t_emb + y_emb

            # Forward through block (already residual)
            x = block(x, c)

            # Save intermediate if requested (for layer-wise supervision)
            if return_intermediates:
                # Project current hidden state to pixel space for supervision
                c_final = self.fixed_t_embs[i:i+1].expand(x.shape[0], -1) + y_emb
                x_proj = self.final_layer(x, c_final)
                x_img = self.unpatchify(x_proj)
                intermediates.append(x_img)

        # Final projection to output
        c_final = self.fixed_t_embs[-1:].expand(x.shape[0], -1) + y_emb
        x = self.final_layer(x, c_final)
        out = self.unpatchify(x)  # (N, C, H, W)

        if return_intermediates:
            return out, intermediates
        return out

    def forward_with_cfg(self, x, y, cfg_scale):
        """Forward with classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, y)
        cond_out, uncond_out = torch.split(model_out, len(model_out) // 2, dim=0)
        half_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        return torch.cat([half_out, half_out], dim=0)


# Model configurations for CIFAR-10
def UnrolledSiT_S(num_classes=10, **kwargs):
    """Small model: 12 blocks, 384 hidden, ~33M params"""
    return UnrolledSiT(
        input_size=32, patch_size=4, in_channels=3,
        hidden_size=384, depth=12, num_heads=6,
        num_classes=num_classes, **kwargs
    )

def UnrolledSiT_B(num_classes=10, **kwargs):
    """Base model: 12 blocks, 768 hidden, ~130M params"""
    return UnrolledSiT(
        input_size=32, patch_size=4, in_channels=3,
        hidden_size=768, depth=12, num_heads=12,
        num_classes=num_classes, **kwargs
    )

def UnrolledSiT_S_deep(num_classes=10, **kwargs):
    """Small model with more blocks: 24 blocks for finer trajectory"""
    return UnrolledSiT(
        input_size=32, patch_size=4, in_channels=3,
        hidden_size=384, depth=24, num_heads=6,
        num_classes=num_classes, **kwargs
    )


if __name__ == "__main__":
    # Quick test
    model = UnrolledSiT_S(num_classes=10)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))

    out = model(x, y)
    print(f"Output shape: {out.shape}")

    out, intermediates = model(x, y, return_intermediates=True)
    print(f"Intermediates: {len(intermediates)} tensors of shape {intermediates[0].shape}")
