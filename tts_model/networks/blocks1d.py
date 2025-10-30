"""1D Network components for TTS adapted from DDN.

This module contains 1D versions of the core DDN blocks for temporal sequence modeling.
Adapted from training/networks.py to work with 1D mel-spectrogram sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


# ----------------------------------------------------------------------------
# Weight initialization utilities

def weight_init(shape, mode, fan_in, fan_out):
    """Initialize weights with various strategies."""
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# 1D Convolution layer

class Conv1d(nn.Module):
    """1D convolution with flexible initialization."""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding='same',
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Handle padding
        if padding == 'same':
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding
            
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel_size, 
                          fan_out=out_channels * kernel_size)
        self.weight = nn.Parameter(
            weight_init([out_channels, in_channels, kernel_size], **init_kwargs) * init_weight
        )
        self.bias = (
            nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if bias else None
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, channels, length]
            
        Returns:
            Output tensor of shape [batch, out_channels, length']
        """
        x = F.conv1d(x, self.weight.to(x.dtype), padding=self.padding, stride=self.stride)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype).view(1, -1, 1)
        return x


# ----------------------------------------------------------------------------
# Group Normalization

class GroupNorm(nn.Module):
    """Group normalization for 1D sequences."""
    
    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, channels, length]
            
        Returns:
            Normalized tensor of same shape
        """
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


# ----------------------------------------------------------------------------
# 1D UNet Block (temporal version)

class UNetBlock1D(nn.Module):
    """1D UNet block for temporal sequence modeling.
    
    Adapted from UNetBlockWoEmb to work with 1D sequences.
    Removes image-specific operations and uses temporal convolutions.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        init_mode="kaiming_normal",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = (
            0 if not attention else (
                num_heads if num_heads is not None else out_channels // channels_per_head
            )
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        
        # First conv path
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            init_mode=init_mode,
        )
        
        # Second conv path
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            init_mode=init_mode,
            init_weight=1e-5,  # Small init for residual
        )
        
        # Skip connection
        self.skip = None
        if out_channels != in_channels:
            self.skip = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                init_mode=init_mode,
            )
        
        # Optional self-attention
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv1d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel_size=1,
                init_mode=init_mode,
            )
            self.proj = Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                init_mode=init_mode,
                init_weight=1e-5,
            )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, channels, length]
            
        Returns:
            Output tensor of shape [batch, out_channels, length]
        """
        orig = x
        
        # First conv block
        x = self.conv0(F.silu(self.norm0(x)))
        
        # Second conv block  
        x = F.silu(self.norm1(x))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        
        # Skip connection
        x = x + (self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        
        # Self-attention
        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1)
                .unbind(2)
            )
            # Simplified attention (no optimized kernel)
            attn = F.softmax(torch.einsum('nci,ncj->nij', q, k) / np.sqrt(q.shape[1]), dim=-1)
            a = torch.einsum('nij,ncj->nci', attn, v)
            a = a.reshape(*x.shape)
            x = x + self.proj(a)
        
        return x


# ----------------------------------------------------------------------------
# Discrete Distribution Output for 1D sequences

class DiscreteDistributionOutput1D(nn.Module):
    """1D version of discrete distribution output for mel-spectrograms.
    
    Generates multiple mel-spectrogram predictions and computes
    distribution-based loss. Adapted from sddn.DiscreteDistributionOutput.
    """
    
    def __init__(
        self,
        k=64,  # Number of discrete outputs
        last_c=128,  # Last layer channels
        predict_c=80,  # Mel bins to predict
        length=None,  # Mel sequence length
        leak_choice=True,
    ):
        super().__init__()
        self.k = k
        self.last_c = last_c
        self.predict_c = predict_c
        self.length = length
        self.leak_choice = leak_choice
        
        # Prediction head: generates k discrete outputs
        self.pred_head = Conv1d(
            in_channels=last_c,
            out_channels=predict_c * k,
            kernel_size=1,
            bias=True,
        )
    
    def forward(self, feat, target=None):
        """Forward pass.
        
        Args:
            feat: Feature tensor [batch, channels, length]
            target: Optional target mel-spectrogram [batch, mel_bins, length]
            
        Returns:
            Dictionary with predictions and losses if target provided
        """
        batch_size, _, length = feat.shape
        
        # Generate k predictions
        pred_all = self.pred_head(feat)  # [batch, predict_c * k, length]
        pred_all = pred_all.reshape(batch_size, self.k, self.predict_c, length)
        
        result = {'predictions': pred_all}
        
        if target is not None:
            # Handle length mismatch between prediction and target
            target_length = target.shape[2]
            if length != target_length:
                # Interpolate predictions to match target length
                pred_all_reshaped = pred_all.reshape(batch_size * self.k, self.predict_c, length)
                pred_all_reshaped = F.interpolate(
                    pred_all_reshaped,
                    size=target_length,
                    mode='linear',
                    align_corners=False
                )
                pred_all = pred_all_reshaped.reshape(batch_size, self.k, self.predict_c, target_length)
            
            # Compute distance to target for each of k predictions
            target_expanded = target.unsqueeze(1)  # [batch, 1, mel_bins, target_length]
            distances = torch.mean((pred_all - target_expanded) ** 2, dim=(2, 3))  # [batch, k]
            
            # Find best match
            best_idx = torch.argmin(distances, dim=1)
            result['best_predictions'] = pred_all[torch.arange(batch_size), best_idx]
            result['distances'] = distances
            
            # Compute loss (mean of minimum distances)
            result['loss'] = torch.mean(distances.min(dim=1)[0])
        
        return result


# ----------------------------------------------------------------------------
# 1D Discrete Distribution Block

class DiscreteDistributionBlock1D(nn.Module):
    """1D version of DiscreteDistributionBlock for TTS.
    
    Wraps a UNet-style block with discrete distribution output
    for mel-spectrogram generation.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        k=64,
        mel_bins=80,
        mel_length=None,
        num_blocks=4,
        attention=False,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mel_bins = mel_bins
        self.mel_length = mel_length
        self.k = k
        
        # Build block stack
        blocks = []
        for i in range(num_blocks):
            blocks.append(UNetBlock1D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                attention=attention and i == num_blocks - 1,  # Attention on last block
                dropout=dropout,
            ))
        self.blocks = nn.ModuleList(blocks)
        
        # Discrete distribution output
        self.ddo = DiscreteDistributionOutput1D(
            k=k,
            last_c=out_channels,
            predict_c=mel_bins,
            length=mel_length,
        )
    
    def forward(self, x, target=None):
        """Forward pass.
        
        Args:
            x: Input features [batch, channels, length]
            target: Optional target mel-spectrogram [batch, mel_bins, length]
            
        Returns:
            Output dictionary from DiscreteDistributionOutput1D
        """
        # Pass through block stack
        for block in self.blocks:
            x = block(x)
        
        # Generate discrete distribution output
        return self.ddo(x, target=target)
    
    def init_features(self, batch_size, device='cuda'):
        """Initialize input features for generation.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Initial feature tensor [batch, channels, length]
        """
        if self.mel_length is None:
            raise ValueError("mel_length must be set to initialize features")
        
        # Simple initialization: learned pattern
        init_feat = torch.randn(
            batch_size, self.in_channels, self.mel_length,
            device=device
        ) * 0.01
        return init_feat


# ----------------------------------------------------------------------------
# Condition Processing for Text/Speaker

class ConditionProcess1D(nn.Module):
    """Processes conditioning information for TTS.
    
    Handles text embeddings, speaker IDs, and other conditioning signals.
    Adapted from ConditionProcess to remove image-specific logic.
    """
    
    def __init__(
        self,
        text_dim=512,
        speaker_dim=0,  # 0 means no speaker conditioning
        hidden_dim=512,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.speaker_dim = speaker_dim
        self.hidden_dim = hidden_dim
        
        total_cond_dim = text_dim + speaker_dim
        
        # Project conditioning to hidden dimension
        self.proj = nn.Sequential(
            nn.Linear(total_cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Speaker embedding if needed
        if speaker_dim > 0:
            self.speaker_embed = nn.Embedding(100, speaker_dim)  # Support up to 100 speakers
    
    def forward(self, text_embed, speaker_id=None):
        """Process conditioning information.
        
        Args:
            text_embed: Text embeddings [batch, length, text_dim]
            speaker_id: Optional speaker IDs [batch]
            
        Returns:
            Processed conditioning [batch, length, hidden_dim]
        """
        batch_size, length, _ = text_embed.shape
        
        # Combine text and speaker embeddings
        cond_list = [text_embed]
        if speaker_id is not None and self.speaker_dim > 0:
            speaker_emb = self.speaker_embed(speaker_id)  # [batch, speaker_dim]
            speaker_emb = speaker_emb.unsqueeze(1).expand(-1, length, -1)  # [batch, length, speaker_dim]
            cond_list.append(speaker_emb)
        
        cond = torch.cat(cond_list, dim=-1)  # [batch, length, total_cond_dim]
        
        # Project
        return self.proj(cond)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple test
    print("Testing 1D network components...")
    
    # Test UNetBlock1D
    block = UNetBlock1D(64, 128, attention=True)
    x = torch.randn(2, 64, 100)
    out = block(x)
    print(f"UNetBlock1D: {x.shape} -> {out.shape}")
    
    # Test DiscreteDistributionBlock1D
    dd_block = DiscreteDistributionBlock1D(
        in_channels=128,
        out_channels=256,
        k=32,
        mel_bins=80,
        mel_length=100,
    )
    x = torch.randn(2, 128, 100)
    target = torch.randn(2, 80, 100)
    out = dd_block(x, target=target)
    print(f"DiscreteDistributionBlock1D predictions: {out['predictions'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    
    print("All tests passed!")
