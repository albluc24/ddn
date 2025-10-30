"""Text encoder and duration model for TTS.

Provides:
- Text/phoneme encoding
- Duration prediction
- Monotonic Alignment Search (MAS) stub with external alignment fallback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ----------------------------------------------------------------------------
# Text Encoder

class TextEncoder(nn.Module):
    """Encodes text/phonemes to conditioning embeddings.
    
    Simple encoder with character embeddings and positional encoding.
    For a production system, replace with a proper phoneme encoder.
    """
    
    def __init__(
        self,
        vocab_size=128,  # ASCII characters for simplicity
        embed_dim=512,
        hidden_dim=512,
        num_layers=4,
        kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Character embedding
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=5000)
        
        # Convolutional encoder layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        
        # Output projection
        self.proj = nn.Conv1d(hidden_dim, hidden_dim, 1)
    
    def forward(self, text_ids):
        """Encode text to embeddings.
        
        Args:
            text_ids: Text token IDs [batch, length]
            
        Returns:
            Text embeddings [batch, length, hidden_dim]
        """
        # Embed and add positional encoding
        x = self.embed(text_ids)  # [batch, length, embed_dim]
        x = self.pos_encoding(x)
        
        # Conv layers expect [batch, channels, length]
        x = x.transpose(1, 2)  # [batch, embed_dim, length]
        
        for conv in self.conv_layers:
            x = conv(x)
        
        x = self.proj(x)
        
        # Return as [batch, length, hidden_dim]
        return x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding.
        
        Args:
            x: Input tensor [batch, length, d_model]
            
        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


# ----------------------------------------------------------------------------
# Duration Predictor

class DurationPredictor(nn.Module):
    """Predicts duration for each text token.
    
    Outputs how many mel frames each token should generate.
    """
    
    def __init__(
        self,
        text_dim=512,
        hidden_dim=256,
        num_layers=2,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = text_dim if i == 0 else hidden_dim
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        
        # Output layer: predict log duration
        self.proj = nn.Conv1d(hidden_dim, 1, 1)
    
    def forward(self, text_embed, text_mask=None):
        """Predict durations.
        
        Args:
            text_embed: Text embeddings [batch, length, text_dim]
            text_mask: Optional mask [batch, length]
            
        Returns:
            Predicted durations [batch, length]
        """
        # Conv expects [batch, channels, length]
        x = text_embed.transpose(1, 2)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        # Predict log duration and exponentiate
        log_duration = self.proj(x).squeeze(1)  # [batch, length]
        duration = torch.exp(log_duration)
        
        # Apply mask if provided
        if text_mask is not None:
            duration = duration * text_mask.float()
        
        return duration


# ----------------------------------------------------------------------------
# Monotonic Alignment Search (MAS) - Stub with External Alignment

class MonotonicAlignmentSearch:
    """MAS implementation stub.
    
    For MVP, this provides a simple stub that can accept external alignments
    or use a basic heuristic. A full MAS implementation would require
    dynamic programming to find optimal monotonic alignment.
    
    As per migration guide: "if not rely on external alignments but make 
    a stub to insert a drop in mas impl."
    """
    
    @staticmethod
    def align(
        text_length: int,
        mel_length: int,
        attn_prior: Optional[torch.Tensor] = None,
        external_alignment: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute alignment between text and mel frames.
        
        Args:
            text_length: Number of text tokens
            mel_length: Number of mel frames
            attn_prior: Optional attention prior [text_length, mel_length]
            external_alignment: Optional pre-computed alignment [text_length]
            
        Returns:
            Duration for each text token [text_length]
        """
        if external_alignment is not None:
            # Use provided external alignment
            return external_alignment
        
        if attn_prior is not None:
            # Use attention prior to guide alignment
            # Simple heuristic: assign mel frames based on attention weights
            attn_sum = attn_prior.sum(dim=1)  # Sum over mel dimension
            durations = (attn_sum / attn_sum.sum() * mel_length).round().long()
            
            # Ensure durations sum to mel_length
            diff = mel_length - durations.sum().item()
            if diff != 0:
                # Adjust the longest duration
                max_idx = durations.argmax()
                durations[max_idx] += diff
            
            return durations
        
        # Fallback: uniform alignment
        base_duration = mel_length // text_length
        remainder = mel_length % text_length
        durations = torch.full((text_length,), base_duration, dtype=torch.long)
        # Distribute remainder evenly
        durations[:remainder] += 1
        
        return durations
    
    @staticmethod
    def expand_with_durations(
        text_embed: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        """Expand text embeddings to mel length using durations.
        
        Args:
            text_embed: Text embeddings [batch, text_length, dim]
            durations: Duration for each token [batch, text_length]
            
        Returns:
            Expanded embeddings [batch, mel_length, dim]
        """
        batch_size, text_length, dim = text_embed.shape
        device = text_embed.device
        
        expanded_list = []
        for b in range(batch_size):
            expanded_tokens = []
            for t in range(text_length):
                dur = int(durations[b, t].item())
                # Repeat token embedding dur times
                token_embed = text_embed[b, t].unsqueeze(0).expand(dur, -1)
                expanded_tokens.append(token_embed)
            
            if len(expanded_tokens) > 0:
                expanded = torch.cat(expanded_tokens, dim=0)  # [mel_length, dim]
            else:
                expanded = torch.zeros(0, dim, device=device)
            
            expanded_list.append(expanded)
        
        # Pad to same length
        max_len = max(e.shape[0] for e in expanded_list)
        padded_list = []
        for expanded in expanded_list:
            if expanded.shape[0] < max_len:
                pad = torch.zeros(max_len - expanded.shape[0], dim, device=device)
                expanded = torch.cat([expanded, pad], dim=0)
            padded_list.append(expanded.unsqueeze(0))
        
        return torch.cat(padded_list, dim=0)  # [batch, mel_length, dim]


# ----------------------------------------------------------------------------
# Duration/Alignment Module

class DurationAlignmentModule(nn.Module):
    """Combined duration prediction and alignment.
    
    During training: Uses MAS or external alignments to learn duration predictor
    During inference: Uses duration predictor to generate alignments
    """
    
    def __init__(
        self,
        text_dim=512,
        use_mas=False,  # Whether to use MAS (stub for now)
    ):
        super().__init__()
        self.duration_predictor = DurationPredictor(text_dim=text_dim)
        self.use_mas = use_mas
        self.mas = MonotonicAlignmentSearch()
    
    def forward(
        self,
        text_embed,
        mel_length=None,
        external_durations=None,
        training=True,
    ):
        """Compute durations and expand text embeddings.
        
        Args:
            text_embed: Text embeddings [batch, text_length, text_dim]
            mel_length: Target mel length (for training with MAS)
            external_durations: Pre-computed durations [batch, text_length]
            training: Whether in training mode
            
        Returns:
            Dictionary with:
                - expanded_embed: Expanded text embeddings [batch, mel_length, text_dim]
                - durations: Computed durations [batch, text_length]
                - duration_loss: Loss for duration predictor (if training)
        """
        batch_size, text_length, _ = text_embed.shape
        
        if training and external_durations is not None:
            # Training with external durations
            target_durations = external_durations
            
            # Predict durations for loss computation
            pred_durations = self.duration_predictor(text_embed)
            duration_loss = F.mse_loss(
                torch.log(pred_durations + 1),
                torch.log(target_durations.float() + 1)
            )
            
            # Use target durations for expansion
            use_durations = target_durations
        else:
            # Inference: use predicted durations
            pred_durations = self.duration_predictor(text_embed)
            use_durations = torch.round(pred_durations).long()
            
            # Ensure at least 1 frame per token
            use_durations = torch.clamp(use_durations, min=1)
            
            duration_loss = None
        
        # Expand embeddings
        expanded_embed = self.mas.expand_with_durations(text_embed, use_durations)
        
        return {
            'expanded_embed': expanded_embed,
            'durations': use_durations,
            'duration_loss': duration_loss,
        }


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing text encoder and duration model...")
    
    # Test TextEncoder
    encoder = TextEncoder(vocab_size=128, embed_dim=256, hidden_dim=512)
    text_ids = torch.randint(1, 128, (2, 50))  # Batch of 2, length 50
    text_embed = encoder(text_ids)
    print(f"TextEncoder: {text_ids.shape} -> {text_embed.shape}")
    
    # Test DurationPredictor
    duration_pred = DurationPredictor(text_dim=512)
    durations = duration_pred(text_embed)
    print(f"DurationPredictor output: {durations.shape}")
    
    # Test MAS alignment
    mas = MonotonicAlignmentSearch()
    align = mas.align(text_length=50, mel_length=200)
    print(f"MAS alignment: {align.shape}, sum={align.sum()}")
    
    # Test expansion (durations are already [batch, length])
    expanded = mas.expand_with_durations(text_embed, durations.round().long())
    print(f"Expanded embeddings: {expanded.shape}")
    
    # Test DurationAlignmentModule
    dur_align = DurationAlignmentModule(text_dim=512)
    result = dur_align(text_embed, external_durations=torch.ones(2, 50).long() * 4)
    print(f"DurationAlignmentModule expanded: {result['expanded_embed'].shape}")
    print(f"Duration loss: {result['duration_loss'].item() if result['duration_loss'] is not None else 'N/A'}")
    
    print("All tests passed!")
