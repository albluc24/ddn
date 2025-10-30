"""Main TTS model combining all components.

This is the top-level TTS model that integrates:
- Text encoder
- Duration/alignment module
- DDN-based mel-spectrogram decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .blocks1d import (
    DiscreteDistributionBlock1D,
    ConditionProcess1D,
    UNetBlock1D,
    Conv1d,
)
from .text_encoder import (
    TextEncoder,
    DurationAlignmentModule,
)


class DDNTTSModel(nn.Module):
    """TTS model based on Discrete Distribution Networks.
    
    Architecture:
    1. Text encoder: text/phonemes -> text embeddings
    2. Duration/alignment: text embeddings -> expanded to mel length
    3. DDN decoder: generates mel-spectrograms with discrete distribution
    4. (External) Vocoder: mel-spectrograms -> waveform (not included here)
    
    This is an MVP implementation following the tts_migration.md guide.
    """
    
    def __init__(
        self,
        # Text encoder params
        vocab_size=128,
        text_embed_dim=512,
        text_hidden_dim=512,
        text_num_layers=4,
        
        # Duration model params
        use_mas=False,
        
        # DDN decoder params
        decoder_channels=256,
        decoder_num_blocks=4,
        decoder_k=64,  # Number of discrete outputs
        mel_bins=80,
        
        # Conditioning params
        speaker_dim=0,  # 0 = no speaker conditioning
        
        # Other params
        dropout=0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.mel_bins = mel_bins
        self.decoder_k = decoder_k
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            dropout=dropout,
        )
        
        # Duration and alignment
        self.duration_alignment = DurationAlignmentModule(
            text_dim=text_hidden_dim,
            use_mas=use_mas,
        )
        
        # Condition processing
        self.condition_process = ConditionProcess1D(
            text_dim=text_hidden_dim,
            speaker_dim=speaker_dim,
            hidden_dim=decoder_channels,
        )
        
        # Initial projection: condition -> decoder input
        self.init_proj = Conv1d(
            in_channels=decoder_channels,
            out_channels=decoder_channels,
            kernel_size=1,
        )
        
        # DDN decoder blocks (multi-scale)
        self.decoder_blocks = nn.ModuleList([
            DiscreteDistributionBlock1D(
                in_channels=decoder_channels,
                out_channels=decoder_channels,
                k=decoder_k,
                mel_bins=mel_bins,
                num_blocks=decoder_num_blocks,
                attention=True,
                dropout=dropout,
            )
        ])
    
    def forward(
        self,
        text_ids,
        target_mel=None,
        external_durations=None,
        speaker_id=None,
        return_intermediate=False,
    ):
        """Forward pass.
        
        Args:
            text_ids: Text token IDs [batch, text_length]
            target_mel: Optional target mel-spectrogram [batch, mel_bins, mel_length]
            external_durations: Optional durations [batch, text_length]
            speaker_id: Optional speaker ID [batch]
            return_intermediate: Whether to return intermediate outputs
            
        Returns:
            Dictionary with predictions, losses, and optionally intermediate outputs
        """
        batch_size = text_ids.shape[0]
        
        # 1. Encode text
        text_embed = self.text_encoder(text_ids)  # [batch, text_length, text_hidden_dim]
        
        # 2. Duration and alignment
        mel_length = target_mel.shape[2] if target_mel is not None else None
        dur_result = self.duration_alignment(
            text_embed,
            mel_length=mel_length,
            external_durations=external_durations,
            training=self.training,
        )
        expanded_text = dur_result['expanded_embed']  # [batch, mel_length, text_hidden_dim]
        durations = dur_result['durations']
        duration_loss = dur_result['duration_loss']
        
        # 3. Process conditioning
        conditioning = self.condition_process(expanded_text, speaker_id)  # [batch, mel_length, decoder_channels]
        
        # 4. Initialize decoder input
        # Conv1d expects [batch, channels, length]
        decoder_input = self.init_proj(conditioning.transpose(1, 2))  # [batch, decoder_channels, mel_length]
        
        # 5. DDN decoder
        decoder_result = self.decoder_blocks[0](decoder_input, target=target_mel)
        
        # Prepare output
        result = {
            'predictions': decoder_result['predictions'],  # [batch, k, mel_bins, mel_length]
            'durations': durations,
        }
        
        # Add losses if target provided
        if target_mel is not None:
            result['mel_loss'] = decoder_result['loss']
            result['best_predictions'] = decoder_result['best_predictions']
            
            # Total loss
            total_loss = decoder_result['loss']
            if duration_loss is not None:
                total_loss = total_loss + 0.1 * duration_loss  # Weight duration loss
                result['duration_loss'] = duration_loss
            
            result['total_loss'] = total_loss
        
        # Add intermediate outputs if requested
        if return_intermediate:
            result['text_embed'] = text_embed
            result['expanded_text'] = expanded_text
            result['conditioning'] = conditioning
        
        return result
    
    def infer(self, text_ids, speaker_id=None, max_mel_length=1000):
        """Inference mode: generate mel-spectrogram from text.
        
        Args:
            text_ids: Text token IDs [batch, text_length]
            speaker_id: Optional speaker ID [batch]
            max_mel_length: Maximum mel length to generate
            
        Returns:
            Generated mel-spectrogram [batch, mel_bins, mel_length]
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(
                text_ids=text_ids,
                speaker_id=speaker_id,
                external_durations=None,
            )
            
            # Use the mean of k predictions
            predictions = result['predictions']  # [batch, k, mel_bins, mel_length]
            mel = predictions.mean(dim=1)  # [batch, mel_bins, mel_length]
            
            # Truncate to actual length based on durations
            durations = result['durations']
            actual_length = durations.sum(dim=1).max().item()
            if actual_length < mel.shape[2]:
                mel = mel[:, :, :int(actual_length)]
            
            return mel


class DDNTTSModelSimple(nn.Module):
    """Simplified TTS model for quick testing.
    
    A minimal version with fewer parameters for faster iteration.
    """
    
    def __init__(
        self,
        vocab_size=128,
        mel_bins=80,
    ):
        super().__init__()
        self.model = DDNTTSModel(
            vocab_size=vocab_size,
            text_embed_dim=256,
            text_hidden_dim=256,
            text_num_layers=2,
            decoder_channels=128,
            decoder_num_blocks=2,
            decoder_k=32,
            mel_bins=mel_bins,
            dropout=0.1,
        )
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def infer(self, *args, **kwargs):
        return self.model.infer(*args, **kwargs)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing DDN TTS Model...")
    
    # Test full model
    model = DDNTTSModel(
        vocab_size=128,
        mel_bins=80,
        decoder_k=32,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass with target (training)
    text_ids = torch.randint(1, 128, (2, 50))
    target_mel = torch.randn(2, 80, 200)
    durations = torch.ones(2, 50).long() * 4  # Each token -> 4 mel frames
    
    result = model(
        text_ids=text_ids,
        target_mel=target_mel,
        external_durations=durations,
    )
    
    print(f"Training output:")
    print(f"  Predictions shape: {result['predictions'].shape}")
    print(f"  Total loss: {result['total_loss'].item():.4f}")
    if 'mel_loss' in result:
        print(f"  Mel loss: {result['mel_loss'].item():.4f}")
    if 'duration_loss' in result:
        print(f"  Duration loss: {result['duration_loss'].item():.4f}")
    
    # Inference
    mel = model.infer(text_ids)
    print(f"\nInference output:")
    print(f"  Generated mel shape: {mel.shape}")
    
    # Test simple model
    print("\nTesting simple model...")
    simple_model = DDNTTSModelSimple()
    print(f"Simple model parameters: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    mel = simple_model.infer(text_ids[:1])
    print(f"Simple model inference: {mel.shape}")
    
    print("\nAll tests passed!")
