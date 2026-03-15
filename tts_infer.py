"""Inference script for DDN TTS model.

Provides utilities for generating mel-spectrograms from text.
In production, add a vocoder to generate waveforms.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
import json

from tts_model.networks.tts_model import DDNTTSModel, DDNTTSModelSimple


class TTSInference:
    """TTS inference interface."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        vocab_size: int = 128,
        mel_bins: int = 80,
    ):
        self.device = device
        self.vocab_size = vocab_size
        self.mel_bins = mel_bins
        
        # Create model
        self.model = DDNTTSModelSimple(
            vocab_size=vocab_size,
            mel_bins=mel_bins,
        ).to(device)
        
        # Load checkpoint if provided
        if model_path is not None:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    
    def text_to_ids(self, text: str) -> torch.Tensor:
        """Convert text to token IDs.
        
        Simple ASCII encoding. In production, use proper phoneme encoding.
        
        Args:
            text: Input text string
            
        Returns:
            Token IDs tensor [length]
        """
        # Convert to ASCII codes, clamp to vocab range
        ids = [min(ord(c), self.vocab_size - 1) for c in text]
        # Filter out padding (0)
        ids = [i if i > 0 else 1 for i in ids]
        return torch.tensor(ids, dtype=torch.long)
    
    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        return_numpy: bool = True,
    ):
        """Synthesize mel-spectrogram from text.
        
        Args:
            text: Input text
            speaker_id: Optional speaker ID for multi-speaker models
            return_numpy: Whether to return numpy array (True) or torch tensor (False)
            
        Returns:
            Mel-spectrogram [mel_bins, length]
        """
        # Convert text to IDs
        text_ids = self.text_to_ids(text).unsqueeze(0).to(self.device)  # [1, length]
        
        # Convert speaker_id to tensor if provided
        speaker_tensor = None
        if speaker_id is not None:
            speaker_tensor = torch.tensor([speaker_id], device=self.device)
        
        # Generate mel
        with torch.no_grad():
            mel = self.model.infer(text_ids, speaker_id=speaker_tensor)
        
        # Return as numpy if requested
        mel = mel.squeeze(0)  # [mel_bins, length]
        if return_numpy:
            mel = mel.cpu().numpy()
        
        return mel
    
    def synthesize_batch(
        self,
        texts: List[str],
        speaker_ids: Optional[List[int]] = None,
        return_numpy: bool = True,
    ):
        """Synthesize batch of texts.
        
        Args:
            texts: List of input texts
            speaker_ids: Optional list of speaker IDs
            return_numpy: Whether to return numpy arrays
            
        Returns:
            List of mel-spectrograms
        """
        mels = []
        for i, text in enumerate(texts):
            speaker_id = speaker_ids[i] if speaker_ids is not None else None
            mel = self.synthesize(text, speaker_id=speaker_id, return_numpy=return_numpy)
            mels.append(mel)
        return mels


def save_mel(mel: np.ndarray, path: str):
    """Save mel-spectrogram to file."""
    np.save(path, mel)
    print(f"Mel-spectrogram saved to {path}")


def load_mel(path: str) -> np.ndarray:
    """Load mel-spectrogram from file."""
    return np.load(path)


# ----------------------------------------------------------------------------
# MAS stub integration point

class MASInterface:
    """Interface for drop-in Monotonic Alignment Search implementations.
    
    This is a stub that can be replaced with actual MAS implementations
    from other repositories (e.g., Glow-TTS, Grad-TTS, etc.).
    """
    
    def __init__(self, implementation='stub'):
        """Initialize MAS interface.
        
        Args:
            implementation: Which MAS implementation to use
                - 'stub': Simple stub implementation (default)
                - 'glow_tts': Use Glow-TTS MAS (requires external repo)
                - 'grad_tts': Use Grad-TTS MAS (requires external repo)
        """
        self.implementation = implementation
        
        if implementation == 'glow_tts':
            try:
                # Placeholder for Glow-TTS MAS import
                # from glow_tts.monotonic_align import maximum_path
                # self.mas_func = maximum_path
                raise ImportError("Glow-TTS MAS not available. Please install Glow-TTS repo.")
            except ImportError as e:
                print(f"Warning: {e}")
                print("Falling back to stub implementation")
                self.implementation = 'stub'
        
        elif implementation == 'grad_tts':
            try:
                # Placeholder for Grad-TTS MAS import
                raise ImportError("Grad-TTS MAS not available. Please install Grad-TTS repo.")
            except ImportError as e:
                print(f"Warning: {e}")
                print("Falling back to stub implementation")
                self.implementation = 'stub'
    
    def compute_alignment(
        self,
        text_length: int,
        mel_length: int,
        attn_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute monotonic alignment.
        
        Args:
            text_length: Number of text tokens
            mel_length: Number of mel frames
            attn_prior: Optional attention prior [text_length, mel_length]
            
        Returns:
            Duration for each text token [text_length]
        """
        if self.implementation == 'stub':
            # Simple uniform alignment
            base_duration = mel_length // text_length
            remainder = mel_length % text_length
            durations = torch.full((text_length,), base_duration, dtype=torch.long)
            durations[:remainder] += 1
            return durations
        
        elif self.implementation == 'glow_tts':
            # Use Glow-TTS MAS
            # return self.mas_func(attn_prior)
            raise NotImplementedError("Glow-TTS MAS not implemented")
        
        elif self.implementation == 'grad_tts':
            # Use Grad-TTS MAS
            raise NotImplementedError("Grad-TTS MAS not implemented")
        
        else:
            raise ValueError(f"Unknown MAS implementation: {self.implementation}")


# ----------------------------------------------------------------------------

def demo_synthesis():
    """Demo: synthesize mel-spectrograms from sample texts."""
    print("DDN TTS Inference Demo")
    print("=" * 50)
    
    # Initialize inference
    print("\nInitializing model...")
    tts = TTSInference(
        model_path=None,  # No pretrained weights for MVP
        device='cpu',  # Use CPU for demo
    )
    
    # Sample texts
    texts = [
        "Hello world",
        "This is a test of the TTS system",
        "The quick brown fox jumps over the lazy dog",
    ]
    
    print("\nSynthesizing...")
    output_dir = Path("outputs/tts_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, text in enumerate(texts):
        print(f"\nText {i + 1}: '{text}'")
        
        # Synthesize
        mel = tts.synthesize(text)
        print(f"Generated mel shape: {mel.shape}")
        
        # Save
        save_path = output_dir / f"mel_{i + 1}.npy"
        save_mel(mel, str(save_path))
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print(f"Outputs saved to {output_dir}")
    print("\nNote: This is an untrained model, so outputs are random.")
    print("To get meaningful results, train the model using tts_train.py")


def demo_mas_interface():
    """Demo: MAS interface with stub and potential drop-in replacements."""
    print("\nMAS Interface Demo")
    print("=" * 50)
    
    # Test stub implementation
    print("\n1. Stub MAS:")
    mas_stub = MASInterface(implementation='stub')
    durations = mas_stub.compute_alignment(text_length=10, mel_length=40)
    print(f"Text length: 10, Mel length: 40")
    print(f"Durations: {durations.tolist()}")
    print(f"Sum: {durations.sum()} (should equal 40)")
    
    # Show how to integrate external MAS
    print("\n2. External MAS integration points:")
    print("   - Set implementation='glow_tts' to use Glow-TTS MAS")
    print("   - Set implementation='grad_tts' to use Grad-TTS MAS")
    print("   - Or implement custom MAS and add to MASInterface")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    # Run demos
    demo_synthesis()
    demo_mas_interface()
