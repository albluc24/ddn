"""Training script for DDN TTS model.

Provides a minimal training loop for the TTS MVP.
For production, consider using PyTorch Lightning or similar frameworks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json

from tts_model.networks.tts_model import DDNTTSModel, DDNTTSModelSimple


class DummyTTSDataset(Dataset):
    """Dummy dataset for testing the training loop.
    
    Generates random text and mel-spectrogram pairs.
    In production, replace with real text-audio dataset.
    """
    
    def __init__(
        self,
        num_samples=1000,
        vocab_size=128,
        mel_bins=80,
        text_length_range=(20, 100),
        mel_length_range=(80, 400),
    ):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.mel_bins = mel_bins
        self.text_length_range = text_length_range
        self.mel_length_range = mel_length_range
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random text length
        text_length = torch.randint(
            self.text_length_range[0],
            self.text_length_range[1] + 1,
            (1,)
        ).item()
        
        # Random mel length (roughly 4x text length)
        mel_length = text_length * 4
        
        # Generate random text IDs (avoid 0 which is padding)
        text_ids = torch.randint(1, self.vocab_size, (text_length,))
        
        # Generate random mel-spectrogram
        mel = torch.randn(self.mel_bins, mel_length)
        
        # Generate durations (uniform for simplicity)
        durations = torch.full((text_length,), mel_length // text_length, dtype=torch.long)
        # Distribute remainder
        remainder = mel_length - durations.sum()
        if remainder > 0:
            durations[:remainder] += 1
        
        return {
            'text_ids': text_ids,
            'mel': mel,
            'durations': durations,
        }


def collate_fn(batch):
    """Collate function for batching variable-length sequences."""
    # Find max lengths
    max_text_len = max(item['text_ids'].shape[0] for item in batch)
    max_mel_len = max(item['mel'].shape[1] for item in batch)
    mel_bins = batch[0]['mel'].shape[0]
    
    # Pad sequences
    text_ids_padded = []
    mel_padded = []
    durations_padded = []
    
    for item in batch:
        # Pad text_ids
        text_len = item['text_ids'].shape[0]
        text_pad = torch.zeros(max_text_len, dtype=torch.long)
        text_pad[:text_len] = item['text_ids']
        text_ids_padded.append(text_pad)
        
        # Pad mel
        mel_len = item['mel'].shape[1]
        mel_pad = torch.zeros(mel_bins, max_mel_len)
        mel_pad[:, :mel_len] = item['mel']
        mel_padded.append(mel_pad)
        
        # Pad durations
        dur_pad = torch.zeros(max_text_len, dtype=torch.long)
        dur_pad[:text_len] = item['durations']
        durations_padded.append(dur_pad)
    
    return {
        'text_ids': torch.stack(text_ids_padded),
        'mel': torch.stack(mel_padded),
        'durations': torch.stack(durations_padded),
    }


class TTSTrainer:
    """Simple trainer for TTS model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints/tts',
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Set to 0 for simplicity
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader) * 100,  # 100 epochs
            eta_min=1e-6,
        )
        
        self.global_step = 0
        self.epoch = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            text_ids = batch['text_ids'].to(self.device)
            target_mel = batch['mel'].to(self.device)
            durations = batch['durations'].to(self.device)
            
            # Forward pass
            result = self.model(
                text_ids=text_ids,
                target_mel=target_mel,
                external_durations=durations,
            )
            
            loss = result['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model."""
        if self.val_dataset is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                text_ids = batch['text_ids'].to(self.device)
                target_mel = batch['mel'].to(self.device)
                durations = batch['durations'].to(self.device)
                
                result = self.model(
                    text_ids=text_ids,
                    target_mel=target_mel,
                    external_durations=durations,
                )
                
                loss = result['total_loss']
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, filename='checkpoint.pt'):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename='checkpoint.pt'):
        """Load training checkpoint."""
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"Checkpoint {path} not found")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded from {path}")
    
    def train(self, num_epochs=100, save_every=10):
        """Train the model."""
        print(f"Training on {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation samples: {len(self.val_dataset)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train loss: {train_loss:.4f}")
            
            # Validate
            if self.val_dataset is not None:
                val_loss = self.validate()
                print(f"Val loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save final checkpoint
        self.save_checkpoint('checkpoint_final.pt')
        print("\nTraining completed!")


# ----------------------------------------------------------------------------

def main():
    """Main training function."""
    print("Setting up TTS training...")
    
    # Configuration
    config = {
        'vocab_size': 128,
        'mel_bins': 80,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 10,  # Small number for MVP testing
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Create datasets
    train_dataset = DummyTTSDataset(
        num_samples=500,
        vocab_size=config['vocab_size'],
        mel_bins=config['mel_bins'],
    )
    
    val_dataset = DummyTTSDataset(
        num_samples=100,
        vocab_size=config['vocab_size'],
        mel_bins=config['mel_bins'],
    )
    
    # Create model (use simple model for faster testing)
    print("\nCreating model...")
    model = DDNTTSModelSimple(
        vocab_size=config['vocab_size'],
        mel_bins=config['mel_bins'],
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = TTSTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        device=config['device'],
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=config['num_epochs'], save_every=5)
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    test_text = torch.randint(1, config['vocab_size'], (1, 50)).to(config['device'])
    with torch.no_grad():
        mel = model.infer(test_text)
    print(f"Generated mel shape: {mel.shape}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
