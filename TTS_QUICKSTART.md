# TTS Model Quick Start Guide

This guide demonstrates how to use the DDN-based TTS model.

## Installation

```bash
# Install dependencies
pip install torch torchvision numpy

# Navigate to the repository
cd /path/to/ddn
```

## Quick Test

Run the test suite to verify everything works:

```bash
python tts_model/test_tts.py
```

Expected output:
```
============================================================
Running TTS Model Test Suite
============================================================
Testing 1D blocks...
✓ All 1D block tests passed

Testing text encoder...
✓ All text encoder tests passed

Testing TTS model...
✓ All TTS model tests passed

Testing training components...
✓ All training component tests passed

Testing inference components...
✓ All inference component tests passed

============================================================
✓ ALL TESTS PASSED
============================================================
```

## Basic Usage

### 1. Inference (Text to Mel-Spectrogram)

```python
from tts_model.networks.tts_model import DDNTTSModelSimple
import torch

# Create model
model = DDNTTSModelSimple(vocab_size=128, mel_bins=80)
model.eval()

# Generate mel-spectrogram from text
text_ids = torch.randint(1, 128, (1, 50))  # Replace with real text encoding
mel = model.infer(text_ids)

print(f"Generated mel-spectrogram shape: {mel.shape}")
# Output: Generated mel-spectrogram shape: torch.Size([1, 80, X])
```

### 2. Using the Inference Script

```bash
python tts_infer.py
```

This will:
- Generate mel-spectrograms for sample texts
- Save outputs to `outputs/tts_demo/`
- Demonstrate MAS interface

### 3. Training

```python
from tts_train import TTSTrainer, DummyTTSDataset
from tts_model.networks.tts_model import DDNTTSModelSimple

# Create datasets (replace with real data)
train_dataset = DummyTTSDataset(num_samples=1000)
val_dataset = DummyTTSDataset(num_samples=100)

# Create model
model = DDNTTSModelSimple(vocab_size=128, mel_bins=80)

# Create trainer
trainer = TTSTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=16,
    learning_rate=1e-4,
)

# Train
trainer.train(num_epochs=100, save_every=10)
```

Or use the training script:

```bash
python tts_train.py
```

## Advanced Features

### MAS Integration

The model includes a stub for Monotonic Alignment Search (MAS) that can be replaced with implementations from other repositories:

```python
from tts_infer import MASInterface

# Use stub (default)
mas = MASInterface(implementation='stub')
durations = mas.compute_alignment(text_length=50, mel_length=200)

# Or integrate external MAS (when available)
# mas = MASInterface(implementation='glow_tts')
# mas = MASInterface(implementation='grad_tts')
```

To add your own MAS implementation:

1. Edit `tts_infer.py` → `MASInterface` class
2. Add your implementation option
3. Import and use the external MAS function

### Model Architecture

The TTS model consists of:

```
Text Input
    ↓
TextEncoder (Conv1D + Positional Encoding)
    ↓
DurationAlignmentModule (Duration Predictor + Expansion)
    ↓
ConditionProcess1D (Text + Speaker Conditioning)
    ↓
DiscreteDistributionBlock1D (DDN Decoder)
    ↓
Mel-Spectrogram Output (k discrete predictions)
```

### Configuration

Key parameters you can adjust:

```python
model = DDNTTSModel(
    # Text encoder
    vocab_size=128,           # Vocabulary size
    text_embed_dim=512,       # Text embedding dimension
    text_hidden_dim=512,      # Hidden dimension
    text_num_layers=4,        # Number of encoder layers
    
    # DDN decoder
    decoder_channels=256,     # Decoder hidden channels
    decoder_num_blocks=4,     # Number of decoder blocks
    decoder_k=64,             # Number of discrete outputs
    mel_bins=80,              # Number of mel bins
    
    # Conditioning
    speaker_dim=0,            # Speaker embedding dimension (0 = no speakers)
    
    # Training
    dropout=0.1,              # Dropout rate
)
```

## Production Checklist

Before using in production, you should:

- [ ] Replace `DummyTTSDataset` with real text-audio dataset (e.g., LJSpeech)
- [ ] Implement proper phoneme encoding (replace ASCII character encoding)
- [ ] Integrate full MAS or use external alignments
- [ ] Add a vocoder for waveform generation (HiFi-GAN, WaveGlow, etc.)
- [ ] Add evaluation metrics (MCD, MOS, intelligibility tests)
- [ ] Tune hyperparameters on your dataset
- [ ] Implement multi-scale refinement if needed

## Troubleshooting

### Issue: Training loss not decreasing

- Check that your dataset has proper alignments/durations
- Try reducing learning rate
- Ensure text and mel lengths match properly

### Issue: Generated mels sound bad

- The model needs training! The untrained model produces random outputs
- Train on real text-audio pairs for at least 50-100 epochs
- Use a proper vocoder to convert mel to audio

### Issue: Out of memory

- Reduce batch size
- Use the Simple model instead of Full model
- Reduce decoder_k (number of discrete outputs)

## Example Workflow

1. **Prepare your dataset**
   ```python
   # Replace DummyTTSDataset with your own dataset
   # that loads text-audio pairs with alignments
   ```

2. **Train the model**
   ```bash
   python tts_train.py  # Edit config in the file
   ```

3. **Generate mel-spectrograms**
   ```python
   from tts_infer import TTSInference
   
   tts = TTSInference(model_path='checkpoints/tts/checkpoint_final.pt')
   mel = tts.synthesize("Hello, how are you?")
   ```

4. **Convert to audio with vocoder**
   ```python
   # Use HiFi-GAN or similar vocoder
   # audio = vocoder(mel)
   ```

## References

- Original DDN paper: https://arxiv.org/abs/2401.00036
- Migration guide: `tts_migration.md` in repository root
- Glow-TTS: https://arxiv.org/abs/2005.11129
- Grad-TTS: https://arxiv.org/abs/2105.06337

## Support

For issues or questions, please check:
- `tts_model/README.md` for detailed documentation
- `tts_model/test_tts.py` for usage examples
- `tts_migration.md` for architecture details
