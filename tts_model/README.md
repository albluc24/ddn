# DDN-based TTS Model (MVP)

This directory contains a trainable MVP implementation of a Text-to-Speech (TTS) model based on Discrete Distribution Networks (DDN), following the migration guide in `tts_migration.md`.

## Overview

This TTS model adapts the DDN architecture from image generation to audio generation:
- **Input**: Text/phonemes → character embeddings
- **Processing**: Duration prediction + alignment → mel-spectrogram generation
- **Output**: Mel-spectrograms (can be converted to audio via vocoder)

## Architecture Components

### 1. Network Modules (`tts_model/networks/`)

- **`blocks1d.py`**: 1D versions of core DDN components
  - `Conv1d`: 1D convolution with flexible initialization
  - `UNetBlock1D`: Temporal U-Net block with optional attention
  - `DiscreteDistributionOutput1D`: Generates k discrete mel predictions
  - `DiscreteDistributionBlock1D`: Full DDN block for mel generation
  - `ConditionProcess1D`: Processes text/speaker conditioning

- **`text_encoder.py`**: Text encoding and duration modeling
  - `TextEncoder`: Converts text/phonemes to embeddings
  - `DurationPredictor`: Predicts mel frame duration per text token
  - `MonotonicAlignmentSearch`: MAS stub with external alignment support
  - `DurationAlignmentModule`: Combined duration prediction and expansion

- **`tts_model.py`**: Main TTS model
  - `DDNTTSModel`: Full TTS model combining all components
  - `DDNTTSModelSimple`: Simplified version for faster testing

### 2. Training (`tts_train.py`)

Provides a minimal training loop:
- `DummyTTSDataset`: Dummy dataset for testing (replace with real data)
- `TTSTrainer`: Training loop with validation and checkpointing

Usage:
```python
python tts_train.py
```

### 3. Inference (`tts_infer.py`)

Tools for generating mel-spectrograms from text:
- `TTSInference`: High-level inference interface
- `MASInterface`: Stub for drop-in MAS implementations

Usage:
```python
python tts_infer.py
```

## Key Features (following tts_migration.md)

### ✅ Completed

1. **Core 1D blocks extracted and adapted**
   - DiscreteDistributionBlock → DiscreteDistributionBlock1D
   - UNetBlockWoEmb → UNetBlock1D
   - ConditionProcess → ConditionProcess1D

2. **Data flow reframed for 1D sequences**
   - Input: text embeddings [batch, length, dim]
   - Output: mel-spectrograms [batch, mel_bins, length]
   - Conditioning: text + optional speaker ID

3. **Duration and alignment strategy**
   - Duration predictor for inference
   - MAS stub with external alignment support
   - Expandable to full MAS implementation

4. **Architectural changes**
   - All 2D convs replaced with 1D temporal convs
   - Skip connections and attention preserved
   - Multi-scale sampling framework ready

5. **Training loop integration**
   - Custom PyTorch training loop
   - Supports external duration/alignment
   - Loss computation for mel + duration

6. **Inference pipeline**
   - Text → embeddings → duration → expanded → mel
   - Integration point for vocoder (external)
   - Supports batch inference

## Model Parameters

- **Simple model**: ~1.8M parameters
- **Full model**: ~8.9M parameters

## Usage Example

```python
from tts_model.networks.tts_model import DDNTTSModelSimple
import torch

# Create model
model = DDNTTSModelSimple(vocab_size=128, mel_bins=80)

# Training
text_ids = torch.randint(1, 128, (2, 50))
target_mel = torch.randn(2, 80, 200)
durations = torch.ones(2, 50).long() * 4

result = model(
    text_ids=text_ids,
    target_mel=target_mel,
    external_durations=durations,
)

print(f"Loss: {result['total_loss'].item()}")

# Inference
mel = model.infer(text_ids)
print(f"Generated mel shape: {mel.shape}")
```

## MAS Integration Point

The `MonotonicAlignmentSearch` class in `text_encoder.py` is a stub that can be replaced with:

1. **Glow-TTS MAS**: Set `implementation='glow_tts'` in `MASInterface`
2. **Grad-TTS MAS**: Set `implementation='grad_tts'` in `MASInterface`
3. **Custom MAS**: Implement in `MonotonicAlignmentSearch.align()`

Example:
```python
from tts_infer import MASInterface

# Use external MAS (when available)
mas = MASInterface(implementation='glow_tts')
durations = mas.compute_alignment(text_length=50, mel_length=200)
```

## Next Steps for Production

1. **Replace dummy dataset** with real text-audio pairs (e.g., LJSpeech)
2. **Add proper phoneme encoder** (replace ASCII character encoding)
3. **Implement full MAS** or integrate from existing repos
4. **Add vocoder** (HiFi-GAN, WaveGlow, etc.) for waveform generation
5. **Add evaluation metrics** (MCD, MOS, etc.)
6. **Multi-scale refinement** (implement coarse-to-fine pyramid)
7. **Training improvements** (better LR scheduling, augmentation, etc.)

## Testing

Run the test scripts to verify everything works:

```bash
# Test individual components
python -m tts_model.networks.blocks1d
python -m tts_model.networks.text_encoder
python -m tts_model.networks.tts_model

# Test inference
python tts_infer.py

# Test training
python tts_train.py
```

## Dependencies

- PyTorch
- NumPy
- (Optional) Real TTS dataset for training
- (Optional) Vocoder for waveform generation

## License

Follows the same license as the parent DDN repository.
