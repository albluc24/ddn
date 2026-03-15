#!/usr/bin/env python3
"""
Complete TTS MVP Demonstration

This script demonstrates:
1. Model creation and architecture
2. Training on dummy data
3. Inference and mel generation
4. MAS integration points

Run with: python tts_demo.py
"""

import torch
import numpy as np
from pathlib import Path


def demo_model_architecture():
    """Demonstrate model architecture and components."""
    print("=" * 70)
    print("DEMO 1: Model Architecture")
    print("=" * 70)
    
    from tts_model.networks.tts_model import DDNTTSModel, DDNTTSModelSimple
    
    # Simple model for quick testing
    print("\n1. Simple Model (for quick testing)")
    simple_model = DDNTTSModelSimple(vocab_size=128, mel_bins=80)
    num_params = sum(p.numel() for p in simple_model.parameters())
    print(f"   Parameters: {num_params:,}")
    print(f"   Memory: ~{num_params * 4 / 1024 / 1024:.1f} MB")
    
    # Full model
    print("\n2. Full Model (for production)")
    full_model = DDNTTSModel(
        vocab_size=128,
        text_embed_dim=512,
        text_hidden_dim=512,
        decoder_channels=256,
        decoder_k=64,
        mel_bins=80,
    )
    num_params = sum(p.numel() for p in full_model.parameters())
    print(f"   Parameters: {num_params:,}")
    print(f"   Memory: ~{num_params * 4 / 1024 / 1024:.1f} MB")
    
    # Show architecture
    print("\n3. Architecture Pipeline:")
    print("   Input: Text tokens [batch, text_length]")
    print("   ↓")
    print("   TextEncoder: Conv1D + Positional Encoding")
    print("   ↓")
    print("   DurationAlignmentModule: Predict + Expand")
    print("   ↓")
    print("   ConditionProcess1D: Text + Speaker Conditioning")
    print("   ↓")
    print("   DiscreteDistributionBlock1D: DDN Decoder")
    print("   ↓")
    print("   Output: Mel-spectrograms [batch, mel_bins, mel_length]")
    print("           (k discrete predictions)")


def demo_training():
    """Demonstrate training on dummy data."""
    print("\n" + "=" * 70)
    print("DEMO 2: Training on Dummy Data")
    print("=" * 70)
    
    from tts_train import DummyTTSDataset, TTSTrainer
    from tts_model.networks.tts_model import DDNTTSModelSimple
    
    # Create small dataset
    print("\n1. Creating dummy dataset...")
    train_dataset = DummyTTSDataset(num_samples=50, vocab_size=128, mel_bins=80)
    val_dataset = DummyTTSDataset(num_samples=20, vocab_size=128, mel_bins=80)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Show sample
    sample = train_dataset[0]
    print(f"\n2. Sample data:")
    print(f"   Text IDs: {sample['text_ids'].shape}")
    print(f"   Mel: {sample['mel'].shape}")
    print(f"   Durations: {sample['durations'].shape}")
    
    # Create model
    print("\n3. Creating model...")
    model = DDNTTSModelSimple(vocab_size=128, mel_bins=80)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train for 2 epochs
    print("\n4. Training for 2 epochs (this may take a minute)...")
    trainer = TTSTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        learning_rate=1e-4,
        device='cpu',
        checkpoint_dir='/tmp/tts_demo_checkpoints',
    )
    
    trainer.train(num_epochs=2, save_every=2)
    print("\n   ✓ Training completed!")


def demo_inference():
    """Demonstrate inference and mel generation."""
    print("\n" + "=" * 70)
    print("DEMO 3: Inference and Mel Generation")
    print("=" * 70)
    
    from tts_infer import TTSInference
    
    # Create inference interface
    print("\n1. Initializing inference...")
    tts = TTSInference(model_path=None, device='cpu', vocab_size=128, mel_bins=80)
    print("   ✓ Model loaded")
    
    # Test texts
    texts = [
        "Hello world",
        "This is a test",
        "The quick brown fox jumps over the lazy dog",
    ]
    
    print("\n2. Generating mel-spectrograms...")
    output_dir = Path("/tmp/tts_demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, text in enumerate(texts):
        print(f"\n   Text {i + 1}: '{text}'")
        
        # Generate mel
        mel = tts.synthesize(text)
        print(f"   Generated mel shape: {mel.shape}")
        
        # Save
        save_path = output_dir / f"mel_{i + 1}.npy"
        np.save(save_path, mel)
        print(f"   Saved to: {save_path}")
    
    # Batch inference
    print("\n3. Batch inference...")
    mels = tts.synthesize_batch(texts)
    print(f"   Generated {len(mels)} mel-spectrograms")
    for i, mel in enumerate(mels):
        print(f"   Mel {i + 1}: {mel.shape}")
    
    print("\n   ✓ Inference completed!")
    print(f"\n   Note: This is an untrained model, outputs are random.")
    print(f"   For real results, train on real text-audio pairs.")


def demo_mas_integration():
    """Demonstrate MAS integration points."""
    print("\n" + "=" * 70)
    print("DEMO 4: MAS (Monotonic Alignment Search) Integration")
    print("=" * 70)
    
    from tts_infer import MASInterface
    from tts_model.networks.text_encoder import MonotonicAlignmentSearch
    
    # Stub implementation
    print("\n1. Stub MAS (default):")
    mas = MASInterface(implementation='stub')
    
    text_length = 20
    mel_length = 80
    durations = mas.compute_alignment(text_length=text_length, mel_length=mel_length)
    
    print(f"   Text length: {text_length}")
    print(f"   Mel length: {mel_length}")
    print(f"   Computed durations: {durations.tolist()[:10]}...")
    print(f"   Sum of durations: {durations.sum()} (should equal {mel_length})")
    
    # Show integration points
    print("\n2. Integration points for external MAS:")
    print("   a) Glow-TTS MAS:")
    print("      - Set implementation='glow_tts' in MASInterface")
    print("      - Import from glow_tts.monotonic_align")
    print("      - Implement in MASInterface.__init__")
    
    print("\n   b) Grad-TTS MAS:")
    print("      - Set implementation='grad_tts' in MASInterface")
    print("      - Import from grad_tts alignment module")
    print("      - Implement in MASInterface.__init__")
    
    print("\n   c) Custom MAS:")
    print("      - Implement in MonotonicAlignmentSearch.align()")
    print("      - Add new implementation option")
    print("      - Use attention prior to guide alignment")
    
    # Show usage in training
    print("\n3. Usage in training:")
    print("   - Training: Use external durations or MAS to generate alignments")
    print("   - Inference: Use duration predictor to generate durations")
    print("   - Both: Use expand_with_durations to map text to mel length")
    
    # Example code
    print("\n4. Example code:")
    print("""
    # In training loop:
    result = model(
        text_ids=text_ids,
        target_mel=target_mel,
        external_durations=durations,  # From MAS or ground truth
    )
    
    # In inference:
    mel = model.infer(text_ids)  # Uses duration predictor
    """)


def demo_complete_workflow():
    """Show complete workflow from text to mel."""
    print("\n" + "=" * 70)
    print("DEMO 5: Complete Workflow (Text → Mel)")
    print("=" * 70)
    
    from tts_model.networks.tts_model import DDNTTSModelSimple
    import torch
    
    print("\n1. Create model")
    model = DDNTTSModelSimple(vocab_size=128, mel_bins=80)
    model.eval()
    
    print("\n2. Prepare input text")
    text = "Hello world"
    # Simple ASCII encoding (replace with phoneme encoder in production)
    text_ids = torch.tensor([min(ord(c), 127) for c in text]).unsqueeze(0)
    print(f"   Text: '{text}'")
    print(f"   Text IDs: {text_ids.shape} {text_ids.tolist()}")
    
    print("\n3. Forward pass (with intermediate outputs)")
    with torch.no_grad():
        result = model.model(
            text_ids=text_ids,
            return_intermediate=True,
        )
    
    print(f"\n4. Intermediate outputs:")
    print(f"   Text embeddings: {result['text_embed'].shape}")
    print(f"   Expanded text: {result['expanded_text'].shape}")
    print(f"   Conditioning: {result['conditioning'].shape}")
    print(f"   Predictions (k={model.model.decoder_k}): {result['predictions'].shape}")
    print(f"   Durations: {result['durations'].shape}")
    
    print("\n5. Final output")
    mel = model.infer(text_ids)
    print(f"   Mel-spectrogram: {mel.shape}")
    print(f"   Duration sum: {result['durations'].sum().item():.0f} frames")
    
    print("\n6. Next steps for production:")
    print("   - Add vocoder (HiFi-GAN, WaveGlow) to convert mel → audio")
    print("   - Train on real dataset (LJSpeech, LibriTTS, etc.)")
    print("   - Use proper phoneme encoder (IPA, ARPAbet, etc.)")
    print("   - Add speaker embeddings for multi-speaker TTS")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("DDN-based TTS Model - Complete Demonstration")
    print("=" * 70)
    print("\nThis demo shows all aspects of the TTS MVP implementation.")
    print("Following the migration guide from tts_migration.md")
    
    # Run demos
    demo_model_architecture()
    demo_training()
    demo_inference()
    demo_mas_integration()
    demo_complete_workflow()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✓ Model architecture demonstrated")
    print("✓ Training pipeline working")
    print("✓ Inference pipeline working")
    print("✓ MAS integration points defined")
    print("✓ Complete workflow validated")
    
    print("\n" + "=" * 70)
    print("MVP COMPLETE!")
    print("=" * 70)
    print("\nThe TTS model is ready for further development:")
    print("1. Replace dummy dataset with real text-audio pairs")
    print("2. Implement or integrate full MAS")
    print("3. Add vocoder for waveform generation")
    print("4. Train on production dataset")
    print("5. Evaluate and tune hyperparameters")
    
    print("\nFor more details, see:")
    print("- tts_model/README.md - Full documentation")
    print("- TTS_QUICKSTART.md - Quick start guide")
    print("- tts_migration.md - Architecture migration guide")
    print()


if __name__ == "__main__":
    main()
