"""Simple test suite for TTS model components.

Run this to validate that all components work correctly.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_blocks1d():
    """Test 1D network blocks."""
    print("Testing 1D blocks...")
    from tts_model.networks.blocks1d import (
        Conv1d, GroupNorm, UNetBlock1D,
        DiscreteDistributionBlock1D, ConditionProcess1D
    )
    
    # Test Conv1d
    conv = Conv1d(64, 128, kernel_size=3)
    x = torch.randn(2, 64, 100)
    out = conv(x)
    assert out.shape == (2, 128, 100), f"Conv1d failed: {out.shape}"
    
    # Test UNetBlock1D
    block = UNetBlock1D(64, 128, attention=True)
    x = torch.randn(2, 64, 100)
    out = block(x)
    assert out.shape == (2, 128, 100), f"UNetBlock1D failed: {out.shape}"
    
    # Test DiscreteDistributionBlock1D
    dd_block = DiscreteDistributionBlock1D(
        in_channels=128, out_channels=256, k=32, mel_bins=80, mel_length=100
    )
    x = torch.randn(2, 128, 100)
    target = torch.randn(2, 80, 100)
    result = dd_block(x, target=target)
    assert 'predictions' in result, "Missing predictions"
    assert 'loss' in result, "Missing loss"
    assert result['predictions'].shape == (2, 32, 80, 100), f"Wrong prediction shape: {result['predictions'].shape}"
    
    # Test ConditionProcess1D
    cond = ConditionProcess1D(text_dim=256, speaker_dim=64)
    text = torch.randn(2, 50, 256)
    speaker = torch.randint(0, 10, (2,))
    out = cond(text, speaker)
    assert out.shape[0] == 2 and out.shape[1] == 50, f"ConditionProcess1D failed: {out.shape}"
    
    print("✓ All 1D block tests passed")


def test_text_encoder():
    """Test text encoder and duration model."""
    print("\nTesting text encoder...")
    from tts_model.networks.text_encoder import (
        TextEncoder, DurationPredictor, MonotonicAlignmentSearch, DurationAlignmentModule
    )
    
    # Test TextEncoder
    encoder = TextEncoder(vocab_size=128, embed_dim=256, hidden_dim=512)
    text_ids = torch.randint(1, 128, (2, 50))
    text_embed = encoder(text_ids)
    assert text_embed.shape == (2, 50, 512), f"TextEncoder failed: {text_embed.shape}"
    
    # Test DurationPredictor
    dur_pred = DurationPredictor(text_dim=512)
    durations = dur_pred(text_embed)
    assert durations.shape == (2, 50), f"DurationPredictor failed: {durations.shape}"
    
    # Test MAS
    mas = MonotonicAlignmentSearch()
    align = mas.align(text_length=50, mel_length=200)
    assert align.shape == (50,), f"MAS align failed: {align.shape}"
    assert align.sum() == 200, f"MAS sum failed: {align.sum()}"
    
    # Test DurationAlignmentModule
    dur_align = DurationAlignmentModule(text_dim=512)
    result = dur_align(text_embed, external_durations=torch.ones(2, 50).long() * 4)
    assert 'expanded_embed' in result, "Missing expanded_embed"
    assert result['expanded_embed'].shape[0] == 2, f"Wrong batch size: {result['expanded_embed'].shape}"
    
    print("✓ All text encoder tests passed")


def test_tts_model():
    """Test full TTS model."""
    print("\nTesting TTS model...")
    from tts_model.networks.tts_model import DDNTTSModel, DDNTTSModelSimple
    
    # Test full model
    model = DDNTTSModel(vocab_size=128, mel_bins=80, decoder_k=32)
    text_ids = torch.randint(1, 128, (2, 50))
    target_mel = torch.randn(2, 80, 200)
    durations = torch.ones(2, 50).long() * 4
    
    # Training forward
    result = model(text_ids=text_ids, target_mel=target_mel, external_durations=durations)
    assert 'predictions' in result, "Missing predictions"
    assert 'total_loss' in result, "Missing total_loss"
    assert result['predictions'].shape[0] == 2, f"Wrong batch size: {result['predictions'].shape}"
    
    # Inference
    mel = model.infer(text_ids)
    assert mel.shape[0] == 2, f"Wrong batch size: {mel.shape}"
    assert mel.shape[1] == 80, f"Wrong mel bins: {mel.shape}"
    
    # Test simple model
    simple_model = DDNTTSModelSimple(vocab_size=128, mel_bins=80)
    mel = simple_model.infer(text_ids[:1])
    assert mel.shape[0] == 1, f"Wrong batch size: {mel.shape}"
    assert mel.shape[1] == 80, f"Wrong mel bins: {mel.shape}"
    
    print("✓ All TTS model tests passed")


def test_training():
    """Test training components."""
    print("\nTesting training components...")
    from tts_train import DummyTTSDataset, TTSTrainer, collate_fn
    from tts_model.networks.tts_model import DDNTTSModelSimple
    
    # Test dataset
    dataset = DummyTTSDataset(num_samples=10, vocab_size=128, mel_bins=80)
    assert len(dataset) == 10, f"Wrong dataset size: {len(dataset)}"
    
    sample = dataset[0]
    assert 'text_ids' in sample and 'mel' in sample and 'durations' in sample, "Missing keys in sample"
    
    # Test collate
    batch = collate_fn([dataset[i] for i in range(3)])
    assert batch['text_ids'].shape[0] == 3, f"Wrong batch size: {batch['text_ids'].shape}"
    
    # Test trainer (just initialization)
    model = DDNTTSModelSimple(vocab_size=128, mel_bins=80)
    trainer = TTSTrainer(
        model=model,
        train_dataset=dataset,
        batch_size=2,
        learning_rate=1e-4,
        device='cpu',
    )
    assert trainer is not None, "Trainer initialization failed"
    
    print("✓ All training component tests passed")


def test_inference():
    """Test inference components."""
    print("\nTesting inference components...")
    from tts_infer import TTSInference, MASInterface
    
    # Test TTSInference
    tts = TTSInference(model_path=None, device='cpu', vocab_size=128, mel_bins=80)
    
    # Test text to IDs
    text_ids = tts.text_to_ids("Hello world")
    assert len(text_ids) > 0, "Text to IDs failed"
    
    # Test synthesis
    mel = tts.synthesize("Hello world")
    assert mel.shape[0] == 80, f"Wrong mel bins: {mel.shape}"
    
    # Test batch synthesis
    mels = tts.synthesize_batch(["Hello", "World"])
    assert len(mels) == 2, f"Wrong batch size: {len(mels)}"
    
    # Test MAS interface
    mas = MASInterface(implementation='stub')
    durations = mas.compute_alignment(text_length=10, mel_length=40)
    assert durations.sum() == 40, f"MAS duration sum failed: {durations.sum()}"
    
    print("✓ All inference component tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running TTS Model Test Suite")
    print("=" * 60)
    
    try:
        test_blocks1d()
        test_text_encoder()
        test_tts_model()
        test_training()
        test_inference()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
