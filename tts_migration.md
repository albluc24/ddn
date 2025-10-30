# DDN to TTS Migration Guide

This guide outlines how to repurpose the discrete distribution network (DDN) components in this repository for a Tacotron/Grad-TTS style text-to-speech architecture. It highlights the files to lift, the dependencies you must preserve, and the changes needed to make the model operate on 1D sequence data instead of 2D images.

## 1. What to Extract
- Core blocks: `DiscreteDistributionBlock`, `PHDDNHandsDense` / `PHDDNHandsSparse`, and their helpers in `training/networks.py`.
- Support utilities: `ConditionProcess`, `ClassEmbeding`, and `UNetBlockWoEmb` if you keep the multi-scale refinement idea.
- External runtime dependencies: `sddn` (for `DiscreteDistributionOutput`, mixed-precision convolutions, and feature initializers) and `boxx` (logging/debug helpers). Replace or stub them if you want a slimmer stack.
- Minimal wrapper logic from `generate.py` or `training/training_loop.py` can be copied only for reference; you will build new TTS-specific driver code.

## 2. Reframe the Data Flow
- **Input tokens**: Replace image tensors with phoneme or character embeddings (`[batch, time, dim]`). Map them to 1D latent features using causal convolutions or transformers. The existing 2D convs in `UNetBlockWoEmb` must be swapped for 1D layers (`Conv1d`, depthwise separable convs, or attention blocks) that respect temporal order.
- **Outputs**: Emit mel-spectrogram frames (`[batch, mel_bins, time]`) instead of RGB pixels. Update `DiscreteDistributionOutput` usage to operate along the time dimension; if the helper assumes square feature maps, clone and adjust it in your project to accept `(channels, length)` tensors.
- **Conditioning**: Create a new `ConditionProcess` variant that injects text embeddings, speaker IDs, or prosody controls. Remove image-specific options such as `edge` or `color`.

## 3. Duration and Alignment Strategy
- Integrate Monotonic Alignment Search (MAS) to align input tokens with output frames. During training, use MAS to produce frame-level alignments and durations that feed into your discrete distribution decoder.
- Train an auxiliary duration predictor. During inference, the predictor replaces MAS to generate alignment paths on the fly.
- Keep the multi-scale sampling ideas if desired: coarse-to-fine duration refinement can mirror the pyramid stages in `PHDDN`.

## 4. Architectural Changes
- Redefine scale loops: the current `PHDDN` expects resolution doubling (`2**scale`). For TTS, drive scale indices with temporal down/up-sampling (e.g., doubling frame rate or using strided convs). Ensure every block knows the 1D length it outputs.
- Re-implement `UNetBlockWoEmb` with temporal convolutions or self-attention suited for sequences. Preserve skip connections and optional attention to keep the hierarchical refinement behavior.
- Adjust feature initialization in `DiscreteDistributionBlock.forward`. Replace calls to `sddn.build_init_feature` with a 1D initializer that matches your mel length.

## 5. Training Loop Integration
- Build a new training loop (PyTorch Lightning, Accelerate, or a custom script) that feeds text/audio pairs, applies MAS, and computes diffusion or NLL style losses.
- Reuse optimizer/EMA utilities from `training/training_loop.py` if helpful, but remove image-centric logging code and dataset loaders.
- Add evaluation hooks specific to TTS (mel cepstral distortion, MOS proxies, etc.).

## 6. Inference Pipeline
1. Text → phoneme encoder to produce conditioning embeddings.
2. Duration predictor (or MAS) to produce time stamps.
3. DDN-based decoder to sample mel frames given the conditioning and alignments.
4. Vocoder (HiFi-GAN, WaveGlow, or BigVGAN) to synthesize waveforms from the mel output.
5. Optional refinement stages mimicking the `refiner` loop in `PHDDN` for de-noising or prosody tweaks.

## 7. Testing Checklist
- Unit-test your 1D `DiscreteDistributionBlock` on synthetic data to ensure shape compatibility and loss stability.
- Validate MAS integration by measuring alignment monotonicity and duration prediction accuracy.
- Run end-to-end TTS smoke tests: short utterances, long-form reading, and corner cases (punctuation, silence).
- Monitor GPU memory use; the original code assumes 2D tensors and may need refactoring for efficient 1D batching.

## 8. Open Questions to Resolve
- How to adapt `DiscreteDistributionOutput` if the public `sddn` package lacks 1D support. You may need to fork or reimplement the quantized mixture head.
- Whether the coarse-to-fine pyramid still offers gains for TTS compared to a single-scale decoder. Profile both options.
- Choice of attention mechanism: convert `AttentionOp` to a causal variant or integrate transformer layers for expressive prosody.

By isolating the reusable DDN abstractions and retooling them around 1D temporal features, you can carry the discrete sampling benefits into a Tacotron/Grad-TTS style pipeline while minimizing coupling to the original image-focused code base.
