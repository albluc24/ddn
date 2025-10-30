"""Network components for TTS model."""

from .blocks1d import (
    Conv1d,
    GroupNorm,
    UNetBlock1D,
    DiscreteDistributionOutput1D,
    DiscreteDistributionBlock1D,
    ConditionProcess1D,
)

from .text_encoder import (
    TextEncoder,
    DurationPredictor,
    MonotonicAlignmentSearch,
    DurationAlignmentModule,
)

from .tts_model import (
    DDNTTSModel,
    DDNTTSModelSimple,
)

__all__ = [
    # 1D blocks
    'Conv1d',
    'GroupNorm',
    'UNetBlock1D',
    'DiscreteDistributionOutput1D',
    'DiscreteDistributionBlock1D',
    'ConditionProcess1D',
    
    # Text encoder
    'TextEncoder',
    'DurationPredictor',
    'MonotonicAlignmentSearch',
    'DurationAlignmentModule',
    
    # TTS model
    'DDNTTSModel',
    'DDNTTSModelSimple',
]
