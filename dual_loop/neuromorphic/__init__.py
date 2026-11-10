"""
Neuromorphic Spiking Brain Module (X-Star HADL v5.5)
=====================================================
Components:
- FastSigmoidSurrogate: Differentiable surrogate gradient for Heaviside step firing.
- LeakyIntegrateAndFireGate: LIF membrane integrator with homeostatic threshold at Layer 7.
- DynamicNeuralReplicator: High-saliency vertical KV-vector replication preserving identity.
- OpenFreezeLiquidPlasticity: Dynamic viscosity mu(t) = tanh(gamma * Surprisal) phase state.
- ReEntrantCrossLayerBus: Cross-layer epistemic feedback bus for deep layers.
- VirtualLatentSandbox: Dry-run latent transition verification enforcing Banach contraction.
- NeuromorphicSpikingBrain: Unified master engine.
- ModelPersonalityAdapter: Dimension-aware scaling laws for model personality.
- PreRouterSpikingPacemaker: Early layer-2 spiking pacemaker emitting driving signal u_drive.
- TourLeaderAttractor: Continuous vector field guidance without rigid prompt templates.
- ConcurrentDualStreamSynthesizer: Parallel System 1 and System 2 with symplectic orthogonal fusion.
"""

from .spiking_gate import (
    FastSigmoidSurrogate,
    spike_fn,
    HomeostaticPreLMHeadVarianceClamp,
    SyntaxAwareCurvatureWhitening,
    LeakyIntegrateAndFireGate,
    DynamicNeuralReplicator,
    OpenFreezeLiquidPlasticity,
    ReEntrantCrossLayerBus,
    VirtualLatentSandbox,
    NeuromorphicSpikingBrain,
    HADLv5SpikingBrainModelWrapper,
    attach_hadl_v5_spiking_brain,
    AnswerFirstLogitsProcessor,
)

from .pacemaker_router import (
    PersonalityProfile,
    ModelPersonalityAdapter,
    PreRouterSpikingPacemaker,
    TourLeaderAttractor,
    ConcurrentDualStreamSynthesizer,
    DirectionalSafetyProjection,
    OuterDynamicScratchDimension,
    LatentThoughtMatrixEngine,
    CentralCognitiveRouter,
)

from .tokenless_latent_engine import (
    IPv6NumericRegister,
    ThoughtPurifiedKVCache,
    EndogenousMetaRLPolicy,
    TokenlessLatentThoughtEngine,
)
from .acoustic_vocal_organ import EndogenousAcousticVocalOrgan
from .latent_imagination_canvas import Progressive2Dto3DImaginationCanvas

__all__ = [
    "FastSigmoidSurrogate",
    "spike_fn",
    "HomeostaticPreLMHeadVarianceClamp",
    "SyntaxAwareCurvatureWhitening",
    "LeakyIntegrateAndFireGate",
    "DynamicNeuralReplicator",
    "OpenFreezeLiquidPlasticity",
    "ReEntrantCrossLayerBus",
    "VirtualLatentSandbox",
    "NeuromorphicSpikingBrain",
    "HADLv5SpikingBrainModelWrapper",
    "attach_hadl_v5_spiking_brain",
    "AnswerFirstLogitsProcessor",
    "PersonalityProfile",
    "ModelPersonalityAdapter",
    "PreRouterSpikingPacemaker",
    "TourLeaderAttractor",
    "ConcurrentDualStreamSynthesizer",
    "DirectionalSafetyProjection",
    "LatentThoughtMatrixEngine",
    "CentralCognitiveRouter",
    "IPv6NumericRegister",
    "ThoughtPurifiedKVCache",
    "EndogenousMetaRLPolicy",
    "TokenlessLatentThoughtEngine",
    "EndogenousAcousticVocalOrgan",
    "Progressive2Dto3DImaginationCanvas",
]



