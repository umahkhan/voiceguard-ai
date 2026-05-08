"""Real-model detectors for VoiceGuard.

Houses the inference wrappers that replace hand-tuned scenario scores when
Live Mode is engaged. Imports of heavy ML deps (torch, transformers, librosa)
happen lazily inside the functions so the scripted demo path costs nothing.
"""

from detectors.voice_clone import detect, MODEL_NAME
from detectors.speaker_verify import similarity as speaker_similarity, embed as speaker_embed

__all__ = ["detect", "MODEL_NAME", "speaker_similarity", "speaker_embed"]
