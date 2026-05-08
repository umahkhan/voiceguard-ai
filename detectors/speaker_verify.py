"""Speaker verification — does this audio match a known reference voice?

Uses Resemblyzer's VoiceEncoder to produce 256-dim L2-normalized speaker
embeddings, then compares them via cosine similarity. This is the
load-bearing signal for "is this caller really our customer, or someone
else (synthetic or otherwise)?" — independent of how good the synthesis
is, because synthesis-of-someone-else has a different voiceprint.

Banks build these embeddings passively over a customer's first few calls
("passive enrollment"). For the demo we accept a single uploaded
reference clip and treat it as a pre-enrolled voiceprint.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np


@functools.lru_cache(maxsize=1)
def _load_encoder():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder(device="cpu", verbose=False)


@functools.lru_cache(maxsize=64)
def _embed(audio_path: str) -> tuple[float, ...]:
    """Embed an audio file. Returned as a tuple for lru_cache hashability."""
    from resemblyzer import preprocess_wav

    encoder = _load_encoder()
    wav = preprocess_wav(Path(audio_path))
    emb = encoder.embed_utterance(wav)
    return tuple(float(x) for x in emb)


def embed(audio_path: str | Path) -> np.ndarray:
    return np.array(_embed(str(audio_path)), dtype=np.float32)


def similarity(reference_path: str | Path, test_path: str | Path) -> float:
    """Cosine similarity in [-1, 1]. Resemblyzer embeddings are L2-normalized,
    so this is just the dot product. Same speaker typically lands 0.70–0.95;
    different speakers typically 0.0–0.55.
    """
    a = embed(reference_path)
    b = embed(test_path)
    return float(np.dot(a, b))
