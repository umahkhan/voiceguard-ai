"""Speaker verification — does this audio match a known reference voice?

Uses SpeechBrain's ECAPA-TDNN model trained on VoxCeleb. ECAPA is the
industry-standard architecture for speaker recognition and gives sharp
separation between speakers — same-speaker comparisons typically land
~0.50–0.85, different-speaker pairs ~-0.10–0.30 (cosine similarity on
192-dim L2-normalized embeddings).

Earlier prototype used Resemblyzer (lighter, ~30 MB) but its
discrimination was insufficient for cross-speaker comparisons,
especially male/female where it returned ~0.55–0.65 for clearly
different speakers. ECAPA-TDNN's larger training set and architecture
push those down toward 0 cleanly.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_MODEL_DIR = Path(__file__).parent / "_models" / "spkrec-ecapa-voxceleb"


@functools.lru_cache(maxsize=1)
def _load_verifier():
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source=_MODEL_SOURCE,
        savedir=str(_MODEL_DIR),
        run_opts={"device": "cpu"},
    )


@functools.lru_cache(maxsize=1)
def _load_encoder():
    """Encoder for one-shot embedding extraction (used when we want the
    embedding itself rather than a pairwise score)."""
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source=_MODEL_SOURCE,
        savedir=str(_MODEL_DIR),
        run_opts={"device": "cpu"},
    )


@functools.lru_cache(maxsize=64)
def _embed(audio_path: str) -> tuple[float, ...]:
    import librosa
    import torch

    encoder = _load_encoder()
    # ECAPA-TDNN expects 16 kHz mono. librosa handles MP3/WAV/M4A/FLAC
    # uniformly without needing torchaudio's codec backend.
    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    signal = torch.from_numpy(y).unsqueeze(0)
    emb = encoder.encode_batch(signal).squeeze().detach().cpu().numpy()
    n = float(np.linalg.norm(emb)) + 1e-9
    emb = emb / n
    return tuple(float(x) for x in emb)


def embed(audio_path: str | Path) -> np.ndarray:
    return np.array(_embed(str(audio_path)), dtype=np.float32)


def similarity(reference_path: str | Path, test_path: str | Path) -> float:
    """Cosine similarity in [-1, 1] between two voice clips. Same speaker
    typically 0.50–0.85; different speakers typically -0.10–0.30."""
    if str(reference_path) == str(test_path):
        return 1.0
    a = embed(reference_path)
    b = embed(test_path)
    return float(np.dot(a, b))
