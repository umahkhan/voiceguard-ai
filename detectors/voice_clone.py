"""Wav2Vec2-based voice-cloning detector.

Replaces `random.uniform` scoring in `agents/node1_voice_cloning.py` with two
honest signals:

  * `spectral_score` — P(synthetic) from a Wav2Vec2 deepfake classifier
    fine-tuned on ASVspoof. Catches TTS / voice-clone artifacts in the
    spectrum that are hard to reproduce naturally.
  * `prosody_score`  — F0-contour anomaly from librosa.yin. Synthetic
    speech tends to have unnaturally regular pitch; we measure the
    coefficient of variation of voiced-frame F0 and invert.

The classifier output is then passed through an FP-aware calibration that
biases scores away from false-positive territory — JPM's cost model puts
false positives at 19% of total fraud cost vs. 7% for missed fraud, so
the system should require higher confidence before pushing a call into
FLAG / BLOCK.
"""

from __future__ import annotations

import functools
from pathlib import Path

MODEL_NAME = "MelodyMachine/Deepfake-audio-detection"

_FAKE_LABEL_KEYS = ("fake", "spoof", "deepfake", "synthetic", "ai", "bonafide")


@functools.lru_cache(maxsize=1)
def _load_pipeline():
    from transformers import pipeline
    # Force CPU. MPS (Apple Silicon) hits a conv1d output-channel cap on
    # this model; CUDA isn't expected on Streamlit Cloud either.
    return pipeline("audio-classification", model=MODEL_NAME, device=-1)


def _extract_fake_prob(results: list[dict]) -> float:
    fake_score = None
    real_score = None
    for r in results:
        label = str(r.get("label", "")).lower()
        score = float(r.get("score", 0.0))
        if "bonafide" in label or "real" in label or "human" in label:
            real_score = score
        elif any(k in label for k in ("fake", "spoof", "deepfake", "synthetic", "ai")):
            fake_score = score
    if fake_score is not None:
        return fake_score
    if real_score is not None:
        return 1.0 - real_score
    return float(results[0].get("score", 0.5))


@functools.lru_cache(maxsize=64)
def _spectral_score(audio_path: str) -> float:
    import librosa

    pipe = _load_pipeline()
    sr = int(pipe.feature_extractor.sampling_rate)
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    out = pipe({"raw": y, "sampling_rate": sr}, top_k=5)
    return _extract_fake_prob(out)


@functools.lru_cache(maxsize=64)
def _prosody_score(audio_path: str) -> float:
    """F0-contour anomaly: flatter pitch contour → higher score.

    Conversational human speech typically lands in CV 0.20–0.40 on natural
    recordings (mp3 codec adds jitter that nudges this up). Older TTS
    (concatenative, formant) is much flatter (CV ~0.10–0.25). Modern
    neural TTS (ElevenLabs, Tortoise, XTTS) deliberately injects
    natural-looking variation, so this signal alone won't catch them —
    that's what the Wav2Vec2 spectral model is for. Prosody is a
    secondary check that flags obvious / older synthesis cheaply.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
    voiced = f0[(f0 > 60) & (f0 < 400)]
    if len(voiced) < 20:
        return 0.5
    cv = float(np.std(voiced) / (np.mean(voiced) + 1e-6))
    # Map: CV ≤ 0.08 → 1.0 (very flat). CV ≥ 0.30 → 0.0 (natural).
    return float(np.clip((0.30 - cv) / 0.22, 0.0, 1.0))


@functools.lru_cache(maxsize=64)
def _duration(audio_path: str) -> float:
    import librosa
    return float(librosa.get_duration(path=audio_path))


def _calibrate_fp_aware(p: float) -> float:
    """Power transform to bias the classifier toward fewer false positives.

    p^1.6 leaves very-confident scores almost unchanged but pulls the
    middling 0.4–0.7 region toward 0, where the FP cost lives.
    """
    return float(max(0.0, min(1.0, p ** 1.6)))


def detect(audio_path: str | Path, fp_tuned: bool = True) -> dict:
    """Run real-model detection on a single audio file.

    Returned dict mirrors the VoiceGuardState fields populated by Stage 1
    (raw_audio_duration_sec, spectral_score, prosody_score) plus a
    `model_used` tag for UI display.
    """
    p = str(audio_path)
    spectral_raw = _spectral_score(p)
    spectral = _calibrate_fp_aware(spectral_raw) if fp_tuned else spectral_raw
    return {
        "raw_audio_duration_sec": round(_duration(p), 2),
        "spectral_score": round(spectral, 3),
        "spectral_score_raw": round(spectral_raw, 3),
        "prosody_score": round(_prosody_score(p), 3),
        "model_used": MODEL_NAME,
        "fp_tuned": fp_tuned,
    }
