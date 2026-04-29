"""Shared Anthropic client helper used by every node.

Falls back to a deterministic stub when ANTHROPIC_API_KEY is not configured so
the demo runs end-to-end offline.
"""

from __future__ import annotations

import os

from anthropic import Anthropic

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 300


def has_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _stub(prompt: str) -> str:
    p = prompt.lower()
    if "ivr entry" in p:
        return "IVR Entry: synthetic-voice probability elevated. Flagging for behavioral analysis."
    if "navigation" in p or "phrasing" in p:
        return ("Navigation timing 180ms avg (human baseline: 600ms). "
                "Phrasing pattern matches synthetic caller profile.")
    if "voiceguard alert" in p or "step-up" in p:
        return ("⚠️ VOICEGUARD ALERT — Caller flagged as likely synthetic. "
                "Step-up authentication required. Do not process account changes until OTP verified.")
    if "outcome summary" in p or "intelligence" in p:
        return "Synthetic-voice attack neutralized at OTP step-up; transaction blocked and threat vector logged."
    return "[demo mode: no API key configured]"


def call_claude(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Single source of truth for Anthropic API calls. All nodes use this."""
    if not has_api_key():
        return _stub(prompt)
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()
