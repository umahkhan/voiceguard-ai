# VoiceGuard — JPM Demo Setup

What to record before the meeting, in order. Total prep time ~15 min.

## What the dashboard shows

A fraud-operations console for a contact-center reviewer. When a call comes
in, the dashboard runs two voice models in real time:

- **Speaker Match** — compares the caller's voiceprint to the customer's
  enrolled baseline (ECAPA-TDNN, the industry standard).
- **Voice Risk** — detects synthesis artifacts in the audio (Wav2Vec2
  deepfake classifier + F0 prosody check).

The reviewer sees both signals plus caller context (claimed identity,
account, transaction in flight), and decides Approve / Step-Up Auth /
Block. That decision actually drives a LangGraph pipeline that pauses at
the human-review step and resumes downstream based on the call.

## Pre-meeting prep — three things to record / generate

### 1. Your enrollment voice (in the dashboard)

In the **Customer Voiceprint** panel, pick **Record**, hit the mic
icon, and read aloud for 5–10 seconds. Any sentence works; a neutral
script is best:

> *"This is John Smith calling from New York. My account number ends
> in zero-zero-four-two."*

This becomes the registered voiceprint — the baseline every scenario
is compared against.

### 2. AI clone of your enrolled voice (uploaded into the dashboard)

This is the *real* test. Generate a clone of your own voice using any
voice-cloning tool (ElevenLabs free tier, Resemble.ai, or similar):

1. Sign up for **ElevenLabs** (free tier is enough)
2. Use **Voice Cloning → Instant Voice Clone**, upload the same
   recording you used for enrollment (or a longer sample if available)
3. Generate a 5–10 second clip of the clone reading something different,
   e.g. *"I need to wire twenty-five thousand dollars to a new account."*
4. Download the result, then drop it into the **Demo Audio · AI Clone of
   Customer** uploader in the dashboard

If you skip this step, Scenario 2 falls back to the existing
`audio/clean.mp3` ElevenLabs sample — still a valid deepfake demo, but
of a *different speaker*, which is the easier case (speaker match
catches it). Uploading your own clone tests the harder case where the
clone may pass speaker match.

### 3. A colleague's voice (file you drop into the repo)

One other person reads the **same sentence** as your enrollment, 5–10
seconds. Phone Voice Memos / laptop mic / anything is fine.

Save the file as:

```
audio/impersonator.wav
```

(any of `.wav` / `.mp3` / `.m4a` is fine — rename and update
`SCENARIOS["3 · Real-Person Impersonator"]["audio"]` in `app.py` if you
use a different extension.)

If you skip this step, scenario 3 falls back to `borderline.mp3` as a
placeholder.

### Already in the repo (no action needed)

- `audio/clean.mp3` — ElevenLabs sample of a different speaker. The
  Scenario 2 fallback when no self-clone is uploaded.
- `audio/ai_voice.wav` — macOS `say` synthesizer. Powers the Robocall
  scenario.

## The 4 scenarios — what each one teaches JPM

| # | Scenario | Audio | Verdict | Pitch point |
|---|---|---|---|---|
| 1 | Authenticated Customer | your enrollment | **PASS** | Legit calls go through with no friction. This is the false-positive cost story (19% of fraud cost is FPs at JPM today). |
| 2 | AI Voice Clone of Customer | clone of your enrolled voice (uploaded) — falls back to `clean.mp3` | **FLAG** (usually) | The hardest case: an AI clone of the *registered customer's actual voice*. May pass speaker match if the clone is good enough — voice-risk artifacts are the backstop. Either way teaches a layered-defense lesson. JPM literally hears what a clone of a known voice sounds like. |
| 3 | Real-Person Impersonator | `impersonator.wav` | **FLAG** | A real human reading off a script. The case where synthesis detection is useless — biometrics carry the load. |
| 4 | Robocall / Crude Bot | `ai_voice.wav` | **BLOCK** | Older TTS. Both signals trip; the easy case but real and worth showing. |
| Live | Live Microphone Input | record on the spot | depends | "Try it now" — JPM speaks into the laptop, sees their own voice authenticate (or fail). The most visceral moment of the demo. |

## Suggested walkthrough order (8–10 minutes)

1. **Step 1 · Enroll Customer Voice** (record yourself live, 30s).
   Establishes that this is a *real* enrollment, not a canned demo.

2. **Scenario 1 · Authenticated Customer** (1 min).
   Click Connect. Audio plays, signals settle, verdict lands on **PASS**.
   Click *Approve* — pipeline resumes, intelligence node logs the case.
   Point: *"This is what 99% of calls look like — no friction for legit
   customers."*

3. **Scenario 4 · Robocall** (1 min).
   Easy case. Both signals scream. Verdict: **BLOCK**. Click Block.
   Point: *"The trivial case — robocalls and crude synthesis. Solved."*

4. **Scenario 2 · AI Voice Clone of Customer** (2 min).
   Audio plays — JPM *hears* a deepfake of *the customer's own voice*.
   Two cases:
   - **If speaker match is high (clone fooled biometrics)**: point to
     the Voice Risk meter and verdict — *"Even when the clone passes
     biometrics, synthesis artifacts are the backstop. And if both
     clear, defense in depth (step-up auth) catches it."*
   - **If speaker match is low (clone didn't fool biometrics)**: point
     to it directly — *"The clone sounds natural to a human ear, but
     ECAPA picks up subtle differences in the voiceprint that even
     ElevenLabs can't fully replicate. This is exactly the case a
     synthesis-only detector would miss."*
   Either outcome makes the layered-defense story.

5. **Scenario 3 · Real-Person Impersonator** (1 min).
   Plays your colleague's recording. No synthesis at all — Voice Risk
   meter is low. But Speaker Match flags it. Verdict: **FLAG**.
   Point: *"And here's the case nobody talks about — a real human
   impersonator. No deepfake classifier in the world catches this.
   Voice biometrics is the only thing that does."*

6. **Live Microphone Input** (1–2 min).
   "Anyone want to try?" — let a JPM person speak into the mic. Speaker
   match drops to ~0; verdict FLAGs. Point: *"This isn't a recording — we
   just analyzed the call you spoke a moment ago, in real time."*

7. **Pipeline status banner + Stage-by-Stage Audit tab** (1 min).
   Show the LangGraph pipeline chips lighting up. Switch to the Audit tab
   and walk through the journey trace. Point: *"The reviewer's button
   click isn't cosmetic — it actually pauses and resumes the pipeline.
   Step-Up triggers OTP at the auth_challenge node. Block routes the case
   to fraud recovery. Approve clears it."*

## Talking points by scenario

- **The 19/7 cost asymmetry**: every PASS in scenario 1 saves ~$X in
  false-positive friction (block-on-legit-customer).
- **Layered defense, not single model**: scenarios 2 + 3 illustrate that
  no single signal works alone — together they cover the failure modes
  each one misses.
- **HITL is real, not theater**: the LangGraph pipeline genuinely pauses
  at human_review and resumes based on the reviewer's decision. JPM can
  ask "where does the click actually go?" and the answer is concrete.
- **What's still mocked**: caller context (claimed name, account, txn)
  is scripted per scenario — there's no real CRM behind it. Behavior +
  Agent Suspicion meters are scripted. IVR analysis is scripted. These
  are the integration projects after the pitch.

## Cold-start

First call in a fresh session takes ~15–25s while the models load. After
that, every subsequent call is sub-second. **Pre-warm by clicking through
all 4 scenarios once before the JPM meeting starts.** They'll then run
snappy in front of the audience.

## Troubleshooting

- **Speaker Match low (~0.6) for your own voice** — your enrollment was
  recorded on a different mic/codec than the live call. Re-enroll using
  the in-app Record button and the cross-condition mismatch goes away
  (~0.85+).
- **Robocall scenario shows lower-than-expected verdict** — `ai_voice.wav`
  is short and from macOS `say`. If you re-record `audio/ai_voice.wav`
  with a different TTS the scores will change. Behavior risk (scripted)
  is still high, so verdict tends to BLOCK regardless.
- **AI Voice Clone scenario shows PASS** — means the deepfake classifier
  was fooled (it sometimes is on ElevenLabs) and your colleague's voice
  happens to embed close to yours. This is itself a teachable moment —
  speaker match sometimes isn't enough either, which is why production
  systems also use cross-channel auth and active challenge.
