# Evaluators

Per-evaluator details — what is measured, on what inputs, with what caveats. For the `Evaluator` ABC contract and how to add a new one, see [Architecture](architecture.md).

## `stats`

Zero-dep conversation-level summary computed from the JSONL alone: duration, segment counts, words/min per role, talk-time ratio, role switches. Use it as a sanity check before paying for an LLM judge.

**Caveat — segment-timestamp quirk:** for some backends, the student's `end_time` reflects when the API delivered the transcript token rather than when audio actually ended, which inflates `student_words_per_min` and biases `talk_ratio`. The evaluators report what's in the JSONL honestly; the fix belongs in `duplex/relay.py` (segment construction), or use an aligned variant via `--align-variant`.

## `turn_taking`

Inspired by [Full-Duplex-Bench (Lin et al., 2025)](https://arxiv.org/abs/2503.04721) — pause handling, smooth turn-taking, backchanneling, interruption — but recomputed from the segment list we already store, so they're free to evaluate:

- `response_latency_s` — mean / median / p90 of the gap between one role finishing and the other starting.
- `overlap_s` — total simultaneous-speech time.
- `silence_ratio` — fraction of conversation with neither role speaking.
- `backchannel_count` — short utterances inside the other role's turn.

Same segment-timestamp caveat as `stats`.

## `naturalness`

DNSMOS via `speechmos` — SIG / BAK / OVRL / P.808 per role. 16 kHz resample done in-process. ONNX weights ship inside the `speechmos` wheel under `site-packages/speechmos/dnsmos_models/` (~2.8 MB), no separate download.

**Why DNSMOS:** non-intrusive (no reference audio needed) and is the predictor used in the Microsoft DNS Challenge series. The `speechmos` PyPI package does NOT bundle UTMOS despite the name — only DNSMOS / PLCMOS / AECMOS — so DNSMOS is what we use. Swapping in NISQA or another predictor later is a one-line change inside `_score`.

## `llm_judge` (5-key naive rubric)

| Key | What it scores |
|---|---|
| `answer_correctness` | Did the tutor reach the right final answer? |
| `scaffolding_quality` | Did the tutor guide rather than give away? |
| `student_realism` | Does the student utterance pattern look like a real learner? |
| `phrasing_naturalness` | Spoken-style phrasing, no markdown / code fences |
| `overall` | Holistic judgment |

Each key is a 1–5 integer score + one-sentence rationale. Rubric prompt at `evaluation/prompts/tutor_judge.txt`. Draws on the [pedagogical Socratic tutoring rubric](https://ceur-ws.org/Vol-4006/paper3short.pdf) (mistake remediation, scaffolding, guidance, coherence/tone) and [TutorBench-style](https://arxiv.org/html/2510.02663) graded criteria.

## `llm_judge_bea` (8-dimension BEA / Maurya et al. taxonomy)

| Key | What it scores |
|---|---|
| `confusion_identification` | Did the tutor recognise the student's misconception? |
| `confusion_location` | Did the tutor pinpoint *where* the confusion is? |
| `answer_withheld` | Did the tutor avoid giving the answer prematurely? |
| `guidance_quality` | Quality of hints and Socratic moves |
| `actionability` | Are next-step suggestions concrete? |
| `coherence` | Conversation flows logically |
| `tutor_tone` | Encouraging, patient, age-appropriate |
| `human_likeness` | Feels like a real tutor, not a script |

Each key is a 1–5 integer score + one-sentence rationale. Rubric prompt at `evaluation/prompts/tutor_judge_bea.txt`. Applies the 8-dimension taxonomy from [Maurya et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.57.pdf), adapted for whole-conversation scoring on MathVista. Keeping both rubrics lets you compare a heuristic 5-axis against a literature-aligned one on the same runs.

## LLM judge providers

`evaluation/evaluators/providers.py` defines a `JudgeProvider` Protocol with three first-class implementations — `OpenAIJudge`, `GeminiJudge`, `ClaudeJudge` — and a `make_judge_provider(provider, model)` factory. Defaults:

| Provider | Default model | SDK | Auth env var |
|---|---|---|---|
| `openai` | `gpt-5-mini` | `openai` | `OPENAI_API_KEY` |
| `gemini` | `gemini-2.5-flash` | `google-genai` | `GEMINI_API_KEY` |
| `claude` | `claude-sonnet-4-6` | `anthropic` | `ANTHROPIC_API_KEY` |

SDK imports are lazy per-class, so an unconfigured provider only fails at construction time, not at module import. Both `llm_judge` and `llm_judge_bea` use the same factory — providers are not rubric-specific.

**Transient-error retry:** `_with_retry` in `providers.py` wraps each provider's `complete()` call and retries on rate-limit / 5xx / connection / overloaded errors with exponential backoff (5 attempts starting at 2 s). Non-transient errors surface immediately. Failed records are stored with `status: "error"` (or `"parse_error"` when the JSON is unparseable) and a plain re-run picks them up automatically — `--force` is only needed to recompute successful results.

**JSON-mode coverage varies by provider:** OpenAI uses `response_format={"type": "json_object"}` and Gemini uses `response_mime_type="application/json"`. Claude has no equivalent flag; the prompt instructs JSON-only output and `_parse_rubric` tolerates stray ```` ```json ```` fences and surrounding prose for all three.

**Claude prefill caveat:** the Claude judge sends a single user message (no assistant-message prefill). The earlier prefill trick caused malformed JSON on `claude-sonnet-4-6`; the prompt instructs JSON-only output and the parser tolerates stray code fences.

## Cost notes

- One API call per `(conversation, judge)`. Running both `llm_judge` + `llm_judge_bea` doubles the call count; the judge sweep across 3 providers × 2 rubrics is 6× the baseline.
- Costs are still cents-scale per 100 conversations on default models (`gpt-5-mini` / `gemini-2.5-flash` / `claude-sonnet-4-6`).
- Rubric prompts live at `prompts/tutor_judge.txt` and `prompts/tutor_judge_bea.txt` and are freely editable — re-run with `--force` after changes.
