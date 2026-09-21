"""Speech-side evaluation of duplex generations.

Two components:

- ``audiobox_scorer`` — Meta Audiobox Aesthetics quality scoring on the
  per-role audio channels (tutor / student / mixed). 4 axes per channel:
  PQ / PC / CE / CU.
- ``turntaking_judge`` — ESPnet ``Turn_taking_prediction_SWBD`` model
  (Whisper-medium encoder + 5-class head) runs on the mono mixdown of the
  conversation and emits a per-window label in {C, T, BC, I, NA}.

See ``speech_eval/README.md`` and ``speech_eval/docs/`` for details.
"""
