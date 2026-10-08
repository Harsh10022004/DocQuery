# Prompt changelog

These were hardcoded as strings in src/engine.py before, so tweaking the wording never
showed up properly in a diff. Moved them out into files.

## system_prompt.txt
- v1: basic "answer only from the context" instruction.
- v2 (current): added the line about ignoring instructions that appear inside the
  retrieved chunks or the question itself. Added this after the adversarial cases in
  data/eval/benchmark.json showed the model sometimes following an instruction buried
  in the question instead of just answering from the docs.

## deep_analysis_prompt.txt
- v1 (current): unchanged. Only used by the optional "Deep AI Analysis" expander in the
  Streamlit UI.
