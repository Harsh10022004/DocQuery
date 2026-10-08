# Prompt changelog

We used to have these hardcoded as python strings inside src/engine.py which meant
we couldn't diff them properly when we tweaked wording. Moved them out here so git
actually shows what changed between versions.

## system_prompt.txt
- v1 (initial): basic "answer only from context" instruction.
- v2 (current): added a line telling the model to ignore any instructions that show
  up inside the retrieved doc chunks or the user question itself. We added this after
  testing a few prompt-injection style questions in our eval set (see
  data/eval/benchmark.json, category: adversarial) where the model would sometimes
  follow an instruction buried in the question instead of just answering from docs.

## deep_analysis_prompt.txt
- v1 (current): unchanged since first version, used only for the optional "Deep AI
  Analysis" expander in the Streamlit UI.
