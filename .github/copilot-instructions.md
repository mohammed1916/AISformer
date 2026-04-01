# GitHub Copilot Instructions for TrAISformer

## Context
- This repository implements TrAISformer trajectory models (standard and interpolation variants).
- Main workspace focus for current work: `src_interpolation/`.
- Key scripts: `src_interpolation/trAISformer.py`, `src_interpolation/trainers.py`, `src_interpolation/models.py`, `src_interpolation/datasets.py`, `src_interpolation/infer_gap.py`.

## Goals
- Improve and maintain interpolation model training & inference.
- Keep alignment with existing dataset formats in `data/ct_dma/` and config definitions in `config_trAISformer.py`.
- Support reproducible experiment semantics, checkpointing, and visualization output to `results_interpolation/`.

## Coding conventions
- Python 3.9+ idiomatic style (PEP8, clear naming, small helper functions).
- Prefer explicit logging and argument validation over silent failure.
- Keep device-agnostic Torch code (`torch.device`, `cuda` optional).

## Task prioritization
1. Bugfixes in interpolation data preparation / window sampling (e.g., past/gap/future alignment, padding masks).
2. Model correctness (loss on gap only, positional embeddings, scaling conversions).
3. Training loop stability (optimizer, scheduler, gradient clipping, checkpoint saving/restoring).
4. Inference pipeline (`infer_gap.py`) and ONNX export support.

## Behavior for Copilot-style suggestions
- Suggest small incremental edits and checks (unit tests, assertions) first.
- Avoid sweeping architecture re-design unless user requests explicit refactor.
- Add inline comments documenting non-obvious coordinate normalization and token type semantics.

## Local workflow hints
- Confirm command paths for execution from repository root (e.g., `python src_interpolation/trainers.py ...` or `python scripts/train_interpolation_from_trial.py ...`).
- Use minimal reproduction cases with `data/ct_dma/` or synthetic 2-3 sample data in tests.
- Keep evaluations in `results_interpolation/` and avoid hardcoded absolute paths.

## Anti-patterns to avoid
- Editing global/trAISformer behavior while working only on `src_interpolation` unless explicitly requested.
- Dropping without further explanation any assumptions about `max_seqlen`, `past/gap/future` configs.
- Unchecked use of `torch.no_grad()` or `model.train()` state in training/eval branches.

## Example prompts for this assistant
- "Fix `src_interpolation/trainers.py` so validation loss uses only gap tokens, and add a test case in `src_interpolation/test_port_context_gpu.py`."
- "Improve `src_interpolation/datasets.py` to support variable-gap sampling from existing trajectory chunks with no repeated endpoints."
- "Add a CLI option to `infer_gap.py` for selecting port-context model variants (none, port only, land+port)."

## Next improvement ideas
- Add `AGENTS.md` to define scoped reducers for data, model, infer, and evaluation tasks.
- Add explicit `readme` section in root for a one-command interpolation training recipe.
