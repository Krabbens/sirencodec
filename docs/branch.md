---
branch: archive/legacy-master
status: archived
base: former master at a5366a3
used_in_thesis: no
install: uv sync
test: python -m pytest -q
---

# Branch manifest

## Role

Former master branch preserving the historical Vocos, MLX and SEANet research line.

This branch is not a source for new development. Its purpose is to preserve an inspectable historical state.

## Variant boundary

The implementation, configurations and commit history on this branch define
the experiment. The normalization commit changes repository organization,
documentation and validation only; it does not replace the model with the
canonical `master` implementation.

## Validated commands

```bash
python scripts/validate_branch_layout.py
python -m compileall -q src tools tests scripts
python -m pytest -q
uv run train --help
```

Commands that train on a corpus or run checkpoint inference require explicit
paths to external datasets and model files. Those assets are intentionally not
stored in Git.

## Repository policy

Active package code belongs in `src/`, command-line utilities in `tools/`,
automation in `scripts/`, tests in `tests/` and experiment settings in
`configs/`. Historical root files are preserved under
`archive/legacy/` and are not part of the supported entrypoint surface.

## Known limitations

Portable CI does not reproduce full GPU training or evaluate private
checkpoints. A passing CI run proves that the tracked source is internally
consistent and that lightweight tests pass; it does not establish speech
quality for this variant.
