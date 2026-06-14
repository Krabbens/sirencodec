---
branch: main
status: archived
base: initial Vocos experiment line
used_in_thesis: no
install: not applicable; archived source snapshot
test: python scripts/validate_branch_layout.py
---

# Branch manifest

## Role

Earliest Vocos and RVQ/FSQ repository snapshot, preserved as an archived experiment.

This branch is not a source for new development. Its purpose is to preserve an inspectable historical state.

## Variant boundary

The implementation, configurations and commit history on this branch define
the experiment. The normalization commit changes repository organization,
documentation and validation only; it does not replace the model with the
canonical `master` implementation.

## Validated commands

```bash
python scripts/validate_branch_layout.py
python -m compileall -q scripts
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
