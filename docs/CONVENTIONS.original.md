Git LFS: install `git-lfs`, run `git lfs install` once per machine so `.gitattributes` patterns apply before push.

Commits tied to training / SYSTEM.md protocol:
- results.tsv: one commit per cycle row (or batch rows), message cites cycle + verdict + last step from experiments/.../log.tsv when relevant.
- experiments/<name>/log.tsv + resume_state.json: commit after milestones (paths in resume_state.json are repo-relative). Checkpoints under experiments/**/checkpoints/*.pt are gitignored—keep on disk or ship via LFS/release elsewhere; do not add them to the index.
- Code / scripts: separate commits from data; use conventional commits (feat/fix/docs).
- Large binaries: Git LFS (*.pt where tracked, spectrograms, *.npy). Root checkpoints/ is gitignored; experiment .pt files stay local unless you override .gitignore intentionally.
