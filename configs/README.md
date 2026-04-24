# Config Templates

Ready-made presets for `uv run train --config ...`.

## Usage

```powershell
uv run train --config configs/ab_50_50.json
```

CLI overrides the JSON file:

```powershell
uv run train --config configs/abd.json --dataset train-clean-100 --epochs 5
```

Each template can include run-length and logging defaults such as:

- `epochs`
- `logs_per_epoch`
- `viz_per_epoch`

## Presets

- `ab_50_50.json`
  - only phases `A` and `B`
  - `A=50%`, `B=50%`
  - no `D`, no `C`
  - default `dataset=train-clean-360`, `epochs=140`, `logs_per_epoch=5`

- `abd.json`
  - phases `A`, `B`, `D`
  - default split `A=15%`, `B=35%`, `D=50%`
  - no `C`
  - default `dataset=train-clean-100`, `epochs=5`, `logs_per_epoch=5`

- `abcd.json`
  - full curriculum `A`, `B`, `C`, `D`
  - default split `A=15%`, `B=35%`, `C=25%`, `D=25%`
  - `C` becomes active only when `lambda_adv > 0`
  - default `dataset=train-clean-100`, `epochs=5`, `logs_per_epoch=5`
