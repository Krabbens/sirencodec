# autoresearch-mlx

**Monorepo sirencodec:** ten katalog to **podprojekt** z własnym `pyproject.toml` i `uv.lock`. Cała procedura eksperymentu jest opisana w [`program.md`](program.md) jako protokół **tylko dla tego monorepo**, nie jako ogólna instrukcja pod zewnętrzny chat.

- Pracuj wyłącznie w `autoresearch-mlx/`.
- Commity: `git add autoresearch-mlx/...` — **nigdy** ślepe `git add -A` z korzenia sirencodec.

MLX / Apple Silicon port idei [autoresearch Karpathy’ego](https://github.com/karpathy/autoresearch): jeden edytowalny `train.py`, metryka **`val_bpb`** (BPB na walidacji — **niżej = lepiej**), budżet 5 min treningu, keep/revert przez git. Bez PyTorch i CUDA. Zakazy (m.in. brak ściągania datasetów z internetu — tylko `prepare.py`): [`program.md`](program.md).

## Quick start

Wymagania: Mac Apple Silicon, Python 3.10+, [uv](https://docs.astral.sh/uv/) (instalacja z dokumentacji uv / Homebrew — bez wiązania tego z pobieraniem danych do eksperymentu).

```bash
cd autoresearch-mlx
uv sync
# Opcjonalnie: lokalne audio pod monorepo — ``../data/**/*.wav|flac|ogg`` → manifest w ~/.cache/autoresearch/
uv run python prepare.py
uv run python train.py   # domyślnie ~300 s budżetu treningu (``AUTORESEARCH_TRAIN_SECONDS``)
```

### Jak to jest zbudowane (sirencodec)

- **`prepare.py`** — nieruszany w pętli eksperymentów: manifest ścieżek audio, podział train/val, **`evaluate_bpb`**, który liczy średnią wartość tego samego skalara co `make_train_fn` w [`tools/train_mlx.py`](../tools/train_mlx.py) (łączny loss kodeka; **niżej = lepiej**). Nazwa **`val_bpb`** jest zgodna z protokołem / `results.tsv`, choć pochodzi z pierwotnego autoresearch LM.
- **`train.py`** — jedyny plik edytowany w eksperymentach: pętla z budżetem czasu, ten sam model / loss co `train_mlx`, stopka `val_bpb`, `training_seconds`, `peak_vram_mb` jak w `program.md`.
- **Bez pobierania danych z sieci** — tylko pliki pod `sirencodec/data/` (lub wyłącznie syntetyczne batche, gdy brak audio).

Do autonomicznej pętli eksperymentów w Cursorze wskaż jako źródło prawdy plik **`program.md`** (ścieżki i ograniczenia są pod sirencodec).

## Co jest ważne

| Plik | Rola |
|------|------|
| `prepare.py` | Dane, tokenizer, dataloader, `evaluate_bpb` — **nie zmieniać**. |
| `train.py` | Model, optimizer, pętla — **jedyne miejsce zmian** w autoresearch. |
| `program.md` | Protokół pętli, git, `results.tsv`, timeouty. |
| `results.tsv` | Historia eksperymentów (TSV z tabulatorem). |

Pętla: edycja `train.py` → `uv run train.py` → odczyt `val_bpb` → keep albo revert wg `program.md`.

## Public baseline results

Tabela poniżej to **zapis historyczny** z obcego / wcześniejszego przebiegu — **nie** zastępuje baseline’u na Twoim sprzęcie (ten ustala się lokalnie, patrz `program.md`).

| Commit | val_bpb | Status | Description |
|---|---:|---|---|
| `383abb4` | 2.667000 | keep | baseline (AdamW, default config) |
| `909dd59` | 2.588904 | keep | halve total batch size to `2^16` |
| `4161af3` | 2.533728 | keep | increase matrix LR to `0.04` |
| `5efc7aa` | 1.807902 | keep | reduce depth from `8` to `4` |

Przy stałym czasie treningu na Apple Silicon mniejsze, szybsze modele często mieszczą więcej kroków optymalizacji niż duży model — stąd różne „wygrane” vs upstream na GPU.

## Dłuższe przebiegi (referencja)

| Machine | Current best | Starting point | Repeated wins |
|---|---:|---:|---|
| M4 Max #1 | 1.294526 | 1.596971 | AdamW-only, low matrix LR, 3x MLP, no logit cap, moderate weight decay |
| M4 Max #2 | 1.330509 | 1.807902 | leaner batch, long anneal, SiLU, lower regularization, no logit cap |
| Mac Mini (long run) | 1.353329 | 1.922472 | Muon, sharper attention, smaller MLP, lower scalar LR |

To ilustruje zależność od sprzętu — w sirencodec liczy się własny baseline na własnym Macu.

## Różnice względem upstream

- MLX zamiast PyTorch/CUDA; pamięć unifikowana.
- Domyślna ścieżka publiczna opiera się na prostszym stacku (np. AdamW); inne warianty były w osobnych przebiegach.
- Mniejszy budżet tokenów w ewaluacji dla szybszej iteracji przy tym samym interfejsie `evaluate_bpb`.
- ~6–7 min na eksperyment (trening + kompilacja + eval).
- MFU — placeholder (brak odpowiednika referencji FLOPs z H100).

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch i nanochat
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx)
- [awni/picochat](https://github.com/awni/picochat)
- [Apple MLX](https://github.com/ml-explore/mlx)

## License

MIT. See [LICENSE](LICENSE).
