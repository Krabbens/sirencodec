# autoresearch-mlx — protokół dla repozytorium **sirencodec**

To **lokalny** opis pracy w podkatalogu `autoresearch-mlx/` w monorepo **sirencodec**. Nie jest to ogólny prompt pod zewnętrzny chat ani szablon „powiedz GPT”. Wykonawcą jest osoba albo **asystent kodu w tym workspace** (np. Cursor), ściśle według poniższych reguł i ścieżek tego repo.

Port na Apple Silicon (MLX) idei autoresearch Karpathy’ego: **jeden plik do edycji** (`train.py`), stały budżet czasu treningu, metryka z `prepare.py`. Bez PyTorch i CUDA.

**Monorepo:** Całe sirencodec może zawierać inne projekty. Pracuj wyłącznie w `autoresearch-mlx/`. Stage’uj tylko `autoresearch-mlx/...`. **Nigdy** `git add -A` z korzenia sirencodec.

**Bez pobierania danych z sieci:** Żadnych `curl`/`wget` ani datasetów z URL pod ten podprojekt. Dane i tokenizer: wyłącznie `uv run prepare.py` → `~/.cache/autoresearch/`. Bez nowych zależności poza `pyproject.toml` / `uv.lock`.

## Metryka: `val_bpb`

**`val_bpb`** = **BPB (bits per byte)** na walidacji (`evaluate_bpb` w `prepare.py`). **Im niżej, tym lepiej.** Nazwa pola musi zostać jak w `train.py` i nagłówku `results.tsv`.

## Setup (ustalić z maintainerem sirencodec)

1. **Tag przebiegu** (np. data: `apr20`). Gałąź `autoresearch/<tag>` musi być nowa.
2. **Gałąź:** `git checkout -b autoresearch/<tag>` od głównej (**u nas: `master`**, chyba że maintainer wskaże inaczej).
3. **Przeczytać:** `autoresearch-mlx/README.md`, `prepare.py` (nie edytować), `train.py` (tu są zmiany).
4. **Dane:** istniejące shardy + tokenizer w `~/.cache/autoresearch/`; inaczej maintainer uruchamia `uv run prepare.py` w `autoresearch-mlx/`.
5. **`results.tsv`:** nagłówek + baseline z **jednego** uruchomienia `uv run train.py` na **tym** Macu — nie przenosić liczb z README ani z innych maszyn.
6. **Start:** krótkie potwierdzenie od człowieka, potem pętla bez pytań „czy kontynuować”.

## Eksperymentacja

- Budżet: **5 min** treningu (wall clock treningu; bez startupu/kompilacji). Komenda: w katalogu `autoresearch-mlx/`: `uv run train.py`.

**Wolno:** zmieniać tylko `train.py`.

**Nie wolno:** dotykać `prepare.py`, dokładać pakietów, zmieniać `evaluate_bpb` ani innych stałych ewaluacji w `prepare.py`.

**Cel:** minimalny **`val_bpb`**, stabilny run, mieści się w czasie.

**Pamięć (MLX):** umiarkowany wzrost RAM OK przy realnej poprawie BPB.

**Prostota:** przy podobnym `val_bpb` preferuj prostszy kod; koszt złożoności liczy się przy marginalnych zyskach.

## Podsumowanie po `train.py`

Kształt wydruku (liczby zależą od sprzętu):

```
---
val_bpb:          X.XXXXXX
training_seconds: ...
total_seconds:    ...
peak_vram_mb:     ...
...
```

Porównania tylko do **własnego baseline’u** na **tym samym** komputerze.

```bash
grep "^val_bpb:" run.log
```

## `results.tsv`

Separator: **TAB**. Kolumny:

```
commit	val_bpb	memory_gb	status	description
```

- `status`: `keep` | `discard` | `crash`
- przy crashu: `val_bpb=0.000000`, `memory_gb=0.0`
- `memory_gb`: z `peak_vram_mb / 1024`, jedno miejsce po przecinku

Przykład **formatu** (nie kopiować hashy z innych przebiegów):

```
commit	val_bpb	memory_gb	status	description
abcdef0	0.000000	0.0	crash	przykład
1234567	9.999999	12.3	keep	krótki opis zmiany w train.py
```

## Pętla

Gałąź np. `autoresearch/apr20`.

1. Stan gita.
2. Edycja `train.py`.
3. `git add autoresearch-mlx/train.py && git commit -m "experiment: <opis>"`
4. `uv run train.py > run.log 2>&1` (bez `tee`; log nie do chatu).
5. `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. Brak linii → crash: `tail -n 50 run.log`; po kilku próbach — `crash` w TSV, następny eksperyment.
7. Wpis do `results.tsv`.
8. Lepszy (niższy) `val_bpb`: `git add autoresearch-mlx/results.tsv && git commit --amend --no-edit`.
9. Tak samo lub gorzej: zapisać commit jako `discard`, `git reset --hard <ostatni keep>`.

**Timeout:** zwykle ~7 min; **> 15 min** — przerwać, uznać za porażkę, revert.

**Po starcie pętli** nie przerywać prośbą o zgodę — działać aż maintainer nie zatrzyma. Brak pomysłów: ponownie pliki w scope, kombinacje wcześniejszych commitów, śmielsze zmiany w `train.py`.
