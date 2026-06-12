# Plan implementacji kodeka dwupasmowego 3,5 kHz

## Zakres

Implementacja powstaje obok istniejacego kodeka jednopasmowego. Nie zmienia
formatu dotychczasowych checkpointow ani zachowania `MLXCodec`.

## Etap 1: model

1. Dodac `FixedComplementaryFIR`:
   - 127 wspolczynnikow, okno Hann, granica 3500 Hz;
   - odbiciowe dopelnienie i zachowanie dlugosci;
   - `split(x) -> (low, high)` oraz `synthesis(low, high)`.
2. Dodac `TwoBandCodecConfig` z jawna konfiguracja galezi low i high.
3. Dodac `TwoBandCodec`:
   - low: `MLXCodec`, latent 512, RVQ `2 x K32`;
   - high: mniejszy `MLXCodec`, latent 256, RVQ `1 x K32`;
   - encoder warunkujacy odtworzony sygnal low;
   - projekcja latentu low i fuzja trzech latentow;
   - dekodowanie high po zdekodowaniu low;
   - osobne zwracanie rekonstrukcji, latentow, indeksow i statystyk VQ.
4. Dodac dekodowanie z indeksow. Dla strumienia indeksowego uzyc
   euklidesowego RVQ, poniewaz obecne cosine VQ przenosi dodatkowa norme
   residualu, ktorej sam indeks nie zawiera.

## Etap 2: strata i trening

1. Dodac funkcje kosztu dla full, low i high:
   - L1, log-STFT, spectral convergence, complex STFT i mel;
   - wagi galezi `1,0 / 0,5 / 1,0`;
   - normalizacja high przez `max(mean(abs(target_high)), 0,02)`;
   - osobne VQ i marginal entropy dla obu galezi.
2. Dodac osobny entrypoint `tools/train_two_band_mlx.py`:
   - dataset `data/train-clean-100`;
   - 330000 krokow, batch 8, segment 16000;
   - warmup 5000 do `3e-5`, potem cosine do `1,2e-7`;
   - log TSV, checkpointy NPZ i pelne safetensors;
   - zapis metadanych z `architecture=two_band`, `split_hz=3500`
     i nominalnym bitrate 937,5 bit/s.
3. Zapewnic `--steps`, `--batch`, `--data-dir`, `--output-dir`,
   `--checkpoint-every`, `--log-every`, `--fast` i `--resume`, aby mozna
   bylo wykonac test i pelny przebieg ta sama sciezka.

## Etap 3: testy

1. Suma pasm odtwarza wejscie z bledem ponizej `1e-5`.
2. Odpowiedz filtra tlumi przeciwne pasmo.
3. Forward zachowuje ksztalt i tworzy indeksy low `[B,T,2]` oraz high
   `[B,T,1]`.
4. Dekodowanie z indeksow zachowuje dlugosc sygnalu.
5. Nominalny bitrate wynosi 937,5 bit/s.
6. Jeden krok backward daje skonczona strate i gradienty.
7. Istniejace testy projektu pozostaja poprawne.

## Etap 4: uruchomienie

1. Wykonac testy jednostkowe.
2. Wykonac 3 kroki na danych syntetycznych.
3. Wykonac 100 krokow na `train-clean-100`.
4. Po poprawnym smoke uruchomic 330000 krokow od zera w tle i zapisac
   komende, PID, log oraz konfiguracje w katalogu przebiegu.
