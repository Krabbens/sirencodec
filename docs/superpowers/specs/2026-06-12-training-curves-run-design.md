# Jednolity przebieg treningowy do wykresow uczenia

## Cel

Uruchomic model docelowy od losowej inicjalizacji i zapisac kompletny,
jednorodny log od kroku 0 do 329999. Przebieg ma dostarczyc danych do wykresow
uczenia w rozdziale 4 pracy, bez laczenia etapow o roznych konfiguracjach.

## Konfiguracja

- dane: LibriSpeech `train-clean-100`, 28539 plikow;
- liczba krokow: 330000;
- efektywny batch: 8;
- segment: 16000 probek;
- architektura: 3-etapowy RVQ `32x32x32`, latent 512, redukcja czasowa 256x,
  jeden blok self-attention przed RVQ, stosy temporalne 2/2, refiner dekodera
  1x oraz post-refiner 2x24 z filtrem gornoprzepustowym;
- harmonogram: linearna rozgrzewka od 0 do `3e-5` przez 5000 krokow, nastepnie
  cosine decay do `1.2e-7`;
- funkcja kosztu: staly zestaw skladnikow uzywany w docelowym wariancie,
  z wylaczona strata semantyczna, aby nie zmieniac kosztu obliczeniowego ani
  czestotliwosci aktualizacji w trakcie przebiegu;
- logowanie: co 25 krokow;
- walidacja, checkpoint i spektrogram: co 2500 krokow;
- seed: 42.

## Artefakty

Kazdy przebieg otrzymuje osobny katalog
`runs/thesis_curves_fresh_cosine_<timestamp>/`, zawierajacy:

- `train.log` z pelnym wyjsciem procesu;
- `train_command.txt` z dokladna komenda;
- `experiment_note.txt` z celem przebiegu;
- `log_mlx.tsv` z SI-SDR i PESQ;
- `results.tsv`;
- katalogi `checkpoints/` i `spectrograms/`.

## Kryteria poprawnego startu

Po uruchomieniu nalezy potwierdzic, ze log zawiera opis modelu, zbior
`train-clean-100`, krok 0 z LR bliskim zeru oraz kolejne skonczone kroki bez
wartosci `NaN`, `Inf` i wyjatkow. Proces ma dzialac w tle i pozostac niezalezny
od sesji terminala.

## Wykresy

Po zakonczeniu przebiegu do rozdzialu 4 zostana przygotowane:

1. wygladzona strata calkowita oraz glowne skladniki rekonstrukcyjne;
2. wykorzystanie RVQ, entropia marginalna i podobienstwo cosinusowe;
3. przebieg learning rate;
4. checkpointowe SI-SDR i PESQ.

Surowe wartosci beda przedstawione jasna linia, a trend srednia kroczaca.
Zmienne o roznych skalach nie beda laczone na jednej osi.
