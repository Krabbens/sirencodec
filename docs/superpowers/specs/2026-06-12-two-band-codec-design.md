# Dwupasmowy kodek mowy o przeplywnosci 937,5 bit/s

## Cel

Rozszerzyc kodek o jawny podzial sygnalu 16 kHz na pasmo niskie 0--3 kHz
i pasmo wysokie 3--8 kHz. Oba pasma maja byc kodowane przez osobne galezie
uczone wspolnie od losowej inicjalizacji. Dekoder pasma wysokiego ma korzystac
z informacji odtworzonej przez galaz niskopasmowa, a laczna nominalna
przeplywnosc obu strumieni ma pozostac bliska 900 bit/s.

## Rozwazane warianty

1. Dwie galezie polaczone stalym komplementarnym filtrem FIR. Rozwiazanie daje
   jawne cele pasmowe, przewidywalne opoznienie i prosty format strumienia.
2. Wspolny encoder z rozdzielonymi kwantyzatorami i dekoderami. Model bylby
   mniejszy, lecz reprezentacje pasm nie bylyby od siebie jednoznacznie
   odseparowane.
3. Podzial i synteza w dziedzinie STFT. Zapewnia dobra kontrole widmowa, ale
   komplikuje przetwarzanie strumieniowe przez buforowanie ramek,
   nakladanie okien i obsluge reprezentacji zespolonej.

Wybrano wariant pierwszy, poniewaz odpowiada docelowej architekturze
dwupasmowej i wymaga najmniejszej zmiany zalozen istniejacego kodeka
falowego.

## Analiza i synteza pasm

Podzial wykorzystuje staly, symetryczny filtr dolnoprzepustowy FIR o 127
wspolczynnikach i czestotliwosci granicznej 3 kHz. Wspolczynniki sa obliczane
raz metoda okienkowa i przechowywane jako stale modelu.

Dla sygnalu wejsciowego `x`:

```text
x_low  = lowpass(x)
x_high = x - x_low
```

Taki zapis tworzy pare komplementarna i ogranicza blad samego podzialu do
bledu numerycznego. Opoznienie grupowe wynosi 63 probki, czyli 3,94 ms przy
16 kHz. Po dekodowaniu obie galezie sa ponownie ograniczane do swoich pasm:

```text
y_low  = lowpass(decoded_low)
y_high = decoded_high - lowpass(decoded_high)
y      = y_low + y_high
```

Filtr ma obslugiwac tensory wsadowe i zachowywac dlugosc sygnalu przez
symetryczne dopelnienie. Ta sama implementacja jest uzywana w treningu,
inferencji i testach.

## Galaz niskopasmowa

Galaz niskopasmowa pozostaje glowna czescia modelu. Korzysta z architektury
encodera i dekodera zgodnej z obecnym kodekiem oraz z latentem o wymiarze 512.
Kwantyzator sklada sie z dwoch etapow RVQ po 32 symbole.

Przy redukcji czasowej 256x i czestotliwosci probkowania 16 kHz powstaje
62,5 ramki na sekunde. Dwa indeksy po 5 bitow daja nominalnie:

```text
62,5 * 2 * log2(32) = 625 bit/s
```

Galaz zwraca odtworzony sygnal niskopasmowy, skwantowany latent i indeksy RVQ.
Latent oraz odtworzony sygnal stanowia kontekst dla dekodera wysokiego pasma.

## Galaz wysokopasmowa

Galaz wysokopasmowa ma mniejszy encoder, latent o wymiarze 256 i jeden etap
RVQ z 32 symbolami. Jej nominalna przeplywnosc wynosi:

```text
62,5 * log2(32) = 312,5 bit/s
```

Encoder przetwarza wylacznie `x_high`. Dekoder otrzymuje trzy skladniki:

- skwantowany latent wysokiego pasma;
- projekcje skwantowanego latentu niskiego pasma z 512 do 256 kanalow;
- cechy odtworzonego sygnalu niskopasmowego uzyskane przez maly encoder
  warunkujacy o redukcji czasowej 256x.

Trzy tensory maja wspolna siatke czasowa i sa laczone przez konkatenacje oraz
projekcje liniowa do 256 kanalow. Dekoder wysokiego pasma nie korzysta
z oryginalnego sygnalu ani z nieskwantowanego latentu dolnej galezi. Dzieki
temu ten sam przebieg jest mozliwy po stronie odbiorczej: najpierw dekodowane
jest pasmo niskie, a nastepnie pasmo wysokie.

Mniejsza galaz wykorzystuje kanaly encodera `(12, 16, 24, 32, 48, 64, 96,
128)`. Struktura redukcji czasowej pozostaje zgodna z glowna galezia, aby nie
bylo potrzebne interpolowanie latentow.

## Budzet bitowy i strumien

Laczna nominalna przeplywnosc wynosi 937,5 bit/s. Jest to wartosc powyzej
900 bit/s o 4,2%, ale zachowuje potegowe rozmiary slownikow i obecny format
kwantyzacji.

Jedna ramka strumienia zawiera kolejno:

1. dwa indeksy niskiego pasma po 5 bitow;
2. jeden indeks wysokiego pasma o dlugosci 5 bitow.

Metadane naglowka okreslaja wersje architektury, czestotliwosc probkowania,
liczbe ramek i dlugosc oryginalnego sygnalu. Dekodowanie jest deterministyczne
i zawsze odbywa sie w kolejnosci low, high, synthesis.

## Funkcja kosztu

Model jest uczony wspolnie. Strata calkowita zawiera:

- rekonstrukcje pasma niskiego wzgledem `x_low`;
- rekonstrukcje pasma wysokiego wzgledem `x_high`;
- rekonstrukcje pelnopasmowa sumy `y_low + y_high` wzgledem `x`;
- skladniki STFT i mel obliczane osobno dla pasm oraz dla sumy;
- straty zobowiazania i ksiegi kodowej obu kwantyzatorow;
- regularyzacje entropii marginalnej liczona osobno dla kazdego etapu RVQ.

Domyslna kombinacja strat rekonstrukcyjnych ma postac:

```text
L_recon = 1,0 * L_full + 0,5 * L_low + 1,0 * L_high_normalized
```

Kazdy skladnik uzywa obecnej receptury bez GAN: `lambda_time=1,0`,
`lambda_stft=0,5`, `lambda_sc=1,0`, `lambda_complex_stft=0,1`
i `lambda_mel_l1=0,12`. Efektywna waga STFT podlega tej samej rozgrzewce
we wszystkich trzech skladnikach. Dla kazdej galezi przyjmowane sa ponadto
`lambda_vq=5,0` i `lambda_marginal=0,35`, a straty semantyczne pozostaja
wylaczone w pierwszym przebiegu.

Strata wysokiego pasma jest dzielona przez srednia wartosc bezwzgledna celu
z dolnym ograniczeniem `0,02`. Zapobiega to zanikowi jej gradientu w cichych
segmentach, bez nieograniczonego wzmacniania szumu. Wszystkie wspolczynniki
sa polami konfiguracji i sa wypisywane w logu startowym.

## Trening i logowanie

Trening rozpoczyna sie od losowej inicjalizacji obu galezi. Uzywa zbioru
LibriSpeech `train-clean-100`, segmentow 16000 probek, batcha 8 i seeda 42.
Harmonogram obejmuje liniowa rozgrzewke do `3e-5` przez 5000 krokow, a potem
cosine decay do `1,2e-7` w kroku 330000.

Jeden optimizer aktualizuje obie galezie. Log zawiera co najmniej:

- `loss_total`, `loss_full`, `loss_low` i `loss_high`;
- skladniki widmowe kazdego pasma;
- straty VQ, entropie i wykorzystanie slownikow oddzielnie dla low i high;
- aktualny learning rate, czas kroku i zuzycie pamieci.

Checkpoint przechowuje obie galezie, filtr, encoder warunkujacy, optimizer
i krok treningu. Konfiguracja checkpointu jednoznacznie oznacza architekture
jako dwupasmowa, aby narzedzia nie probowaly wczytac jej jako zwyklego
`MLXCodec`.

## Obsluga bledow i zgodnosc

Nowy model ma osobny typ konfiguracji i nie zmienia znaczenia istniejacych
checkpointow. Narzedzie inferencyjne rozpoznaje wariant modelu z metadanych.
Brak pola architektury oznacza dotychczasowy kodek jednopasmowy.

Walidowane sa niezgodne liczby codebookow, rozmiary slownikow, dlugosci
latentow i brak indeksow jednej z galezi. Komunikaty bledow wskazuja pole
konfiguracji lub element pakietu, ktory nie zgadza sie z modelem.

## Testy i kryteria akceptacji

Implementacja jest gotowa do pelnego treningu po spelnieniu nastepujacych
warunkow:

1. filtr rozdziela sygnal, a suma pasm odtwarza wejscie z bledem ponizej
   `1e-5`;
2. oba latenty maja identyczna liczbe ramek dla dlugosci obslugiwanych przez
   obecny kodek;
3. model wykonuje forward i backward bez `NaN` oraz zwraca wszystkie
   skladniki strat;
4. indeksy maja ksztalt odpowiednio `[B, T, 2]` i `[B, T, 1]`;
5. obliczona nominalna przeplywnosc wynosi 937,5 bit/s;
6. dekodowanie z samych indeksow daje sygnal o pierwotnej dlugosci;
7. co najmniej 100 krokow testowego treningu konczy sie bez wyjatku, utraty
   gradientu jednej z galezi lub stalego wyboru jednego symbolu;
8. istniejace testy kodeka jednopasmowego nadal przechodza.

Po spelnieniu testow uruchamiany jest nowy przebieg 330000 krokow w osobnym
katalogu `runs/two_band_937bps_fresh_cosine_<timestamp>/`.
