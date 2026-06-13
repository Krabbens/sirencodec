# Czyszczenie historii wszystkich gałęzi

## Cel

Wszystkie gałęzie i tagi repozytorium mają zachować swój kod oraz przeznaczenie
eksperymentalne, ale nie mogą zawierać danych lokalnych, nagrań, checkpointów,
wyeksportowanych modeli ani regenerowalnych wyników. Historia Git zostanie
przepisana, a zdalne referencje zaktualizowane przez wymuszony push.

## Zakres usuwania

Z całej historii usuwane są:

- katalogi `data/`, `artifacts/`, `runs/`, `experiments/`, `c++outputs/`;
- katalogi wynikowe pasujące do `infer_*`;
- pliki zaczynające się od `sztyks`, `szweka`, `czat`, `adam`, `goralski`
  lub `góralski`, niezależnie od położenia;
- pliki audio i pomocnicze: `*.wav`, `*.mp3`, `*.flac`, `*.asd`;
- checkpointy i eksporty modeli: `*.pt`, `*.pth`, `*.ckpt`,
  `*.safetensors`, `*.npz`, `*.tflite`;
- archiwa danych: `*.tar.gz`.

Zachowane zostają źródła, konfiguracje, testy, dokumentacja, pliki LaTeX oraz
materiały graficzne pracy, w tym `*.pdf`, `*.png` i `*.eps`.

## Procedura

1. Zapisać bieżące lokalne i zdalne referencje oraz utworzyć pełny bare mirror
   poza katalogiem projektu.
2. Utworzyć drugi mirror roboczy i przepisać w nim wszystkie referencje za
   pomocą `git-filter-repo`.
3. Dodać na każdej końcówce gałęzi reguły `.gitignore`, które blokują ponowne
   dodanie usuniętych klas artefaktów.
4. Zweryfikować, że żadna osiągalna historia nie zawiera zabronionych ścieżek
   ani dużych binarnych obiektów z usuniętych klas.
5. Dla każdej gałęzi utworzyć osobny worktree lub checkout i uruchomić
   dostępne lekkie kontrole: kompilację plików Python, testy oraz konfigurację
   CMake, jeśli dana gałąź je zawiera.
6. Wymusić aktualizację wszystkich gałęzi i tagów na `origin`.
7. Ponownie pobrać zdalne referencje i porównać ich identyfikatory z lokalnym
   mirrorem.

## Bezpieczeństwo i rollback

Bieżący katalog roboczy nie jest używany do filtrowania. Nieśledzone pliki
lokalne pozostają na dysku. Kopia bezpieczeństwa przechowuje wszystkie
referencje sprzed operacji i nie jest automatycznie usuwana. W razie błędu
można przywrócić z niej gałęzie i tagi przez ponowny wymuszony push.

## Kryteria ukończenia

- wszystkie lokalne i zdalne gałęzie oraz tagi są przepisane;
- zabronione ścieżki nie występują w żadnym osiągalnym commicie;
- końcówki gałęzi mają zabezpieczenia `.gitignore`;
- wynik walidacji każdej gałęzi jest zapisany;
- zdalne identyfikatory referencji odpowiadają oczyszczonemu mirrorowi;
- lokalna kopia bezpieczeństwa i raport z operacji mają podane ścieżki.
