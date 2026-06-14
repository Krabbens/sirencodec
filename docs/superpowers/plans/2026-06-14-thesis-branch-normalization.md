# Plan uporządkowania gałęzi pracy magisterskiej

## Cel

Uporządkować wszystkie zachowane gałęzie bez zmiany znaczenia wariantów
eksperymentalnych. `7m-best` jest wzorcem, `master` staje się gałęzią
kanoniczną, a dotychczasowy `master` zostaje zachowany jako
`archive/legacy-master`.

## Etap 1: zabezpieczenie i inwentaryzacja

1. Zapisać hashe wszystkich lokalnych i zdalnych gałęzi.
2. Utworzyć bare backup przed zmianami.
3. Dla każdej gałęzi zapisać:
   - strukturę katalogu głównego;
   - liczbę źródeł, testów, narzędzi i konfiguracji;
   - dostępne komendy instalacji i testów;
   - pliki wymagające klasyfikacji lub archiwizacji.
4. Utworzyć raport początkowy w backupie, poza historią projektu.

## Etap 2: warstwa wzorcowa na `7m-best`

1. Skrócić README do stabilnego opisu projektu i odsyłacza do manifestu
   gałęzi.
2. Dodać `docs/branch.md` ze statusem `canonical-reference`.
3. Dodać skrypt `scripts/validate_branch_layout.py`, który sprawdza:
   - obecność wymaganych plików;
   - poprawność statusu manifestu;
   - zakazane artefakty;
   - dozwolone elementy katalogu głównego.
4. Uzupełnić CI o walidację struktury oraz osobny build C++.
5. Usunąć z README opisy funkcji nieobecnych w śledzonej zawartości.
6. Uruchomić kompilację Python, instalację, pytest, walidator i Release build
   C++.

## Etap 3: role kanoniczne

1. Utworzyć `archive/legacy-master` na dotychczasowym końcu `master`.
2. Ustawić `master` na zweryfikowany koniec `7m-best`.
3. Zachować `7m-best` jako nazwany punkt odniesienia.
4. Nadać `7m-best-rozdzial4` status `thesis` i zachować dodatkowe materiały
   rozdziału czwartego.

## Etap 4: normalizacja rodzin gałęzi

Zmiany będą wykonywane osobnymi commitami na każdej gałęzi.

### Rodzina współczesna

Gałęzie:

- `codex/3khz-600bps-novasr8k`;
- `codex/8khz-lowpass3k-600bps`;
- `codex/cuda-1to1-baseline`;
- `codex/cuda-minimal-karpathy`;
- `models/13m`;
- `models/13m-2`;
- `models/13m-2d`;
- `models/20m`;
- `models/20m-no-phases-scaled`;
- `models/6m`;
- `q/turboquant`;
- `refactor/production-cleanup`.

Działania:

1. Przenieść wspólny README, walidator i CI.
2. Wygenerować `docs/branch.md` z różnic na podstawie historii, konfiguracji
   i kodu danej gałęzi.
3. Przenieść luźne wykresy do `archive/legacy/diagnostics/`.
4. Zachować konfiguracje i implementacje charakterystyczne dla wariantu.

### Rodzina historycznego CUDA/Vocos

Gałęzie:

- `cuda`;
- `cuda-optimizing`;
- dotychczasowy `master`, zachowany jako `archive/legacy-master`;
- `main`.

Działania:

1. Sprawdzić odwołania do wszystkich plików w katalogu głównym.
2. Aktywne entrypointy przenieść do `tools/` lub `scripts/` i poprawić
   odwołania.
3. Zastąpione skrypty, notatki, obrazy i wyniki przenieść do odpowiednich
   podkatalogów `archive/legacy/`.
4. Uzupełnić manifest o informację, że jest to historyczna rodzina
   Vocos/RVQ/FSQ.
5. Zachować sprawdzalne komendy historycznego wariantu.

### Gałąź `Q_unsmooth`

1. Zachować osobną implementację PyTorch Vocos/SEANet.
2. Przenieść rootowy `run.py` do `tools/`, jeżeli nie jest entrypointem
   pakietu, albo udokumentować go jako jedyny aktywny launcher.
3. Dodać minimalne testy struktury i manifest wariantu.

## Etap 5: walidacja zbiorcza

Dla każdej gałęzi:

1. Uruchomić walidator układu.
2. Skompilować wszystkie pliki Python.
3. Zainstalować pakiet editable bez ciężkich zależności, jeśli manifest na to
   pozwala.
4. Uruchomić testy niewymagające datasetu, checkpointu ani GPU.
5. Zbudować C++ w Release, jeśli gałąź zawiera `cpp/sirencodec_infer`.
6. Sprawdzić, czy README nie odwołuje się do nieistniejących ścieżek.
7. Zapisać wynik i świadome wyjątki w raporcie TSV.

Push jest dozwolony wyłącznie wtedy, gdy:

- każda gałąź przechodzi walidator;
- nie ma nieudokumentowanych błędów kompilacji lub testów;
- `master` i `7m-best` są zgodne funkcjonalnie;
- `archive/legacy-master` wskazuje pierwotny stan `master`;
- bieżący worktree użytkownika nie został zmieniony.

## Etap 6: publikacja i kontrola

1. Wypchnąć nową gałąź archiwalną.
2. Wypchnąć uporządkowane gałęzie.
3. Zaktualizować `master` do stanu `7m-best`.
4. Pobrać zdalne referencje i porównać wszystkie hashe z raportem lokalnym.
5. Zachować backup oraz końcowy raport walidacji.

## Cofnięcie zmian

Backup bare zachowuje początkowe referencje. Przywrócenie polega na
wymuszonym wypchnięciu wybranego refa z backupu. Nieśledzone dane i wyniki
użytkownika pozostają poza worktree używanymi do normalizacji.
