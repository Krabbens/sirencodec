# Uporządkowanie gałęzi na potrzeby pracy magisterskiej

## Cel

Repozytorium ma zachować wszystkie warianty eksperymentalne, ale każda gałąź
powinna jednoznacznie przedstawiać odtwarzalny etap badań. Kod aktywny,
materiały pracy, konfiguracje eksperymentów i pliki historyczne nie mogą być
przemieszane w katalogu głównym ani opisane sprzecznie z rzeczywistą
zawartością gałęzi.

Gałąź `7m-best` jest wzorcem struktury i kanonicznym wariantem modelu.
Po zakończeniu porządkowania `master` ma wskazywać ten sam stan funkcjonalny.

## Role gałęzi

- `master` - kanoniczna, uruchamialna wersja projektu zgodna z modelem
  opisywanym w pracy;
- `7m-best` - zachowany punkt odniesienia dla najlepszego wariantu modelu,
  funkcjonalnie zgodny z `master`;
- `7m-best-rozdzial4` - gałąź materiałów i analiz rozdziału czwartego,
  rozwijana względem kanonicznego kodu;
- pozostałe gałęzie - odtwarzalne warianty eksperymentalne, których kod,
  konfiguracja i wyniki opisowe zachowują znaczenie badawcze;
- `archive/legacy-master` - zachowany stan dotychczasowego `master` sprzed
  zastąpienia go zawartością `7m-best`.

Gałęzie eksperymentalne nie będą rebase'owane na nowy `master`. Porządkowanie
ma zmienić ich organizację i dokumentację, ale nie może zacierać różnic
modelu ani zmieniać znaczenia przeprowadzonych eksperymentów.

## Docelowa struktura

Każda aktywna gałąź używa, o ile dane elementy są dla niej potrzebne,
następującego układu:

```text
.
|-- README.md
|-- pyproject.toml
|-- uv.lock
|-- src/
|-- tests/
|-- tools/
|-- scripts/
|-- configs/
|-- docs/
|   `-- branch.md
|-- cpp/
|-- overleaf/
`-- archive/
    `-- legacy/
```

Katalog główny może zawierać wyłącznie pliki wejściowe i konfiguracyjne
projektu, takie jak README, manifest pakietu, lockfile, konfiguracja CI,
Dockerfile, licencja i reguły Git. Aktywny kod biblioteczny należy do `src/`,
testy do `tests/`, narzędzia użytkowe do `tools/`, automatyzacja do `scripts/`,
a konfiguracje eksperymentów do `configs/`.

Katalogi `cpp/` i `overleaf/` występują tylko na gałęziach, które rzeczywiście
ich potrzebują. Materiały LaTeX pracy pozostają na `7m-best`,
`7m-best-rozdzial4` i kanonicznym `master`; nie będą kopiowane do wszystkich
wariantów eksperymentalnych.

## Manifest gałęzi

Każda gałąź otrzyma plik `docs/branch.md` zawierający:

- status: `canonical`, `thesis`, `experimental` albo `archived`;
- rolę gałęzi oraz jej relację do `master`;
- opis rzeczywistych zmian modelu lub procesu uczenia;
- konfigurację modelu i ograniczenia przepływności istotne dla wariantu;
- wymagania sprzętowe i zależności opcjonalne;
- sprawdzone komendy instalacji, treningu, inferencji i testów;
- znane ograniczenia oraz informację, czy wariant został wykorzystany w pracy.

Wspólny `README.md` będzie krótkim punktem wejścia do projektu. Nie może
reklamować funkcji, których dana gałąź nie zawiera. Szczegóły wariantu będą
umieszczane w `docs/branch.md`, dzięki czemu README nie będzie rozchodzić się
między gałęziami.

## Zasady archiwizacji

Historyczne pliki pozostaną w repozytorium, lecz zostaną przeniesione do
`archive/legacy/` i pogrupowane według przeznaczenia, na przykład:

```text
archive/legacy/
|-- entrypoints/
|-- training/
|-- research-notes/
|-- diagnostics/
`-- tooling/
```

Przeniesieniu podlegają w szczególności stare entrypointy z katalogu głównego,
robocze roadmapy, instrukcje agentów, obrazy diagnostyczne, jednorazowe
benchmarki i narzędzia zastąpione przez kod z `src/`, `tools/` lub `scripts/`.

Plik nie może zostać zarchiwizowany wyłącznie na podstawie nazwy. Przed
przeniesieniem należy sprawdzić importy, wywołania w skryptach, dokumentacji
i CI. Jeżeli plik jest nadal jedyną implementacją potrzebnej funkcji,
najpierw należy przenieść go do właściwego aktywnego katalogu i poprawić
odwołania. Każdy katalog `archive/legacy/` otrzyma README opisujące pochodzenie
plików i powód archiwizacji.

## Wspólne zabezpieczenia

Każda aktywna gałąź otrzyma:

- jednolite reguły `.gitignore` blokujące datasety, checkpointy, nagrania,
  eksporty modeli, logi i wyniki generowane;
- CI wykonujące kompilację plików Pythona, instalację pakietu i testy możliwe
  do uruchomienia bez prywatnych danych oraz checkpointów;
- kontrolę obecności i podstawowej kompletności `docs/branch.md`;
- kontrolę zakazanych artefaktów i luźnych plików w katalogu głównym;
- Release build C++ na gałęziach zawierających runtime C++.

Test wymagający zewnętrznego datasetu, checkpointu lub środowiska GPU musi być
jawnie oznaczony jako integracyjny i udokumentowany, zamiast udawać test
jednostkowy wykonywany przez domyślne CI.

## Procedura wdrożenia

1. Utworzyć pełny backup referencji i zapisać początkowy raport 19 gałęzi.
2. Utworzyć `archive/legacy-master` ze stanu dotychczasowego `master`.
3. Ustawić `master` na aktualny stan `7m-best`.
4. Na podstawie `7m-best` przygotować wspólny README, manifest gałęzi,
   walidator struktury, CI i zasady archiwizacji.
5. Przenieść wspólną warstwę porządkową na każdą gałąź osobnym commitem,
   rozwiązując różnice bez zmiany semantyki eksperymentu.
6. Na każdej gałęzi sklasyfikować luźne pliki jako aktywne, dokumentacyjne
   albo historyczne i odpowiednio je przenieść.
7. Uruchomić walidację wszystkich gałęzi i zapisać raport zbiorczy.
8. Wypchnąć zmiany dopiero po przejściu kontroli i porównać lokalne oraz
   zdalne identyfikatory referencji.

Prace będą prowadzone w osobnych worktree. Bieżący katalog roboczy oraz jego
nieśledzone datasety, logi, checkpointy i materiały pracy nie będą usuwane ani
przenoszone.

## Kryteria ukończenia

- istnieje 19 uporządkowanych gałęzi oraz `archive/legacy-master`;
- `master` i `7m-best` mają ten sam kanoniczny kod oraz strukturę;
- każda gałąź ma poprawny `docs/branch.md` i README zgodny z zawartością;
- katalog główny nie zawiera luźnych implementacji, roboczych notatek ani
  wyników diagnostycznych;
- historyczne pliki są zachowane w opisanym `archive/legacy/`;
- warianty eksperymentalne zachowują swoje różnice i nie zostały
  znormalizowane semantycznie;
- domyślne testy nie wymagają lokalnych danych ani checkpointów;
- gałęzie z C++ przechodzą Release build;
- raport walidacji podaje wynik każdej gałęzi oraz świadome wyjątki;
- wszystkie zdalne referencje odpowiadają zatwierdzonym lokalnym końcówkom.

## Ryzyka i ograniczenia

Największym ryzykiem jest uznanie historycznego pliku za nieaktywny mimo
ukrytego odwołania albo przypadkowe ujednolicenie kodu, które zmieni znaczenie
wariantu badawczego. Z tego powodu migracja będzie wykonywana osobno dla
każdej gałęzi, z kontrolą odwołań przed przeniesieniem i testami po zmianie.

Porządkowanie nie obejmuje poprawiania wyników eksperymentów, przebudowy
architektury modelu ani dopisywania brakujących badań. Celem jest czytelność,
odtwarzalność i zgodność repozytorium z materiałem pracy magisterskiej.
