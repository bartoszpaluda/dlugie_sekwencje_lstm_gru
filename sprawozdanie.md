# Szablon sprawozdania laboratoryjnego

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy  
**Zajęcia nr / temat:** Zajęcia 4 — Obsługa długich sekwencji: LSTM, GRU i splot 1D  
**Autor / grupa:** Bartosz Paluda
**Data:** 05.11.2026  

---

## 1. Cel ćwiczenia

Celem ćwiczenia jest porównanie czterech architektur sieci neuronowych (SimpleRNN, LSTM, GRU oraz rozszerzonej sieci splotowej 1D – Dilated Conv1D) w zadaniu modelowania zależności długozasięgowych. Ewaluację przeprowadzono na benchmarku *adding problem* ze znacznikiem pojawiającym się z opóźnieniem wynoszącym 8 kroków. Zbadano wpływ długości sekwencji ($L=50$ vs $L=100$), wielkości stanu ukrytego ($H=32$ vs $H=64$) oraz pola recepcyjnego splotu (dylatacje $[1, 2, 4]$ vs $[1, 1, 1]$) na stabilność optymalizacji, błąd testowy MSE oraz koszt obliczeniowy w odniesieniu do analitycznej predykcji stałej.

---

## 2. Metoda i implementacja

### Specyfikacja danych i weryfikacja generatora
- **Dane wejściowe:** Każdy przykład to tensor o kształcie `[L, 2]`. Kanał 0 zawiera liczby losowane jednostajnie $v_t \sim U[0, 1]$. Kanał 1 zawiera binarne znaczniki $m_t \in \{0, 1\}$, gdzie dokładnie dwa elementy mają wartość $1$.
- **Opóźnienie (Delay = 8):** Pozycje znaczników $p_1, p_2$ losowane są bez zwracania z zakresu $[8, L-1]$ (numeracja $0\dots L-1$). Cel $y$ stanowi sumę wartości $v_{p_1 - 8} + v_{p_2 - 8}$.
- **Weryfikacja działania na 1 przykładzie (indeksowanie od 1 zgodnie z treścią zadania):**
  - Wygenerowana sekwencja ($L=20$):
  - Wylosowane kroki znaczników $m(t)=1$: **krok 10** oraz **krok 18** (indeksy 9 i 17).
  - Wskazywane kroki wartości ($t - 8$): **krok 2** oraz **krok 10** (indeksy 1 i 9).
  - Pobrane wartości: $v(2) = 0{,}7124$, $v(10) = 0{,}4318$.
  - Rzeczywista suma (Cel): $0{,}7124 + 0{,}4318 = \mathbf{1{,}1442}$ (identyczna z tensorem wyjściowym $y$).

### Architektury i odczyt wyniku
- **Modele rekurencyjne (SimpleRNN, LSTM, GRU):** Przetwarzają sekwencję krok po kroku i przekazują wyłącznie **ostatni stan ukryty** $h_L$ do warstwy liniowej (`nn.Linear(hidden, 1)`).
- **Model splotowy (Conv1D):** Stos warstw splotowych z dopełnieniem symetrycznym (`padding = dilation`) zachowującym długość sekwencji. Wyjście podlega globalnemu uśrednianiu w czasie (`mean(dim=2)`), po czym trafia do warstwy liniowej.
- **Parametry treningu:** $N_{train} = 4000$, $N_{test} = 800$, epoki = 30, batch = 128, optymalizator Adam ($lr = 0{,}005$), obcinanie normy gradientu do $1{,}0$ (`clip_grad_norm_`). Zbiory danych wygenerowano raz i współdzielono między modelami danej długości.

---

## 3. Wyniki

### Punkt odniesienia (Baseline)
Dla $y = v_{t_1-8} + v_{t_2-8}$ oczekiwana wartość $\mathbb{E}[y] = 0{,}5 + 0{,}5 = 1{,}0$, a wariancja $\text{Var}(y) = \frac{1}{12} + \frac{1}{12} = \frac{1}{6} \approx \mathbf{0{,}1667}$. Stała predykcja równa średniej ze zbioru treningowego daje błąd testowy:
- Dla $L=50$: $\text{MSE}_{base} = \mathbf{0{,}1672}$
- Dla $L=100$: $\text{MSE}_{base} = \mathbf{0{,}1669}$

### Tabela zbiorcza wyników

| Model | Długość $L$ | Parametry | Test MSE | Czas treningu [s] | Uwagi / Konfiguracja |
|---|:---:|:---:|:---:|:---:|---|
| **Baseline (Średnia)** | 50 | 0 | 0.1672 | — | Predykcja stała $\hat{y} \approx 1{,}0$ |
| **SimpleRNN** | 50 | 1 185 | 0.1741 | 14.2 s | $H=32$, nie uczy się |
| **LSTM** | 50 | 4 609 | 0.0012 | 44.8 s | $H=32$, bramki |
| **GRU** | 50 | 3 457 | 0.0009 | 41.5 s | $H=32$, bramki |
| **Conv1D [1,2,4]** | 50 | 25 217 | 0.0004 | 18.6 s | $ch=64$, RF = 15 |
| **Baseline (Średnia)** | 100 | 0 | 0.1669 | — | Predykcja stała $\hat{y} \approx 1{,}0$ |
| **SimpleRNN** | 100 | 1 185 | 0.1789 | 24.5 s | $H=32$, zanikający gradient |
| **SimpleRNN [MOD]** | 100 | 4 417 | 0.1724 | 26.8 s | $H=64$, zwiększona szerokość |
| **LSTM** | 100 | 4 609 | 0.0028 | 85.1 s | $H=32$, bramki |
| **GRU** | 100 | 3 457 | 0.0019 | 79.3 s | $H=32$, bramki |
| **Conv1D [1,2,4]** | 100 | 25 217 | 0.0007 | 34.2 s | $ch=64$, dylatacje $[1,2,4]$, RF = 15 |
| **Conv1D [1,1,1] [MOD]**| 100 | 25 217 | 0.1648 | 33.9 s | $ch=64$, dylatacje $[1,1,1]$, RF = 7 |

<img width="989" height="490" alt="mse test" src="https://github.com/user-attachments/assets/fd6d9fc3-80d6-4503-a6fa-1c82179de3eb" />

---

## 4. Analiza i interpretacja

### 1. Zdolność do pokonania punktu odniesienia i wpływ długości sekwencji
- Modele **LSTM, GRU oraz Conv1D [1, 2, 4]** bez trudu pokonały barierę wariancji szumu ($0{,}167$), osiągając błędy rzędu $0{,}0004\dots 0{,}0028$.
- **SimpleRNN** zarówno dla $L=50$ ($0{,}1741$), jak i $L=100$ ($0{,}1789$) nie zdołał pobić stałej predykcji. Wynika to z problemu zanikającego gradientu podczas BPTT — wielokrotne mnożenie przez macierz wag $W_{hh}$ i pochodne $\tanh \le 1$ tłumi sygnał błędu, uniemożliwiając powiązanie znacznika z wartością odległą o 8 kroków i utrzymanie tej sumy do końca sekwencji.

### 2. Eksperyment modyfikacyjny dla $L=100$
- **Wpływ rozmiaru stanu w SimpleRNN (H=32 vs H=64):** Zwiększenie liczby jednostek ukrytych z 32 do 64 prawie czterokrotnie zwiększyło liczbę parametrów (z 1 185 do 4 417), ale błąd testowy spadł nieznacznie (z $0{,}1789$ do $0{,}1724$) i wciąż pozostał na poziomie naiwnej średniej. Potwierdza to, że wąskim gardłem prostej rekurencji nie jest pojemność pamięci (szerokość), lecz niestabilność propagacji gradientu w czasie (głębokość BPTT).
- **Wpływ dylatacji i pola recepcyjnego w Conv1D:**
  - Wariant bazowy $[1, 2, 4]$ posiada pole recepcyjne $R = 1 + 2 \cdot (1 + 2 + 4) = 15$ pozycji. Ponieważ $15 > 8$, pojedyncza pozycja w mapie cech obejmuje jednocześnie znacznik $m_t$ oraz wskazywaną wartość $v_{t-8}$. Model osiąga znakomite $\text{MSE} = 0{,}0007$.
  - Wariant zmodyfikowany $[1, 1, 1]$ posiada pole recepcyjne $R = 1 + 2 \cdot (1 + 1 + 1) = 7$ pozycji. Ponieważ $7 < 8$, jądra splotowe fizycznie nie są w stanie skojarzyć jedynki z wartością sprzed 8 kroków. W efekcie błąd wyniósł $\text{MSE} = 0{,}1648$ (poziom stałej predykcji), mimo identycznej liczby parametrów (25 217).

### 3. Analiza próbki testowej o największym błędzie (Worst-case Sample)
Dla modelu `SimpleRNN (h=32)` dla $L=100$ wyodrębniono próbkę testową o maksymalnym błędzie kwadratowym:
- **Pozycje znaczników (kroki 1..100):** krok 12 oraz krok 25 (indeksy 11 i 24).
- **Wskazywane wartości (kroki 1..100):** krok 4 ($v_4 = 0{,}9412$) oraz krok 17 ($v_{17} = 0{,}8920$).
- **Wartość rzeczywista (Cel):** $0{,}9412 + 0{,}8920 = \mathbf{1{,}8332}$.
- **Predykcja SimpleRNN:** $\mathbf{0{,}9841}$.
- **Błąd bezwzględny:** $|1{,}8332 - 0{,}9841| = \mathbf{0{,}8491}$ (błąd kwadratowy: $0{,}7210$).
- **Interpretacja:** Obie wartości wystąpiły w pierwszej ćwiartce sekwencji ($t \le 25$). Przez pozostałe 75 kroków stan ukryty SimpleRNN był nieustannie nadpisywany losowym szumem z kolejnych kroków. Model całkowicie utracił informację z początku sekwencji i wygenerował zachowawczą wartość zbliżoną do średniej ($\approx 1{,}0$), co przy skrajnej sumie rzeczywistej dało potężny błąd.

---

## 5. Wnioski

1. **Przewaga komórek bramkowych nad SimpleRNN:** Mechanizmy bramek w LSTM i GRU (bramka zapominania/aktualizacji) tworzą addytywne drogi przepływu gradientu, eliminując zanikanie gradientu i umożliwiając bezbłędne przeniesienie sumy przez 100 kroków czasowych.
2. **Rola pola recepcyjnego w Conv1D:** Model splotowy 1D jest niezwykle szybki w treningu (blisko trzykrotnie szybszy niż LSTM dzięki równolegleniu obliczeń wzdłuż osi czasu), jednak jego skuteczność zależy krytycznie od dylatacji. Aby model splotowy wykrył zależność o opóźnieniu $D$, jego pole recepcyjne musi spełniać warunek $R > D$.
3. **Wpływ mechanizmu agregacji:** Znakomity wynik Conv1D wynika z faktu, że zadanie polega na sumowaniu elementów – globalne uśrednianie (`mean(dim=2)`) naturalnie akumuluje wyekstrahowane lokalnie składniki bez konieczności pamiętania ich sekwencyjnego porządku.
4. **Skalowalność:** Zwiększenie długości sekwencji z 50 do 100 nie wpłynęło negatywnie na zbieżność LSTM, GRU i Dilated Conv1D, powodując jedynie liniowy wzrost czasu obliczeń.
