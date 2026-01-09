# Seminarska naloga – Celovita podatkovna analiza in optimizacija procesa 
---
## 1.1 Izbor podatkov
- Izbrana *javno dostopna* baza podatkov
- Za seminarsko nalogo sva izbrala javno dostopno bazo podatkov iz Kaggle:
- Ime podatkovne baze: *Cardiovascular Diseases Risk Prediction Dataset*
- Velikost podatkov: 308,854 zapisov × 19 spremenljivk
- Vsebina: anketni (samoporočani) podatki o zdravju in življenjskem slogu odraslih, z indikatorjem prisotnosti srčne bolezni.

### Namen analize
- Namen analize je razumeti povezave med dejavniki življenjskega sloga/zdravja in srčno-žilnimi tveganji ter pripraviti podatkovno osnovo za gradnjo napovednih modelov in kasnejšo simulacijo optimizacijskih ukrepov.

### Opis spremenljivk (ime, pomen, tip)

| Spremenljivka                    | Prevod                              | Pomen / opis                                                       | Tip spremenljivke        | Možne vrednosti / merilo                                                                                     |
| -------------------------------- | ----------------------------------- | ------------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| **General_Health**               | **Splošno zdravje**                 | Samoocena splošnega zdravstvenega stanja.                          | kategorialna (ordinalna) | `Poor`, `Fair`, `Good`, `Very Good`, `Excellent`                                                             |
| **Checkup**                      | **Zadnji preventivni pregled**      | Kdaj je bil zadnji rutinski zdravstveni pregled.                   | kategorialna (ordinalna) | `Within the past year`, `Within the past 2 years`, `Within the past 5 years`, `5 or more years ago`, `Never` |
| **Exercise**                     | **Telesna aktivnost**               | Ali je oseba izvajala telesno aktivnost.                           | kategorialna (binarna)   | `Yes` / `No`                                                                                                 |
| **Heart_Disease**                | **Srčna bolezen**                   | Prisotnost srčne bolezni (ciljna spremenljivka za klasifikacijo).  | kategorialna (binarna)   | `Yes` / `No`                                                                                                 |
| **Skin_Cancer**                  | **Kožni rak**                       | Prisotnost/zgodovina kožnega raka.                                 | kategorialna (binarna)   | `Yes` / `No`                                                                                                 |
| **Other_Cancer**                 | **Drugi rak**                       | Prisotnost/zgodovina druge vrste raka.                             | kategorialna (binarna)   | `Yes` / `No`                                                                                                 |
| **Depression**                   | **Depresija**                       | Prisotnost/diagnoza depresije.                                     | kategorialna (binarna)   | `Yes` / `No`                                                                                                 |
| **Diabetes**                     | **Diabetes (sladkorna bolezen)**    | Status diabetesa.                                                  | kategorialna (nominalna) | `No`, `Yes`, `No, pre-diabetes or borderline diabetes`, `Yes, but female told only during pregnancy`         |
| **Arthritis**                    | **Artritis**                        | Prisotnost artritisa.                                              | kategorialna (binarna)   | `Yes` / `No`                                                                                                 |
| **Sex**                          | **Spol**                            | Spol anketiranca.                                                  | kategorialna (nominalna) | `Female` / `Male`                                                                                            |
| **Age_Category**                 | **Starostna skupina**               | Starost v kategorijah.                                             | kategorialna (ordinalna) | `18-24`, `25-29`, …, `75-79`, `80+` (skupaj 13 kategorij)                                                    |
| **Height_(cm)**                  | **Višina (cm)**                     | Višina osebe v centimetrih.                                        | numerična (kontinuirna)  | realne vrednosti; v naboru približno **91–241 cm**                                                           |
| **Weight_(kg)**                  | **Teža (kg)**                       | Teža osebe v kilogramih.                                           | numerična (kontinuirna)  | realne vrednosti; v naboru približno **24.95–293.02 kg**                                                     |
| **BMI**                          | **ITM / BMI (indeks telesne mase)** | Indeks telesne mase (kg/m²).                                       | numerična (kontinuirna)  | realne vrednosti; v naboru približno **12.02–99.33**                                                         |
| **Smoking_History**              | **Kadilska zgodovina**              | Ali je oseba kadila/kadi (zgodovina kajenja).                      | kategorialna (binarna)   | `Yes` / `No`                                                                                                 |
| **Alcohol_Consumption**          | **Uživanje alkohola**               | Pogostost/mera uživanja alkohola (številčno kodirano).             | numerična (diskretna)    | vrednosti **0–30**                                                                                           |
| **Fruit_Consumption**            | **Uživanje sadja**                  | Pogostost uživanja sadja (številčno kodirano).                     | numerična (diskretna)    | vrednosti **0–120**                                                                                          |
| **Green_Vegetables_Consumption** | **Uživanje zelene zelenjave**       | Pogostost uživanja zelene zelenjave (številčno kodirano).          | numerična (diskretna)    | vrednosti **0–128**                                                                                          |
| **FriedPotato_Consumption**      | **Uživanje ocvrtega krompirja**     | Pogostost uživanja ocvrtih krompirjevih jedi (številčno kodirano). | numerična (diskretna)    | vrednosti **0–128**                                                                                          |
---
## 1.2 Pregled in čiščenje podatkov

Čeprav v opisu podatkovne baze piše da je očiščena, sva se odločila da podatke tudi midva preveriva. 
Vse korake, izpise in preveritve sva dokumentirala v Python zvezku: `1_2_pregled_in_ciscenje.ipynb`.

### Preverjanje manjkajočih vrednosti (NA)
- Preverila sva manjkajoče vrednosti po vseh spremenljivkah.
- *Rezultat*: skupno število manjkajočih vrednosti je *0*.
- *Odločitev*: ker manjkajočih vrednosti ni, *nadomeščanje ni potrebno*.
### Preverjanje podvojenih zapisov (duplikati)
- Preverila sva popolne duplikate vrstic, kar pomeni: vrstica je identična drugi vrstici v vseh 19 stolpcih (ne gre za “ponavljanje” v enem stolpcu, ampak za 100% enak zapis).
- *Rezultat*: 80 ponovitev (tj. ponovitve po prvem pojavu)
- *Odločitev*: *duplikatov ne odstranjujeva*, ker nabor nima identifikatorja posameznika (ID) in gre za anketne podatke — popolnoma enaki odgovori so zato lahko realni in predstavljajo različne osebe z enakimi karakteristikami.
prosim editaj to v lepo markdown notacijo
### Preverjanje ekstremnih vrednosti (outliers)
- Pregledala sva razpone numeričnih spremenljivk (min/max) in porazdelitve.
- Rezultat (min–max):
- Height_(cm): `91 – 241`
- Weight_(kg): `24.95 – 293.02`
- BMI: `12.02 – 99.33`
- Alcohol_Consumption: `0 – 30`
- Fruit_Consumption: `0 – 120`
- Green_Vegetables_Consumption: `0 – 128`
- FriedPotato_Consumption: `0 – 128`
- *Odločitev*: *ekstremnih vrednosti ne odstranjujeva*, ker so lahko realne (npr. zelo visok BMI ali teža) in lahko pomembno vplivajo na raziskovanje in napovedovanje tveganj.
  
---
## 1.3 Deskriptivna statistika z grafi

Za vsako spremenljivko sva pripravila:
- osnovne statistike: numerične spremenljivke: mean ± SD ali mediana (Q1–Q3) ter min–max, kategorialne spremenljivke: n (%) po kategorijah,
- grafični prikaz: numerične: histogram (in po potrebi boxplot), kategorialne: barplot deležev,
- kratko interpretacijo opažanj (porazdelitve, odstopanja, posebnosti).

Vse izračune, tabele in grafe sva izvedla v Python zvezku: `1_3_deskriptivna_statistika_grafi.ipynb`.
### Grafi (avtomatsko generirano)

### Numerične spremenljivke (histogram + boxplot)

<table>
  <tr>
    <td align="center"><b>Height_(cm)</b><br>
      <img src="figures/1_3/hist_Height_cm.png" width="360"><br>
      <img src="figures/1_3/box_Height_cm.png" width="360">
    </td>
    <td align="center"><b>Weight_(kg)</b><br>
      <img src="figures/1_3/hist_Weight_kg.png" width="360"><br>
      <img src="figures/1_3/box_Weight_kg.png" width="360">
    </td>
  </tr>
  <tr>
    <td align="center"><b>BMI</b><br>
      <img src="figures/1_3/hist_BMI.png" width="360"><br>
      <img src="figures/1_3/box_BMI.png" width="360">
    </td>
    <td align="center"><b>Alcohol_Consumption</b><br>
      <img src="figures/1_3/hist_Alcohol_Consumption.png" width="360"><br>
      <img src="figures/1_3/box_Alcohol_Consumption.png" width="360">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Fruit_Consumption</b><br>
      <img src="figures/1_3/hist_Fruit_Consumption.png" width="360"><br>
      <img src="figures/1_3/box_Fruit_Consumption.png" width="360">
    </td>
    <td align="center"><b>Green_Vegetables_Consumption</b><br>
      <img src="figures/1_3/hist_Green_Vegetables_Consumption.png" width="360"><br>
      <img src="figures/1_3/box_Green_Vegetables_Consumption.png" width="360">
    </td>
  </tr>
  <tr>
    <td align="center"><b>FriedPotato_Consumption</b><br>
      <img src="figures/1_3/hist_FriedPotato_Consumption.png" width="360"><br>
      <img src="figures/1_3/box_FriedPotato_Consumption.png" width="360">
    </td>
    <td></td>
  </tr>
</table>

### Kategorialne spremenljivke (barplot deležev)

<table>
  <tr>
    <td align="center"><b>General_Health</b><br><img src="figures/1_3/bar_General_Health.png" width="360"></td>
    <td align="center"><b>Checkup</b><br><img src="figures/1_3/bar_Checkup.png" width="360"></td>
  </tr>
  <tr>
    <td align="center"><b>Exercise</b><br><img src="figures/1_3/bar_Exercise.png" width="360"></td>
    <td align="center"><b>Heart_Disease</b><br><img src="figures/1_3/bar_Heart_Disease.png" width="360"></td>
  </tr>
  <tr>
    <td align="center"><b>Skin_Cancer</b><br><img src="figures/1_3/bar_Skin_Cancer.png" width="360"></td>
    <td align="center"><b>Other_Cancer</b><br><img src="figures/1_3/bar_Other_Cancer.png" width="360"></td>
  </tr>
  <tr>
    <td align="center"><b>Depression</b><br><img src="figures/1_3/bar_Depression.png" width="360"></td>
    <td align="center"><b>Diabetes</b><br><img src="figures/1_3/bar_Diabetes.png" width="360"></td>
  </tr>
  <tr>
    <td align="center"><b>Arthritis</b><br><img src="figures/1_3/bar_Arthritis.png" width="360"></td>
    <td align="center"><b>Sex</b><br><img src="figures/1_3/bar_Sex.png" width="360"></td>
  </tr>
  <tr>
    <td align="center"><b>Age_Category</b><br><img src="figures/1_3/bar_Age_Category.png" width="360"></td>
    <td align="center"><b>Smoking_History</b><br><img src="figures/1_3/bar_Smoking_History.png" width="360"></td>
  </tr>
</table>
---

## 1.4 Bivariatna analiza (KLASIFIKACIJA: Heart_Disease)

V okviru bivariatne analize preveriva povezavo **vsake neodvisne spremenljivke X** z odvisno spremenljivko **Heart_Disease (Yes/No)**. Cilj je:
- razumeti, **katere spremenljivke so najbolj povezane** s pojavnostjo srčne bolezni,
- dobiti **interpretabilne rezultate** (učinek + grafi),
- dobiti osnovo za nadaljnji korak **1.5 (feature selection)** in kasnejše modeliranje.

### Uporabljeni statistični testi (izbor po tipu spremenljivk)

**Numerične spremenljivke (X numerična, Y binarna):**
- preverjanje normalnosti porazdelitve po skupinah (Yes/No),
- preverjanje enakosti varianc (Levenov test),
- nato:
  - **Studentov t-test** (normalnost + enake variance),
  - **Welchov t-test** (normalnost + neenake variance),
  - **Mann–Whitney U** (če normalnost ni izpolnjena).

**Kategorialne spremenljivke (X kategorialna, Y binarna):**
- **χ² test neodvisnosti**,
- če je tabela **2×2** in so pričakovane frekvence **< 5**, uporabiva **Fisherjev eksaktni test**.

Ker je vzorec zelo velik, so p-vrednosti pogosto zelo majhne (**p < 0.001**), zato za realno pomembnost upoštevava tudi **mere učinka**:
- **Cramérjev V** (kategorialne spremenljivke),
- **Cohenov d** ali **rank-biserial r** (numerične spremenljivke).

### TOP povezave (po meri učinka)

Spodaj je prikaz najmočnejših povezav med X in Heart_Disease (povzetek iz skupne tabele):

| X | Tip X | Test | p | Mera učinka | Učinek |
|---|---|---|---|---|---|
| General_Health | kategorialna | χ² test neodvisnosti | <0.001 | Cramérjev V | 0.251 |
| Age_Category | kategorialna | χ² test neodvisnosti | <0.001 | Cramérjev V | 0.242 |
| Diabetes | kategorialna | χ² test neodvisnosti | <0.001 | Cramérjev V | 0.184 |
| Arthritis | kategorialna | χ² test neodvisnosti | <0.001 | Cramérjev V | 0.158 |
| Exercise | kategorialna | χ² test neodvisnosti | <0.001 | Cramérjev V | 0.153 |
| Alcohol_Consumption | numerična | Mann–Whitney U | <0.001 | Rank-biserial r | -0.152 |
| Smoking_History | kategorialna | χ² test neodvisnosti | <0.001 | Cramérjev V | 0.139 |
| Weight_(kg) | numerična | Mann–Whitney U | <0.001 | Rank-biserial r | 0.109 |
| BMI | numerična | Mann–Whitney U | <0.001 | Rank-biserial r | 0.107 |
| Checkup | kategorialna | χ² test neodvisnosti | <0.001 | Cramérjev V | 0.102 |

**Interpretacija (v kontekstu procesa):**
- Najmočnejše povezave s srčno boleznijo so pri **General_Health** in **Age_Category** (največji Cramérjev V) → slabše splošno zdravje in višja starostna skupina sta povezana z višjo pojavnostjo Heart_Disease.
- Pomembne povezave imajo tudi **Diabetes**, **Arthritis**, **Exercise** in **Smoking_History**, kar je smiselno glede na znane dejavnike tveganja.
- Pri numeričnih spremenljivkah so razlike statistično značilne, vendar so učinki večinoma manjši (npr. Heart_Disease=Yes ima višji BMI/težo). Pri tako velikem N je zato bolj pomembno gledati **velikost učinka** in ne samo p-vrednost.

---

# 2. Klasifikacija – gradnja in ocenjevanje modelov (cilj: Heart_Disease)

V tem delu obravnavava klasifikacijski problem, kjer je odvisna spremenljivka Heart_Disease (binarno: Yes/No). Celoten postopek (split, modeli, 10-fold CV, metrike, grafi) je izveden v zvezku `classification.ipynb`, izvoz rezultatov pa je v `2_3_cv_tabela_klasifikacija`.csv in mapah s slikami.

## 2.1 Priprava podatkov za modeliranje

- Ciljna spremenljivka (Y): `Heart_Disease` (`Yes` → 1, `No` → 0).
- **Učna/testna** delitev: 80% učna množica in 20% testna množica.
- Ker podatki niso časovno odvisni, uporabiva naključno delitev.
- Pri delitvi uporabiva stratify, da se razmerje razredov ohrani v obeh množicah (pomembno zaradi neuravnoteženosti razredov).

## 2.2 Gradnja modelov

Zgradila sva 5 različnih klasifikacijskih modelov (vsi imajo eksplicitno nastavljene hiperparametre, ne privzete), med njimi tudi obvezni logistični regresijski model:

1. LogisticRegression

- solver="lbfgs", max_iter=500, class_weight="balanced"

- (zaradi kompatibilnosti z različnimi verzijami sklearn: penalty je nastavljen tako, da model deluje; cilj je “približno ne-regularizirana” LR)

2. LinearSVC

- `C=0.8`, `class_weight="balanced"`, `max_iter=5000`

3. SGDClassifier (log-loss)

- `loss="log_loss"`, `penalty="elasticnet"`, `alpha=1e-4`, `l1_ratio=0.15`, `max_iter=2000`, `class_weight="balanced"`

4. ExtraTreesClassifier

- `n_estimators=500`, `max_depth=16`, `min_samples_split=4`, `min_samples_leaf=2`, `max_features="sqrt"`, `class_weight="balanced"`

5. RandomForestClassifier

- `n_estimators=300`, `max_depth=14`, `min_samples_split=4`, `min_samples_leaf=2`, `max_features="sqrt"`, `class_weight="balanced_subsample"`

## 2.3 Metrike napovedne uspešnosti + rezultati (10-fold CV)

Uporabiva 10-fold stratified cross-validacijo na učni množici (seed nastavljen), ter izračunava zahtevane metrike:

- **AUC, Accuracy, Sensitivity (TPR), Specificity (TNR), PPV, NPV, F1**

Tabela: primerjava modelov (validacijska množica, mean ± SD)

| Model              | AUC (mean±SD) | Accuracy (mean±SD) | Sensitivity (mean±SD) | Specificity (mean±SD) | PPV (mean±SD) | NPV (mean±SD) |  F1 (mean±SD) | AIC/BIC                    | Izbor |
| ------------------ | ------------: | -----------------: | --------------------: | --------------------: | ------------: | ------------: | ------------: | -------------------------- | :---: |
| LogisticRegression | 0.834 ± 0.007 |      0.919 ± 0.001 |         0.061 ± 0.006 |         0.995 ± 0.001 | 0.509 ± 0.032 | 0.923 ± 0.000 | 0.109 ± 0.009 | AIC=109856.5, BIC=110367.0 |   DA  |
| LinearSVC          | 0.834 ± 0.007 |      0.732 ± 0.003 |         0.790 ± 0.012 |         0.726 ± 0.003 | 0.247 ± 0.002 | 0.972 ± 0.001 | 0.323 ± 0.006 | —                          |   DA  |
| SGDClassifier      | 0.832 ± 0.006 |      0.729 ± 0.018 |         0.790 ± 0.024 |         0.723 ± 0.021 | 0.245 ± 0.017 | 0.972 ± 0.002 | 0.321 ± 0.017 | —                          |   DA  |
| ExtraTrees         | 0.826 ± 0.007 |      0.756 ± 0.003 |         0.740 ± 0.010 |         0.758 ± 0.003 | 0.221 ± 0.004 | 0.969 ± 0.001 | 0.329 ± 0.009 | —                          |       |
| RandomForest       | 0.826 ± 0.007 |      0.781 ± 0.004 |         0.695 ± 0.011 |         0.788 ± 0.003 | 0.232 ± 0.004 | 0.965 ± 0.001 | 0.339 ± 0.009 | —                          |       |

**Kriterij izbora (validacija)**: kot “TOP 3” sva označila modele z najvišjim AUC_mean (ker je AUC neodvisen od praga klasifikacije).

**Grafična primerjava modelov**

AUC (10-fold CV):
<img src="figures/2_classification/cv_auc.png" width="650">

F1 (10-fold CV):
<img src="figures/2_classification/cv_f1.png" width="650">

### Kratka interpretacija rezultatov (v kontekstu procesa)

Vsi modeli dosegajo **podobne AUC** (≈ 0.826–0.834), kar pomeni, da imajo primerljivo sposobnost rangiranja primerov (bolni dobivajo višje “score” kot zdravi).

Pri metriki **F1** pa se pojavijo večje razlike, kar je pričakovano, ker je **ciljna spremenljivka neuravnotežena** (razred Yes je bistveno redkejši).

**LogisticRegression** ima najvišji AUC, vendar zelo nizko **Sensitivity** (≈ 0.061) in posledično zelo nizek **F1** (≈ 0.109). To kaže, da model pri izbranem pragu klasifikacije skoraj nikogar ne označi kot pozitivnega (visoka Specificity, veliko FN).

**RandomForest** in **ExtraTrees** imata nekoliko nižji AUC, vendar bistveno boljši **F1** (≈ 0.33) ter višjo **Sensitivity**, kar je pogosto bolj uporabno v zdravstvenem kontekstu, kjer želimo zaznati čim več pozitivnih primerov.

# NEED TO ADD COMMENTS TO THE TABLE !!!!!!!!! 
# !!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!

