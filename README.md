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
# Klasifikacija
---
## 1.4 Bivariatna analiza (KLASIFIKACIJA: Heart_Disease)

V okviru bivariatne analize preveriva povezavo **vsake neodvisne spremenljivke (X)** z odvisno spremenljivko **Heart_Disease (Yes/No)**. Cilj je:
- razumeti, katere spremenljivke so najbolj povezane s pojavnostjo srčne bolezni,
- dobiti interpretabilne rezultate (test + mera učinka + graf),
- dobiti osnovo za nadaljnji korak **1.5 (feature selection)**.

### Uporabljeni testi (izbor po tipu spremenljivk)
- **Numerične spremenljivke (X numerična, Y binarna):**
  - preverjanje normalnosti po skupinah (Yes/No) in enakosti varianc,
  - nato: Studentov t-test / Welchov t-test / **Mann–Whitney U** (če normalnost ni izpolnjena).
- **Kategorialne spremenljivke (X kategorialna, Y binarna):**
  - **χ² test neodvisnosti**,
  - Fisherjev eksaktni test bi uporabila pri 2×2 tabelah z zelo majhnimi pričakovanimi frekvencami (v tem naboru to zaradi velikega N praviloma ni potrebno).

Ker je vzorec zelo velik, so p-vrednosti pogosto ekstremno majhne, zato za praktično pomembnost upoštevava tudi **mere učinka**:
- **Cramérjev V** (kategorialne spremenljivke),
- **rank-biserial r** (numerične spremenljivke pri Mann–Whitney).

### TOP povezave (razvrščeno po |učinku|)
Spodaj je TOP 10 spremenljivk glede na absolutno vrednost mere učinka (podatki iz skupne tabele):

| X | Tip X | Test | Mera učinka | Učinek |
|---|---|---|---|---:|
| General_Health | kategorialna | χ² | Cramérjev V | 0.250 |
| Age_Category | kategorialna | χ² | Cramérjev V | 0.242 |
| Diabetes | kategorialna | χ² | Cramérjev V | 0.184 |
| Arthritis | kategorialna | χ² | Cramérjev V | 0.154 |
| Alcohol_Consumption | numerična | Mann–Whitney | rank-biserial r | -0.152 |
| Weight_(kg) | numerična | Mann–Whitney | rank-biserial r | 0.109 |
| Smoking_History | kategorialna | χ² | Cramérjev V | 0.108 |
| BMI | numerična | Mann–Whitney | rank-biserial r | 0.107 |
| Exercise | kategorialna | χ² | Cramérjev V | 0.096 |
| Checkup | kategorialna | χ² | Cramérjev V | 0.094 |

### Kratka interpretacija (v kontekstu procesa)
- Najmočnejše povezave s srčno boleznijo sta **General_Health** in **Age_Category**: slabša samoocena zdravja in višja starostna skupina sta povezani z višjim deležem `Heart_Disease=Yes`.
- Pomembne povezave imajo tudi **Diabetes** in **Arthritis**, kar je smiselno glede na znane dejavnike tveganja.
- Pri **BMI** in **Weight_(kg)** so razlike statistično značilne, vendar z manjšim učinkom (pri velikem N je zato bolj pomembna **velikost učinka** kot sama p-vrednost).
- Pri **Alcohol_Consumption** je učinek negativen (nižja poraba pri `Yes`)

## 1.5 Izbor spremenljivk (Feature Selection – klasifikacija)

Za izbor pomembnih spremenljivk uporabiva več metod:
- **Random Forest feature importance**
- **Logistična regresija z Elastic Net regularizacijo** (upoštevava absolutno vrednost koeficientov)
- **RFE** (Recursive Feature Elimination)

Nato rezultate normalizirava in agregirava v skupni “combined score” (povprečje 3 metod).

### Končni izbor spremenljivk za modeliranje
V modele vključiva spremenljivke z oznako `Izbrano = DA` (agregirano po metodah):
- `Age_Category`
- `General_Health`
- `Diabetes`
- `Arthritis`
- `Sex`
- `BMI`
- `Weight_(kg)`

Graf TOP 10 (agregirano):
<img src="figures/feature_selection_top10.png" width="750">

---
## 2. DRUGI DEL: Gradnja in ocenjevanje modelov (KLASIFIKACIJA: Heart_Disease)

### 2.1 Priprava podatkov za modeliranje
- Ciljna spremenljivka: **Heart_Disease** (binarno: `Yes/No`).
- Podatke sva razdelila na:
  - **učno množico (80%)**
  - **testno množico (20%)**
- Ker podatki niso časovno odvisni (anketni presečni podatki), sva uporabila **naključno delitev s stratifikacijo** po razredih (`stratify=y`) za ohranitev razmerja `Yes/No` v obeh množicah.
- Predobdelava:
  - numerične spremenljivke: **standardizacija** (StandardScaler)
  - kategorialne spremenljivke: **one-hot encoding** (OneHotEncoder)

### 2.2 Gradnja modelov
Zgradila sva **vsaj 5 različnih klasifikacijskih modelov**, pri čemer je eden izmed njih **logistična regresija**:

1. **Logistic Regression** (linearni/logistični regresijski model)
2. **LinearSVC**
3. **SGDClassifier** (log-loss, elastic net)
4. **ExtraTreesClassifier**
5. **RandomForestClassifier**

Pri vseh modelih sva nastavila **hiperparametre** (torej ne uporabljava privzetih nastavitev).

Za ocenjevanje modelov sva uporabila **10-fold stratificirano cross-validacijo** na učni množici:
- `StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)`
- s fiksnim semenom za ponovljivost.

Ker je razred `Heart_Disease=Yes` bistveno redkejši od `No`, sva naredila **dve verziji eksperimenta**:
- **(A) brez uteževanja razredov** (baseline)
- **(B) z uteževanjem razredov** (`class_weight="balanced"` oz. `balanced_subsample`)

### 2.3 Metrike napovedne uspešnosti (10-fold CV)
Za klasifikacijo sva izračunala:
- **AUC**
- **Accuracy**
- **Sensitivity (TPR)**
- **Specificity (TNR)**
- **PPV (Precision)**
- **NPV**
- **F1**

Rezultate predstavljava kot **mean ± SD čez 10 foldov** in grafično (AUC in F1).

---

## 2.3.1 Rezultati (brez uteževanja razredov)
**Ključna ugotovitev:** modeli dosegajo zelo visoko **Accuracy** in **Specificity**, vendar skoraj ne zaznajo pozitivnih primerov (`Yes`) → **zelo nizka Sensitivity** in **F1**.  
To je tipična posledica **neuravnoteženih razredov** (model se “nauči”, da je skoraj vedno najbolj varno napovedati `No`).

**Grafi:**
- <img src="figures/2_classification/cv_auc_unweighted.png" width="750">
- <img src="figures/2_classification/cv_f1_unweighted.png" width="750">

---

## 2.3.2 Rezultati (z uteževanjem razredov)
**Ključna ugotovitev:** uteževanje razredov bistveno izboljša zaznavo pozitivnih (`Yes`):
- **Sensitivity** naraste (model zazna več `Yes`)
- **F1** naraste (boljši kompromis med precision in recall)
- **Accuracy** in **Specificity** običajno padeta (več lažno pozitivnih), kar je pričakovan trade-off.

**AUC** ostaja približno enak, ker je AUC neodvisen od izbranega praga (threshold), uteževanje pa predvsem premakne odločanje pri napovedih.

## 2.3 Primerjava modelov (validacijska množica) – **weighted** (10-fold CV)

| Model | Tip | Parametri | Metrike (mean±SD) | AIC/BIC | Komentar | Izbor |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | LogisticRegression | solver=lbfgs, penalty=l2, C=0.8, max_iter=500, class_weight=balanced | AUC: 0.810 ± 0.005<br>Accuracy: 0.699 ± 0.004<br>Sensitivity: 0.777 ± 0.009<br>Specificity: 0.700 ± 0.004<br>PPV: 0.185 ± 0.002<br>NPV: 0.973 ± 0.001<br>F1: 0.299 ± 0.004 | 267276.6/267641.2 | Visoka občutljivost, a več lažno pozitivnih (nižja specifičnost). | DA (TOP 3) |
| LinearSVC | LinearSVC | C=1.2, max_iter=5000, class_weight=balanced | AUC: 0.806 ± 0.005<br>Accuracy: 0.704 ± 0.004<br>Sensitivity: 0.772 ± 0.010<br>Specificity: 0.704 ± 0.004<br>PPV: 0.187 ± 0.002<br>NPV: 0.973 ± 0.001<br>F1: 0.300 ± 0.004 | — | Visoka občutljivost, a več lažno pozitivnih (nižja specifičnost). |  |
| SGDClassifier | SGDClassifier | loss=log_loss, penalty=elasticnet, alpha=0.0001, l1_ratio=0.15, max_iter=2000, class_weight=balanced | AUC: 0.805 ± 0.005<br>Accuracy: 0.707 ± 0.004<br>Sensitivity: 0.748 ± 0.012<br>Specificity: 0.711 ± 0.005<br>PPV: 0.186 ± 0.002<br>NPV: 0.971 ± 0.001<br>F1: 0.297 ± 0.004 | — | Visoka občutljivost, a več lažno pozitivnih (nižja specifičnost). |  |
| ExtraTrees | ExtraTreesClassifier | n_estimators=200, max_depth=16, min_samples_split=4, min_samples_leaf=2, max_features=sqrt, class_weight=balanced | AUC: 0.804 ± 0.005<br>Accuracy: 0.725 ± 0.004<br>Sensitivity: 0.740 ± 0.009<br>Specificity: 0.725 ± 0.004<br>PPV: 0.199 ± 0.002<br>NPV: 0.972 ± 0.001<br>F1: 0.304 ± 0.004 | — | Visoka občutljivost, a več lažno pozitivnih (nižja specifičnost). | DA (TOP 3) |
| RandomForest | RandomForestClassifier | n_estimators=200, max_depth=14, min_samples_split=4, min_samples_leaf=2, max_features=sqrt, class_weight=balanced_subsample | AUC: 0.803 ± 0.005<br>Accuracy: 0.744 ± 0.004<br>Sensitivity: 0.711 ± 0.013<br>Specificity: 0.744 ± 0.004<br>PPV: 0.205 ± 0.002<br>NPV: 0.970 ± 0.001<br>F1: 0.308 ± 0.005 | — | Visoka občutljivost, a več lažno pozitivnih (nižja specifičnost). | DA (TOP 3) |

**Grafi:**
- <img src="figures/2_classification/cv_auc_weighted.png" width="750">
- <img src="figures/2_classification/cv_f1_weighted.png" width="750">

---
### Tabela: TOP 3 modeli – testna množica (weighted)

| Model | Tip | Parametri | Metrike | AIC/BIC | Komentar | Izbor |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest | RandomForestClassifier | n_estimators=200, max_depth=14, min_samples_split=4, min_samples_leaf=2, max_features=sqrt, class_weight=balanced_subsample | AUC: 0.810; Acc: 0.742; Sens: 0.720; Spec: 0.744; PPV: 0.186; NPV: 0.967; F1: 0.311; Conf(TP/FP/TN/FN): 3595/15778/36210/1391 | — | Visoka občutljivost, več lažno pozitivnih (nižja specifičnost). | DA (TOP 3) |
| ExtraTrees | ExtraTreesClassifier | n_estimators=200, max_depth=16, min_samples_split=4, min_samples_leaf=2, max_features=sqrt, class_weight=balanced | AUC: 0.812; Acc: 0.727; Sens: 0.750; Spec: 0.725; PPV: 0.177; NPV: 0.970; F1: 0.308; Conf(TP/FP/TN/FN): 3747/17372/34616/1239 | — | Visoka občutljivost, več lažno pozitivnih (nižja specifičnost). | DA (TOP 3) |
| LogisticRegression | LogisticRegression | solver=lbfgs, penalty=l2, C=0.8, max_iter=500, class_weight=balanced | AUC: 0.816; Acc: 0.708; Sens: 0.788; Spec: 0.701; PPV: 0.188; NPV: 0.974; F1: 0.304; Conf(TP/FP/TN/FN): 3937/16997/34991/1049 | — | Visoka občutljivost, več lažno pozitivnih (nižja specifičnost). | DA (TOP 3) |
--- 
# REGRESIJSKI PRIMER (Y = BMI)

--- 
## 1.4 Bivariatna analiza (REGRESIJA: BMI)

V okviru bivariatne analize preveriva povezavo **vsake neodvisne spremenljivke (X)** z odvisno spremenljivko **BMI (numerična)**. Cilj je:
- ugotoviti, katere spremenljivke imajo statistično in praktično pomembno povezavo z BMI,
- dobiti interpretabilne rezultate (test + korelacija / p-vrednost) in vizualizacije,
- pripraviti osnovo za korak **1.5 (feature selection)**.

### Uporabljeni testi (izbor po tipu spremenljivk)
- **Numerične / ordinalne spremenljivke:**  
  - **Spearmanova korelacija (ρ)** in **Pearsonova korelacija (r)** (za oceno monotone / linearne povezave). :contentReference[oaicite:0]{index=0}
- **Binarne spremenljivke (0/1):**
  - **Mann–Whitney U test** (primerjava porazdelitev BMI med skupinama 0 vs 1). :contentReference[oaicite:1]{index=1}
- **Normalnost (diagnostika):**
  - Shapiro–Wilk test (na vzorcu do 5000 vrednosti, zaradi velikega N) + po potrebi Q–Q plot. 

### Grafični prikazi
Za vizualno interpretacijo:
- pri binarnih X: **boxplot + stripplot**,
- pri “kontinuirnih” X: **regplot (LOWESS)** za trend. :contentReference[oaicite:3]{index=3}


## 1.5 Izbor spremenljivk (Feature Selection – regresija)

V kodi sta definirana:
- **širši nabor kandidatov (`all_features`)**:  
  `Age_Category, Sex, Smoking_History, Exercise, Alcohol_Consumption, Fruit_Consumption, Green_Vegetables_Consumption, FriedPotato_Consumption, Diabetes, Depression, Arthritis, Checkup, Heart_Disease` :contentReference[oaicite:4]{index=4}
- **ožji “interpretabilen” nabor (`important_features`)**:  
  `Age_Category, Sex, Exercise, Diabetes, Arthritis, General_Health` :contentReference[oaicite:5]{index=5}

Končno množico spremenljivk za regresijske modele določiva na podlagi kombinacije:
- bivariatnih rezultatov (korelacije / Mann–Whitney),
- pomembnosti značilk pri drevesnih modelih (feature importance),
- interpretabilnosti in procesne smiselnosti (kaj lahko dejansko optimiziramo v praksi). 


## 2. DRUGI DEL: Gradnja in ocenjevanje modelov (REGRESIJA: BMI)

### 2.1 Priprava podatkov za modeliranje
- Ciljna spremenljivka: **BMI**. :contentReference[oaicite:7]{index=7}
- Podatke razdeliva na:
  - **učno množico (80%)**
  - **testno množico (20%)**
- Ker podatki niso časovno odvisni, uporabiva naključno delitev. 

### 2.2 Gradnja modelov (vsaj 5)
Zgrajenih je 5 regresijskih modelov z nastavljenimi hiperparametri:
1. **Linear Regression (OLS / statsmodels)** – linearni model. :contentReference[oaicite:10]{index=10}  
2. **Ridge Regression** – `alpha=1.0`, z **StandardScaler** v pipeline.   
3. **Random Forest Regressor** – `n_estimators=50`, `max_depth=3`. :contentReference[oaicite:12]{index=12}  
4. **Gradient Boosting Regressor** – `n_estimators=50`, `max_depth=3`.   
5. **XGBoost Regressor** – `n_estimators=100`, `max_depth=5`.   

### 2.3 Metrike napovedne uspešnosti (CV)
Za regresijo poročava:
- **R²**
- **RMSE**
- **MAE** (pri testiranju)

### 2.3 Primerjava modelov (validacijska množica)
| Model | RMSE (CV mean ± std) | R² | Parameters |
|-------|----------------------|----|------------|
| 3 Gradient Boosting | 5.953 ± 0.039 | 0.155221 | n_estimators=100, max_depth=5 [SAVED] |
| 4 XGBoost | 5.990 ± 0.036 | 0.146536 | n_estimators=200, max_depth=6 [SAVED] |
| 2 Random Forest | 6.042 ± 0.046 | 0.127633 | n_estimators=100, max_depth=5 |
| 1 Ridge Regression | 6.129 ± 0.042 | 0.105729 | alpha=1.0 [SAVED] |
| 0 Linear Regression | 37.569 ± 0.512 | 0.105729 | - |

## 3. Izbor najboljših modelov in testiranje
| Model               | RMSE      | MAE       | R²        |
|--------------------|-----------|-----------|-----------|
| Gradient Boosting  | 5.9968    | 4.4333    | 0.1552    |
| XGBoost            | 6.0275    | 4.4498    | 0.1465    |
| Ridge Regression   | 6.1797    | 4.5976    | 0.1029    |

Na podlagi validacijskih rezultatov izbereva **TOP 3 regresijske modele**:
- **XGBoost**
- **Gradient Boosting**
- **Ridge Regression** 

Modele shraniva in jih uporabiva tudi v aplikaciji:
- `models/ridge_model.joblib`
- `models/gradient_boosting_model.joblib`
- `models/xgboost_model.joblib` 

---
## 4. Interaktivna aplikacija (Simulacija in optimizacija)

Aplikacija je implementirana v **Streamlit** in pokriva oba primera:
- **Klasifikacija:** napoved `Heart_Disease (Yes/No)`
- **Regresija:** napoved `BMI`

### Zagon aplikacije
1. Namestitev odvisnosti:
   ```bash
   pip install -r requirements.txt
  
2. Zagon aplikacije:
  ```bash
  streamlit run app.py
  ```
### 4.1 Izbira modela
Uporabnik lahko izbere enega izmed 3 najboljših modelov:
- posebej za klasifikacijo (TOP 3),
- posebej za regresijo (TOP 3).

### 4.2 Napoved za posamezen ali skupinski vzorec
Aplikacija omogoča:
- vnos vrednosti spremenljivk (polja / drsniki / izbor vrednosti),
- izračun napovedi Y,
- grafični prikaz rezultatov,
- (opcijsko) napoved za več vzorcev z uvozom CSV ter izpisom rezultatov.

### 4.3 Simulacija sprememb (optimizacija procesa)
Uporabnik lahko:
- spreminja izbrano neodvisno spremenljivko,
- opazuje grafični prikaz vpliva spremembe na napoved Y,
- primerja napoved “pred” in “po” spremembi,
- izvozi rezultate simulacije v CSV.

---
# 5. Povzetek ugotovitev in priporočila za optimizacijo

Spodaj podajava zaključne ugotovitve in priporočila **ločeno** za:
- **a) Klasifikacijo** (napoved `Heart_Disease`)
- **b) Regresijo** (napoved `BMI`)

---

## 5A) KLASIFIKACIJA – `Heart_Disease` (binarno)

### Izbor najboljšega modela (testna množica)
Na testni množici med TOP3 modeli izbereva **LogisticRegression (weighted)** kot najboljši model, ker dosega:
- **najvišji AUC = 0.816**
- **najvišjo občutljivost (Sensitivity) = 0.788**  
(kar je ključno pri odkrivanju srčne bolezni, kjer želimo ujeti čim več pozitivnih primerov)

> Opomba: RandomForest in ExtraTrees imata nekoliko višji Accuracy/F1, vendar zaznata manj pozitivnih primerov (nižja Sensitivity). Pri “screening” uporabi je zato bolj smiselno dati prednost višji občutljivosti.

### Spremenljivke z največjim vplivom (in smer vpliva)
Najbolj vplivne spremenljivke (na podlagi bivariatne analize + feature selection + modelov) in interpretacija:

| Spremenljivka | Vpliv na tveganje srčne bolezni | Pomen za proces |
|---|---|---|
| **Age_Category** | **pozitiven** (višja starost → več tveganja) | starost je močan, ne-spremenljiv dejavnik → ciljanje preventivnih ukrepov |
| **General_Health** | **pozitiven** (slabša ocena → več tveganja) | proxy splošnega stanja → potreben sistematičen pregled |
| **Diabetes** | **pozitiven** | diabetes je znan rizični faktor → preventivni programi in kontrola |
| **Arthritis** | **pozitiven** (posredno) | pogosto povezan z manj gibanja/višjim BMI → sekundarni vpliv |
| **BMI / Weight_(kg)** | **pozitiven** (višje → več tveganja) | ključno področje optimizacije (prehrana + gibanje) |
| **Sex** | običajno **višje tveganje pri moških** (v povprečju) | segmentacija komunikacije in programov |

### Procesno smiselne spremembe (kaj lahko realno izboljšamo)
Spremenljivke, na katere lahko vplivamo procesno (v praksi):
- **BMI / Weight_(kg)**: prehrana, gibanje, programi hujšanja
- **Diabetes**: preventiva (preddiabetski programi), kontrola glikemije, redni pregledi
- **General_Health**: zgodnja obravnava kroničnih težav, checkup programi
- **Exercise** (čeprav ni v finalnem seznamu spremenljivk, je procesno zelo relevanten): povečanje telesne aktivnosti

### Priporočila vodstvu (konkretno)
1. **Vzpostavite “risk-screening” proces**: z modelom prepoznajte skupine z višjim tveganjem in jih povabite na preventivne preglede.
2. **Programi za znižanje BMI** (največji “leverage”):
   - ciljane delavnice prehrane + gibanja,
   - spremljanje napredka (npr. 8–12 tednov program).
3. **Diabetes preventiva in upravljanje**:
   - prioritetna obravnava oseb z diabetesom / preddiabetesom,
   - redni monitoring in edukacija.
4. **Optimizacija praga (threshold)**:
   - če želite ujeti več “Yes”, nastavite prag bolj “agresivno” (več občutljivosti, več lažno pozitivnih),
   - prag lahko prilagodite glede na kapacitete (koliko ljudi lahko obravnavate).

---

## 5B) REGRESIJA – `BMI` (numerično)

### Izbor najboljšega modela
Kot najboljši model za napoved BMI izbereva **XGBoost**, ker ima najboljši rezultat med primerjanimi regresijskimi modeli (najnižji RMSE v primerjavi modelov, ter je med shranjenimi TOP3).

> Opomba za interpretacijo: R² pri regresiji ni zelo visok (okoli ~0.15), kar pomeni, da model pojasni le del variabilnosti BMI. Kljub temu je uporaben za **simulacije “kaj-če”** in za primerjavo učinkov ukrepov na ravni populacije.

### Spremenljivke z največjim vplivom (in smer vpliva)
Pri regresiji so pomembne predvsem spremenljivke, ki so:
- povezane z življenjskim slogom,
- procesno vplivljive (cilj optimizacije).

Tipična interpretacija (podprta s smerjo iz simulacij v aplikaciji):

| Spremenljivka | Vpliv na BMI | Kaj to pomeni |
|---|---|---|
| **Exercise** | **negativen** (več gibanja → nižji BMI) | največji praktični vzvod za optimizacijo |
| **FriedPotato_Consumption** | **pozitiven** (več → višji BMI) | prehranski ukrepi (manj ocvrte hrane) |
| **Depression** | **pozitiven** (prisotnost → višji BMI) | pomemben posreden vpliv (motivacija, navade, gibanje) |
| **Diabetes** | **pozitiven** | povezava z metabolnim tveganjem in telesno maso |
| **Age_Category** | pogosto **pozitiven** | višja starost → večje tveganje za višji BMI (v povprečju) |
| **General_Health** | slabše zdravje → **višji BMI** | splošno stanje in navade se odražajo v BMI |
| **Arthritis** | **pozitiven** (posredno) | bolečine → manj gibanja → višji BMI |
| **Sex** | razlike po spolu | omogoča segmentacijo programov |

### Procesno smiselne spremembe
- **Povečanje telesne aktivnosti** (najbolj vplivno + izvedljivo)
- **Prehranske intervencije** (zmanjšanje ocvrte hrane, izboljšanje navad)
- **Podpora duševnemu zdravju** (depresija kot “multiplikator” slabih navad)
- **Ciljani programi za rizične skupine** (starejši, diabetes, arthritis)

### Priporočila vodstvu (konkretno)
1. **Program “Aktiven življenjski slog”**: cilj na redno telesno aktivnost (npr. tedenski cilji + spremljanje).
2. **Prehranski ukrepi**: zmanjšanje pogoste ocvrte hrane (izobraževanje + substitucije).
3. **Celostni pristop**: vključitev podpore pri depresiji/stresu (ker vpliva na prehrano in gibanje).
4. **Segmentacija ukrepov**: starostne skupine in kronična stanja (diabetes/arthritis) naj imajo prilagojene programe.

---

## Skupni zaključek (razumljivo vodstvu)
- Za **zmanjšanje srčno-žilnega tveganja** je ključno zgodnje prepoznavanje rizičnih skupin (model) in ciljani ukrepi.
- Največji procesni učinek imata:
  - **zniževanje BMI (prehrana + gibanje)** in
  - **upravljanje diabetesa / splošnega zdravja**.
- Modeli omogočajo **merljiv pristop**: predlagane ukrepe lahko testiramo s simulacijami (6) in pokažemo izboljšanje metrik ter sigma nivoja (7).
---
# 6. Testiranje globalnih sprememb (Simulacije)

V tem koraku sva izvedla globalne “kaj-če” simulacije, kjer na celotni testni množici hkrati spremenimo izbrane vhodne spremenljivke (npr. izboljšanje splošnega zdravja, znižanje BMI, izboljšanje diabetesa …). Namen je pokazati, kako bi se spremenile napovedi modela in ključne metrike, če bi uvedli izbran optimizacijski ukrep na ravni populacije.

### 6A) Klasifikacija – Heart_Disease (model: LogisticRegression weighted)

Merjeno:
- povprečno tveganje P(Yes),
- delež napovedanih Yes (threshold = 0.5),
- število napovedanih Yes.
Spodaj so prikazani samo scenariji, ki so imeli vpliv (tj. sprememba ni 0).
Tabela: Pred–Potem (Klasifikacija)

| Model              | Metrika                                                      |     Pred |       Po |   Razlika | Interpretacija                                             |
| ------------------ | ------------------------------------------------------------ | -------: | -------: | --------: | ---------------------------------------------------------- |
| LogisticRegression | Povp. tveganje (P(Yes)) — General_Health +1                  | 0.361618 | 0.278083 | -0.083535 | Nižje je bolje (nižja povprečna verjetnost srčne bolezni). |
| LogisticRegression | Delež napovedanih Yes (threshold=0.5) — General_Health +1    | 0.310923 | 0.187289 | -0.123634 | Nižje je bolje (manj pričakovanih pozitivnih primerov).    |
| LogisticRegression | Št. napovedanih Yes (threshold=0.5) — General_Health +1      |    19206 |    11569 |     -7637 | Nižje je bolje (manj napovedanih primerov bolezni).        |
| LogisticRegression | Povp. tveganje (P(Yes)) — Diabetes izboljšanje               | 0.361618 | 0.350718 | -0.010900 | Nižje je bolje (nižja povprečna verjetnost srčne bolezni). |
| LogisticRegression | Delež napovedanih Yes (threshold=0.5) — Diabetes izboljšanje | 0.310923 | 0.293989 | -0.016934 | Nižje je bolje (manj pričakovanih pozitivnih primerov).    |
| LogisticRegression | Št. napovedanih Yes (threshold=0.5) — Diabetes izboljšanje   |    19206 |    18160 |     -1046 | Nižje je bolje (manj napovedanih primerov bolezni).        |
| LogisticRegression | Povp. tveganje (P(Yes)) — BMI −2                             | 0.361618 | 0.360810 | -0.000807 | Nižje je bolje (nižja povprečna verjetnost srčne bolezni). |
| LogisticRegression | Delež napovedanih Yes (threshold=0.5) — BMI −2               | 0.310923 | 0.309903 | -0.001020 | Nižje je bolje (manj pričakovanih pozitivnih primerov).    |
| LogisticRegression | Št. napovedanih Yes (threshold=0.5) — BMI −2                 |    19206 |    19143 |       -63 | Nižje je bolje (manj napovedanih primerov bolezni).        |
| LogisticRegression | Povp. tveganje (P(Yes)) — Teža −5%                           | 0.361618 | 0.361031 | -0.000586 | Nižje je bolje (nižja povprečna verjetnost srčne bolezni). |
| LogisticRegression | Delež napovedanih Yes (threshold=0.5) — Teža −5%             | 0.310923 | 0.310146 | -0.000777 | Nižje je bolje (manj pričakovanih pozitivnih primerov).    |
| LogisticRegression | Št. napovedanih Yes (threshold=0.5) — Teža −5%               |    19206 |    19158 |       -48 | Nižje je bolje (manj napovedanih primerov bolezni).        |

Kratek komentar učinkov:
- Največji vpliv ima scenarij General_Health +1, kjer se povprečno tveganje in število napovedanih Yes izrazito zmanjšata.
- Opazen učinek ima tudi izboljšanje diabetesa, kar je skladno z znanimi dejavniki tveganja.
- Spremembe pri BMI/teži so v smeri izboljšanja, vendar so relativno majhne (pri threshold=0.5 se malo primerov premakne čez odločilno mejo).
### 6B) Regresija – BMI (model: XGBoost, scenarij: Vsi se gibamo)
V regresijskem primeru sva simulirala scenarij “Vsi se gibamo”, kjer sva za vse zapise v testni množici nastavila Exercise = Yes. Nato sva primerjala napovedi BMI pred in po spremembi.
| Metric   |    Before |     After | Difference |
| -------- | --------: | --------: | ---------: |
| MAE      |  4.429150 |  4.429673 |   0.000523 |
| RMSE     |  5.962129 |  6.014610 |   0.052481 |
| Mean BMI | 28.662680 | 28.323908 |  -0.338772 |

Kratek komentar učinkov:
- Povprečni napovedani BMI se pri scenariju “vsi telovadijo” zniža za približno 0.34.
- MAE/RMSE ostaneta praktično podobna (rahlo višja), kar je pričakovano pri globalni simulaciji, ker primerjamo napovedi z nespremenjenimi dejanskimi vrednostmi.

---
# 7. Six Sigma analiza – PREJ in POTEM

### 7.1 Klasifikacija (Heart_Disease) – napake = napačne klasifikacije
V klasifikacijskem primeru napako (defect) definiramo kot napačno klasifikacijo (model napove Yes/No narobe). Ker ima vsak zapis eno odločitev, vzamemo 1 opportunity na zapis. Na tej osnovi izračunamo:
- DPMO = (število napak / (št. zapisov × 1)) × 1,000,000
- Sigma nivo iz izkoristka (yield = 1 − DPMO/1e6)
Spodnja tabela prikazuje stanje PREJ in POTEM za izbrane scenarije.

| Model              | Scenarij             | DPMO PREJ | Sigma PREJ | DPMO POTEM | Sigma POTEM | Izboljšava (ΔSigma) |
| ------------------ | -------------------- | --------: | ---------: | ---------: | ----------: | ------------------: |
| LogisticRegression | General_Health +1    |    264072 |   0.630841 |     170905 |    0.950593 |        **0.319752** |
| LogisticRegression | Diabetes izboljšanje |    264072 |   0.630841 |     243172 |    0.696134 |            0.065293 |
| LogisticRegression | BMI -2               |    264072 |   0.630841 |     259863 |    0.643768 |            0.012926 |
| LogisticRegression | Teža -5%             |    264072 |   0.630841 |     267019 |    0.621855 |           -0.008986 |

Interpretacija izboljšave procesa:

- Največjo izboljšavo doseže scenarij “General_Health +1”, kjer se DPMO zmanjša iz 264072 na 170905, sigma nivo pa se poveča iz 0.631 na 0.951 (ΔSigma ≈ +0.320). To pomeni opazno manj napačnih klasifikacij na milijon odločitev in s tem boljši “proces” odločanja modela.
- Scenarij “Diabetes izboljšanje” prinese zmerno izboljšavo (ΔSigma ≈ +0.065).
- Scenarija “BMI -2” in “Teža -5%” imata zelo majhen oziroma negativen učinek na sigma nivo (pri teži -5% se DPMO celo rahlo poveča).
Statistična in procesna pomembnost:
- Za glavni scenarij (General_Health +1) je McNemar test pokazal p ≈ 0, kar pomeni, da je razlika v napakah statistično značilna.
- Procesno je sprememba smiselna predvsem tam, kjer je ΔSigma dovolj velik (tukaj je to General_Health +1). Pri minimalnih ΔSigma (BMI -2) gre za majhno izboljšavo, ki je v praksi manj prepričljiva.
Ali upravičuje implementacijo v praksi?
- Scenarij General_Health +1 kaže največji potencial, vendar predstavlja “globalno izboljšanje” splošnega zdravja, kar v praksi pomeni potrebo po širših preventivnih programih (checkup, zgodnja obravnava kroničnih težav, vodenje rizičnih skupin). Glede na velik ΔSigma je tak ukrep iz vidika modela najbolj upravičen.
- Pri drugih scenarijih je učinek manjši; implementacija je bolj smiselna kot dopolnilo, ne kot glavni nosilec optimizacije.


### 7.2 Regresija (BMI) – napake = odstopanje nad toleranco
V regresijskem primeru napako (defect) definiramo preko tolerance: zapis je “defekten”, če je absolutna napaka |BMI − napoved| večja od izbrane tolerance. Na tej osnovi nato izračunamo DPMO in sigma nivo pred in po spremembi.

| Metric      |        Before |         After |   Difference |
| ----------- | ------------: | ------------: | -----------: |
| Defects     | 215810.000000 | 214768.000000 | -1042.000000 |
| DPMO        | 698744.390553 | 695370.628193 | -3373.762360 |
| Sigma Level |     -0.520793 |     -0.511132 |     0.009661 |


## 8. Končni povzetek (Executive Summary)

V seminarski nalogi sva obravnavala problem **napovedovanja srčno-žilnega tveganja** in hkrati pripravila osnovo za **optimizacijo procesa** z uporabo podatkovno-podprte simulacije “kaj-če”. Uporabila sva javno dostopno bazo *Cardiovascular Diseases Risk Prediction Dataset* (308,854 zapisov × 19 spremenljivk), ki vsebuje anketne (samoporočane) podatke o zdravju in življenjskem slogu odraslih ter ciljno spremenljivko **Heart_Disease (Yes/No)**. Poleg klasifikacije sva za prikaz regresijskega pristopa modelirala še **BMI** kot numerični izhod (regresijski primer).

### Kaj je bil problem

Ključni izziv je bil:

1. zgraditi **zanesljiv napovedni model** za odkrivanje primerov z višjim tveganjem srčne bolezni (screening pristop), in
2. preveriti, **katere spremembe vhodnih spremenljivk** (ukrepi) bi lahko procesno zmanjšale napovedano tveganje ter kako se to odrazi v **Six Sigma metrikah** (DPMO, sigma nivo).

### Kateri modeli so najboljši in zakaj

**Klasifikacija (Heart_Disease):** Med TOP3 modeli na testni množici izbereva **LogisticRegression (weighted)** kot najboljši kompromis, ker dosega:

* **najvišji AUC = 0.816** in
* **najvišjo občutljivost (Sensitivity) = 0.788**, kar je ključno pri odkrivanju pozitivnih primerov (želimo “ujeti” čim več oseb z boleznijo/tveganjem).

**Regresija (BMI):** Kot najboljši model za napoved BMI izbereva **XGBoost**, ker v primerjavi uporabljenih regresijskih modelov dosega najboljšo napovedno moč (najnižji RMSE med TOP3) in je primeren za simulacije “kaj-če” na populacijskem nivoju.

### Katere spremenljivke je smiselno optimizirati

Iz bivariatne analize, izbora značilk in procesne interpretacije izstopajo spremenljivke, ki imajo močan vpliv in so hkrati **vsaj delno vplivljive**:

* **General_Health** (splošno zdravje): najbolj izrazito povezano s srčno boleznijo; procesno pomeni potrebo po preventivi, checkup programih in zgodnji obravnavi kroničnih težav.
* **Diabetes**: pomemben dejavnik tveganja; smiselni so programi preventive, nadzor glikemije in edukacija.
* **BMI / Weight_(kg)**: neposredno povezano s tveganjem; optimizacija preko prehranskih in gibalnih intervencij.
* **Exercise** (v regresiji): glavni praktični vzvod za zniževanje BMI (več gibanja → nižji BMI).
* **Age_Category** in **Sex** sta pomembna, vendar **neoptimizabilna**; uporabljata se predvsem za segmentacijo in ciljanje ukrepov (kdo je prioriteten za preventivo).

### Kakšni so učinki sprememb (simulacije “kaj-če”)

Pri klasifikaciji (LogisticRegression weighted) se pri globalnih simulacijah najbolj izkaže scenarij:

* **General_Health +1**, kjer se povprečno napovedano tveganje in število napovedanih pozitivnih primerov bistveno znižata; opazen učinek ima tudi **Diabetes izboljšanje**, medtem ko sta vpliva **BMI −2** in **Teža −5%** precej manjša pri pragu 0.5.

Pri regresiji (BMI) scenarij **“Vsi se gibamo”** pokaže:

* znižanje **povprečno napovedanega BMI za približno 0.34**,
* MAE/RMSE ostaneta praktično podobna (rahlo višja), kar je pričakovano pri globalni simulaciji, ker primerjamo napovedi z nespremenjenimi dejanskimi vrednostmi.

### Kako se je sigma stopnja izboljšala (Six Sigma PREJ/POTEM)

**Klasifikacija (napake = napačne klasifikacije):**
Največjo izboljšavo doseže scenarij **General_Health +1**:

* **DPMO** se zmanjša iz **264,072** na **170,905**,
* **sigma nivo** se poveča iz **0.631** na **0.951** (ΔSigma ≈ **+0.320**).
  To pomeni občutno manj napak na milijon odločitev in boljši “proces” napovedovanja. Za ta glavni scenarij je bila sprememba napak tudi **statistično značilna** (McNemar p ≈ 0).

**Regresija (napake = odstopanje nad toleranco):**
Po spremembi se število “defects” rahlo zmanjša (−1042), DPMO se zniža za ~3374, sigma nivo pa se minimalno izboljša (ΔSigma ≈ +0.0097), kar pomeni majhno izboljšavo procesa glede na izbrano toleranco.

### Priporočilo za implementacijo sprememb

Priporočava **implementacijo sprememb predvsem tam, kjer je učinek procesno največji in operativno izvedljiv**:

1. **Prioriteta 1: Programi za izboljšanje General_Health (preventiva + checkup + zgodnja obravnava)**

   * ker daje največji učinek v simulacijah in največjo izboljšavo sigma nivoja (ΔSigma ≈ +0.320).
   * v praksi to pomeni sistematično vodenje preventivnih pregledov, zgodnje prepoznavanje kroničnih težav in aktivno spremljanje rizičnih skupin.

2. **Prioriteta 2: Diabetes preventiva in upravljanje**

   * zmeren, a stabilen učinek (ΔSigma ≈ +0.065), visok procesni smisel (znan rizični faktor).

3. **Podporno: zniževanje BMI/teže + spodbujanje gibanja**

   * vpliv na klasifikacijo je pri pragu 0.5 manjši, vendar je ukrep dolgoročno smiseln in se v regresijskem primeru kaže kot znižanje povprečnega BMI (≈ −0.34), zato ga priporočava kot del celostnega programa zdravja.

Skupno: modeli omogočajo **merljivo odločanje** (koga ciljati) in **kvantificiranje učinkov ukrepov** (kaj se izboljša in za koliko). Zato je smiselno spremembe uvajati postopno: najprej ukrepi z največjim vplivom (General_Health, Diabetes), nato razširitev na celostne programe (BMI/Exercise), ob sprotnem spremljanju KPI-jev in sigma metrik.


