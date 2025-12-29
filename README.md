# Seminarska naloga – Celovita podatkovna analiza in optimizacija procesa 

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
