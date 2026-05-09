# 📊 Analiza și Prognoza Consumului de Gaze Naturale

### Proiect: Analiza activității unei organizații utilizând Python

Acest proiect reprezintă o soluție avansată pentru monitorizarea și prognozarea consumului de gaze naturale. Aplicația corelează datele istorice de consum cu indicatori meteorologici (temperatură, nebulozitate) și utilizează algoritmi de tip **State-of-the-Art** pentru a genera previziuni precise pe o durată de 7 zile.

## 🎯 Obiectivul Proiectului

Implementarea unei aplicații interactive în Streamlit capabilă să execute întreg fluxul de analiză a datelor: de la curățare și prelucrare statistică, până la modelare predictivă complexă folosind **HistGradientBoostingRegressor** (implementarea optimizată a scikit-learn) și validare prin modele econometrice.

---

## 🛠️ Facilități Implementate (Conform Cerințelor)

Proiectul integrează următoarele funcționalități obligatorii și adiționale:

| Facilitate                               | Implementare în Proiect                                                                          | Status |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------- | :----: |
| **Metode Streamlit**               | Interfață dinamică: sidebar, metrici de consum, tab-uri și grafice interactive Plotly.        |   ✅   |
| **Tratare Valori Lipsă/Extreme**  | Detectarea anomaliilor în datele senzorilor și tratarea valorilor lipsă (imputare/eliminare). |   ✅   |
| **Codificarea Datelor**            | Transformarea variabilelor temporale (zile, luni, weekend) în vectori numerici.                 |   ✅   |
| **Metode de Scalare**              | Utilizarea StandardScaler/MinMaxScaler pentru normalizarea factorilor meteo.                     |   ✅   |
| **Grupare și Agregare (Pandas)**  | Analiza consumului mediu pe paliere de temperatură și ferestre calendaristice.                 |   ✅   |
| **Utilizarea Funcțiilor de Grup** | Aplicarea metodelor de tip groupby().transform() pentru normalizarea consumului.                 |   ✅   |
| **Scikit-learn (Clustering)**      | Segmentarea automată a zilelor în profiluri de consum (Optim/Critic) prin K-Means.             |   ✅   |
| **Statsmodels (Regr. Multiplă)**  | Analiza statistică a impactului temperaturii și a Cloud Cover asupra consumului.               |   ✅   |
| **Scikit-learn (Forecast)**        | Implementarea **HistGradientBoosting** pentru prognoza consumului pe 7 zile.                     |   ✅   |

---

## 📂 Structura Seturilor de Date

* **Sursa Datelor:** Date extrase din platforma Refinitiv.
* **Istoric Consum & Meteo:** Set de date cuprinzând consumul zilnic, temperatura medie și gradul de acoperire a norilor (Cloud Cover).
* **Date Forecast:** Date meteorologice prognozate pentru următoarele 7 zile, utilizate ca input pentru modelul de Gradient Boosting.

---

## 💻 Tehnologii Utilizate

* **Limbaj:** Python 3.10+
* **Modele Predictive:** 
    * **HistGradientBoostingRegressor** (Scikit-learn) pentru prognoză.
    * **K-Means** (Scikit-learn) pentru clustering.
* **Analiză Statistică:** Statsmodels, Pandas, NumPy.
* **Vizualizare:** Plotly (grafice interactive).
* **Web Framework:** Streamlit.

---

## 🚀 Instrucțiuni de Instalare și Rulare

1. **Clonarea proiectului:**
   git clone https://github.com/andreeasandu006/proiect_pachete.git
   cd proiect_pachete
2. **Instalarea dependențelor:**
   pip install -r requirements.txt
3. **Pornirea aplicației:**
   streamlit run app.py

---

## 👤 Autor

* **Nume:** Sandu Andreea-Mihaela, Rasmerita Andra-Elena
* **Grupa: 1094**
