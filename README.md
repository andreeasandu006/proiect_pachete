# 📊 Analiza și Prognoza Consumului de Gaze Naturale

### Proiect: Analiza activității unei organizații utilizând Python și SAS

Acest proiect reprezintă o soluție avansată pentru monitorizarea și prognozarea consumului de gaze naturale. Aplicația corelează datele istorice de consum cu indicatori meteorologici (temperatură, nebulozitate) și utilizează algoritmi de tip State-of-the-Art pentru a genera previziuni precise pe o durată de 7 zile.

## 🎯 Obiectivul Proiectului

Implementarea unei aplicații interactive în Streamlit capabilă să execute întreg fluxul de analiză a datelor: de la curățare și prelucrare statistică, până la modelare predictivă complexă folosind XGBoost și validare prin modele econometrice clasice.

---

## 🛠️ Facilități Implementate (Conform Cerințelor)

Proiectul integrează următoarele funcționalități obligatorii și adiționale:

| Facilitate                               | Implementare în Proiect                                                                         | Status |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------- | :----: |
| **Metode Streamlit**               | Interfață dinamică: sidebar, metrici de consum, tab-uri și grafice interactive Plotly.       |   ✅   |
| **Tratare Valori Lipsă/Extreme**  | Detectarea anomaliilor în datele senzorilor și tratarea valorilor lipsă (imputare/eliminare). |   ✅   |
| **Codificarea Datelor**            | Transformarea variabilelor temporale (zile, luni, weekend) în vectori numerici.                 |   ✅   |
| **Metode de Scalare**              | Utilizarea StandardScaler/MinMaxScaler pentru normalizarea factorilor meteo.                     |   ✅   |
| **Grupare și Agregare (Pandas)**  | Analiza consumului mediu pe paliere de temperatură și ferestre calendaristice.                 |   ✅   |
| **Utilizarea Funcțiilor de Grup** | Aplicarea metodelor de tip groupby().transform() pentru normalizarea consumului.                 |   ✅   |
| **Scikit-learn (Clustering)**      | Segmentarea automată a zilelor în profiluri de consum (Optim/Critic) prin K-Means.             |   ✅   |
| **Scikit-learn (Regresie Log.)**   | Predicția probabilității ca o zi viitoare să depășească pragul de consum critic.          |   ✅   |
| **Statsmodels (Regr. Multiplă)**  | Analiza statistică a impactului temperaturii și a Cloud Cover asupra consumului.               |   ✅   |
| **XGBoost (Forecast)**             | Implementarea modelului de Gradient Boosting pentru prognoza consumului pe 7 zile.               |   ✅   |

---

## 📂 Structura Seturilor de Date

* **Istoric Consum & Meteo:** Set de date cuprinzând consumul zilnic, temperatura medie și gradul de acoperire a norilor (Cloud Cover).
* **Date Forecast:** Date meteorologice prognozate pentru următoarele 7 zile, utilizate ca input pentru modelul XGBoost.

---

## 💻 Tehnologii Utilizate

* **Limbaj:** Python 3.10+
* **Modele Predictive:** XGBoost (pentru forecast), Scikit-learn (pentru clasificare și clustering)
* **Analiză Statistică:** Statsmodels, Pandas, NumPy
* **Vizualizare:** Plotly, Seaborn, Matplotlib
* **Web Framework:** Streamlit
* **Validare Externă:** SAS

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
