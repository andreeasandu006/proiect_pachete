# proiect_pachete
# 📊 Analiza și Prognoza Consumului de Gaze Naturale
### Proiect: Analiza activității unei organizații utilizând Python și SAS

Acest proiect reprezintă o soluție integrată pentru monitorizarea, analiza și prognozarea consumului de resurse energetice (gaze naturale) ale unei organizații. Aplicația utilizează date istorice de consum corelate cu indicatori meteorologici (temperatură și grad de acoperire a norilor) pentru a genera previziuni pe o durată de 7 zile.

## 🎯 Obiectivul Proiectului
Implementarea unei aplicații interactive de tip Streamlit care să permită prelucrarea datelor brute, vizualizarea tendințelor și aplicarea modelelor de Machine Learning și Statistică pentru optimizarea procesului de extindere și planificare a activității organizației.

---

## 🛠️ Facilități Implementate (Conform Cerințelor)

Proiectul integrează următoarele funcționalități obligatorii:

| Cerință | Implementare în Proiect | Status |
| :--- | :--- | :---: |
| **Metode Streamlit** | Utilizarea elementelor de tip st.sidebar, st.metric, st.plotly_chart și layout-uri pe coloane. | ✅ |
| **Tratare Valori Lipsă/Extreme** | Gestionarea erorilor de înregistrare din senzori și eliminarea valorilor aberante prin metode statistice. | ✅ |
| **Codificarea Datelor** | Aplicarea LabelEncoder pentru variabilele temporale și categorice (ex: tipul zilei). | ✅ |
| **Metode de Scalare** | Utilizarea StandardScaler pentru normalizarea variabilelor de temperatură și consum. | ✅ |
| **Grupare și Agregare (Pandas)** | Calculul mediilor de consum pe categorii de temperatură și ferestre temporale. | ✅ |
| **Utilizarea Funcțiilor de Grup** | Aplicarea funcțiilor de tip groupby().agg() și transform() pentru analize granulare. | ✅ |
| **Scikit-learn (Clustering)** | Segmentarea zilelor de consum în 3 profiluri distincte folosind algoritmul K-Means. | ✅ |
| **Scikit-learn (Regresie Logistică)** | Clasificarea zilelor cu risc de depășire a pragului critic de consum energetic. | ✅ |
| **Statsmodels (Regresie Multiplă)** | Determinarea influenței directe a temperaturii și a nebulozității asupra volumului de gaz consumat. | ✅ |

---

## 📂 Structura Seturilor de Date
Analiza se bazează pe următoarele surse de date:
* **Istoric Consum & Meteo:** Date despre consumul zilnic, temperatura medie și cloud_cover.
* **Date Forecast:** Prognoza meteorologică pentru următoarele 7 zile utilizată pentru predicția consumului viitor.

---

## 💻 Tehnologii Utilizate
* **Limbaj principal:** Python 3.10+
* **Biblioteci Data Science:** Pandas, NumPy, Scikit-learn, Statsmodels
* **Vizualizare:** Matplotlib, Seaborn, Plotly
* **Web Framework:** Streamlit
* **Validare Statistică:** SAS (analiză complementară a activității organizațiilor)

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
* **Grupa:** 1094

---
*Proiect realizat pentru disciplina: Pachete software pentru analiza activității organizațiilor.*