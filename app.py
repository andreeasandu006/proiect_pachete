# =============================================================================
# APLICATIE STREAMLIT - ANALIZA CONSUM GAZE NATURALE
# Flux complet: incarcare date → curatare → EDA → modelare → prognoza
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import holidays
import statsmodels.api as sm
from forecast_gas import train_and_forecast
import warnings
warnings.filterwarnings("ignore")

# Biblioteci scikit-learn pentru preprocesare, clustering si metrici
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score

# Diagnostic pentru regresia OLS din statsmodels
from statsmodels.stats.stattools import durbin_watson


# =============================================================================
# CONFIGURARE PAGINA STREAMLIT
# layout="wide" foloseste toata latimea ecranului pentru grafice mai clare
# =============================================================================
st.set_page_config(
    page_title="Sistem Expert Gaze Naturale",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS PERSONALIZAT
# Stilizeaza sidebar-ul, cardurile de metrici, taburile si butoanele
# =============================================================================
st.markdown("""
<style>
    /* Sidebar: fundal inchis cu gradient vertical */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #111827 100%);
        border-right: 1px solid #1f2a3e;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-weight: 500; padding: 0.4rem 0.5rem;
        border-radius: 0.5rem; transition: 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover { background: #1e293b; }

    /* Carduri metrice: fundal alb cu bordura colorata si efect hover */
    div[data-testid="metric-container"] {
        background: #ffffff; border-radius: 1rem; padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 4px solid #2c7da0; transition: 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px); box-shadow: 0 8px 18px rgba(0,0,0,0.1);
    }

    /* Tabele: colturi rotunjite si umbra subtila */
    .stDataFrame { border-radius: 0.75rem; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }

    /* Butoane: stil inchis cu tranzitie */
    .stButton button {
        background: #1e293b; border: none; border-radius: 2rem;
        padding: 0.4rem 1.2rem; font-weight: 500; transition: 0.2s;
    }
    .stButton button:hover { background: #2d3a4e; transform: scale(0.98); }

    /* Taburi: linie albastra sub tabul activ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem; background: transparent; border-bottom: 1px solid #1f2a3e;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0; padding: 0.5rem 1.2rem; font-weight: 500;
        background: transparent !important; color: #94a3b8 !important;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important; color: #e2e8f0 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* Spatiere generala a paginii */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    hr { margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNCTII AJUTATOARE
# =============================================================================

def add_holiday_features(index):
    """
    Genereaza trei coloane binare legate de sarbatorile legale din Romania:
      - is_holiday:        ziua curenta este sarbatoare
      - is_before_holiday: ziua urmatoare este sarbatoare (ajun)
      - is_after_holiday:  ziua precedenta a fost sarbatoare
    Consumul de gaze variaza tipic in jurul sarbatorilor legale.
    """
    ro_holidays = holidays.Romania()
    return pd.DataFrame({
        "is_holiday":        [1 if d in ro_holidays else 0 for d in index],
        "is_before_holiday": [1 if (d + pd.Timedelta(days=1)) in ro_holidays else 0 for d in index],
        "is_after_holiday":  [1 if (d - pd.Timedelta(days=1)) in ro_holidays else 0 for d in index],
    }, index=index)


@st.cache_data
def load_raw_data():
    """
    Incarca cele trei fisiere CSV din directorul 'date/':
      - gas_history.csv              : consum zilnic + temperatura + nebulozitate
      - gas_temp_forecast.csv        : prognoza temperatura pentru orizontul viitor
      - gas_cloud_cover_forecast.csv : prognoza nebulozitate pentru orizontul viitor

    Decoratorul @st.cache_data evita re-citirea fisierelor la fiecare interactiune,
    accelerand semnificativ aplicatia.
    """
    path = "date"
    df_h = pd.read_csv(os.path.join(path, "gas_history.csv"))
    df_t = pd.read_csv(os.path.join(path, "gas_temp_forecast.csv"))
    df_c = pd.read_csv(os.path.join(path, "gas_cloud_cover_forecast.csv"))

    # Parsare date cu format explicit pentru a evita ambiguitatile zz/ll/aa
    df_h["record_date"] = pd.to_datetime(df_h["record_date"], format="%d-%m-%y")
    df_t["target_date"] = pd.to_datetime(df_t["target_date"])
    df_c["target_date"] = pd.to_datetime(df_c["target_date"])
    return df_h, df_t, df_c


@st.cache_data
def clean_data(df_raw):
    """
    Pipeline complet de curatare si inginerie de caracteristici:

    1. Sortare cronologica si eliminare duplicate pe coloana de data
    2. Imputare valori lipsa cu mediana (robusta la outlieri)
    3. Detectare si eliminare outlieri prin metoda IQR (Tukey fences)
    4. Extragere caracteristici calendaristice: luna, zi saptamana, trimestru
    5. Codificare sezon prin LabelEncoder (sklearn)
    6. Adaugare caracteristici pentru sarbatori legale
    7. Scalare StandardScaler pe variabilele numerice principale
    8. Calcul functii de grup: medie lunara si abatere fata de medie
    9. Calcul HDD (Heating Degree Days) - indicator standard al nevoii de incalzire

    Returneaza:
      df        - DataFrame curatat si imbogatit cu caracteristici noi
      df_scaled - DataFrame cu variabilele numerice standardizate
      meta      - dictionar cu statistici intermediare pentru afisare in UI
    """
    df = df_raw.copy()

    # --- Pas 1: Sortare cronologica si deduplicare ---
    df = df.sort_values("record_date").drop_duplicates(subset="record_date").reset_index(drop=True)

    # --- Pas 2: Tratare valori lipsa ---
    # Mediana este preferata mediei deoarece este robusta la valorile extreme
    missing_before = df.isnull().sum()
    df = df.fillna(df.median(numeric_only=True))
    missing_after = df.isnull().sum()

    # --- Pas 3: Detectare si eliminare outlieri (metoda IQR) ---
    # Limitele Tukey: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    n_before = len(df)
    Q1, Q3 = df["gas_consumption"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[
        (df["gas_consumption"] >= Q1 - 1.5 * IQR) &
        (df["gas_consumption"] <= Q3 + 1.5 * IQR)
    ].reset_index(drop=True)
    n_after = len(df)

    # --- Pas 4: Caracteristici calendaristice ---
    df["luna"]      = df["record_date"].dt.month        # 1-12
    df["zi_sapt"]   = df["record_date"].dt.dayofweek    # 0=Luni, 6=Duminica
    df["trimestru"] = df["record_date"].dt.quarter      # 1-4

    # --- Pas 5: Codificare sezon cu LabelEncoder ---
    # LabelEncoder transforma etichetele text in valori numerice intregi (0,1,2,3)
    le = LabelEncoder()
    sezon_labels = df["luna"].map({
        12: "Iarna", 1: "Iarna", 2: "Iarna",
        3: "Primavara", 4: "Primavara", 5: "Primavara",
        6: "Vara", 7: "Vara", 8: "Vara",
        9: "Toamna", 10: "Toamna", 11: "Toamna",
    })
    df["sezon_enc"] = le.fit_transform(sezon_labels)

    # --- Pas 6: Caracteristici sarbatori legale romanesti ---
    hols = add_holiday_features(pd.DatetimeIndex(df["record_date"]))
    hols.index = df.index
    df = pd.concat([df, hols], axis=1)

    # --- Pas 7: Scalare StandardScaler ---
    # Formula: z = (x - medie) / abatere_standard  →  medie=0, std=1
    # Esential pentru algoritmi bazati pe distante (KMeans, SVM, etc.)
    num_cols = ["gas_consumption", "temp_daily", "cloud_cover_daily"]
    scaler = StandardScaler()
    df_scaled_arr = scaler.fit_transform(df[num_cols])
    df_scaled = pd.DataFrame(df_scaled_arr, columns=[c + "_scaled" for c in num_cols])

    # --- Pas 8: Functii de grup (groupby + transform) ---
    # transform("mean") returneaza o serie aliniata cu df-ul original,
    # permitand adaugarea directa ca coloana noua (fara merge)
    df["medie_lunara"]   = df.groupby("luna")["gas_consumption"].transform("mean")
    df["abatere_lunara"] = df["gas_consumption"] - df["medie_lunara"]

    # --- Pas 9: Heating Degree Days (HDD) ---
    # HDD = max(18 - temp, 0) — indicator standard pentru nevoia de incalzire
    # Cu cat temperatura scade sub pragul de 18°C, cu atat creste nevoia de incalzire
    df["hdd"] = np.maximum(18 - df["temp_daily"], 0)

    # Pachet cu metadate utile pentru afisare in interfata utilizatorului
    meta = {
        "missing_before": missing_before,
        "missing_after":  missing_after,
        "n_before":       n_before,
        "n_after":        n_after,
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "scaler":   scaler,
        "num_cols": num_cols,
    }
    return df, df_scaled, meta


# =============================================================================
# INCARCARE SI CURATARE DATE
# Apelate o singura data la pornirea aplicatiei (rezultatele sunt in cache)
# =============================================================================
df_raw, df_t, df_c = load_raw_data()
df, df_scaled, meta = clean_data(df_raw)


# =============================================================================
# SIDEBAR - NAVIGARE SI INFORMATII RAPIDE
# =============================================================================
with st.sidebar:
    st.markdown("Analiza Consum Gaze Naturale")
    st.markdown("---")

    # Meniu principal de navigare — fiecare optiune = un pas din fluxul de analiza
    choice = st.radio(
        "Navigare:",
        [
            "Prezentare",
            "Valori Lipsa si Outlieri",
            "Codificare si Scalare",
            "EDA - Statistici si Grafice",
            "Modele ML si Stats",
            "Forecast HGBR",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Buton pentru invalidarea cache-ului (util la actualizarea fisierelor de date)
    if st.button("Reset Cache"):
        st.cache_data.clear()
        st.rerun()

    # Informatii sumare despre setul de date curatat
    st.caption(f"Perioada: {df['record_date'].min().date()} → {df['record_date'].max().date()}")
    st.caption(f"Inregistrari curate: {len(df):,}")


# =============================================================================
# PAGINA 1: PREZENTARE GENERALA
# =============================================================================
if choice == "Prezentare":

    # Titlu cu gradient CSS inline — compatibil cu toate temele Streamlit
    # Nota: clasa CSS separata nu functioneaza consistent; stilul inline este mai robust
    st.markdown(
        """
        <h1 style="
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            background: linear-gradient(90deg, #3b82f6, #60a5fa, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">
         Analiza Consum Gaze Naturale
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Flux complet: EDA → Prelucrare → Modelare → Prognoza")

    st.info(
        "Aplicatie interactiva care parcurge intregul flux de analiza a consumului de gaze naturale: "
        "de la inspectia valorilor lipsa si tratarea outlierilor, codificare si scalare, "
        "pana la modele ML (KMeans, HGBR) si prognoze recursive cu date meteo."
    )

    # Tabel cu toate facilitatile implementate
    st.subheader("Facilitati implementate")
    fac = pd.DataFrame({
        "Nr.": list(range(1, 10)),
        "Facilitate": [
            "Metode Streamlit interactive (tabs, slider, selectbox, metric, cache)",
            "Detectare & eliminare valori lipsa (imputare cu mediana)",
            "Detectare & eliminare outlieri (metoda IQR / Tukey fences)",
            "Codificare variabile calendaristice (LabelEncoder, ordinal, binar)",
            "Scalare numerica (StandardScaler → medie=0, std=1)",
            "Grupare si agregare lunara (groupby + agg)",
            "Functii de grup (transform pentru medie si abatere lunara)",
            "Statsmodels — Regresie OLS multipla + diagnostic complet reziduuri",
            "Scikit-learn — Clusterizare KMeans + Forecast HGBR",
        ],
        "Status": ["✅"] * 9,
    })
    st.dataframe(fac, use_container_width=True, hide_index=True)

    # Metrici sumare despre seturile de date disponibile
    st.subheader("Date disponibile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zile brute",          len(df_raw))
    c2.metric("Zile dupa curatare",  len(df), delta=f"-{len(df_raw)-len(df)}")
    c3.metric("Zile forecast temp.", len(df_t))
    c4.metric("Zile forecast nori",  len(df_c))

    st.markdown("---")
    st.markdown(
        """
        **Structura aplicatiei:**
        - **Pasul 1** – Valori lipsa si outlieri: vizualizare si tratare
        - **Pasul 2** – Codificare si scalare: preprocesare pentru modele ML
        - **Pasul 3** – EDA: statistici descriptive, agregate si corelatii
        - **Pasul 4** – Modele ML si Stats: regresie OLS + clusterizare KMeans
        - **Pasul 5** – Forecast: prognoza recursiva cu HistGradientBoosting
        """
    )


# =============================================================================
# PAGINA 2: VALORI LIPSA SI OUTLIERI
# =============================================================================
elif choice == "Valori Lipsa si Outlieri":
    st.header("Pasul 1 – Valori Lipsa si Outlieri")

    # ── Sectiunea 1.1: Valori lipsa ─────────────────────────────────────────
    st.markdown("### 1.1 Analiza valorilor lipsa")
    st.markdown(
        "Valorile lipsa sunt identificate per coloana si imputate cu **mediana** — "
        "alegere robusta fata de medie deoarece nu este influentata de valorile extreme."
    )

    nan_df = pd.DataFrame({
        "Coloana":         meta["missing_before"].index,
        "Lipsa (inainte)": meta["missing_before"].values,
        "Lipsa (dupa)":    meta["missing_after"].values,
        "% lipsa":         (meta["missing_before"] / len(df_raw) * 100).round(2).values,
    })

    col_a, col_b = st.columns([1, 1.2])
    with col_a:
        st.dataframe(nan_df, use_container_width=True, hide_index=True)
    with col_b:
        fig_nan = px.bar(
            nan_df, x="Coloana", y="% lipsa",
            color="% lipsa", color_continuous_scale="Reds",
            title="Procent valori lipsa per coloana", text_auto=True,
        )
        fig_nan.update_layout(height=380, margin=dict(l=0, r=0))
        st.plotly_chart(fig_nan, use_container_width=True)

    st.divider()

    # ── Sectiunea 1.2: Outlieri ──────────────────────────────────────────────
    st.markdown("### 1.2 Detectarea si eliminarea outlierilor")
    st.markdown(
        "Se aplica metoda **IQR (Interquartile Range)** — Tukey fences: "
        "valorile in afara intervalului `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` sunt considerate outlieri si eliminate."
    )

    col1, col2 = st.columns(2)
    with col1:
        iqr_info = pd.DataFrame({
            "Indicator": ["Q1 (25%)", "Q3 (75%)", "IQR", "Limita inferioara", "Limita superioara"],
            "Valoare (MWh)": [
                f"{meta['Q1']:.2f}", f"{meta['Q3']:.2f}", f"{meta['IQR']:.2f}",
                f"{meta['Q1'] - 1.5*meta['IQR']:.2f}",
                f"{meta['Q3'] + 1.5*meta['IQR']:.2f}",
            ],
        })
        st.dataframe(iqr_info, use_container_width=True, hide_index=True)
        st.metric(
            "Inregistrari eliminate",
            meta["n_before"] - meta["n_after"],
            delta=f"-{((meta['n_before']-meta['n_after'])/meta['n_before']*100):.1f}%",
        )
    with col2:
        # Box plot comparativ: date brute vs date curatate
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df_raw["gas_consumption"].dropna(), name="Brut",
            marker_color="#f97316",
        ))
        fig_box.add_trace(go.Box(
            y=df["gas_consumption"], name="Curatat",
            marker_color="#22c55e",
        ))
        fig_box.update_layout(
            title="Distributia consumului: inainte vs dupa eliminarea outlierilor",
            height=420,
        )
        st.plotly_chart(fig_box, use_container_width=True)


# =============================================================================
# PAGINA 3: CODIFICARE SI SCALARE
# =============================================================================
elif choice == "Codificare si Scalare":
    st.header("Pasul 2 – Codificare si Scalare")

    tab1, tab2 = st.tabs(["Codificare variabile", "Scalare numerica"])

    # ── Tab 1: Codificare ────────────────────────────────────────────────────
    with tab1:
        st.markdown(
            "Variabilele calendaristice sunt transformate in reprezentari numerice "
            "pentru a putea fi folosite de algoritmii de machine learning."
        )
        enc_info = pd.DataFrame({
            "Variabila originala": ["luna", "zi_sapt", "trimestru", "sezon", "sarbatoare"],
            "Tip codificare":      ["Ordinal (1-12)", "Ordinal (0-6)", "Ordinal (1-4)", "Label Encoding", "Binar (0/1)"],
            "Descriere": [
                "Luna calendaristica",
                "Ziua saptamanii (0=Luni, 6=Duminica)",
                "Trimestrul anului",
                "Iarna/Primavara/Vara/Toamna → 0/1/2/3",
                "1 daca e sarbatoare legala, 0 altfel",
            ],
        })
        st.dataframe(enc_info, use_container_width=True, hide_index=True)

        colA, colB = st.columns(2)
        with colA:
            # Pie chart cu etichete reale ale sezoanelor (nu coduri numerice)
            sezon_map = {
                12: "Iarna", 1: "Iarna", 2: "Iarna",
                3: "Primavara", 4: "Primavara", 5: "Primavara",
                6: "Vara", 7: "Vara", 8: "Vara",
                9: "Toamna", 10: "Toamna", 11: "Toamna",
            }
            df_pie = df.copy()
            df_pie["sezon_label"] = df_pie["luna"].map(sezon_map)
            fig_sezon = px.pie(
                df_pie, names="sezon_label",
                title="Distributia zilelor pe sezon",
                hole=0.4,
                color="sezon_label",
                color_discrete_map={
                    "Iarna":    "#3b82f6",
                    "Primavara":"#22c55e",
                    "Vara":     "#f97316",
                    "Toamna":   "#a855f7",
                },
            )
            fig_sezon.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_sezon, use_container_width=True)

        with colB:
            # Bar chart consum mediu lunar pentru un an selectabil cu slider
            ani_disponibili = sorted(df["record_date"].dt.year.unique().tolist())
            an_selectat = st.select_slider(
                "Selecteaza anul pentru vizualizare:",
                options=ani_disponibili,
                value=ani_disponibili[0],
            )
            df_an = df[df["record_date"].dt.year == an_selectat].copy()
            lunar = df_an.groupby("luna")["gas_consumption"].mean().reset_index()
            lunar.columns = ["Luna", "Consum mediu (MWh)"]
            nume_luni = {
                1:"Ian", 2:"Feb", 3:"Mar", 4:"Apr", 5:"Mai", 6:"Iun",
                7:"Iul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
            }
            lunar["Luna"] = lunar["Luna"].map(nume_luni)
            fig_bar_an = px.bar(
                lunar, x="Luna", y="Consum mediu (MWh)",
                title=f"Consum mediu lunar – {an_selectat}",
                color="Consum mediu (MWh)",
                color_continuous_scale="blues",
                text_auto=".0f",
            )
            fig_bar_an.update_layout(
                height=370,
                coloraxis_showscale=False,
                xaxis=dict(
                    categoryorder="array",
                    categoryarray=list(nume_luni.values()),
                ),
            )
            fig_bar_an.update_traces(textposition="outside")
            st.plotly_chart(fig_bar_an, use_container_width=True)

    # ── Tab 2: Scalare ───────────────────────────────────────────────────────
    with tab2:
        st.markdown(
            "**StandardScaler** (sklearn) standardizeaza fiecare variabila: "
            "`z = (x − μ) / σ`, astfel incat **media devine 0** si **abaterea standard devine 1**. "
            "Acest pas este esential pentru algoritmii sensibili la scala (ex: KMeans, regresie logistica)."
        )
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Inainte de scalare – statistici descriptive**")
            st.dataframe(df[meta["num_cols"]].describe().round(2), use_container_width=True)
        with col_s2:
            st.markdown("**Dupa scalare (StandardScaler) – statistici descriptive**")
            st.dataframe(df_scaled.describe().round(4), use_container_width=True)

        st.info(
            "Observatie: dupa scalare, media fiecarei variabile este ~0 si "
            "abaterea standard este ~1, indiferent de unitatea de masura initiala."
        )
        st.divider()

        # Comparatie histograme inainte / dupa scalare pentru variabila selectata
        st.markdown("**Distributia variabilelor inainte si dupa scalare**")
        var_selectata = st.selectbox(
            "Selecteaza variabila pentru comparatie:",
            options=meta["num_cols"],
            format_func=lambda x: {
                "gas_consumption":   "Consum gaze (MWh)",
                "temp_daily":        "Temperatura zilnica (°C)",
                "cloud_cover_daily": "Nebulozitate zilnica (%)",
            }.get(x, x),
        )
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_before = px.histogram(
                df, x=var_selectata, nbins=40,
                title="Inainte de scalare",
                color_discrete_sequence=["#3b82f6"],
                labels={var_selectata: var_selectata},
            )
            fig_before.update_layout(height=350, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_before, use_container_width=True)
        with col_v2:
            scaled_col = var_selectata + "_scaled"
            fig_after = px.histogram(
                df_scaled, x=scaled_col, nbins=40,
                title="Dupa scalare (valori standardizate)",
                color_discrete_sequence=["#22c55e"],
                labels={scaled_col: "valoare standardizata (z-score)"},
            )
            fig_after.update_layout(height=350, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_after, use_container_width=True)


# =============================================================================
# PAGINA 4: EDA - EXPLORAREA DATELOR
# =============================================================================
elif choice == "EDA - Statistici si Grafice":
    st.header("Pasul 3 – Explorarea Datelor (EDA)")
    st.markdown(
        "Analiza exploratorie evidentiaza distributia consumului, sezonalitatea, "
        "si corelatiile cu factorii meteo (temperatura, nebulozitate)."
    )

    tabs = st.tabs([
        "Statistici descriptive",
        "Agregare lunara",
        "Temperatura vs Consum",
        "Nebulozitate vs Consum",
    ])

    # ── Tab 1: Statistici descriptive + serie temporala ─────────────────────
    with tabs[0]:
        st.markdown("**Statistici descriptive pentru variabilele principale**")
        st.dataframe(
            df[["gas_consumption", "temp_daily", "cloud_cover_daily", "hdd"]].describe().round(2),
            use_container_width=True,
        )
        st.markdown("**Serie temporala – consum zilnic gaze naturale**")
        df_sorted = df.sort_values("record_date")
        fig_ts = px.line(
            df_sorted, x="record_date", y="gas_consumption",
            title="Evolutia consumului zilnic de gaze naturale",
            template="plotly_white",
            color_discrete_sequence=["#3b82f6"],
        )
        fig_ts.update_layout(height=450, xaxis_title="Data", yaxis_title="Consum (MWh)")
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Tab 2: Agregare lunara cu groupby + agg ──────────────────────────────
    with tabs[1]:
        st.markdown(
            "Consumul este **agregat lunar** folosind `groupby + agg(['mean', 'std'])`. "
            "Barele de eroare reprezinta abaterea standard, indicand variabilitatea din fiecare luna."
        )
        # Agregare cu doua functii simultan: medie si abatere standard
        monthly = df.groupby("luna")["gas_consumption"].agg(["mean", "std"]).reset_index()

        # Paleta de culori sezonala pentru lunile calendaristice
        culori_luna = {
            1:"#1d4ed8", 2:"#2563eb", 3:"#16a34a",
            4:"#22c55e", 5:"#84cc16", 6:"#f97316",
            7:"#ef4444", 8:"#f97316", 9:"#a855f7",
            10:"#7c3aed", 11:"#4f46e5", 12:"#1d4ed8",
        }
        monthly["culoare"] = monthly["luna"].map(culori_luna)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=monthly["luna"], y=monthly["mean"],
            error_y=dict(type="data", array=monthly["std"], visible=True, color="rgba(255,255,255,0.4)"),
            marker_color=monthly["culoare"].tolist(),
            text=monthly["mean"].round(0).astype(int),
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Consum mediu lunar cu bare de eroare (abatere standard)",
            xaxis=dict(
                title="Luna", tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=["Ian","Feb","Mar","Apr","Mai","Iun",
                          "Iul","Aug","Sep","Oct","Nov","Dec"],
            ),
            yaxis_title="Consum mediu (MWh)",
            height=480, template="plotly_white", showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tab 3: Corelatie Temperatura vs Consum ───────────────────────────────
    with tabs[2]:
        st.markdown(
            "Scatter plot colorat pe sezon cu **trendline LOWESS** (non-parametric). "
            "Se observa clar relatia inversa: cu cat temperatura scade, cu atat consumul creste."
        )
        sezon_map = {
            12:"Iarna",1:"Iarna",2:"Iarna",
            3:"Primavara",4:"Primavara",5:"Primavara",
            6:"Vara",7:"Vara",8:"Vara",
            9:"Toamna",10:"Toamna",11:"Toamna",
        }
        df_sc = df.copy()
        df_sc["Sezon"] = df_sc["luna"].map(sezon_map)
        fig_scatter = px.scatter(
            df_sc, x="temp_daily", y="gas_consumption",
            color="Sezon",
            color_discrete_map={
                "Iarna":"#3b82f6", "Primavara":"#22c55e",
                "Vara":"#f97316",  "Toamna":"#a855f7",
            },
            trendline="lowess",
            title="Corelatie Temperatura – Consum (colorat pe sezon)",
            labels={"temp_daily":"Temperatura (°C)", "gas_consumption":"Consum (MWh)"},
            opacity=0.6,
        )
        fig_scatter.update_layout(height=480, template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Tab 4: Corelatie Nebulozitate vs Consum ──────────────────────────────
    with tabs[3]:
        st.markdown(
            "Scatter plot nebulozitate vs consum, colorat dupa temperatura. "
            "Nebulozitatea singura explica putine din variatia consumului — "
            "efectul dominant ramane temperatura."
        )
        fig_cloud = px.scatter(
            df, x="cloud_cover_daily", y="gas_consumption",
            color="temp_daily",
            color_continuous_scale="RdBu_r",
            trendline="lowess",
            title="Corelatie Nebulozitate – Consum (colorat dupa temperatura)",
            labels={
                "cloud_cover_daily": "Nebulozitate (%)",
                "gas_consumption":   "Consum (MWh)",
                "temp_daily":        "Temperatura (°C)",
            },
            opacity=0.6,
        )
        fig_cloud.update_layout(height=480, template="plotly_white")
        st.plotly_chart(fig_cloud, use_container_width=True)
        st.info(
            "Nebulozitatea are o corelatie slaba cu consumul (p-value > 0.05 in modelul OLS). "
            "Colorarea dupa temperatura confirma ca factorul principal ramane temperatura."
        )


# =============================================================================
# PAGINA 5: MODELE ML SI STATISTICE
# =============================================================================
elif choice == "Modele ML si Stats":
    st.header("Pasul 4 – Modelare si Diagnostic")

    tab_ols, tab_kmeans = st.tabs([
        "Regresie OLS (statsmodels)",
        "Clusterizare KMeans (sklearn)",
    ])

    # ── TAB 1: Regresie OLS (statsmodels) ───────────────────────────────────
    with tab_ols:
        st.markdown(
            "Regresia **OLS (Ordinary Least Squares)** estimeaza relatia liniara dintre "
            "variabilele independente (temperatura, nebulozitate) si consumul de gaze. "
            "Se minimizeaza suma patratelor reziduurilor: `min Σ(y - ŷ)²`."
        )

        # Constructia matricei de design X cu constanta adaugata (intercept)
        X_ols = sm.add_constant(df[["temp_daily", "cloud_cover_daily"]])
        res   = sm.OLS(df["gas_consumption"], X_ols).fit()

        coef_df = pd.DataFrame({
            "Variabila":    ["Intercept", "Temperatura", "Nebulozitate"],
            "Coeficient":   res.params.values.round(4),
            "p-value":      res.pvalues.values.round(4),
            "Semnificatie": ["✅ Da (p<0.05)" if p < 0.05 else "❌ Nu" for p in res.pvalues],
        })
        st.subheader("Coeficienti estimati")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        # Metrici de calitate ale modelului OLS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²",               f"{res.rsquared:.4f}")
        c2.metric("R² ajustat",       f"{res.rsquared_adj:.4f}")
        dw = durbin_watson(res.resid)
        c3.metric("Durbin-Watson",    f"{dw:.3f}", help="Valoare ideala ~2 (fara autocorelare)")
        c4.metric("p-value (F-stat)", f"{res.f_pvalue:.2e}")

        st.info(
            "**Interpretare:** R² indica proportia variantei consumului explicata de model. "
            "Durbin-Watson aproape de 2 sugereaza absenta autocorelarii reziduurilor — "
            "ipoteza importanta in regresia OLS clasica."
        )

        st.divider()
        st.subheader("Diagnostic reziduuri OLS")
        st.markdown(
            "Reziduurile `e = y - ŷ` trebuie sa fie **aleatoare, centrate in 0** "
            "si **distribuite normal** pentru ca ipotezele OLS sa fie respectate."
        )

        rezid  = res.resid.values
        fitted = res.fittedvalues.values

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            # Reziduuri vs Valori estimate — detecteaza heterocedasticitate
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(
                x=fitted, y=rezid, mode="markers",
                marker=dict(color="#3b82f6", size=4, opacity=0.5),
                name="Reziduuri",
            ))
            fig_rv.add_hline(y=0, line_dash="dash", line_color="#f97316", line_width=1.5)
            fig_rv.update_layout(
                title="Reziduuri vs Valori Estimate",
                xaxis_title="Valori estimate (MWh)", yaxis_title="Reziduuri",
                height=400, template="plotly_white",
            )
            st.plotly_chart(fig_rv, use_container_width=True)

        with col_r2:
            # Histograma reziduuri — verifica normalitatea distributiei erorilor
            fig_rh = px.histogram(
                x=rezid, nbins=40,
                title="Distributia Reziduurilor (ar trebui ~Normala)",
                labels={"x": "Reziduuri", "y": "Frecventa"},
                color_discrete_sequence=["#3b82f6"],
            )
            fig_rh.update_layout(height=400, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_rh, use_container_width=True)

        # Reziduuri in timp — detecteaza autocorelare sau structura temporala reziduala
        fig_rt = go.Figure()
        fig_rt.add_trace(go.Scatter(
            x=df.sort_values("record_date")["record_date"], y=rezid,
            mode="lines", line=dict(color="#3b82f6", width=1), name="Reziduuri",
        ))
        fig_rt.add_hline(y=0, line_dash="dash", line_color="#f97316", line_width=1.5)
        fig_rt.update_layout(
            title="Reziduuri in Timp (structura temporala ramasa)",
            xaxis_title="Data", yaxis_title="Reziduuri",
            height=350, template="plotly_white",
        )
        st.plotly_chart(fig_rt, use_container_width=True)

    # ── TAB 2: Clusterizare KMeans (sklearn) ─────────────────────────────────
    with tab_kmeans:
        st.markdown(
            "**KMeans** este un algoritm de clusterizare nesupervizata — grupeaza zilele in `k` "
            "categorii similare, fara etichete predefinite. Fiecare cluster reprezinta un "
            "profil tipic de zi (ex: 'zi de iarna cu consum mare', 'zi de vara cu consum mic'). "
            "Algoritmul minimizeaza **inertia** (suma distantelor patrate fata de centroid)."
        )

        # Slider interactiv: utilizatorul alege numarul de clustere
        n_clusters = st.slider(
            "Numarul de clustere (k) — modifica si graficele se actualizeaza automat:",
            min_value=2, max_value=6, value=3, step=1,
        )

        # Pregatire date: temperatura + consum ca variabile de clustering
        # Scalarea este OBLIGATORIE pentru KMeans — algoritm bazat pe distante euclidiene
        X_km = df[["temp_daily", "gas_consumption"]].dropna()
        scaler_km = StandardScaler()
        X_km_scaled = scaler_km.fit_transform(X_km)

        # Antrenare KMeans cu n_init=10 (ruleaza de 10 ori, pastreaza cel mai bun rezultat)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_km_scaled)

        df_km = X_km.copy()
        df_km["Cluster"] = [f"Cluster {l+1}" for l in labels]

        # Metrici de calitate a clusterizarii
        sil     = silhouette_score(X_km_scaled, labels)
        inertia = kmeans.inertia_

        cm1, cm2, cm3 = st.columns(3)
        cm1.metric(
            "Silhouette Score", f"{sil:.4f}",
            help="Intre -1 si 1. Aproape de 1 = clustere bine separate si compacte.",
        )
        cm2.metric(
            "Inertie (WCSS)", f"{inertia:.1f}",
            help="Suma distantelor patrate fata de centroid. Mai mica = clustere mai compacte.",
        )
        cm3.metric("Nr. clustere selectate", n_clusters)

        st.divider()

        col_k1, col_k2 = st.columns([1.6, 1])
        with col_k1:
            # Scatter principal: fiecare punct = o zi, colorat pe cluster
            fig_km = px.scatter(
                df_km, x="temp_daily", y="gas_consumption",
                color="Cluster",
                title=f"Clustere KMeans (k={n_clusters}): Temperatura vs Consum",
                labels={"temp_daily": "Temperatura (°C)", "gas_consumption": "Consum (MWh)"},
                opacity=0.65,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            # Adauga centroizii reali (inversam scalarea pentru a reveni la unitatile originale)
            centroids_orig = scaler_km.inverse_transform(kmeans.cluster_centers_)
            fig_km.add_trace(go.Scatter(
                x=centroids_orig[:, 0], y=centroids_orig[:, 1],
                mode="markers",
                marker=dict(symbol="x", size=16, color="black", line=dict(width=2)),
                name="Centroizi",
            ))
            fig_km.update_layout(height=460, template="plotly_white")
            st.plotly_chart(fig_km, use_container_width=True)

        with col_k2:
            # Tabel cu statistici agregate per cluster (groupby pe coloana Cluster)
            st.markdown("**Statistici per cluster**")
            stats_km = df_km.groupby("Cluster").agg(
                Nr_zile=("gas_consumption", "count"),
                Temp_medie=("temp_daily", "mean"),
                Consum_mediu=("gas_consumption", "mean"),
            ).round(2).reset_index()
            stats_km.columns = ["Cluster", "Nr. zile", "Temp. medie (°C)", "Consum mediu (MWh)"]
            st.dataframe(stats_km, use_container_width=True, hide_index=True)

            # Pie chart: distributia zilelor pe clustere
            fig_pie_km = px.pie(
                stats_km, names="Cluster", values="Nr. zile",
                title="Distributia zilelor pe cluster",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_pie_km.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie_km.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_pie_km, use_container_width=True)

        st.divider()

        # Metoda Elbow: ruleaza KMeans pentru k=2..8 si traseaza inertia
        st.markdown("**Metoda Elbow – alegerea numarului optim de clustere**")
        st.markdown(
            "Se ruleaza KMeans pentru mai multe valori de k si se traseaza inertia. "
            "Numarul optim de clustere se afla la **codul cotului** — punctul unde "
            "scaderea inertiei incetineste brusc (randament marginal descrescator)."
        )
        inertii = []
        k_range = range(2, 9)
        for k in k_range:
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_tmp.fit(X_km_scaled)
            inertii.append(km_tmp.inertia_)

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=inertii,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=8, color="#3b82f6"),
            name="Inertie",
        ))
        # Linie verticala marcand k-ul ales curent de utilizator
        fig_elbow.add_vline(
            x=n_clusters, line_dash="dash",
            line_color="#f97316", line_width=1.5,
            annotation_text=f"k selectat = {n_clusters}",
            annotation_position="top right",
        )
        fig_elbow.update_layout(
            title="Elbow Method – Inertie (WCSS) vs Numar Clustere",
            xaxis_title="Numar clustere (k)",
            yaxis_title="Inertie (WCSS)",
            height=380, template="plotly_white",
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.info(
            "**Interpretare:** Silhouette Score confirma calitatea separarii — valori peste 0.5 "
            "indica clustere bine definite. Metoda Elbow ajuta la identificarea vizuala a lui k optim."
        )


# =============================================================================
# PAGINA 6: FORECAST HGBR
# =============================================================================
elif choice == "Forecast HGBR":
    st.header("Pasul 5 – Prognoza Recursiva HGBR")
    st.markdown(
        "**HistGradientBoostingRegressor** (sklearn) este un model de gradient boosting "
        "optimizat pentru seturi mari de date. Prognoza este **recursiva**: "
        "predictia zilei `t+1` devine input pentru ziua `t+2`, si asa mai departe, "
        "pe intregul orizont de prognoza."
    )

    with st.spinner("Se antreneaza modelul si se genereaza prognoza... ⏳"):
        forecast_df, model, X_train, y_train, feature_names = train_and_forecast()
        y_pred_train = model.predict(X_train)

        # Calcul metrici de performanta pe setul de antrenare
        r2   = r2_score(y_train, y_pred_train)
        mae  = mean_absolute_error(y_train, y_pred_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # Afisare metrici principale ale modelului
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("R² (train)",       f"{r2:.4f}",       help="Proportia variantei explicate de model")
    col_m2.metric("MAE",              f"{mae:.2f} MWh",  help="Eroarea absoluta medie")
    col_m3.metric("RMSE",             f"{rmse:.2f} MWh", help="Radacina erorii patrate medii")
    col_m4.metric("Orizont prognoza", f"{len(forecast_df)} zile")

    # Grafic: ultimele 30 zile din istoric + prognoza viitoare
    # Continuitatea vizuala ajuta la evaluarea coerentei prognozei
    df_sorted    = df.sort_values("record_date")
    istoric_tail = df_sorted.tail(30)

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(
        x=istoric_tail["record_date"],
        y=istoric_tail["gas_consumption"],
        name="Istoric (ultimele 30 zile)",
        line=dict(color="#3b82f6", width=2),
    ))
    fig_f.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["forecast"],
        name="Prognoza HGBR",
        line=dict(color="#f97316", width=3, dash="dot"),
        mode="lines+markers",
        marker=dict(size=7),
    ))
    fig_f.update_layout(
        title="Evolutie consum – Istoric vs Prognoza HGBR",
        xaxis_title="Data", yaxis_title="Consum (MWh)",
        height=480, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # Importanta variabilelor (feature importance din HGBR)
    if hasattr(model, "feature_importances_"):
        st.markdown("**Importanta variabilelor in modelul HGBR**")
        st.markdown(
            "Feature importance indica cat de mult contribuie fiecare variabila "
            "la predictiile modelului. Valorile sunt normalizate (suma = 1)."
        )
        fi = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        fi = fi.sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            fi.tail(15), x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="blues",
            title="Top 15 variabile dupa importanta (HGBR)",
        )
        fig_fi.update_layout(height=420)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Comparatie Real vs Estimat pe setul de antrenare
    st.markdown("**Comparatie Real vs Estimat pe setul de antrenare**")
    st.markdown(
        "Suprapunerea celor doua serii indica cat de bine a invatat modelul "
        "tiparul consumului din datele istorice."
    )
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(
        x=list(range(len(y_train))), y=y_train.values,
        name="Consum real", opacity=0.6, line=dict(color="#3b82f6", width=1),
    ))
    fig_fit.add_trace(go.Scatter(
        x=list(range(len(y_pred_train))), y=y_pred_train,
        name="Consum estimat (HGBR)", opacity=0.8,
        line=dict(color="#f97316", width=2, dash="dot"),
    ))
    fig_fit.update_layout(
        title="Real vs Estimat – Set de antrenare",
        xaxis_title="Index esantion", yaxis_title="Consum (MWh)", height=420,
    )
    st.plotly_chart(fig_fit, use_container_width=True)

    # Tabel detaliat cu valorile prognozate zilnic
    st.subheader("📋 Tabel prognoza consum zilnic")
    df_show = forecast_df.reset_index()
    df_show.columns = ["Data", "Consum Prognozat (MWh)"]
    df_show["Consum Prognozat (MWh)"] = df_show["Consum Prognozat (MWh)"].round(2)
    st.dataframe(df_show, use_container_width=True, hide_index=True, height=400)