import os
from pathlib import Path
import html

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 0) PAGE + THEME
# =========================
st.set_page_config(
    page_title="HAVI – Product Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #f3f4f6 0, #e5e7eb 40%, #d1d5db 100%);
        color: #111827;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #0f172a;
        margin-top: 1.4rem;
        margin-bottom: 0.35rem;
        border-left: 6px solid #22c55e;
        padding-left: 0.75rem;
    }
    .section-subtitle {
        color: #334155;
        font-size: 0.92rem;
        margin-bottom: 1.0rem;
        opacity: 0.95;
    }
    button[data-baseweb="tab"] {
        font-weight: 650;
        border-radius: 999px !important;
        padding: 0.35rem 0.9rem;
        margin-right: 0.35rem;
        color: #e2e8f0;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.35);
        transition: all 0.2s ease-in-out;
    }
    button[data-baseweb="tab"]:hover {
        border-color: #60a5fa;
        background: rgba(30, 64, 175, 0.85);
        color: #f9fafb;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #1d4ed8, #22c55e);
        color: #f9fafb;
        border-color: transparent;
        box-shadow: 0 12px 30px rgba(34, 197, 94, 0.35);
    }
    .kpi-grid{
        display:flex;
        gap: 10px;
        width: 100%;
        align-items: stretch;
        flex-wrap: wrap;
    }
    .kpi-card{
        flex: 1 1 220px;
        background: rgba(255,255,255,0.60);
        border: 1px solid rgba(15, 23, 42, 0.10);
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 112px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
    }
    .kpi-label{
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.15;
        white-space: normal;
    }
    .kpi-value{
        margin-top: 8px;
        font-size: clamp(1.15rem, 2.1vw, 1.60rem);
        font-weight: 650;
        color: #0f172a;
        line-height: 1.1;
        word-break: break-word;
    }
    .pill {
        display:inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        border: 1px solid rgba(15,23,42,0.12);
        background: rgba(255,255,255,0.55);
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 1) PATHS + LOADING (SINGLE SOURCE OF TRUTH)
# =========================

# folder projektu (tam gdzie app.py)
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return -1.0

@st.cache_data(show_spinner=False)
def _read_parquet(path_str: str, mtime: float) -> pd.DataFrame:
    # mtime jest tylko po to, żeby cache się unieważniał po podmianie pliku
    return pd.read_parquet(path_str)

def load_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _read_parquet(str(path), _mtime(path))

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(exist_ok=True)

    # Jedyna historia do UI: master_model_ready.parquet
    hist = load_parquet_if_exists(DATA_DIR / "master_model_ready.parquet")

    # Jedyny forecast: forecast_future.parquet
    future = load_parquet_if_exists(DATA_DIR / "forecast_future.parquet")

    # Backtest + portfolio metrics (opcjonalne)
    backtest = load_parquet_if_exists(DATA_DIR / "forecast_backtest.parquet")
    metrics = load_parquet_if_exists(DATA_DIR / "metrics_summary.parquet")

    return hist, future, backtest, metrics

df_hist, df_future, df_backtest, df_metrics = load_inputs()

# =========================
# 2) HELPERS
# =========================
def section_header(title: str, subtitle: str | None = None):
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)

def kpi_cards_row(cards: list[tuple[str, str]]):
    parts = ['<div class="kpi-grid">']
    for label, value in cards:
        parts.append(
            f'<div class="kpi-card">'
            f'  <div class="kpi-label">{html.escape(str(label))}</div>'
            f'  <div class="kpi-value">{html.escape(str(value))}</div>'
            f'</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "-"
    try:
        return f"{int(round(float(x))):,}".replace(",", " ")
    except Exception:
        return str(x)

def fmt_float(x, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "-"
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)

def safe_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def require_cols(df: pd.DataFrame, cols: list[str], df_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Brak kolumn w {df_name}: {missing}")
        st.stop()

def _mape_thresh(y_true, y_pred, min_y=5.0):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt >= float(min_y)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)

# =========================
# 3) STANDARDIZE INPUTS (dates/keys/columns)
# =========================

# Normalizacja kluczy country/sku we wszystkich DF
for _df in [df_hist, df_future, df_backtest, df_metrics]:
    if _df is None or _df.empty:
        continue
    if "country" in _df.columns:
        _df["country"] = _df["country"].astype(str).str.strip()
    if "sku" in _df.columns:
        _df["sku"] = _df["sku"].astype(str).str.strip()

# --- HISTORY (master_model_ready.parquet) ---
if not df_hist.empty:
    req_hist = {"country", "sku"}
    miss = req_hist - set(df_hist.columns)
    if miss:
        st.error(f"Brak wymaganych kolumn w master_model_ready.parquet: {sorted(miss)}")
        st.stop()

    df_hist = df_hist.copy()

    # accept either 'date' or 'week_start'
    if "date" not in df_hist.columns and "week_start" in df_hist.columns:
        df_hist = df_hist.rename(columns={"week_start": "date"})
    if "date" in df_hist.columns:
        df_hist["date"] = safe_dt(df_hist["date"])
        df_hist = df_hist[df_hist["date"].notna()].copy()
    else:
        st.error("master_model_ready.parquet musi mieć kolumnę date (lub week_start).")
        st.stop()

    # demand column: accept demand_raw or demand
    if "demand_raw" not in df_hist.columns and "demand" in df_hist.columns:
        df_hist = df_hist.rename(columns={"demand": "demand_raw"})
    if "demand_raw" in df_hist.columns:
        df_hist["demand_raw"] = pd.to_numeric(df_hist["demand_raw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        st.error("master_model_ready.parquet musi mieć kolumnę demand_raw (lub demand).")
        st.stop()

    if "product_name" not in df_hist.columns:
        df_hist["product_name"] = df_hist["sku"].astype(str)

    if "dc_id" not in df_hist.columns:
        df_hist["dc_id"] = "ALL"

# --- FUTURE (forecast_future.parquet) ---
if not df_future.empty:
    req_fut = {"country", "sku", "date", "y_pred"}
    miss = req_fut - set(df_future.columns)
    if miss:
        st.error(f"Brak wymaganych kolumn w forecast_future.parquet: {sorted(miss)}")
        st.stop()

    df_future = df_future.copy()
    df_future["date"] = safe_dt(df_future["date"])
    df_future = df_future[df_future["date"].notna()].copy()

    for c in ["y_pred", "PI10", "PI90"]:
        if c in df_future.columns:
            df_future[c] = pd.to_numeric(df_future[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    if "product_name" not in df_future.columns:
        df_future["product_name"] = df_future["sku"].astype(str)
    if "dc_id" not in df_future.columns:
        df_future["dc_id"] = "ALL"
    if "model_type" not in df_future.columns:
        df_future["model_type"] = "unknown"

# --- BACKTEST (forecast_backtest.parquet) ---
if not df_backtest.empty:
    df_backtest = df_backtest.copy()
    if "date" in df_backtest.columns:
        df_backtest["date"] = safe_dt(df_backtest["date"])
    if "cutoff_date" in df_backtest.columns:
        df_backtest["cutoff_date"] = safe_dt(df_backtest["cutoff_date"])

    for c in ["y_true", "y_pred_raw", "y_pred_cal"]:
        if c in df_backtest.columns:
            df_backtest[c] = pd.to_numeric(df_backtest[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    if "dc_id" not in df_backtest.columns:
        df_backtest["dc_id"] = "ALL"

# =========================
# 4) PRODUCT UNIVERSE + SIDEBAR (product-first)
# =========================
def make_product_label(df: pd.DataFrame) -> pd.Series:
    return (
        df["product_name"].astype(str)
        + "  |  "
        + df["sku"].astype(str)
        + "  |  "
        + df["country"].astype(str)
    )

# products universe from history if available, else from future
if not df_hist.empty:
    prod_universe = df_hist[["product_name", "sku", "country"]].drop_duplicates().copy()
elif not df_future.empty:
    prod_universe = df_future[["product_name", "sku", "country"]].drop_duplicates().copy()
else:
    prod_universe = pd.DataFrame(columns=["product_name", "sku", "country"])

if not prod_universe.empty:
    prod_universe["product_label"] = make_product_label(prod_universe)
    prod_universe = prod_universe.sort_values(["product_name", "country", "sku"]).reset_index(drop=True)

st.sidebar.header("Filtry (product-first)")

if st.sidebar.button("Odśwież dane (clear cache)", key="btn_clear_cache"):
    st.cache_data.clear()
    st.rerun()

if prod_universe.empty:
    st.sidebar.error("Brak danych do wyboru produktu (sprawdź pliki w data/).")
    st.stop()

# Optional: filter by country first
country_opts = sorted(prod_universe["country"].dropna().astype(str).unique())
country_sel = st.sidebar.multiselect("Kraj (opcjonalnie)", options=country_opts, default=[])

prod_view = prod_universe.copy()
if country_sel:
    prod_view = prod_view[prod_view["country"].astype(str).isin(country_sel)].copy()

prod_labels = prod_view["product_label"].tolist()
if not prod_labels:
    st.sidebar.error("Brak produktów po zastosowaniu filtrów (np. kraj).")
    st.stop()

product_label_sel = st.sidebar.selectbox(
    "Produkt (nazwa | sku | kraj)",
    options=prod_labels,
    index=0,
)

# decode selection to series key
sel_row = prod_view.loc[prod_view["product_label"] == product_label_sel].iloc[0]
SEL_COUNTRY = str(sel_row["country"])
SEL_SKU = str(sel_row["sku"])
SEL_PNAME = str(sel_row["product_name"])

# model type filter for forecast (optional)
model_type_sel = None
if not df_future.empty and "model_type" in df_future.columns:
    mt_opts = sorted(df_future["model_type"].dropna().astype(str).unique())
    if mt_opts:
        model_type_sel = st.sidebar.selectbox(
            "Model type (opcjonalnie)",
            options=["(all)"] + mt_opts,
            index=0,
        )

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <span class="pill"><b>Produkt:</b> {html.escape(SEL_PNAME)}</span>
    <span class="pill"><b>SKU:</b> {html.escape(SEL_SKU)}</span>
    <span class="pill"><b>Kraj:</b> {html.escape(SEL_COUNTRY)}</span>
    """,
    unsafe_allow_html=True,
)

# =========================
# 5) DATE FILTERS (history + forecast)
# =========================
hist_range = None
if not df_hist.empty:
    df_hist_prod_all = df_hist[
        (df_hist["country"].astype(str) == SEL_COUNTRY) & (df_hist["sku"].astype(str) == SEL_SKU)
    ].copy()

    if not df_hist_prod_all.empty:
        dmin = df_hist_prod_all["date"].min().date()
        dmax = df_hist_prod_all["date"].max().date()
        hist_range = st.sidebar.date_input(
            "Zakres dat (historia)",
            value=(dmin, dmax),
            min_value=dmin,
            max_value=dmax,
        )
    else:
        st.sidebar.info("Brak historii dla wybranego produktu w master_model_ready.parquet.")
else:
    st.sidebar.info("Brak danych historycznych (master_model_ready.parquet).")

fut_range = None
if not df_future.empty:
    df_fut_prod_all = df_future[
        (df_future["country"].astype(str) == SEL_COUNTRY) & (df_future["sku"].astype(str) == SEL_SKU)
    ].copy()

    if model_type_sel and model_type_sel != "(all)" and "model_type" in df_fut_prod_all.columns:
        df_fut_prod_all = df_fut_prod_all[df_fut_prod_all["model_type"].astype(str) == str(model_type_sel)].copy()

    if not df_fut_prod_all.empty:
        fmin = df_fut_prod_all["date"].min().date()
        fmax = df_fut_prod_all["date"].max().date()
        fut_range = st.sidebar.date_input(
            "Zakres dat (forecast)",
            value=(fmin, fmax),
            min_value=fmin,
            max_value=fmax,
        )
    else:
        st.sidebar.info("Brak forecastu dla wybranego produktu w forecast_future.parquet.")
else:
    st.sidebar.info("Brak danych forecast (forecast_future.parquet).")

# =========================
# 6) TABS (skeleton)
# =========================
tab_overview, tab_history, tab_forecast, tab_backtest, tab_export = st.tabs(
    ["Portfolio", "Historia produktu", "Forecast 2026", "Backtest & jakość", "Eksport"]
)

# =========================
# 7) PREPARE FILTERED VIEWS (single selected product)
# =========================
df_hist_prod = pd.DataFrame()
if not df_hist.empty:
    df_hist_prod = df_hist[
        (df_hist["country"].astype(str) == SEL_COUNTRY) & (df_hist["sku"].astype(str) == SEL_SKU)
    ].copy()
    if hist_range and isinstance(hist_range, tuple) and len(hist_range) == 2:
        d0, d1 = hist_range
        df_hist_prod = df_hist_prod[
            (df_hist_prod["date"] >= pd.to_datetime(d0)) & (df_hist_prod["date"] <= pd.to_datetime(d1))
        ].copy()
    df_hist_prod = df_hist_prod.sort_values("date")

df_fut_prod = pd.DataFrame()
if not df_future.empty:
    df_fut_prod = df_future[
        (df_future["country"].astype(str) == SEL_COUNTRY) & (df_future["sku"].astype(str) == SEL_SKU)
    ].copy()
    if model_type_sel and model_type_sel != "(all)" and "model_type" in df_fut_prod.columns:
        df_fut_prod = df_fut_prod[df_fut_prod["model_type"].astype(str) == str(model_type_sel)].copy()
    if fut_range and isinstance(fut_range, tuple) and len(fut_range) == 2:
        f0, f1 = fut_range
        df_fut_prod = df_fut_prod[
            (df_fut_prod["date"] >= pd.to_datetime(f0)) & (df_fut_prod["date"] <= pd.to_datetime(f1))
        ].copy()
    df_fut_prod = df_fut_prod.sort_values("date")

df_bt_prod = pd.DataFrame()
if not df_backtest.empty:
    df_bt_prod = df_backtest[
        (df_backtest["country"].astype(str) == SEL_COUNTRY) & (df_backtest["sku"].astype(str) == SEL_SKU)
    ].copy()
    if "date" in df_bt_prod.columns and "cutoff_date" in df_bt_prod.columns:
        df_bt_prod = df_bt_prod[df_bt_prod["date"].notna() & df_bt_prod["cutoff_date"].notna()].copy()
        df_bt_prod = df_bt_prod.sort_values(["cutoff_date", "date"])
    elif "date" in df_bt_prod.columns:
        df_bt_prod = df_bt_prod[df_bt_prod["date"].notna()].copy()
        df_bt_prod = df_bt_prod.sort_values("date")

# =========================
# 8) TAB: PORTFOLIO (overview)
# =========================
with tab_overview:
    section_header("Portfolio", "Szybki przegląd danych oraz wybieranej serii (product-first).")

    if not df_hist.empty:
        n_products = int(prod_universe.shape[0])
        n_countries = int(prod_universe["country"].nunique())
        n_sku = int(prod_universe["sku"].nunique())
        dmin = df_hist["date"].min().date() if df_hist["date"].notna().any() else "-"
        dmax = df_hist["date"].max().date() if df_hist["date"].notna().any() else "-"
        total = float(df_hist["demand_raw"].sum()) if "demand_raw" in df_hist.columns else np.nan
        zero_share = float((df_hist["demand_raw"] == 0).mean()) if "demand_raw" in df_hist.columns else np.nan

        kpi_cards_row([
            ("Produkty/serie (country+sku)", fmt_int(n_products)),
            ("Kraje", fmt_int(n_countries)),
            ("SKU (unikalne)", fmt_int(n_sku)),
            ("Zakres historii", f"{dmin} → {dmax}"),
        ])
        kpi_cards_row([
            ("Suma popytu (historia)", fmt_int(total) if np.isfinite(total) else "-"),
            ("Udział zer (historia)", fmt_float(zero_share, 3) if np.isfinite(zero_share) else "-"),
            ("Wybrany produkt", SEL_PNAME),
            ("Wybrana seria", f"{SEL_SKU} | {SEL_COUNTRY}"),
        ])

        if "demand_raw" in df_hist.columns:
            rank_cols = ["product_name", "sku", "country"]
            agg = (
                df_hist.groupby(rank_cols, as_index=False, observed=True)["demand_raw"]
                .sum()
                .rename(columns={"demand_raw": "sum_demand"})
                .sort_values("sum_demand", ascending=False)
            )
            st.markdown("---")
            st.subheader("Ranking produktów wg wolumenu (historia)")
            st.dataframe(agg.head(30), use_container_width=True, hide_index=True)

    else:
        st.warning("Brak danych historycznych (master_model_ready.parquet). Portfolio ograniczone do forecastu.")
        if not df_future.empty:
            n_products = int(prod_universe.shape[0])
            dmin = df_future["date"].min().date() if df_future["date"].notna().any() else "-"
            dmax = df_future["date"].max().date() if df_future["date"].notna().any() else "-"
            kpi_cards_row([
                ("Produkty/serie (country+sku)", fmt_int(n_products)),
                ("Zakres forecastu", f"{dmin} → {dmax}"),
                ("Wybrany produkt", SEL_PNAME),
                ("Wybrana seria", f"{SEL_SKU} | {SEL_COUNTRY}"),
            ])

    st.markdown("---")
    section_header("Wybrany produkt", "Podstawowe statystyki dla konkretnej serii.")

    if not df_hist_prod.empty and "demand_raw" in df_hist_prod.columns and "date" in df_hist_prod.columns:
        n_weeks = int(df_hist_prod["date"].nunique())
        span_weeks = int(((df_hist_prod["date"].max() - df_hist_prod["date"].min()).days // 7) + 1)
        missing_weeks = max(0, span_weeks - n_weeks)
        zshare = float((df_hist_prod["demand_raw"] == 0).mean())

        kpi_cards_row([
            ("Tygodnie obserwacji", fmt_int(n_weeks)),
            ("Zakres (tyg. span)", fmt_int(span_weeks)),
            ("Brakujące tygodnie", fmt_int(missing_weeks)),
            ("Udział zer", fmt_float(zshare, 3)),
        ])

        g = (
            df_hist_prod[["date", "demand_raw"]]
            .groupby("date", as_index=False)["demand_raw"]
            .sum()
            .sort_values("date")
        )
        fig = px.line(
            g, x="date", y="demand_raw",
            title=f"Historia popytu: {SEL_PNAME}",
            labels={"date": "Data", "demand_raw": "Popyt"},
        )
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Brak historii dla wybranego produktu (albo brak kolumny demand_raw/date).")

    if not df_fut_prod.empty:
        fmin = df_fut_prod["date"].min().date()
        fmax = df_fut_prod["date"].max().date()
        total_fc = float(df_fut_prod["y_pred"].sum()) if "y_pred" in df_fut_prod.columns else np.nan
        mt = str(df_fut_prod["model_type"].iloc[0]) if "model_type" in df_fut_prod.columns and len(df_fut_prod) else "-"
        kpi_cards_row([
            ("Forecast zakres", f"{fmin} → {fmax}"),
            ("Suma prognozy", fmt_int(total_fc) if np.isfinite(total_fc) else "-"),
            ("Model type", mt),
            ("PI dostępne", "TAK" if ("PI10" in df_fut_prod.columns and "PI90" in df_fut_prod.columns) else "NIE"),
        ])
    else:
        st.info("Brak forecastu dla wybranego produktu w forecast_future.parquet.")

# =========================
# 9) TAB: HISTORY (single product)
# =========================
with tab_history:
    section_header("Historia produktu", "Wizualizacja i statystyki tylko dla wybranego produktu/serii.")

    if df_hist_prod.empty:
        st.warning("Brak historii dla wybranego produktu (sprawdź master_model_ready.parquet).")
    else:
        show_cols = [c for c in ["date", "demand_raw", "product_name", "sku", "country", "dc_id"] if c in df_hist_prod.columns]
        dfh = df_hist_prod[show_cols].copy()
        dfh["demand_raw"] = pd.to_numeric(dfh["demand_raw"], errors="coerce").fillna(0.0).clip(lower=0.0)

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        y = dfh["demand_raw"]
        c1.metric("Średnia", fmt_float(float(y.mean()), 2))
        c2.metric("Mediana", fmt_float(float(y.median()), 2))
        c3.metric("Max", fmt_float(float(y.max()), 2))
        c4.metric("Udział zer", fmt_float(float((y == 0).mean()), 3))

        g = dfh.groupby("date", as_index=False)["demand_raw"].sum().sort_values("date")
        fig = px.line(
            g,
            x="date",
            y="demand_raw",
            title=f"Historia: {SEL_PNAME}  |  {SEL_SKU}  |  {SEL_COUNTRY}",
            labels={"date": "Data", "demand_raw": "Popyt"},
        )
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Wygładzenie (rolling mean)"):
            win = st.slider("Okno (tygodnie)", min_value=2, max_value=26, value=8, step=1)
            g2 = g.copy()
            g2["roll_mean"] = g2["demand_raw"].rolling(window=win, min_periods=1).mean()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=g2["date"], y=g2["demand_raw"], mode="lines", name="Raw"))
            fig2.add_trace(go.Scatter(x=g2["date"], y=g2["roll_mean"], mode="lines", name=f"Rolling mean ({win})"))
            fig2.update_layout(title="Historia: raw vs rolling mean", xaxis_title="Date", yaxis_title="Demand", legend_title_text="")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Tabela historii (do 5000 wierszy)")
        st.dataframe(dfh.sort_values("date").head(5000), use_container_width=True, hide_index=True)

# =========================
# 10) TAB: FORECAST (single product) — FIXED PI
# =========================
with tab_forecast:
    section_header("Forecast 2026", "Prognoza i przedziały niepewności dla wybranego produktu/serii.")

    show_pi = st.checkbox("Pokaż przedziały PI (P10–P90)", value=True)
    scale_to_forecast = st.checkbox("Skaluj oś Y do prognozy (lepiej widać trend)", value=True)

    if df_fut_prod.empty:
        st.warning("Brak forecastu dla wybranego produktu (sprawdź forecast_future.parquet).")
    else:
        dff = df_fut_prod.copy()
        dff["y_pred"] = pd.to_numeric(dff["y_pred"], errors="coerce").fillna(0.0).clip(lower=0.0)

        # jeśli PI brak, to fallback +/-20%
        if "PI10" not in dff.columns or "PI90" not in dff.columns:
            dff["PI10"] = (dff["y_pred"] * 0.8).clip(lower=0.0)
            dff["PI90"] = (dff["y_pred"] * 1.2).clip(lower=0.0)
        else:
            dff["PI10"] = pd.to_numeric(dff["PI10"], errors="coerce").fillna(0.0).clip(lower=0.0)
            dff["PI90"] = pd.to_numeric(dff["PI90"], errors="coerce").fillna(0.0).clip(lower=0.0)

        fmin = dff["date"].min().date()
        fmax = dff["date"].max().date()
        total_fc = float(dff["y_pred"].sum())
        mean_fc = float(dff["y_pred"].mean())
        mean_pi_width = float((dff["PI90"] - dff["PI10"]).mean())
        mt = str(dff["model_type"].iloc[0]) if "model_type" in dff.columns and len(dff) else "-"

        kpi_cards_row([
            ("Zakres", f"{fmin} → {fmax}"),
            ("Horyzont (tyg.)", fmt_int(dff["date"].nunique())),
            ("Suma prognozy", fmt_int(total_fc)),
            ("Śr. PI width (P90-P10)", fmt_int(mean_pi_width)),
        ])
        kpi_cards_row([
            ("Śr. prognoza / tydz.", fmt_float(mean_fc, 2)),
            ("Model type", mt),
            ("Produkt", SEL_PNAME),
            ("Seria", f"{SEL_SKU} | {SEL_COUNTRY}"),
        ])

        fig = go.Figure()

        # Forecast zawsze
        fig.add_trace(go.Scatter(
            x=dff["date"], y=dff["y_pred"],
            mode="lines+markers",
            name="Forecast"
        ))

        # PI tylko jeśli user chce (FIX: oba trace w if)
        if show_pi:
            fig.add_trace(go.Scatter(
                x=dff["date"], y=dff["PI10"],
                mode="lines",
                name="P10",
                line=dict(width=0),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=dff["date"], y=dff["PI90"],
                mode="lines",
                name="P90",
                fill="tonexty",
                opacity=0.25,
                line=dict(width=0),
                showlegend=True
            ))

        fig.update_layout(
            title=f"Forecast + PI: {SEL_PNAME}  |  {SEL_SKU}  |  {SEL_COUNTRY}",
            xaxis_title="Date",
            yaxis_title="Forecast",
            legend_title_text="",
        )

        if scale_to_forecast:
            ymin = float(dff["y_pred"].min())
            ymax = float(dff["y_pred"].max())
            pad = max(1.0, 0.05 * (ymax - ymin) if ymax > ymin else 1.0)
            fig.update_yaxes(range=[max(0.0, ymin - pad), ymax + pad])

        st.plotly_chart(fig, use_container_width=True)

        # Optional overlay: last N weeks of history
        if not df_hist_prod.empty:
            with st.expander("Porównanie z końcówką historii (overlay)"):
                n_last = st.slider("Ile ostatnich tygodni historii pokazać", min_value=8, max_value=104, value=26, step=1)
                hist_tail = (
                    df_hist_prod.sort_values("date")
                    .tail(n_last)[["date", "demand_raw"]]
                    .groupby("date", as_index=False)["demand_raw"]
                    .sum()
                    .sort_values("date")
                )
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=hist_tail["date"], y=hist_tail["demand_raw"], mode="lines+markers", name="History (tail)"))
                fig3.add_trace(go.Scatter(x=dff["date"], y=dff["y_pred"], mode="lines+markers", name="Forecast"))
                fig3.update_layout(title="Historia (tail) vs Forecast", xaxis_title="Date", yaxis_title="Demand", legend_title_text="")
                st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Tabela forecast (do 5000 wierszy)")
        show_cols = [c for c in ["date", "y_pred", "PI10", "PI90", "model_type", "product_name", "sku", "country"] if c in dff.columns]
        st.dataframe(dff[show_cols].sort_values("date").head(5000), use_container_width=True, hide_index=True)

# =========================
# 11) BACKTEST METRICS HELPERS
# =========================
def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    if denom <= 1e-9:
        return np.nan
    return float(np.abs(y_true - y_pred).sum() / denom)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))

def calculate_metrics_basic(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    mask = np.abs(y_true) > 1e-9
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        mpe = float(np.mean(((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)
    else:
        mape = np.nan
        mpe = np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MPE": mpe}

# =========================
# 12) TAB: BACKTEST (product-first)
# =========================
with tab_backtest:
    section_header("Backtest & jakość", "Metryki tylko dla wybranego produktu/serii + ranking portfela (opcjonalnie).")

    if df_backtest.empty:
        st.warning("Brak data/forecast_backtest.parquet — bez tego nie pokażemy backtestu i metryk.")
    else:
        require_cols(
            df_backtest,
            ["country", "dc_id", "sku", "date", "cutoff_date", "y_true", "y_pred_raw", "y_pred_cal"],
            "forecast_backtest.parquet",
        )

        if df_bt_prod.empty:
            st.info("Brak rekordów backtest dla wybranego produktu/serii.")
        else:
            dfb = df_bt_prod.copy()
            dfb["date"] = safe_dt(dfb["date"])
            dfb["cutoff_date"] = safe_dt(dfb["cutoff_date"])

            for c in ["y_true", "y_pred_raw", "y_pred_cal"]:
                dfb[c] = pd.to_numeric(dfb[c], errors="coerce").fillna(0.0).clip(lower=0.0)

            y_true = dfb["y_true"].to_numpy(float)
            y_raw = dfb["y_pred_raw"].to_numpy(float)
            y_cal = dfb["y_pred_cal"].to_numpy(float)

            m_raw = calculate_metrics_basic(y_true, y_raw)
            m_cal = calculate_metrics_basic(y_true, y_cal)
            mape5_raw = _mape_thresh(y_true, y_raw, min_y=5.0)
            mape5_cal = _mape_thresh(y_true, y_cal, min_y=5.0)

            kpi_cards_row([
                ("RMSE_raw (seria)", fmt_float(m_raw["RMSE"], 3)),
                ("RMSE_cal (seria)", fmt_float(m_cal["RMSE"], 3)),
                ("MAPE_raw % (seria)", fmt_float(m_raw["MAPE"], 2)),
                ("MAPE_cal % (seria)", fmt_float(m_cal["MAPE"], 2)),
            ])
            kpi_cards_row([
                ("MPE_raw % (seria)", fmt_float(m_raw["MPE"], 2)),
                ("MPE_cal % (seria)", fmt_float(m_cal["MPE"], 2)),
                ("WAPE_raw (seria)", fmt_float(_wape(y_true, y_raw), 4)),
                ("WAPE_cal (seria)", fmt_float(_wape(y_true, y_cal), 4)),
            ])

            metrics_compare = pd.DataFrame([
                {"Metoda": "RAW", "MAE": m_raw["MAE"], "MAPE": m_raw["MAPE"], "MAPE>=5": mape5_raw, "MPE": m_raw["MPE"], "RMSE": m_raw["RMSE"]},
                {"Metoda": "CAL", "MAE": m_cal["MAE"], "MAPE": m_cal["MAPE"], "MAPE>=5": mape5_cal, "MPE": m_cal["MPE"], "RMSE": m_cal["RMSE"]},
            ])
            st.subheader("Dodatkowe metryki (ALL): MAE / MAPE / MPE / RMSE")
            st.dataframe(metrics_compare.round(4), use_container_width=True, hide_index=True)

            # metrics per cutoff_date
            rows = []
            for cd, g in dfb.groupby("cutoff_date", observed=True):
                yt = g["y_true"].to_numpy(float)
                yr = g["y_pred_raw"].to_numpy(float)
                yc = g["y_pred_cal"].to_numpy(float)
                rows.append({
                    "cutoff_date": cd,
                    "n_obs": int(len(g)),
                    "WAPE_raw": _wape(yt, yr),
                    "WAPE_cal": _wape(yt, yc),
                    "MAE_raw": _mae(yt, yr),
                    "MAE_cal": _mae(yt, yc),
                    "Bias_raw": _bias(yt, yr),
                    "Bias_cal": _bias(yt, yc),
                })
            metrics_cutoff = pd.DataFrame(rows).sort_values("cutoff_date")

            st.subheader("Metryki per cutoff (wybrany produkt)")
            st.dataframe(
                metrics_cutoff.assign(
                    cutoff_date=lambda d: d["cutoff_date"].dt.date,
                    WAPE_raw=lambda d: d["WAPE_raw"].round(6),
                    WAPE_cal=lambda d: d["WAPE_cal"].round(6),
                    MAE_raw=lambda d: d["MAE_raw"].round(3),
                    MAE_cal=lambda d: d["MAE_cal"].round(3),
                    Bias_raw=lambda d: d["Bias_raw"].round(3),
                    Bias_cal=lambda d: d["Bias_cal"].round(3),
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            sel_metric = st.selectbox("Wykres metryki (wybrany produkt)", options=["WAPE", "MAE", "Bias"], index=0)
            if sel_metric == "WAPE":
                y_raw_col, y_cal_col, ylab = "WAPE_raw", "WAPE_cal", "WAPE"
            elif sel_metric == "MAE":
                y_raw_col, y_cal_col, ylab = "MAE_raw", "MAE_cal", "MAE"
            else:
                y_raw_col, y_cal_col, ylab = "Bias_raw", "Bias_cal", "Bias"

            figm = go.Figure()
            figm.add_trace(go.Scatter(x=metrics_cutoff["cutoff_date"], y=metrics_cutoff[y_raw_col], mode="lines+markers", name=f"{ylab}_raw"))
            figm.add_trace(go.Scatter(x=metrics_cutoff["cutoff_date"], y=metrics_cutoff[y_cal_col], mode="lines+markers", name=f"{ylab}_cal"))
            figm.update_layout(title=f"{ylab} per cutoff: {SEL_PNAME}", xaxis_title="cutoff_date", yaxis_title=ylab, legend_title_text="")
            st.plotly_chart(figm, use_container_width=True)

            st.markdown("---")
            st.subheader("Backtest: y_true vs pred (próbka)")
            if metrics_cutoff["cutoff_date"].notna().any():
                last_cd = metrics_cutoff["cutoff_date"].max()
                cd_opts = sorted(metrics_cutoff["cutoff_date"].dropna().unique())
                cd_sel = st.selectbox("cutoff_date", options=cd_opts, index=cd_opts.index(last_cd))
                df_show = dfb[dfb["cutoff_date"] == cd_sel].copy().sort_values("date")
            else:
                df_show = dfb.copy().sort_values("date")

            figbt = go.Figure()
            figbt.add_trace(go.Scatter(x=df_show["date"], y=df_show["y_true"], mode="lines+markers", name="y_true"))
            figbt.add_trace(go.Scatter(x=df_show["date"], y=df_show["y_pred_raw"], mode="lines+markers", name="y_pred_raw"))
            figbt.add_trace(go.Scatter(x=df_show["date"], y=df_show["y_pred_cal"], mode="lines+markers", name="y_pred_cal"))
            figbt.update_layout(title="Backtest: y_true vs predictions", xaxis_title="date", yaxis_title="demand", legend_title_text="")
            st.plotly_chart(figbt, use_container_width=True)

            st.dataframe(
                df_show[["cutoff_date", "date", "y_true", "y_pred_raw", "y_pred_cal"]].head(2000),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        with st.expander("Ranking portfela (opcjonalnie) – metryki per produkt/seria"):
            dfp = df_backtest.copy()

            if "product_name" not in dfp.columns:
                if not df_hist.empty:
                    dim = df_hist[["country", "sku", "product_name"]].drop_duplicates()
                    dfp = dfp.merge(dim, on=["country", "sku"], how="left")
                if "product_name" not in dfp.columns:
                    dfp["product_name"] = dfp["sku"].astype(str)

            rows = []
            for (cty, sku), g in dfp.groupby(["country", "sku"], observed=True):
                yt = pd.to_numeric(g["y_true"], errors="coerce").fillna(0.0).to_numpy(float)
                yc = pd.to_numeric(g["y_pred_cal"], errors="coerce").fillna(0.0).to_numpy(float)
                yr = pd.to_numeric(g["y_pred_raw"], errors="coerce").fillna(0.0).to_numpy(float)
                pname = str(g["product_name"].dropna().iloc[0]) if g["product_name"].notna().any() else str(sku)
                rows.append({
                    "product_name": pname,
                    "sku": str(sku),
                    "country": str(cty),
                    "n_obs": int(len(g)),
                    "WAPE_raw": _wape(yt, yr),
                    "WAPE_cal": _wape(yt, yc),
                    "MAE_cal": _mae(yt, yc),
                    "Bias_cal": _bias(yt, yc),
                })
            rank = pd.DataFrame(rows).sort_values("WAPE_cal", ascending=True)

            st.dataframe(
                rank.assign(
                    WAPE_raw=lambda d: d["WAPE_raw"].round(6),
                    WAPE_cal=lambda d: d["WAPE_cal"].round(6),
                    MAE_cal=lambda d: d["MAE_cal"].round(3),
                    Bias_cal=lambda d: d["Bias_cal"].round(3),
                ),
                use_container_width=True,
                hide_index=True,
            )

# =========================
# 13) TAB: EXPORT (product-first)
# =========================
with tab_export:
    section_header("Eksport", "Pobierz dane tylko dla wybranego produktu/serii (historia, forecast, backtest).")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Historia")
        if df_hist_prod.empty:
            st.info("Brak danych historii dla eksportu.")
        else:
            out_h = df_hist_prod.copy()
            out_h["date"] = safe_dt(out_h["date"]).dt.date
            csv_h = out_h.to_csv(index=False, sep=";", encoding="utf-8-sig")
            st.download_button(
                "Pobierz historię (CSV)",
                data=csv_h,
                file_name=f"havi_history_{SEL_COUNTRY}_{SEL_SKU}.csv".replace(" ", "_"),
                mime="text/csv",
            )

    with c2:
        st.subheader("Forecast")
        if df_fut_prod.empty:
            st.info("Brak danych forecast dla eksportu.")
        else:
            out_f = df_fut_prod.copy()
            out_f["date"] = safe_dt(out_f["date"]).dt.date
            csv_f = out_f.to_csv(index=False, sep=";", encoding="utf-8-sig")
            st.download_button(
                "Pobierz forecast (CSV)",
                data=csv_f,
                file_name=f"havi_forecast_{SEL_COUNTRY}_{SEL_SKU}.csv".replace(" ", "_"),
                mime="text/csv",
            )

    with c3:
        st.subheader("Backtest")
        if df_bt_prod.empty:
            st.info("Brak backtest dla eksportu.")
        else:
            out_b = df_bt_prod.copy()
            out_b["date"] = safe_dt(out_b["date"]).dt.date
            if "cutoff_date" in out_b.columns:
                out_b["cutoff_date"] = safe_dt(out_b["cutoff_date"]).dt.date
            csv_b = out_b.to_csv(index=False, sep=";", encoding="utf-8-sig")
            st.download_button(
                "Pobierz backtest (CSV)",
                data=csv_b,
                file_name=f"havi_backtest_{SEL_COUNTRY}_{SEL_SKU}.csv".replace(" ", "_"),
                mime="text/csv",
            )

# =========================
# 14) FOOTER: DATA AVAILABILITY DIAGNOSTIC
# =========================
st.markdown("---")
with st.expander("Diagnostyka plików (data/)"):
    st.write("Ścieżka DATA_DIR:", str(DATA_DIR.resolve()) if DATA_DIR.exists() else str(DATA_DIR))
    st.write("Pliki oczekiwane:")
    for p in ["master_model_ready.parquet", "forecast_future.parquet", "forecast_backtest.parquet", "metrics_summary.parquet"]:
        st.write(f"- data/{p} : ", "OK" if (DATA_DIR / p).exists() else "BRAK")
    st.write("Rozmiary ramek danych:")
    st.write({
        "hist_rows": int(len(df_hist)) if not df_hist.empty else 0,
        "future_rows": int(len(df_future)) if not df_future.empty else 0,
        "backtest_rows": int(len(df_backtest)) if not df_backtest.empty else 0,
        "metrics_rows": int(len(df_metrics)) if not df_metrics.empty else 0,
    })