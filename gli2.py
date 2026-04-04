"""
Global Liquidity Index vs SPY & BTC  —  Streamlit App
=======================================================
Install:
    pip install streamlit pandas yfinance plotly fredapi requests

Run:
    streamlit run global_liquidity_index.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime

st.set_page_config(page_title="Global Liquidity Index vs SPY & BTC", layout="wide")
st.title("Global Liquidity Index vs SPY & BTC")
st.markdown("**GLI** = FED − TGA − RRP + ECB + BOJ + BOC + RBA + BOE + SNB")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    FRED_API_KEY = st.text_input(
        "FRED API Key", type="password",
        help="Free key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )
    START_DATE = st.date_input("Start Date", value=datetime(2015, 1, 1))
    if not FRED_API_KEY:
        st.warning("Enter your FRED API key to continue.")
        st.stop()

RESAMPLE = "W-FRI"
start_str = START_DATE.strftime("%Y-%m-%d")
end_str   = datetime.today().strftime("%Y-%m-%d")

# ── Fred client (one instance per key, not cached globally) ────────────────────
# Store in session_state so cache_data functions can reference it via closure
# without Streamlit trying to hash the Fred object.
if "fred" not in st.session_state or st.session_state.get("fred_key") != FRED_API_KEY:
    st.session_state.fred     = Fred(api_key=FRED_API_KEY)
    st.session_state.fred_key = FRED_API_KEY

fred: Fred = st.session_state.fred

# ── Helpers ────────────────────────────────────────────────────────────────────
def resample(s: pd.Series) -> pd.Series:
    return s.resample(RESAMPLE).last().ffill()

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fred(series_id: str, start: str) -> pd.Series:
    """Fetch a single FRED series and resample weekly."""
    try:
        s = fred.get_series(series_id, observation_start=start)
        # FRED occasionally returns an int64 positional index instead of
        # DatetimeIndex (seen with BOEBSTASGBP and a few others). Force it.
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s.index = s.index.tz_localize(None)  # strip tz if present
        return resample(s).rename(series_id)
    except Exception as e:
        st.warning(f"FRED `{series_id}` failed: {e}")
        return pd.Series(dtype=float, name=series_id)

def fx(fred_id: str) -> pd.Series:
    return fetch_fred(fred_id, start_str)

# ── Central bank fetchers ──────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_fed(start: str):
    """FED total assets (WALCL) in USD millions → keep as-is, divide later."""
    return fetch_fred("WALCL", start)

@st.cache_data(ttl=86400, show_spinner=False)
def get_tga(start: str):
    return fetch_fred("WTREGEN", start)

@st.cache_data(ttl=86400, show_spinner=False)
def get_rrp(start: str):
    return fetch_fred("RRPONTSYD", start)

@st.cache_data(ttl=86400, show_spinner=False)
def get_ecb(start: str) -> pd.Series:
    """ECB total assets: FRED ECBASSETSW (EUR millions) → USD billions."""
    ecb_eur = fetch_fred("ECBASSETSW", start)   # EUR millions
    eur_usd = fx("DEXUSEU")                      # USD per 1 EUR
    aligned = ecb_eur.reindex(eur_usd.index, method="ffill")
    return (aligned * eur_usd / 1_000).rename("ECB")  # → USD billions

@st.cache_data(ttl=86400, show_spinner=False)
def get_boj(start: str) -> pd.Series:
    """BOJ total assets via FRED JPNASSETS (JPY trillions) → USD billions."""
    # JPNASSETS = JPY trillions; multiply by 1e3 to get billions JPY
    boj_jpy = fetch_fred("JPNASSETS", start) * 1_000   # JPY billions
    jpy_per_usd = fx("DEXJPUS")
    aligned = boj_jpy.reindex(jpy_per_usd.index, method="ffill")
    return (aligned / jpy_per_usd).rename("BOJ")        # → USD billions

def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    """Coerce any index to tz-naive DatetimeIndex. Guards against FRED quirks."""
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.to_datetime(s.index, errors="coerce")
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s

@st.cache_data(ttl=86400, show_spinner=False)
def get_boe(start: str) -> pd.Series:
    """BOE total assets: FRED BOEBSTASGBP (GBP millions) → USD billions."""
    boe_gbp = _ensure_datetime_index(fetch_fred("BOEBSTASGBP", start))
    gbp_usd = _ensure_datetime_index(fx("DEXUSUK"))
    aligned = boe_gbp.reindex(gbp_usd.index, method="ffill")
    return (aligned * gbp_usd / 1_000).rename("BOE")

@st.cache_data(ttl=86400, show_spinner=False)
def get_boc(start: str) -> pd.Series:
    """BOC: try Bank of Canada valet API, fall back to FRED CAALTSASSETS."""
    try:
        url = (
            "https://www.bankofcanada.ca/valet/observations/group/b2_weekly/csv"
            "?start_date=" + start
        )
        df = pd.read_csv(url, skiprows=1)
        df.columns = df.columns.str.strip()
        date_col = next(c for c in df.columns if "date" in c.lower())
        asset_col = next(
            (c for c in df.columns
             if "total asset" in c.lower() and "liability" not in c.lower()),
            None
        )
        if not asset_col:
            raise ValueError("Total assets column not found in BOC CSV.")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        assets_cad_m = pd.to_numeric(
            df[asset_col].astype(str).str.replace(",", ""), errors="coerce"
        ).dropna()
        assets_cad_b = resample(assets_cad_m) / 1_000   # CAD millions → billions
        cad_per_usd  = fx("DEXCAUS")
        aligned = assets_cad_b.reindex(cad_per_usd.index, method="ffill")
        return (aligned / cad_per_usd).rename("BOC")    # → USD billions
    except Exception as e:
        st.warning(f"BOC valet API failed ({e}). Falling back to FRED CAALTSASSETS.")
        try:
            boc_cad_m = fetch_fred("CAALTSASSETS", start)  # CAD millions
            cad_per_usd = fx("DEXCAUS")
            aligned = boc_cad_m.reindex(cad_per_usd.index, method="ffill")
            return (aligned / 1_000 / cad_per_usd).rename("BOC")
        except Exception as e2:
            st.warning(f"BOC FRED fallback also failed: {e2}")
            return pd.Series(dtype=float, name="BOC")

@st.cache_data(ttl=86400, show_spinner=False)
def get_rba(start: str) -> pd.Series:
    """RBA total assets: FRED RBATOTASSETS (AUD millions) → USD billions."""
    rba_aud_m = fetch_fred("RBATOTASSETS", start)  # AUD millions
    aud_usd   = fx("DEXUSAL")                       # USD per 1 AUD
    aligned   = rba_aud_m.reindex(aud_usd.index, method="ffill")
    return (aligned * aud_usd / 1_000).rename("RBA")

@st.cache_data(ttl=86400, show_spinner=False)
def get_snb(start: str) -> pd.Series:
    """SNB total assets: FRED SNBASSETS (CHF billions) → USD billions."""
    snb_chf_b  = fetch_fred("SNBASSETS", start)   # CHF billions
    chf_per_usd = fx("DEXSZUS")
    aligned = snb_chf_b.reindex(chf_per_usd.index, method="ffill")
    return (aligned / chf_per_usd).rename("SNB")

# ── GLI assembly ───────────────────────────────────────────────────────────────
def build_gli(components: dict[str, pd.Series]) -> pd.Series:
    """
    GLI = FED(USD B) - TGA - RRP + ECB + BOJ + BOC + RBA + BOE + SNB
    All inputs should be in USD billions before calling this.
    """
    def get(key) -> pd.Series:
        return components.get(key, pd.Series(dtype=float))

    # WALCL is in USD millions; convert to billions here
    fed = get("FED") * 0.001
    tga = get("TGA")
    rrp = get("RRP")

    addends = [get(k) for k in ("ECB", "BOJ", "BOC", "RBA", "BOE", "SNB")]
    gli = fed.sub(tga, fill_value=0).sub(rrp, fill_value=0)
    for s in addends:
        gli = gli.add(s, fill_value=0)

    return gli.rename("GLI_USD_B")

# ── Market data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_market(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(["SPY", "BTC-USD"], start=start, end=end, progress=False)
    return raw["Close"].resample(RESAMPLE).last().ffill()

# ── Chart ──────────────────────────────────────────────────────────────────────
def plot_gli(gli: pd.Series, market: pd.DataFrame) -> go.Figure | None:
    df = pd.concat([gli, market], axis=1).sort_index().dropna(how="all")
    if df.empty:
        st.error("No data to plot.")
        return None

    # Guard against zero/NaN in base row before indexing
    base = df.iloc[0].replace(0, np.nan)
    idx  = df.div(base) * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
        subplot_titles=("Indexed performance (start = 100)", "GLI (USD trillions)")
    )

    palette = {"GLI_USD_B": "#3266ad", "SPY": "#1D9E75", "BTC-USD": "#D85A30"}
    labels  = {"GLI_USD_B": "Global Liquidity Index", "SPY": "SPY", "BTC-USD": "BTC/USD"}

    for col in ["GLI_USD_B", "SPY", "BTC-USD"]:
        if col not in idx.columns or idx[col].isna().all():
            continue
        fig.add_trace(go.Scatter(
            x=idx.index, y=idx[col].round(1),
            name=labels[col], mode="lines",
            line=dict(color=palette[col], width=2.5 if col == "GLI_USD_B" else 1.5)
        ), row=1, col=1)

    if "GLI_USD_B" in df.columns and df["GLI_USD_B"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=(df["GLI_USD_B"] / 1_000).round(2),
            name="GLI (raw)", mode="lines",
            line=dict(color="#3266ad", width=1.5), showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        title="Global Liquidity Index vs SPY & BTC",
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=750
    )
    fig.update_yaxes(title_text="Indexed (start=100)", row=1, col=1)
    fig.update_yaxes(title_text="USD Trillions",       row=2, col=1)
    fig.update_xaxes(title_text="Date",                row=2, col=1)
    return fig

# ── Main ───────────────────────────────────────────────────────────────────────
with st.status("Loading data...", expanded=True) as status:
    components = {}
    steps = [
        ("FED assets (WALCL)",    "FED", lambda: get_fed(start_str)),
        ("TGA (WTREGEN)",         "TGA", lambda: get_tga(start_str)),
        ("RRP (RRPONTSYD)",       "RRP", lambda: get_rrp(start_str)),
        ("ECB (ECBASSETSW)",      "ECB", lambda: get_ecb(start_str)),
        ("BOJ (JPNASSETS)",       "BOJ", lambda: get_boj(start_str)),
        ("BOE (BOEBSTASGBP)",     "BOE", lambda: get_boe(start_str)),
        ("BOC",                   "BOC", lambda: get_boc(start_str)),
        ("RBA (RBATOTASSETS)",    "RBA", lambda: get_rba(start_str)),
        ("SNB (SNBASSETS)",       "SNB", lambda: get_snb(start_str)),
    ]
    for label, key, fn in steps:
        st.write(f"Fetching {label}...")
        components[key] = fn()

    st.write("Building GLI...")
    gli = build_gli(components)

    st.write("Fetching SPY & BTC-USD...")
    market = get_market(start_str, end_str)

    status.update(label="Done.", state="complete", expanded=False)

fig = plot_gli(gli, market)
if fig:
    st.plotly_chart(fig, use_container_width=True)

# ── Debug / data table ─────────────────────────────────────────────────────────
with st.expander("Coverage summary"):
    summary = {k: f"{v.notna().sum()} weeks" for k, v in components.items()}
    summary["GLI"] = f"{gli.notna().sum()} weeks"
    st.table(pd.Series(summary, name="Non-null observations"))

if st.checkbox("Show raw data"):
    combined = pd.concat([gli, market], axis=1).dropna(how="all")
    st.dataframe(combined.style.format("{:,.2f}"))

st.caption(
    "Coverage: FED · ECB · BOJ · BOC · RBA · BOE · SNB | "
    "Missing: PBC · RBI · CBR · BCB · BOK · RBNZ · Riksbank · BNM"
)
