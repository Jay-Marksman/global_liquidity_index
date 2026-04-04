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

RESAMPLE   = "W-FRI"
start_str  = START_DATE.strftime("%Y-%m-%d")
end_str    = datetime.today().strftime("%Y-%m-%d")

# ── Fred client ────────────────────────────────────────────────────────────────
if "fred" not in st.session_state or st.session_state.get("fred_key") != FRED_API_KEY:
    st.session_state.fred     = Fred(api_key=FRED_API_KEY)
    st.session_state.fred_key = FRED_API_KEY

fred: Fred = st.session_state.fred

# ── Index normalisation ────────────────────────────────────────────────────────
def to_datetime_index(s: pd.Series) -> pd.Series:
    """
    Guarantee a tz-naive DatetimeIndex.
    st.cache_data round-trips Series through Arrow/Parquet which can silently
    convert DatetimeIndex → int64 (nanoseconds). Fix it unconditionally here
    rather than only on cache-miss inside fetch_fred.
    """
    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, unit="ns", errors="coerce")
    elif idx.tz is not None:
        idx = idx.tz_localize(None)
    s = s.copy()
    s.index = idx
    return s

def safe_reindex(source: pd.Series, target: pd.Series) -> pd.Series:
    """Reindex source onto target's index with ffill, fixing dtypes first."""
    source = to_datetime_index(source)
    target = to_datetime_index(target)
    return source.reindex(target.index, method="ffill")

def resample(s: pd.Series) -> pd.Series:
    s = to_datetime_index(s)
    return s.resample(RESAMPLE).last().ffill()

# ── FRED fetch ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fred(series_id: str, start: str) -> pd.Series:
    try:
        s = fred.get_series(series_id, observation_start=start)
        return resample(s).rename(series_id)
    except Exception as e:
        st.warning(f"FRED `{series_id}` failed: {e}")
        return pd.Series(dtype=float, name=series_id)

def fx(fred_id: str) -> pd.Series:
    return fetch_fred(fred_id, start_str)

# ── Central bank fetchers ──────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_fed(start: str) -> pd.Series:
    return fetch_fred("WALCL", start)           # USD millions

@st.cache_data(ttl=86400, show_spinner=False)
def get_tga(start: str) -> pd.Series:
    return fetch_fred("WTREGEN", start)         # USD billions

@st.cache_data(ttl=86400, show_spinner=False)
def get_rrp(start: str) -> pd.Series:
    return fetch_fred("RRPONTSYD", start)       # USD billions

@st.cache_data(ttl=86400, show_spinner=False)
def get_ecb(start: str) -> pd.Series:
    """ECBASSETSW: EUR millions → USD billions."""
    ecb_eur = fetch_fred("ECBASSETSW", start)
    eur_usd = fx("DEXUSEU")
    return (safe_reindex(ecb_eur, eur_usd) * eur_usd / 1_000).rename("ECB")

@st.cache_data(ttl=86400, show_spinner=False)
def get_boj(start: str) -> pd.Series:
    """JPNASSETS: JPY trillions → USD billions."""
    boj_jpy_b = fetch_fred("JPNASSETS", start) * 1_000   # JPY trillions → billions
    jpy_per_usd = fx("DEXJPUS")
    return (safe_reindex(boj_jpy_b, jpy_per_usd) / jpy_per_usd).rename("BOJ")

@st.cache_data(ttl=86400, show_spinner=False)
def get_boe(start: str) -> pd.Series:
    """
    BOE: FRED ECBASSETSW sister series doesn't exist for BOE, so we use the
    BOE's Interactive Database CSV export for series RPQB55A (total assets,
    GBP millions, weekly). The URL below is the direct CSV download link.
    Falls back to FRED UKASSETS (discontinued 2014) for early history.
    """
    import io, requests
    try:
        # RPQB55A = Bank of England total assets (Bank Return), GBP millions, weekly
        url = (
            "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
            "?Travel=NIxAZxSUx&FromSeries=1&ToSeries=50&DAT=RNG"
            "&FD=1&FM=Jan&FY=2015&TD=31&TM=Dec&TY=2030"
            "&FNY=Y&CSVF=TT&html.x=66&html.y=26"
            "&SeriesCodes=RPQB55A&UsingCodes=Y"
        )
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        # BOE CSV: first row = title, second row = units, third row = date header
        # Use comment='#' and skip metadata rows by finding where dates start
        lines = resp.text.strip().splitlines()
        # Find first line where col 0 parses as a date
        data_start = next(
            i for i, ln in enumerate(lines)
            if pd.to_datetime(ln.split(",")[0].strip(), dayfirst=True, errors="coerce")
            is not pd.NaT
        )
        df = pd.read_csv(
            io.StringIO("\n".join(lines[data_start:])),
            header=None, names=["date", "value"]
        )
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        assets_gbp_m = pd.to_numeric(df["value"], errors="coerce").dropna()
        assets_gbp_m = resample(assets_gbp_m)
        gbp_usd = fx("DEXUSUK")
        return (safe_reindex(assets_gbp_m, gbp_usd) * gbp_usd / 1_000).rename("BOE")
    except Exception as e:
        st.warning(f"BOE direct API failed ({e}). Falling back to FRED UKASSETS (ends ~2014).")
        try:
            boe_gbp = fetch_fred("UKASSETS", start)   # GBP millions, monthly
            gbp_usd = fx("DEXUSUK")
            return (safe_reindex(boe_gbp, gbp_usd) * gbp_usd / 1_000).rename("BOE")
        except Exception as e2:
            st.warning(f"BOE FRED fallback also failed: {e2}")
            return pd.Series(dtype=float, name="BOE")

@st.cache_data(ttl=86400, show_spinner=False)
def get_boc(start: str) -> pd.Series:
    """
    Bank of Canada total assets via Valet API (JSON).
    The JSON structure is:
      { "seriesDetail": {...}, "observations": [ {"d": "YYYY-MM-DD", "B2_TOTA": {"v": "123"}}, ... ] }
    Series B2_TOTA = total assets, CAD millions, weekly.
    """
    import requests
    try:
        url = (
            "https://www.bankofcanada.ca/valet/observations/B2_TOTA/json"
            f"?start_date={start}"
        )
        data = requests.get(url, timeout=30).json()
        # Key is "observations" — a list of dicts with "d" and series name
        obs    = data["observations"]          # list of {"d": date, "B2_TOTA": {"v": val}}
        dates  = pd.to_datetime([o["d"] for o in obs])
        values = pd.to_numeric([o["B2_TOTA"]["v"] for o in obs], errors="coerce")
        assets_cad_m = pd.Series(values, index=dates).dropna()
        assets_cad_b = resample(assets_cad_m) / 1_000   # CAD millions → billions
        cad_per_usd  = fx("DEXCAUS")
        return (safe_reindex(assets_cad_b, cad_per_usd) / cad_per_usd).rename("BOC")
    except Exception as e:
        st.warning(f"BOC Valet API failed: {e}")
        return pd.Series(dtype=float, name="BOC")

@st.cache_data(ttl=86400, show_spinner=False)
def get_rba(start: str) -> pd.Series:
    """
    RBA total assets from RBA Table A1 CSV.
    The RBA CSV has a metadata block at the top. The structure is:
      Row 0: table title
      Row 1: series descriptions  (e.g. "Total assets")
      Row 2: series IDs           (e.g. ARBALCRAW)
      Row 3: frequency
      Row 4: units                (e.g. $ millions)
      Row 5: source
      Row 6: publication date
      Row 7+: blank then data with "Date" in col 0
    We find the data start by scanning for the first parseable date in col 0.
    Total assets series ID is ARBALCRAW.
    """
    import io, requests
    try:
        url  = "https://www.rba.gov.au/statistics/tables/csv/a1-data.csv"
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()

        # Find the row index of the series-ID row (contains "ARBALCRAW")
        series_id_row = next(
            (i for i, ln in enumerate(lines) if "ARBALCRAW" in ln), None
        )
        # Find the header row (contains "Date" in first column)
        header_row = next(
            (i for i, ln in enumerate(lines)
             if ln.split(",")[0].strip().lower() == "date"), None
        )
        if series_id_row is None or header_row is None:
            raise ValueError("Could not find series ID row or Date header in RBA CSV.")

        # Read column positions from the series-ID row
        id_cols = lines[series_id_row].split(",")
        total_assets_col = next(
            (i for i, v in enumerate(id_cols) if "ARBALCRAW" in v), None
        )
        if total_assets_col is None:
            raise ValueError("ARBALCRAW not found in series ID row.")

        # Read the data block
        df = pd.read_csv(
            io.StringIO("\n".join(lines[header_row:])),
            header=0, dtype=str
        )
        date_col  = df.columns[0]
        value_col = df.columns[total_assets_col]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        assets_aud_m = pd.to_numeric(df[value_col].str.replace(",", ""), errors="coerce").dropna()
        assets_aud_m = resample(assets_aud_m)
        aud_usd = fx("DEXUSAL")
        return (safe_reindex(assets_aud_m, aud_usd) * aud_usd / 1_000).rename("RBA")
    except Exception as e:
        st.warning(f"RBA fetch failed: {e}")
        return pd.Series(dtype=float, name="RBA")

@st.cache_data(ttl=86400, show_spinner=False)
def get_snb(start: str) -> pd.Series:
    """
    SNB total assets via SNB data portal CSV API.
    Cube: snbbipo (SNB balance sheet), dimension D0=ATASSET (total assets), CHF millions, monthly.
    API pattern: https://data.snb.ch/api/cube/{cube}/data/csv/en?dimSel=D0(ATASSET)&fromDate=YYYY-MM
    Falls back to FRED SNBFORCURPOS (foreign currency positions, ~95% of SNB assets).
    """
    import io, requests
    try:
        from_date = start[:7]   # YYYY-MM
        url = (
            f"https://data.snb.ch/api/cube/snbbipo/data/csv/en"
            f"?dimSel=D0(ATASSET)&fromDate={from_date}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        # SNB CSV format: semicolon-delimited, columns: Date;D0;Value
        df = pd.read_csv(io.StringIO(resp.text), sep=";")
        df.columns = df.columns.str.strip()
        # Expect columns: Date, D0 (dimension label), Value
        date_col  = next(c for c in df.columns if "date" in c.lower())
        value_col = next(c for c in df.columns if "value" in c.lower())
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        snb_chf_m = pd.to_numeric(df[value_col], errors="coerce").dropna()
        snb_chf_b = resample(snb_chf_m) / 1_000   # CHF millions → billions
        chf_per_usd = fx("DEXSZUS")
        return (safe_reindex(snb_chf_b, chf_per_usd) / chf_per_usd).rename("SNB")
    except Exception as e:
        st.warning(f"SNB primary API failed ({e}). Falling back to FRED SNBFORCURPOS proxy.")
        try:
            # SNBFORCURPOS = foreign currency investments, CHF millions (~95% of total assets)
            snb_chf_m   = fetch_fred("SNBFORCURPOS", start)
            chf_per_usd = fx("DEXSZUS")
            return (safe_reindex(snb_chf_m, chf_per_usd) / chf_per_usd / 1_000).rename("SNB")
        except Exception as e2:
            st.warning(f"SNB FRED fallback also failed: {e2}")
            return pd.Series(dtype=float, name="SNB")

# ── GLI assembly ───────────────────────────────────────────────────────────────
def build_gli(components: dict) -> pd.Series:
    """GLI = FED(USD B) - TGA - RRP + ECB + BOJ + BOC + RBA + BOE + SNB"""
    def get(key) -> pd.Series:
        s = components.get(key, pd.Series(dtype=float))
        return to_datetime_index(s)

    fed = get("FED") * 0.001               # USD millions → billions
    tga = get("TGA")
    rrp = get("RRP")

    gli = fed.sub(tga, fill_value=0).sub(rrp, fill_value=0)
    for key in ("ECB", "BOJ", "BOC", "RBA", "BOE", "SNB"):
        gli = gli.add(get(key), fill_value=0)

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
        ("FED assets (WALCL)",  "FED", lambda: get_fed(start_str)),
        ("TGA (WTREGEN)",       "TGA", lambda: get_tga(start_str)),
        ("RRP (RRPONTSYD)",     "RRP", lambda: get_rrp(start_str)),
        ("ECB (ECBASSETSW)",    "ECB", lambda: get_ecb(start_str)),
        ("BOJ (JPNASSETS)",     "BOJ", lambda: get_boj(start_str)),
        ("BOE (BOEBSTASGBP)",   "BOE", lambda: get_boe(start_str)),
        ("BOC",                 "BOC", lambda: get_boc(start_str)),
        ("RBA (RBATOTASSETS)",  "RBA", lambda: get_rba(start_str)),
        ("SNB (SNBASSETS)",     "SNB", lambda: get_snb(start_str)),
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

# ── Coverage summary & raw data ────────────────────────────────────────────────
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
