import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime
import requests

st.set_page_config(page_title="Global Liquidity Index vs SPY & BTC", layout="wide")
st.title("🌍 Global Liquidity Index vs SPY & BTC")
st.markdown("**GLI** = FED − TGA − RRP + ECB + BOJ + BOC + RBA + BOE + SNB")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    FRED_API_KEY = st.text_input("FRED API Key", type="password",
                                 help="Free key at https://fred.stlouisfed.org/docs/api/api_key.html")
    START_DATE = st.date_input("Start Date", value=datetime(2015, 1, 1))
    if not FRED_API_KEY:
        st.warning("Enter your FRED API key to continue.")
        st.stop()

RESAMPLE = "W-FRI"
start_str = START_DATE.strftime("%Y-%m-%d")
end_str = datetime.today().strftime("%Y-%m-%d")

# ── Fred client ────────────────────────────────────────────────────────────────
if "fred" not in st.session_state or st.session_state.get("fred_key") != FRED_API_KEY:
    st.session_state.fred = Fred(api_key=FRED_API_KEY)
    st.session_state.fred_key = FRED_API_KEY
fred: Fred = st.session_state.fred

# ── Helper functions (index safety) ───────────────────────────────────────────
def to_datetime_index(s: pd.Series) -> pd.Series:
    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
    elif idx.tz is not None:
        idx = idx.tz_localize(None)
    s = s.copy()
    s.index = idx
    return s

def safe_reindex(source: pd.Series, target: pd.Series) -> pd.Series:
    source = to_datetime_index(source)
    target = to_datetime_index(target)
    return source.reindex(target.index, method="ffill")

def resample(s: pd.Series) -> pd.Series:
    return to_datetime_index(s).resample(RESAMPLE).last().ffill()

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

# ── Central bank fetchers (fixed) ──────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def get_fed(start: str) -> pd.Series:
    return fetch_fred("WALCL", start)

@st.cache_data(ttl=86400, show_spinner=False)
def get_tga(start: str) -> pd.Series:
    return fetch_fred("WTREGEN", start)

@st.cache_data(ttl=86400, show_spinner=False)
def get_rrp(start: str) -> pd.Series:
    return fetch_fred("RRPONTSYD", start)

@st.cache_data(ttl=86400, show_spinner=False)
def get_ecb(start: str) -> pd.Series:
    ecb_eur = fetch_fred("ECBASSETSW", start)
    eur_usd = fx("DEXUSEU")
    return (safe_reindex(ecb_eur, eur_usd) * eur_usd / 1000).rename("ECB")

@st.cache_data(ttl=86400, show_spinner=False)
def get_boj(start: str) -> pd.Series:
    """FIXED: JPNASSETS unit = 100 Million Yen → multiplier 0.1 for billions Yen"""
    boj = fetch_fred("JPNASSETS", start) * 0.1          # ← corrected multiplier
    jpy_usd = fx("DEXJPUS")
    return (safe_reindex(boj, jpy_usd) / jpy_usd).rename("BOJ")

@st.cache_data(ttl=86400, show_spinner=False)
def get_boe(start: str) -> pd.Series:
    """Stable CSV endpoint (official BOE weekly report)"""
    try:
        url = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp?Travel=NIxAZxSUx&FromSeries=1&ToSeries=50&DAT=RNG&FD=1&FM=Jan&FY=2018&TD=31&TM=Dec&TY=2027&FNY=Y&CSVF=TT&html.x=66&html.y=26&SeriesCodes=RPWB55A,RPWB56A,RPWB59A,RPWB67A,RPWZ4TJ,RPWZ4TK,RPWZOQ4,RPWZ4TL,RPWZ4TM,RPWZOI7,RPWZ4TN&UsingCodes=Y&Filter=N&title=Bank%20of%20England%20Weekly%20Report&VPD=Y"
        df = pd.read_csv(url, skiprows=1)
        df.columns = df.columns.str.strip()
        date_col = next(c for c in df.columns if "date" in c.lower() or "period" in c.lower())
        asset_col = next(c for c in df.columns if "total asset" in str(c).lower() and "liability" not in str(c).lower())
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        assets = pd.to_numeric(df[asset_col].astype(str).str.replace(",", ""), errors="coerce").dropna()
        assets = resample(assets)
        gbp_usd = fx("DEXUSUK")
        return (safe_reindex(assets, gbp_usd) * gbp_usd / 1000).rename("BOE")
    except Exception as e:
        st.warning(f"BOE: {e} (using zero for now)")
        return pd.Series(dtype=float, name="BOE")

# (BOC, RBA, SNB functions are unchanged from your file — they already work well)

@st.cache_data(ttl=86400, show_spinner=False)
def get_boc(start: str) -> pd.Series:
    # ... (your original BOC function – unchanged) ...
    # (copy-paste your original get_boc here if you want; it works)

@st.cache_data(ttl=86400, show_spinner=False)
def get_rba(start: str) -> pd.Series:
    # ... (your original RBA function – unchanged) ...

@st.cache_data(ttl=86400, show_spinner=False)
def get_snb(start: str) -> pd.Series:
    # ... (your original SNB function – unchanged) ...

# ── GLI assembly ───────────────────────────────────────────────────────────────
def build_gli(components: dict) -> pd.Series:
    def get(key): 
        s = components.get(key, pd.Series(dtype=float))
        return to_datetime_index(s)

    fed = get("FED") * 0.001
    tga = get("TGA")
    rrp = get("RRP")

    gli = fed.sub(tga, fill_value=0).sub(rrp, fill_value=0)
    for key in ("ECB", "BOJ", "BOC", "RBA", "BOE", "SNB"):
        gli = gli.add(get(key), fill_value=0)

    gli.name = "GLI_USD_B"
    return gli

# ── Market data & Plot (unchanged from your file) ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_market(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(["SPY", "BTC-USD"], start=start, end=end, progress=False)
    return raw["Close"].resample(RESAMPLE).last().ffill()

def plot_gli(gli: pd.Series, market: pd.DataFrame) -> go.Figure | None:
    # (your original plot_gli function – unchanged; it's excellent)

# ── Main ───────────────────────────────────────────────────────────────────────
with st.status("Loading data...", expanded=True) as status:
    components = {}
    steps = [
        ("FED assets", "FED", lambda: get_fed(start_str)),
        ("TGA",        "TGA", lambda: get_tga(start_str)),
        ("RRP",        "RRP", lambda: get_rrp(start_str)),
        ("ECB",        "ECB", lambda: get_ecb(start_str)),
        ("BOJ",        "BOJ", lambda: get_boj(start_str)),
        ("BOE",        "BOE", lambda: get_boe(start_str)),
        ("BOC",        "BOC", lambda: get_boc(start_str)),
        ("RBA",        "RBA", lambda: get_rba(start_str)),
        ("SNB",        "SNB", lambda: get_snb(start_str)),
    ]
    for label, key, fn in steps:
        st.write(f"Fetching {label}...")
        components[key] = fn()

    st.write("Building GLI...")
    gli = build_gli(components)

    st.write("Fetching SPY & BTC...")
    market = get_market(start_str, end_str)

    status.update(label="✅ Done", state="complete", expanded=False)

# Latest GLI diagnostic (so you can see the real number instantly)
latest_gli = gli.iloc[-1] if not gli.empty else np.nan
st.metric("Latest Global Liquidity Index (USD billions)", f"{latest_gli:,.0f}")

fig = plot_gli(gli, market)
if fig:
    st.plotly_chart(fig, use_container_width=True)

# (rest of your expander + raw data table remains exactly as you had it)

with st.expander("Coverage summary"):
    summary = {k: f"{v.notna().sum()} weeks" for k, v in components.items()}
    summary["GLI"] = f"{gli.notna().sum()} weeks"
    st.table(pd.Series(summary, name="Non-null observations"))

if st.checkbox("Show raw data"):
    combined = pd.concat([gli, market], axis=1).dropna(how="all")
    st.dataframe(combined.style.format("{:,.2f}"))

st.caption("Coverage: FED · ECB · BOJ · BOC · RBA · BOE · SNB | Next: PBC, RBI, etc.")
