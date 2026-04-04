import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime
import numpy as np
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def resample(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    return s.resample(RESAMPLE).last().ffill()

def safe_reindex(source: pd.Series, target: pd.Series) -> pd.Series:
    source = resample(source)
    target = resample(target)
    return source.reindex(target.index, method="ffill")

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

# ── Central Bank Fetchers ──────────────────────────────────────────────────────
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
    """Corrected scaling: JPNASSETS in 100 million JPY"""
    boj = fetch_fred("JPNASSETS", start)
    jpy_usd = fx("DEXJPUS")
    return (safe_reindex(boj, jpy_usd) * 0.1 / jpy_usd).rename("BOJ")

@st.cache_data(ttl=86400, show_spinner=False)
def get_boe(start: str) -> pd.Series:
    """BOE - simplified due to frequent 403 errors"""
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
        st.warning(f"BOE: HTTP Error 403 Forbidden (using zero for now)")
        return pd.Series(dtype=float, name="BOE")

@st.cache_data(ttl=86400, show_spinner=False)
def get_boc(start: str) -> pd.Series:
    try:
        url = f"https://www.bankofcanada.ca/valet/observations/group/b2_weekly/json?start_date={start}"
        data = requests.get(url, timeout=30).json()
        obs = data["observations"]
        dates = pd.to_datetime([o["d"] for o in obs])
        values = pd.to_numeric([o["V36610"]["v"] for o in obs], errors="coerce")
        assets = pd.Series(values, index=dates).dropna()
        assets = resample(assets) / 1000
        cad_usd = fx("DEXCAUS")
        return (safe_reindex(assets, cad_usd) / cad_usd).rename("BOC")
    except Exception as e:
        st.warning(f"BOC: {e}")
        return pd.Series(dtype=float, name="BOC")

@st.cache_data(ttl=86400, show_spinner=False)
def get_rba(start: str) -> pd.Series:
    try:
        url = "https://www.rba.gov.au/statistics/tables/csv/a1-data.csv"
        df = pd.read_csv(url, skiprows=11, header=None)
        df.columns = ["date"] + [f"col{i}" for i in range(1, len(df.columns))]
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        assets = pd.to_numeric(df.iloc[:, 13], errors="coerce").dropna()
        assets = resample(assets) / 1000
        aud_usd = fx("DEXUSAL")
        return (safe_reindex(assets, aud_usd) * aud_usd).rename("RBA")
    except Exception as e:
        st.warning(f"RBA: {e}")
        return pd.Series(dtype=float, name="RBA")

@st.cache_data(ttl=86400, show_spinner=False)
def get_snb(start: str) -> pd.Series:
    try:
        s = fetch_fred("SNBFORCURPOS", start)
        chf_usd = fx("DEXSZUS")
        return (safe_reindex(s, chf_usd) / chf_usd / 1000).rename("SNB")
    except Exception as e:
        st.warning(f"SNB: {e}")
        return pd.Series(dtype=float, name="SNB")

# ── Build GLI (robust) ─────────────────────────────────────────────────────────
def build_gli(components: dict) -> pd.Series:
    def get(key):
        s = components.get(key, pd.Series(dtype=float))
        return resample(s)

    fed = get("FED") * 0.001
    tga = get("TGA")
    rrp = get("RRP")

    gli = fed.sub(tga, fill_value=0).sub(rrp, fill_value=0)
    for key in ("ECB", "BOJ", "BOC", "RBA", "BOE", "SNB"):
        gli = gli.add(get(key), fill_value=0)

    return gli.rename("GLI_USD_B")

# ── Market data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_market(start: str, end: str):
    raw = yf.download(["SPY", "BTC-USD"], start=start, end=end, progress=False)
    return raw["Close"].resample(RESAMPLE).last().ffill()

# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_gli(gli: pd.Series, market: pd.DataFrame):
    df = pd.concat([gli, market], axis=1).sort_index().dropna(how="all")
    if df.empty:
        st.error("No data available.")
        return None

    base = df.iloc[0].replace(0, np.nan)
    idx = df.div(base) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04,
                        subplot_titles=("Indexed performance (start = 100)", "GLI (USD trillions)"))

    colors = {"GLI_USD_B": "#3266ad", "SPY": "#1D9E75", "BTC-USD": "#D85A30"}
    names = {"GLI_USD_B": "Global Liquidity Index", "SPY": "SPY", "BTC-USD": "BTC/USD"}

    for col in ["GLI_USD_B", "SPY", "BTC-USD"]:
        if col in idx.columns and not idx[col].isna().all():
            fig.add_trace(go.Scatter(x=idx.index, y=idx[col].round(1),
                                     name=names[col], mode="lines",
                                     line=dict(color=colors[col], width=2.5 if col == "GLI_USD_B" else 1.5)),
                          row=1, col=1)

    if "GLI_USD_B" in df.columns and not df["GLI_USD_B"].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=(df["GLI_USD_B"]/1000).round(2),
                                 name="GLI (raw)", mode="lines",
                                 line=dict(color="#3266ad", width=1.5), showlegend=False),
                      row=2, col=1)

    fig.update_layout(title="Global Liquidity Index vs SPY & BTC",
                      hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      height=750)
    fig.update_yaxes(title_text="Indexed (start=100)", row=1, col=1)
    fig.update_yaxes(title_text="USD Trillions", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig

# ── Main ───────────────────────────────────────────────────────────────────────
with st.status("Loading data... (first load can take 40–80 seconds)", expanded=True) as status:
    components = {}
    steps = [
        ("FED assets", "FED", lambda: get_fed(start_str)),
        ("TGA", "TGA", lambda: get_tga(start_str)),
        ("RRP", "RRP", lambda: get_rrp(start_str)),
        ("ECB", "ECB", lambda: get_ecb(start_str)),
        ("BOJ", "BOJ", lambda: get_boj(start_str)),
        ("BOE", "BOE", lambda: get_boe(start_str)),
        ("BOC", "BOC", lambda: get_boc(start_str)),
        ("RBA", "RBA", lambda: get_rba(start_str)),
        ("SNB", "SNB", lambda: get_snb(start_str)),
    ]
    for label, key, fn in steps:
        st.write(f"Fetching {label}...")
        components[key] = fn()

    st.write("Building GLI...")
    gli = build_gli(components)

    st.write("Fetching SPY & BTC...")
    market = get_market(start_str, end_str)

    status.update(label="✅ Done", state="complete", expanded=False)

# Latest GLI metric
latest_gli = gli.iloc[-1] if not gli.empty else np.nan
st.metric("Latest Global Liquidity Index", f"{latest_gli:,.0f} billion USD")

fig = plot_gli(gli, market)
if fig:
    st.plotly_chart(fig, use_container_width=True)

# Coverage summary
with st.expander("Coverage summary"):
    summary = {k: f"{v.notna().sum()} weeks" for k, v in components.items()}
    summary["GLI"] = f"{gli.notna().sum()} weeks"
    st.table(pd.Series(summary, name="Non-null observations"))

if st.checkbox("Show raw data table"):
    combined = pd.concat([gli, market], axis=1).dropna(how="all")
    st.dataframe(combined.style.format("{:,.2f}"))

st.caption("Coverage: FED · ECB · BOJ · BOC · RBA · BOE · SNB | BOE currently using zero due to access restriction.")
