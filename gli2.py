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
st.markdown("**Global Liquidity Index** = FED − TGA − RRP + ECB + BOJ + BOC + RBA + SNB")

# Sidebar
with st.sidebar:
    st.header("Settings")
    FRED_API_KEY = st.text_input("FRED API Key", type="password", 
                                 help="Free key from https://fred.stlouisfed.org/docs/api/api_key.html")
    START_DATE = st.date_input("Start Date", value=datetime(2015, 1, 1))
    
    if not FRED_API_KEY:
        st.warning("Enter your FRED API key to load data")
        st.stop()

RESAMPLE_FREQ = "W-FRI"

# Fred client
@st.cache_resource
def get_fred_client(_api_key):
    return Fred(api_key=_api_key)

fred = get_fred_client(FRED_API_KEY)

# ── Cached fetchers ─────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_fred_series(series_id: str, name: str, start: str):
    try:
        s = fred.get_series(series_id, observation_start=start)
        s = s.resample(RESAMPLE_FREQ).last().ffill()
        s.name = name
        return s
    except Exception as e:
        st.warning(f"FRED {series_id} ({name}) failed: {e}")
        return pd.Series(dtype=float, name=name)

@st.cache_data(ttl=86400)
def get_fx(fred_id: str, start: str):
    return get_fred_series(fred_id, fred_id, start)

@st.cache_data(ttl=86400)
def get_ecb_total_assets(start: str):
    try:
        s = get_fred_series("ECBASSETSW", "ECB_EUR", start)
        eur_usd = get_fx("DEXUSEU", start)
        df_aligned = s.reindex(eur_usd.index, method="ffill") * eur_usd / 1000
        return df_aligned.rename("ECB")
    except Exception as e:
        st.warning(f"ECB fetch failed: {e}")
        return pd.Series(dtype=float, name="ECB")

@st.cache_data(ttl=86400)
def get_boj_total_assets(start: str):
    try:
        # Primary: BOJ official CSV
        url = "https://www.stat-search.boj.or.jp/ssi/mtshtml/bs01_m_1_en.csv"
        df = pd.read_csv(url, skiprows=4, encoding="shift_jis")
        df.columns = df.columns.str.strip()
        df = df.iloc[:, :2]
        df.columns = ["date", "BOJ_JPY"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna().set_index("date")["BOJ_JPY"]
        df = pd.to_numeric(df.str.replace(",", ""), errors="coerce").dropna()
        df = df.resample(RESAMPLE_FREQ).last().ffill()

        jpy_usd = get_fx("DEXJPUS", start)
        df_aligned = df.reindex(jpy_usd.index, method="ffill") / jpy_usd * 0.01
        return df_aligned.rename("BOJ")
    except Exception:
        try:
            s = get_fred_series("JPNASSETS", "BOJ_JPYT", start) * 0.1   # Correct scaling
            jpy_usd = get_fx("DEXJPUS", start)
            return (s.reindex(jpy_usd.index, method="ffill") / jpy_usd).rename("BOJ")
        except Exception as e:
            st.warning(f"BOJ fetch failed: {e}")
            return pd.Series(dtype=float, name="BOJ")

@st.cache_data(ttl=86400)
def get_boc_total_assets(start: str):
    try:
        url = f"https://www.bankofcanada.ca/valet/observations/group/b2_weekly/json?start_date={start}"
        data = requests.get(url, timeout=30).json()
        obs = data["observations"]
        dates = pd.to_datetime([o["d"] for o in obs])
        values = pd.to_numeric([o.get("V36610", {}).get("v", 0) for o in obs], errors="coerce")
        assets = pd.Series(values, index=dates).dropna()
        assets = assets.resample(RESAMPLE_FREQ).last().ffill() / 1000
        cad_usd = get_fx("DEXCAUS", start)
        return (assets.reindex(cad_usd.index, method="ffill") / cad_usd).rename("BOC")
    except Exception as e:
        st.warning(f"BOC: {e}")
        return pd.Series(dtype=float, name="BOC")

@st.cache_data(ttl=86400)
def get_rba_total_assets(start: str):
    try:
        url = "https://www.rba.gov.au/statistics/tables/csv/a1-data.csv"
        df = pd.read_csv(url, skiprows=11, header=None)
        df.columns = ["date"] + [f"col{i}" for i in range(1, len(df.columns))]
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        assets = pd.to_numeric(df.iloc[:, 13], errors="coerce").dropna()
        assets = assets.resample(RESAMPLE_FREQ).last().ffill() / 1000
        aud_usd = get_fx("DEXUSAL", start)
        return (assets.reindex(aud_usd.index, method="ffill") * aud_usd).rename("RBA")
    except Exception as e:
        st.warning(f"RBA: {e}")
        return pd.Series(dtype=float, name="RBA")

@st.cache_data(ttl=86400)
def get_snb_total_assets(start: str):
    try:
        s = get_fred_series("SNBFORCURPOS", "SNB_CHF", start)
        chf_usd = get_fx("DEXSZUS", start)
        return (s.reindex(chf_usd.index, method="ffill") / chf_usd / 1000).rename("SNB")
    except Exception as e:
        st.warning(f"SNB: {e}")
        return pd.Series(dtype=float, name="SNB")

# ── Build GLI ───────────────────────────────────────────────────────────────
def build_gli(df: pd.DataFrame) -> pd.Series:
    fed = df.get("FED_ASSETS", pd.Series(0)) * 0.001
    tga = df.get("TGA", pd.Series(0))
    rrp = df.get("RRP", pd.Series(0))
    ecb = df.get("ECB", pd.Series(0))
    boj = df.get("BOJ", pd.Series(0))
    boc = df.get("BOC", pd.Series(0))
    rba = df.get("RBA", pd.Series(0))
    snb = df.get("SNB", pd.Series(0))

    gli = fed - tga - rrp + ecb + boj + boc + rba + snb
    return gli.rename("GLI_USD_B")

# ── Market data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_market_data(start: str, end: str):
    tickers = yf.download(["SPY", "BTC-USD"], start=start, end=end, progress=False)
    prices = tickers["Close"].resample(RESAMPLE_FREQ).last().ffill()
    return prices

# ── Plot ────────────────────────────────────────────────────────────────────
def plot_gli(gli: pd.Series, market: pd.DataFrame):
    df = pd.concat([gli, market], axis=1).sort_index().dropna(how="all")
    
    if df.empty:
        st.error("No data available.")
        return None

    st.caption(f"Data shape: {df.shape} | GLI non-NaN weeks: {gli.notna().sum()}")

    first = df.index[0]
    idx = df.loc[first:].div(df.loc[first]) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        subplot_titles=("Indexed Performance (start=100)", "GLI (USD Trillions)"))

    colors = {"GLI_USD_B": "#3266ad", "SPY": "#1D9E75", "BTC-USD": "#D85A30"}
    names = {"GLI_USD_B": "Global Liquidity Index", "SPY": "SPY", "BTC-USD": "BTC/USD"}

    for col in ["GLI_USD_B", "SPY", "BTC-USD"]:
        if col in idx.columns:
            fig.add_trace(go.Scatter(x=idx.index, y=idx[col].round(1),
                                     name=names[col], mode="lines",
                                     line=dict(color=colors[col], width=2.5 if col=="GLI_USD_B" else 1.5)),
                          row=1, col=1)

    if "GLI_USD_B" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=(df["GLI_USD_B"]/1000).round(2),
                                 name="GLI raw", mode="lines",
                                 line=dict(color="#3266ad", width=1.5), showlegend=False),
                      row=2, col=1)

    fig.update_layout(title="Global Liquidity Index vs SPY & BTC",
                      hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                      height=750)
    fig.update_yaxes(title="Indexed (start=100)", row=1, col=1)
    fig.update_yaxes(title="USD Trillions", row=2, col=1)

    return fig

# ── Main ────────────────────────────────────────────────────────────────
start_str = START_DATE.strftime("%Y-%m-%d")
end_str = datetime.today().strftime("%Y-%m-%d")

FRED_SERIES = {
    "WALCL": "FED_ASSETS",
    "WTREGEN": "TGA",
    "RRPONTSYD": "RRP",
    "ECBASSETSW": "ECB_EUR",
    "JPNASSETS": "BOJ_JPYT",
    "DEXUSEU": "EUR_USD",
    "DEXJPUS": "JPY_USD",
    "DEXCAUS": "CAD_USD_INV",
    "DEXUSAL": "AUD_USD",
    "DEXSZUS": "CHF_USD_INV",
}

with st.spinner("Fetching data... (first run can take 30-70 seconds)"):
    raw = {}
    for sid, name in FRED_SERIES.items():
        raw[name] = get_fred_series(sid, name, start_str)

    raw["ECB"] = get_ecb_total_assets(start_str)
    raw["BOJ"] = get_boj_total_assets(start_str)
    raw["BOC"] = get_boc_total_assets(start_str)
    raw["RBA"] = get_rba_total_assets(start_str)
    raw["SNB"] = get_snb_total_assets(start_str)

    df_raw = pd.DataFrame(raw).sort_index().ffill()
    gli = build_gli(df_raw)
    market = get_market_data(start_str, end_str)

    fig = plot_gli(gli, market)

if fig:
    st.plotly_chart(fig, width="stretch")   # Fixed deprecation warning

if st.checkbox("Show raw data"):
    combined = pd.concat([gli, market], axis=1).dropna(how="all")
    st.dataframe(combined.style.format("{:,.2f}"))

# Latest GLI metric
latest = gli.iloc[-1] if not gli.empty else 0
st.metric("Latest Global Liquidity Index", f"{latest:,.0f} billion USD")

st.caption("""Current coverage: FED + ECB + BOJ + BOC + RBA + SNB  
BOE skipped due to access issues. More banks (PBC, RBI, etc.) can be added later.""")
