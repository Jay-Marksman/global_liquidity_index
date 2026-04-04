import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime

st.set_page_config(page_title="Global Liquidity Index vs SPY & BTC", layout="wide")
st.title("🌍 Global Liquidity Index vs SPY & BTC")
st.markdown("**Global Liquidity Index** (FED + ECB + BOJ + others) vs SPY and Bitcoin.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    FRED_API_KEY = st.text_input("FRED API Key", type="password",
                                 help="Free key: https://fred.stlouisfed.org/docs/api/api_key.html")
    START_DATE = st.date_input("Start Date", value=datetime(2015, 1, 1))

    if not FRED_API_KEY:
        st.warning("Enter your FRED API key")
        st.stop()

# Fred client
@st.cache_resource
def get_fred_client(_api_key):
    return Fred(api_key=_api_key)

fred = get_fred_client(FRED_API_KEY)
RESAMPLE_FREQ = "W-FRI"

# Cached fetchers
@st.cache_data(ttl=86400)
def get_fred_series(series_id: str, name: str, start: str):
    try:
        s = fred.get_series(series_id, observation_start=start)
        s = s.resample(RESAMPLE_FREQ).last().ffill()
        s.name = name
        return s
    except Exception as e:
        st.warning(f"FRED {series_id} failed: {e}")
        return pd.Series(dtype=float, name=name)

@st.cache_data(ttl=86400)
def get_fx(fred_id: str, start: str):
    return get_fred_series(fred_id, fred_id, start)

@st.cache_data(ttl=86400)
def get_ecb_total_assets(start: str):
    try:
        s = get_fred_series("ECBASSETSW", "ECB_EUR", start)
        eur_usd = get_fx("DEXUSEU", start)
        return (s.reindex(eur_usd.index, method="ffill") * eur_usd / 1000).rename("ECB")
    except Exception as e:
        st.warning(f"ECB: {e}")
        return pd.Series(dtype=float, name="ECB")

@st.cache_data(ttl=86400)
def get_boj_total_assets(start: str):
    try:
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
        return (df.reindex(jpy_usd.index, method="ffill") / jpy_usd * 0.01).rename("BOJ")
    except Exception:
        try:
            s = get_fred_series("JPNASSETS", "BOJ_JPYT", start) * 1000
            jpy_usd = get_fx("DEXJPUS", start)
            return (s.reindex(jpy_usd.index, method="ffill") / jpy_usd).rename("BOJ")
        except Exception as e:
            st.warning(f"BOJ: {e}")
            return pd.Series(dtype=float, name="BOJ")

@st.cache_data(ttl=86400)
def get_boe_total_assets(start: str):
    """Simplified BOE - disabled for now due to 403. Can be re-enabled with better scraper later."""
    st.info("BOE direct fetch temporarily disabled (403 error).")
    return pd.Series(dtype=float, name="BOE")

# GLI Calculation
def build_gli(df: pd.DataFrame) -> pd.Series:
    fed = df.get("FED_ASSETS", pd.Series(0)) * 0.001
    tga = df.get("TGA", pd.Series(0))
    rrp = df.get("RRP", pd.Series(0))
    ecb = df.get("ECB", pd.Series(0))
    boj = df.get("BOJ", pd.Series(0))
    boe = df.get("BOE", pd.Series(0))

    gli = fed - tga - rrp + ecb + boj + boe
    return gli.rename("GLI_USD_B")

# Market data
@st.cache_data(ttl=3600)
def get_market_data(start: str, end: str):
    tickers = yf.download(["SPY", "BTC-USD"], start=start, end=end, progress=False)
    return tickers["Close"].resample(RESAMPLE_FREQ).last().ffill()

# Plot (fixed index type issue)
def plot_gli(gli: pd.Series, market: pd.DataFrame):
    df = pd.concat([gli, market], axis=1)
    
    # Force datetime index to prevent mixed type error
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors="coerce")
    
    df = df.sort_index().dropna(how="all")
    
    if df.empty:
        st.error("No data available after alignment.")
        return None

    st.caption(f"Data shape: {df.shape} | GLI non-NaN: {gli.notna().sum()}")

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
                                     line=dict(color=colors[col], width=2.5 if col == "GLI_USD_B" else 1.5)),
                          row=1, col=1)

    if "GLI_USD_B" in df.columns:
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

# Main
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
}

with st.spinner("Fetching data... (first load can take 30-60 seconds)"):
    raw = {}
    for sid, name in FRED_SERIES.items():
        raw[name] = get_fred_series(sid, name, start_str)

    raw["ECB"] = get_ecb_total_assets(start_str)   # redundant safety
    raw["BOJ"] = get_boj_total_assets(start_str)
    raw["BOE"] = get_boe_total_assets(start_str)

    df_raw = pd.DataFrame(raw).sort_index().ffill()
    gli = build_gli(df_raw)
    market = get_market_data(start_str, end_str)

    fig = plot_gli(gli, market)

if fig:
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show raw data table"):
    combined = pd.concat([gli, market], axis=1).dropna(how="all")
    st.dataframe(combined.style.format("{:,.2f}"))

st.caption("Current reliable coverage: **FED + ECB + BOJ**. BOE temporarily skipped due to access issue.")
