import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime

st.set_page_config(page_title="Global Liquidity Index vs SPY & BTC", layout="wide")
st.title("🌍 Global Liquidity Index vs SPY & BTC")
st.markdown("""
This app computes a **Global Liquidity Index** from major central bank balance sheets and compares it to SPY and Bitcoin.  
**Paste your free FRED API key** below to fetch the data.
""")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    FRED_API_KEY = st.text_input("FRED API Key", type="password", 
                                 help="Get one at https://fred.stlouisfed.org/docs/api/api_key.html")
    
    START_DATE = st.date_input("Start Date", value=datetime(2015, 1, 1))
    RESAMPLE_FREQ = "W-FRI"

    if not FRED_API_KEY:
        st.warning("⚠️ Please enter your FRED API key")
        st.stop()

# Create Fred client once (outside cached functions)
@st.cache_resource
def get_fred_client(_api_key: str):
    return Fred(api_key=_api_key)

fred = get_fred_client(FRED_API_KEY)

# ── Cached data fetching functions ──────────────────────────────────────────
@st.cache_data(ttl=86400)  # 24 hours
def get_fred_series(series_id: str, name: str, start: str):
    s = fred.get_series(series_id, observation_start=start)
    s = s.resample(RESAMPLE_FREQ).last().ffill()
    s.name = name
    return s

@st.cache_data(ttl=86400)
def get_fx(fred_id: str, start: str):
    return get_fred_series(fred_id, fred_id, start)

@st.cache_data(ttl=86400)
def get_ecb_total_assets(start: str):
    url = "https://data-api.ecb.europa.eu/service/data/ILM/W.U2.C.A1.U4.EUR?format=csvdata&startPeriod=2015-01-01"
    try:
        df = pd.read_csv(url)
        df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
        df = df.set_index("TIME_PERIOD")["OBS_VALUE"]
        df = df.resample(RESAMPLE_FREQ).last().ffill()
        df.name = "ECB"
        
        eur_usd = get_fx("DEXUSEU", start)
        df_aligned = df.reindex(eur_usd.index, method="ffill") * eur_usd / 1000
        return df_aligned
    except Exception as e:
        st.warning(f"ECB fetch failed: {e}")
        return pd.Series(dtype=float, name="ECB")

@st.cache_data(ttl=86400)
def get_boj_total_assets(start: str):
    try:
        # Primary source
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
        # FRED fallback
        try:
            s = get_fred_series("JPNASSETS", "BOJ_JPYT", start) * 1000
            jpy_usd = get_fx("DEXJPUS", start)
            return (s.reindex(jpy_usd.index, method="ffill") / jpy_usd).rename("BOJ")
        except Exception as e:
            st.warning(f"BOJ fetch failed: {e}")
            return pd.Series(dtype=float, name="BOJ")

@st.cache_data(ttl=86400)
def get_pboc_total_assets(start: str):
    try:
        s = get_fred_series("CHASSETS", "PBC_CNY", start)
        cny_usd = get_fx("DEXCHUS", start)
        return (s.reindex(cny_usd.index, method="ffill") / cny_usd).rename("PBC")
    except Exception as e:
        st.warning(f"PBC fetch failed: {e}")
        return pd.Series(dtype=float, name="PBC")

@st.cache_data(ttl=86400)
def get_rba_total_assets(start: str):
    try:
        s = get_fred_series("RBATOTASSETS", "RBA_AUD", start)
        aud_usd = get_fx("DEXUSAL", start)
        return (s.reindex(aud_usd.index, method="ffill") * aud_usd).rename("RBA")
    except Exception as e:
        st.warning(f"RBA fetch failed: {e}")
        return pd.Series(dtype=float, name="RBA")

@st.cache_data(ttl=3600)  # shorter TTL for market data
def get_market_data(start: str, end: str):
    tickers = yf.download(["SPY", "BTC-USD"], start=start, end=end, progress=False)
    prices = tickers["Close"].resample(RESAMPLE_FREQ).last().ffill()
    return prices

# ── GLI calculation ─────────────────────────────────────────────────────────
def build_gli(df: pd.DataFrame) -> pd.Series:
    fed = df.get("FED_ASSETS", 0) * 0.001
    tga = df.get("TGA", 0)
    rrp = df.get("RRP", 0)
    
    boe = (df.get("BOE_GBP", 0) * 0.001 * df.get("GBP_USD", 1)) if "BOE_GBP" in df and "GBP_USD" in df else 0
    boc = (df.get("BOC_CAD", 0) * 0.001 / df.get("CAD_USD_INV", 1)) if "BOC_CAD" in df and "CAD_USD_INV" in df else 0
    snb = (df.get("SNB_CHF", 0) / df.get("CHF_USD_INV", 1)) if "SNB_CHF" in df and "CHF_USD_INV" in df else 0

    gli = (fed - tga - rrp +
           df.get("ECB", 0) + df.get("BOJ", 0) + df.get("PBC", 0) +
           df.get("RBA", 0) + boe + boc + snb)
    return gli.rename("GLI_USD_B")

def plot_gli(gli: pd.Series, market: pd.DataFrame):
    df = pd.concat([gli, market], axis=1).dropna(how="all")
    if df.empty:
        st.error("No data available after alignment.")
        return None
    
    first = df.dropna().index[0]
    idx = df.loc[first:].div(df.loc[first]) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04,
                        subplot_titles=("Indexed Performance (start = 100)", "GLI (USD Trillions)"))

    colors = {"GLI_USD_B": "#3266ad", "SPY": "#1D9E75", "BTC-USD": "#D85A30"}
    names = {"GLI_USD_B": "Global Liquidity Index", "SPY": "SPY", "BTC-USD": "BTC/USD"}

    for col in ["GLI_USD_B", "SPY", "BTC-USD"]:
        if col in idx.columns:
            fig.add_trace(go.Scatter(
                x=idx.index, y=idx[col].round(1),
                name=names[col], mode="lines",
                line=dict(color=colors[col], width=2.5 if col == "GLI_USD_B" else 1.5)
            ), row=1, col=1)

    if "GLI_USD_B" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=(df["GLI_USD_B"] / 1000).round(2),
            name="GLI (raw)", mode="lines",
            line=dict(color="#3266ad", width=1.5), showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        title="Global Liquidity Index vs SPY & BTC",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=750
    )
    fig.update_yaxes(title_text="Indexed (start=100)", row=1, col=1)
    fig.update_yaxes(title_text="USD Trillions", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig

# ── Main execution ──────────────────────────────────────────────────────────
start_str = START_DATE.strftime("%Y-%m-%d")
end_str = datetime.today().strftime("%Y-%m-%d")

FRED_SERIES = {
    "WALCL": "FED_ASSETS",
    "WTREGEN": "TGA",
    "RRPONTSYD": "RRP",
    "BOEBSTASGBP": "BOE_GBP",
    "CAALTSASSETS": "BOC_CAD",
    "SNBASSETS": "SNB_CHF",
    "DEXUSUK": "GBP_USD",
    "DEXCAUS": "CAD_USD_INV",
    "DEXSZUS": "CHF_USD_INV",
}

with st.spinner("Fetching central bank and market data... (first load can take 20–50 seconds)"):
    raw = {}
    for series_id, name in FRED_SERIES.items():
        try:
            raw[name] = get_fred_series(series_id, name, start_str)
        except Exception as e:
            st.warning(f"FRED series {series_id} failed: {e}")

    raw["ECB"] = get_ecb_total_assets(start_str)
    raw["BOJ"] = get_boj_total_assets(start_str)
    raw["PBC"] = get_pboc_total_assets(start_str)
    raw["RBA"] = get_rba_total_assets(start_str)

    df = pd.DataFrame(raw).sort_index().ffill()
    gli = build_gli(df)
    market = get_market_data(start_str, end_str)

    fig = plot_gli(gli, market)

if fig:
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show raw data"):
    combined = pd.concat([gli, market], axis=1).dropna(how="all")
    st.dataframe(combined.style.format("{:,.2f}"))

st.caption("Data is cached for 24 hours (central banks) and 1 hour (market). Some smaller central banks are still missing.")
