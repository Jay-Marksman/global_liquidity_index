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
    """ECB total assets - updated endpoint (ECB Data Portal / SDW replacement)"""
    # Try modern ECB API first
    urls = [
        "https://data-api.ecb.europa.eu/service/data/BSI/W.U2.C.A.T0.A1.Z5._T._T._Z.XDC._T._X.V._T._T._T._T?format=csvdata",  # Consolidated balance sheet attempt
        "https://data-api.ecb.europa.eu/service/data/ILM/W.U2.C.A1.U4.EUR?format=csvdata&startPeriod=2015-01-01"  # old one
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if "TIME_PERIOD" in df.columns and "OBS_VALUE" in df.columns:
                df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
                df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
                df = df.set_index("TIME_PERIOD")["OBS_VALUE"]
                df = df.resample(RESAMPLE_FREQ).last().ffill()
                df.name = "ECB"
                
                eur_usd = get_fx("DEXUSEU", start)  # renamed for clarity
                df_aligned = df.reindex(eur_usd.index, method="ffill") * eur_usd / 1000   # EUR millions → USD billions
                return df_aligned
        except Exception:
            continue
    st.warning("ECB fetch failed from both endpoints. Using zeros for ECB.")
    return pd.Series(dtype=float, name="ECB")

@st.cache_data(ttl=86400)
def get_boj_total_assets(start: str):
    """BOJ - keep your original + fallback"""
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
        df_aligned = df.reindex(jpy_usd.index, method="ffill") / jpy_usd * 0.01
        return df_aligned.rename("BOJ")
    except Exception as e:
        st.warning(f"Primary BOJ source failed: {e}. Trying FRED fallback...")
        try:
            s = get_fred_series("JPNASSETS", "BOJ", start) * 1000  # adjust multiplier if needed
            jpy_usd = get_fx("DEXJPUS", start)
            return (s.reindex(jpy_usd.index, method="ffill") / jpy_usd).rename("BOJ")
        except Exception as e2:
            st.warning(f"BOJ completely failed: {e2}")
            return pd.Series(dtype=float, name="BOJ")

@st.cache_data(ttl=86400)
def get_pboc_total_assets(start: str):
    """PBC (People's Bank of China) - updated series if available, else warn"""
    try:
        # Try common series for Chinese central bank assets
        s = get_fred_series("CHASSETS", "PBC_CNY", start)   # this may still fail
        cny_usd = get_fx("DEXCHUS", start)
        return (s.reindex(cny_usd.index, method="ffill") / cny_usd).rename("PBC")
    except Exception:
        st.warning("PBC (CHASSETS) not available on FRED. Consider direct PBoC source later.")
        return pd.Series(dtype=float, name="PBC")

@st.cache_data(ttl=86400)
def get_rba_total_assets(start: str):
    """RBA - similar situation"""
    try:
        s = get_fred_series("RBATOTASSETS", "RBA_AUD", start)
        aud_usd = get_fx("DEXUSAL", start)
        return (s.reindex(aud_usd.index, method="ffill") * aud_usd).rename("RBA")
    except Exception:
        st.warning("RBA (RBATOTASSETS) not found on FRED.")
        return pd.Series(dtype=float, name="RBA")
        
@st.cache_data(ttl=3600)  # shorter TTL for market data
def get_market_data(start: str, end: str):
    tickers = yf.download(["SPY", "BTC-USD"], start=start, end=end, progress=False)
    prices = tickers["Close"].resample(RESAMPLE_FREQ).last().ffill()
    return prices

# ── GLI calculation ─────────────────────────────────────────────────────────
def build_gli(df: pd.DataFrame) -> pd.Series:
    fed = df.get("FED_ASSETS", pd.Series(0)) * 0.001
    tga = df.get("TGA", pd.Series(0))
    rrp = df.get("RRP", pd.Series(0))

    # BOE (if available and in %GDP, this is imperfect - we'll improve later)
    boe = df.get("BOE_GBP", pd.Series(0))

    # Other banks
    ecb = df.get("ECB", pd.Series(0))
    boj = df.get("BOJ", pd.Series(0))
    pbc = df.get("PBC", pd.Series(0))
    rba = df.get("RBA", pd.Series(0))

    # FX conversions where needed
    if "GBP_USD" in df.columns:
        boe = boe * df["GBP_USD"] * 0.001 if "BOE_GBP" in df.columns else boe  # rough scaling

    gli = fed - tga - rrp + ecb + boj + pbc + rba + boe
    return gli.rename("GLI_USD_B")
    
def plot_gli(gli: pd.Series, market: pd.DataFrame):
    if gli.empty and market.empty:
        st.error("No data was fetched for GLI or markets.")
        return None

    df = pd.concat([gli, market], axis=1).sort_index()
    
    # Diagnostic info
    st.caption(f"Data shape: {df.shape} | GLI non-NaN: {gli.notna().sum()} | Market non-NaN: {market.notna().sum().sum()}")

    # Drop rows where ALL columns are NaN, but keep rows with partial data
    df = df.dropna(how="all")
    
    if df.empty:
        st.error("No overlapping data available after alignment. Check date range or API key.")
        return None

    # Find the first date where we have at least the GLI or one market
    valid_df = df.dropna(subset=["GLI_USD_B", "SPY", "BTC-USD"], how="all")
    if valid_df.empty:
        st.warning("No date has both GLI and market data. Showing available series separately.")
        valid_df = df  # fallback

    first = valid_df.index[0]

    # Index to 100 from the first valid date
    idx = df.loc[first:].div(df.loc[first]) * 100

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=("Indexed Performance (start = 100)", "GLI (USD Trillions)")
    )

    colors = {"GLI_USD_B": "#3266ad", "SPY": "#1D9E75", "BTC-USD": "#D85A30"}
    names = {"GLI_USD_B": "Global Liquidity Index", "SPY": "SPY", "BTC-USD": "BTC/USD"}

    for col in ["GLI_USD_B", "SPY", "BTC-USD"]:
        if col in idx.columns:
            fig.add_trace(go.Scatter(
                x=idx.index, 
                y=idx[col].round(1),
                name=names[col], 
                mode="lines",
                line=dict(color=colors[col], width=2.5 if col == "GLI_USD_B" else 1.5)
            ), row=1, col=1)

    # Bottom panel: raw GLI in trillions
    if "GLI_USD_B" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=(df["GLI_USD_B"] / 1000).round(2),
            name="GLI (raw)", 
            mode="lines",
            line=dict(color="#3266ad", width=1.5), 
            showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        title="Global Liquidity Index vs SPY & BTC",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=750,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    fig.update_yaxes(title_text="Indexed (start=100)", row=1, col=1)
    fig.update_yaxes(title_text="USD Trillions", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig
# ── Main execution ──────────────────────────────────────────────────────────
start_str = START_DATE.strftime("%Y-%m-%d")
end_str = datetime.today().strftime("%Y-%m-%d")

FRED_SERIES = {
    # Reliable FED components (these still work)
    "WALCL": "FED_ASSETS",      # Fed Total Assets (millions USD)
    "WTREGEN": "TGA",           # Treasury General Account
    "RRPONTSYD": "RRP",         # Overnight Reverse Repo

    # BOE - use the long-running one (though it ends in 2016, we'll add a better source later)
    "BOEBSTAUKA": "BOE_GBP",    # Bank of England Total Assets (% of GDP, but we'll adjust)

    # FX rates (these are usually stable)
    "DEXUSUK": "GBP_USD",       # USD per GBP
    "DEXCAUS": "CAD_USD_INV",   # CAD per USD (invert)
    "DEXSZUS": "CHF_USD_INV",   # CHF per USD (invert)
    "DEXUSEU": "EUR_USD",       # USD per EUR (for ECB)
    "DEXJPUS": "JPY_USD",       # JPY per USD (for BOJ)
    "DEXCHUS": "CNY_USD",       # CNY per USD (for PBC)
    "DEXUSAL": "AUD_USD",       # USD per AUD (for RBA)
}

with st.spinner("Fetching central bank and market data... (first load can take 30–60 seconds)"):
    raw = {}
    for series_id, name in FRED_SERIES.items():
        try:
            raw[name] = get_fred_series(series_id, name, start_str)
        except Exception as e:
            st.warning(f"FRED {series_id} failed: {e}")

    raw["ECB"] = get_ecb_total_assets(start_str)
    raw["BOJ"] = get_boj_total_assets(start_str)
    raw["PBC"] = get_pboc_total_assets(start_str)
    raw["RBA"] = get_rba_total_assets(start_str)

    df_raw = pd.DataFrame(raw).sort_index().ffill()
    gli = build_gli(df_raw)
    market = get_market_data(start_str, end_str)

    fig = plot_gli(gli, market)

if fig:
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show raw data table"):
    combined = pd.concat([gli, market], axis=1).dropna(how="all")
    if not combined.empty:
        st.dataframe(combined.style.format("{:,.2f}"))
    else:
        st.info("No combined data available yet.")
        
st.caption("Data is cached for 24 hours (central banks) and 1 hour (market). Some smaller central banks are still missing.")
