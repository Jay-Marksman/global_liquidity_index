import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime

st.set_page_config(page_title="Global Liquidity Index vs SPY & BTC", layout="wide")
st.title("🌍 Global Liquidity Index vs SPY & BTC")
st.markdown("**Global Liquidity Index** (major central bank assets) compared to SPY and Bitcoin. Paste your FRED API key below.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    FRED_API_KEY = st.text_input("FRED API Key", type="password", 
                                 help="Free key from https://fred.stlouisfed.org/docs/api/api_key.html")
    START_DATE = st.date_input("Start Date", value=datetime(2015, 1, 1))
    
    if not FRED_API_KEY:
        st.warning("Enter your FRED API key to load data")
        st.stop()

# Fred client (cached as resource)
@st.cache_resource
def get_fred_client(_api_key):
    return Fred(api_key=_api_key)

fred = get_fred_client(FRED_API_KEY)

RESAMPLE_FREQ = "W-FRI"

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
    """ECB - Use reliable FRED weekly series instead of direct API"""
    try:
        # ECBASSETSW = Central Bank Assets for Euro Area, weekly, millions of EUR
        s = get_fred_series("ECBASSETSW", "ECB_EUR", start)
        eur_usd = get_fx("DEXUSEU", start)
        # Convert EUR millions → USD billions
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
            # FRED fallback (JPNASSETS in trillions JPY)
            s = get_fred_series("JPNASSETS", "BOJ_JPYT", start) * 1000
            jpy_usd = get_fx("DEXJPUS", start)
            return (s.reindex(jpy_usd.index, method="ffill") / jpy_usd).rename("BOJ")
        except Exception as e:
            st.warning(f"BOJ fetch failed: {e}")
            return pd.Series(dtype=float, name="BOJ")
            
@st.cache_data(ttl=86400)
def get_boe_total_assets(start: str):
    """Bank of England weekly balance sheet - direct from BOE site"""
    try:
        # BOE Weekly Report data (total assets column)
        url = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp?Travel=NIxAZxSUx&FromSeries=1&ToSeries=50&DAT=RNG&FD=1&FM=Jan&FY=2018&TD=31&TM=Dec&TY=2027&FNY=Y&CSVF=TT&html.x=66&html.y=26&SeriesCodes=RPWB55A,RPWB56A,RPWB59A,RPWB67A,RPWZ4TJ,RPWZ4TK,RPWZOQ4,RPWZ4TL,RPWZ4TM,RPWZOI7,RPWZ4TN&UsingCodes=Y&Filter=N&title=Bank%20of%20England%20Weekly%20Report&VPD=Y"
        
        df = pd.read_csv(url, skiprows=1)  # adjust skiprows if needed after testing
        # The exact column names change; we look for total assets (usually around "Total assets" or code RPW... )
        # For robustness, we'll use a common pattern - this may need one tweak after first run
        df.columns = df.columns.str.strip()
        date_col = [col for col in df.columns if "date" in col.lower() or "period" in col.lower()][0]
        asset_col = None
        for col in df.columns:
            if "total asset" in str(col).lower() or "assets" in str(col).lower() and "liability" not in str(col).lower():
                asset_col = col
                break
        if not asset_col:
            raise ValueError("Could not find assets column")
        
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        df = pd.to_numeric(df[asset_col].astype(str).str.replace(",", ""), errors="coerce").dropna()
        df = df.resample(RESAMPLE_FREQ).last().ffill()
        
        gbp_usd = get_fx("DEXUSUK", start)
        df_aligned = df.reindex(gbp_usd.index, method="ffill") * gbp_usd / 1000  # assume millions GBP → USD billions
        return df_aligned.rename("BOE")
    except Exception as e:
        st.warning(f"BOE direct fetch failed: {e}. Using zero for now.")
        return pd.Series(dtype=float, name="BOE")

@st.cache_data(ttl=86400)
def get_rba_total_assets(start: str):
    try:
        s = get_fred_series("RBATOTASSETS", "RBA_AUD", start)  # may still fail
        aud_usd = get_fx("DEXUSAL", start) if "DEXUSAL" in FRED_SERIES else pd.Series(1.0)
        return (s.reindex(aud_usd.index, method="ffill") * aud_usd).rename("RBA")
    except Exception:
        st.info("RBA assets not available on FRED yet.")
        return pd.Series(dtype=float, name="RBA")

@st.cache_data(ttl=86400)
def get_pboc_total_assets(start: str):
    try:
        s = get_fred_series("CHASSETS", "PBC_CNY", start)
        cny_usd = get_fx("DEXCHUS", start) if "DEXCHUS" in FRED_SERIES else pd.Series(1.0)
        return (s.reindex(cny_usd.index, method="ffill") / cny_usd).rename("PBC")
    except Exception:
        st.info("PBC assets not reliably available on FRED.")
        return pd.Series(dtype=float, name="PBC")

# ── Build GLI ────────────────────────
def build_gli(df: pd.DataFrame) -> pd.Series:
    fed = df.get("FED_ASSETS", pd.Series(0)) * 0.001
    tga = df.get("TGA", pd.Series(0))
    rrp = df.get("RRP", pd.Series(0))
    
    ecb = df.get("ECB", pd.Series(0))
    boj = df.get("BOJ", pd.Series(0))
    boe = df.get("BOE", pd.Series(0))
    snb = df.get("SNB_CHF", pd.Series(0)) / df.get("CHF_USD_INV", pd.Series(1.0)) if "SNB_CHF" in df.columns and "CHF_USD_INV" in df.columns else pd.Series(0)
    # rba and pbc are small/optional for now
    rba = df.get("RBA", pd.Series(0))
    pbc = df.get("PBC", pd.Series(0))

    gli = fed - tga - rrp + ecb + boj + boe + snb + rba + pbc
    return gli.rename("GLI_USD_B")
    
# ── Market data ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_market_data(start: str, end: str):
    tickers = yf.download(["SPY", "BTC-USD"], start=start, end=end, progress=False)
    prices = tickers["Close"].resample(RESAMPLE_FREQ).last().ffill()
    return prices

# ── Plot function (robust) ──────────────────────────────────────────────
def plot_gli(gli: pd.Series, market: pd.DataFrame):
    df = pd.concat([gli, market], axis=1).sort_index().dropna(how="all")
    
    if df.empty:
        st.error("No data available. Check API key and internet.")
        return None

    st.caption(f"Data shape: {df.shape} | GLI non-NaN days: {gli.notna().sum()}")

    first = df.dropna(how="all").index[0]
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
    "WALCL": "FED_ASSETS",      # Fed Total Assets (millions USD)
    "WTREGEN": "TGA",           # Treasury General Account
    "RRPONTSYD": "RRP",         # Overnight RRP
    "ECBASSETSW": "ECB_EUR",    # ECB weekly assets (millions EUR) - confirmed working
    "JPNASSETS": "BOJ_JPYT",    # BOJ fallback (trillions JPY)
    "SNBASSETS": "SNB_CHF",     # SNB total assets (billions CHF) - let's try this
    # FX rates
    "DEXUSEU": "EUR_USD",
    "DEXJPUS": "JPY_USD",
    "DEXSZUS": "CHF_USD_INV",   # CHF per USD → invert for USD conversion
    "DEXUSUK": "GBP_USD",       # For future BOE
}

with st.spinner("Fetching data... (first run can take 30-60 seconds)"):

raw = {}
    for sid, name in FRED_SERIES.items():
        raw[name] = get_fred_series(sid, name, start_str)

    raw["ECB"] = get_ecb_total_assets(start_str)   # keep for safety, though ECBASSETSW is now in FRED_SERIES
    raw["BOJ"] = get_boj_total_assets(start_str)
    raw["BOE"] = get_boe_total_assets(start_str)
    raw["RBA"] = get_rba_total_assets(start_str)
    raw["PBC"] = get_pboc_total_assets(start_str)

    df_raw = pd.DataFrame(raw).sort_index().ffill()
    gli = build_gli(df_raw)
    market = get_market_data(start_str, end_str)

    fig = plot_gli(gli, market)

if fig:
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show raw data"):
    combined = pd.concat([gli, market], axis=1).dropna(how="all")
    st.dataframe(combined.style.format("{:,.2f}"))

st.caption("""Current coverage: FED (assets - TGA - RRP) + ECB + BOJ.  
Many other central banks need direct sources (Bank of England weekly report, etc.).""")
