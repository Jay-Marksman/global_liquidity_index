import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime

st.set_page_config(page_title="Global Liquidity Index vs SPY & BTC", layout="wide")
st.title("🌍 Global Liquidity Index vs SPY & BTC")
st.markdown("**Global Liquidity Index** = FED – TGA – RRP + ECB + BOJ + BOC + RBA + BOE + SNB + ...")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    FRED_API_KEY = st.text_input("FRED API Key", type="password",
                                 help="Free key from https://fred.stlouisfed.org/docs/api/api_key.html")
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

# ── Core Fetchers ───────────────────────────────────────────────────────
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
def get_boc_total_assets(start: str):
    """Bank of Canada - official weekly CSV"""
    try:
        url = "https://www.bankofcanada.ca/valet/observations/group/b2_weekly/csv?start_date=2015-01-01"
        df = pd.read_csv(url, skiprows=1)
        df.columns = df.columns.str.strip()
        date_col = [c for c in df.columns if "date" in c.lower()][0]
        # Look for total assets column (names vary slightly)
        asset_col = next((c for c in df.columns if "total asset" in c.lower() or "assets" in c.lower() and "liability" not in c.lower()), None)
        if not asset_col:
            asset_col = df.columns[-1]  # fallback to last numeric column
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        assets = pd.to_numeric(df[asset_col].astype(str).str.replace(",", ""), errors="coerce").dropna()
        assets = assets.resample(RESAMPLE_FREQ).last().ffill()
        cad_usd = get_fx("DEXCAUS", start)  # CAD per USD
        return (assets.reindex(cad_usd.index, method="ffill") / cad_usd).rename("BOC")
    except Exception as e:
        st.warning(f"BOC: {e}")
        return pd.Series(dtype=float, name="BOC")

@st.cache_data(ttl=86400)
def get_rba_total_assets(start: str):
    """RBA - official weekly CSV (much more reliable than Excel)"""
    try:
        url = "https://www.rba.gov.au/statistics/tables/csv/a1-data.csv"
        df = pd.read_csv(url, skiprows=1)
        df.columns = df.columns.str.strip()
        date_col = [c for c in df.columns if "date" in c.lower() or "period" in c.lower()][0]
        asset_col = next((c for c in df.columns if "total asset" in str(c).lower() or "total assets" in str(c).lower()), None)
        if not asset_col:
            asset_col = [c for c in df.columns if "assets" in str(c).lower()][-1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        assets = pd.to_numeric(df[asset_col].astype(str).str.replace(",", ""), errors="coerce").dropna()
        assets = assets.resample(RESAMPLE_FREQ).last().ffill()
        aud_usd = get_fx("DEXUSAL", start)  # USD per AUD
        return (assets.reindex(aud_usd.index, method="ffill") * aud_usd).rename("RBA")
    except Exception as e:
        st.warning(f"RBA: {e}")
        return pd.Series(dtype=float, name="RBA")

@st.cache_data(ttl=86400)
def get_boe_total_assets(start: str):
    """Bank of England - official weekly report CSV"""
    try:
        url = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp?Travel=NIxAZxSUx&FromSeries=1&ToSeries=50&DAT=RNG&FD=1&FM=Jan&FY=2018&TD=31&TM=Dec&TY=2027&FNY=Y&CSVF=TT&html.x=66&html.y=26&SeriesCodes=RPWB55A,RPWB56A,RPWB59A,RPWB67A,RPWZ4TJ,RPWZ4TK,RPWZOQ4,RPWZ4TL,RPWZ4TM,RPWZOI7,RPWZ4TN&UsingCodes=Y&Filter=N&title=Bank%20of%20England%20Weekly%20Report&VPD=Y"
        df = pd.read_csv(url, skiprows=1)
        df.columns = df.columns.str.strip()
        date_col = next((c for c in df.columns if "date" in c.lower() or "period" in c.lower()), None)
        asset_col = next((c for c in df.columns if "total asset" in str(c).lower() and "liability" not in str(c).lower()), None)
        if not date_col or not asset_col:
            raise ValueError("Column not found")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        assets = pd.to_numeric(df[asset_col].astype(str).str.replace(",", ""), errors="coerce").dropna()
        assets = assets.resample(RESAMPLE_FREQ).last().ffill()
        gbp_usd = get_fx("DEXUSUK", start)
        return (assets.reindex(gbp_usd.index, method="ffill") * gbp_usd / 1000).rename("BOE")  # assume millions GBP
    except Exception as e:
        st.warning(f"BOE: {e} (often 403 - skipping)")
        return pd.Series(dtype=float, name="BOE")

@st.cache_data(ttl=86400)
def get_snb_total_assets(start: str):
    try:
        s = get_fred_series("SNBREMBALPOS", "SNB_CHF", start)  # best available proxy
        chf_usd = get_fx("DEXSZUS", start)
        return (s.reindex(chf_usd.index, method="ffill") / chf_usd).rename("SNB")
    except Exception as e:
        st.warning(f"SNB: {e}")
        return pd.Series(dtype=float, name="SNB")

# ── GLI Calculation (robust) ────────────────────────────────────────────
def build_gli(df: pd.DataFrame) -> pd.Series:
    common_idx = df.index
    fed = df.get("FED_ASSETS", pd.Series(0, index=common_idx)) * 0.001
    tga = df.get("TGA", pd.Series(0, index=common_idx))
    rrp = df.get("RRP", pd.Series(0, index=common_idx))
    ecb = df.get("ECB", pd.Series(0, index=common_idx))
    boj = df.get("BOJ", pd.Series(0, index=common_idx))
    boc = df.get("BOC", pd.Series(0, index=common_idx))
    rba = df.get("RBA", pd.Series(0, index=common_idx))
    boe = df.get("BOE", pd.Series(0, index=common_idx))
    snb = df.get("SNB", pd.Series(0, index=common_idx))

    gli = fed - tga - rrp + ecb + boj + boc + rba + boe + snb
    gli.name = "GLI_USD_B"
    return gli

# Market data
@st.cache_data(ttl=3600)
def get_market_data(start: str, end: str):
    tickers = yf.download(["SPY", "BTC-USD"], start=start, end=end, progress=False)
    return tickers["Close"].resample(RESAMPLE_FREQ).last().ffill()

# Plot (same as before, with extra safety)
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
        if col in idx.columns and idx[col].notna().any():
            fig.add_trace(go.Scatter(x=idx.index, y=idx[col].round(1),
                                     name=names[col], mode="lines",
                                     line=dict(color=colors[col], width=2.5 if col == "GLI_USD_B" else 1.5)),
                          row=1, col=1)

    if "GLI_USD_B" in df.columns and df["GLI_USD_B"].notna().any():
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

# ── Main Execution ──────────────────────────────────────────────────────
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
    "DEXUSUK": "GBP_USD",
    "DEXSZUS": "CHF_USD_INV",
}

with st.spinner("Fetching data... (first load ~40-70 seconds with new banks)"):
    raw = {}
    for sid, name in FRED_SERIES.items():
        raw[name] = get_fred_series(sid, name, start_str)

    raw["ECB"] = get_ecb_total_assets(start_str)
    raw["BOJ"] = get_boj_total_assets(start_str)
    raw["BOC"] = get_boc_total_assets(start_str)
    raw["RBA"] = get_rba_total_assets(start_str)
    raw["BOE"] = get_boe_total_assets(start_str)
    raw["SNB"] = get_snb_total_assets(start_str)

    df_raw = pd.DataFrame(raw).sort_index().ffill()
    gli = build_gli(df_raw)
    market = get_market_data(start_str, end_str)

    fig = plot_gli(gli, market)

if fig:
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show raw data table"):
    combined = pd.concat([gli, market], axis=1).dropna(how="all")
    st.dataframe(combined.style.format("{:,.2f}"))

st.caption("**Current coverage**: FED + ECB + BOJ + BOC + RBA + BOE + SNB. Data shape should now be significantly larger.")
