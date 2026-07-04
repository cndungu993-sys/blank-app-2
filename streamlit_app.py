import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta, timezone

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="USDINTEL — FX Strength & Levels", layout="wide",
                    initial_sidebar_state="expanded")

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

# USD-base pairs: pair UP means USD strengthens.
# USD-quote pairs: pair UP means the OTHER currency strengthens (USD weakens).
USD_IS_BASE = {"USDJPY": True, "USDCHF": True, "USDCAD": True,
               "EURUSD": False, "GBPUSD": False, "AUDUSD": False, "NZDUSD": False}
OTHER_CCY = {"EURUSD": "EUR", "GBPUSD": "GBP", "AUDUSD": "AUD", "NZDUSD": "NZD",
             "USDJPY": "JPY", "USDCHF": "CHF", "USDCAD": "CAD"}

# Only these five timeframes are supported anywhere in the app
TIMEFRAMES = ["M15", "H4", "D1", "W1", "MN1"]

TD_TF_MAP = {"M15": "15min", "H4": "4h", "D1": "1day", "W1": "1week", "MN1": "1month"}
TD_PAIRS  = {"EURUSD": "EUR/USD", "USDJPY": "USD/JPY", "GBPUSD": "GBP/USD",
             "USDCHF": "USD/CHF", "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "NZDUSD": "NZD/USD"}
YF_TICKERS = {"EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "GBPUSD": "GBPUSD=X",
              "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X"}
# Yahoo has no native 4h bucket — we pull 1h and resample.
YF_TF_MAP = {"M15": "15m", "H4": "1h", "D1": "1d", "W1": "1wk", "MN1": "1mo"}
YF_LOOKBACK_DAYS = {"M15": 5, "H4": 59, "D1": 730, "W1": 1825, "MN1": 3650}

BASE_PRICES = {"EURUSD": 1.0850, "USDJPY": 154.20, "GBPUSD": 1.2650,
               "USDCHF": 0.9020, "AUDUSD": 0.6480, "USDCAD": 1.3640, "NZDUSD": 0.5920}

FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXT    = [1.272, 1.618]

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
_defaults = {"last_refresh": 0, "cache_buster": 0, "last_minute": -1}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

current_minute = int(time.time() // 60)
if st.session_state.last_minute != current_minute:
    st.session_state.last_minute = current_minute
    st.session_state.cache_buster = current_minute

# ═══════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════
def fetch_td(pair, tf, td_key, limit=300):
    """Twelve Data — real OHLC (no volume needed anymore)."""
    if not td_key:
        return pd.DataFrame()
    params = {"symbol": TD_PAIRS.get(pair, pair), "interval": TD_TF_MAP.get(tf, "1day"),
              "outputsize": min(limit, 5000), "timezone": "UTC", "order": "ASC",
              "format": "JSON", "apikey": td_key}
    try:
        r = requests.get("https://api.twelvedata.com/time_series", params=params,
                          headers={"Cache-Control": "no-cache", "Pragma": "no-cache"}, timeout=20)
        d = r.json()
        if d.get("status") == "error":
            return pd.DataFrame()
        values = d.get("values", [])
        if not values:
            return pd.DataFrame()
        df = pd.DataFrame(values).rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"])
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["time", "open", "high", "low", "close"]].dropna()
        return df.sort_values("time").tail(limit).reset_index(drop=True)
    except Exception as e:
        print(f"[TD] {pair} {tf}: {e}")
        return pd.DataFrame()

def _resample(df, rule):
    d = df.set_index("time")
    out = d.resample(rule).agg({"open": "first", "high": "max",
                                 "low": "min", "close": "last"}).dropna()
    return out.reset_index()

def fetch_yf(pair, tf, limit=300):
    """Yahoo Finance HTTP fallback. H4 is built by resampling 1h bars."""
    ticker = YF_TICKERS.get(pair, f"{pair}=X")
    yf_tf = YF_TF_MAP.get(tf, "1d")
    lookback_days = YF_LOOKBACK_DAYS.get(tf, 59)
    now = int(datetime.now(timezone.utc).replace(tzinfo=None).timestamp())
    period1 = now - lookback_days * 86400
    period2 = now

    for host in ["query2.finance.yahoo.com", "query1.finance.yahoo.com"]:
        url = (f"https://{host}/v8/finance/chart/{ticker}"
               f"?interval={yf_tf}&period1={period1}&period2={period2}"
               f"&includePrePost=false&corsDomain=finance.yahoo.com")
        try:
            r = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Cache-Control": "no-cache, no-store", "Pragma": "no-cache"}, timeout=15)
            d = r.json()
            res = d.get("chart", {}).get("result", [])
            if not res:
                continue
            data_raw = res[0]
            times = data_raw.get("timestamp", [])
            q = data_raw["indicators"]["quote"][0]
            if not times:
                continue
            df = pd.DataFrame({
                "time": pd.to_datetime(times, unit="s"),
                "open": q.get("open", [None] * len(times)),
                "high": q.get("high", [None] * len(times)),
                "low": q.get("low", [None] * len(times)),
                "close": q.get("close", [None] * len(times)),
            }).dropna(subset=["open", "high", "low", "close"])

            base = BASE_PRICES[pair]
            df = df[(df["close"] > base * 0.5) & (df["close"] < base * 2.0)]
            if df.empty:
                continue

            if tf == "H4":
                df = _resample(df, "4h")

            df = df.sort_values("time").tail(limit).reset_index(drop=True)
            return df
        except Exception as e:
            print(f"[YF] {pair} {tf} from {host}: {e}")
            continue
    return pd.DataFrame()

def simulate(pair, tf, n=300):
    rng = np.random.default_rng(abs(hash(pair + tf)) % 9999)
    base = BASE_PRICES[pair]
    step = {"M15": 0.0002, "H4": 0.0009, "D1": 0.002, "W1": 0.005, "MN1": 0.012}.get(tf, 0.001)
    closes = base * np.cumprod(1 + rng.normal(0, step, n))
    noise = rng.uniform(step * 0.3, step * 1.5, n)
    opens = np.roll(closes, 1)
    opens[0] = base
    freq = {"M15": timedelta(minutes=15), "H4": timedelta(hours=4), "D1": timedelta(days=1),
            "W1": timedelta(weeks=1), "MN1": timedelta(days=30)}.get(tf, timedelta(hours=1))
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    times = [now - freq * (n - i) for i in range(n)]
    return pd.DataFrame({"time": times, "open": opens,
                          "high": closes * (1 + noise), "low": closes * (1 - noise),
                          "close": closes})

def get_candles(pair, tf, td_key, limit=300):
    df = fetch_td(pair, tf, td_key, limit) if td_key else pd.DataFrame()
    if df.empty:
        df = fetch_yf(pair, tf, limit)
    if df.empty:
        df = simulate(pair, tf, limit)
    return df

def get_live_price(pair, td_key):
    base = BASE_PRICES[pair]
    if td_key:
        try:
            sym = TD_PAIRS.get(pair, pair)
            r = requests.get("https://api.twelvedata.com/price",
                              params={"symbol": sym, "apikey": td_key}, timeout=8)
            d = r.json()
            if "price" in d:
                p = float(d["price"])
                if base * 0.5 < p < base * 2.0:
                    return round(p, 5)
        except Exception:
            pass
    try:
        ticker = YF_TICKERS.get(pair, f"{pair}=X")
        now = int(datetime.now(timezone.utc).replace(tzinfo=None).timestamp())
        for host in ["query2.finance.yahoo.com", "query1.finance.yahoo.com"]:
            r = requests.get(
                f"https://{host}/v8/finance/chart/{ticker}?interval=1m&period1={now-600}&period2={now}",
                headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}, timeout=10)
            d = r.json()
            res = d.get("chart", {}).get("result", [])
            if res:
                closes = [c for c in res[0]["indicators"]["quote"][0].get("close", []) if c]
                if closes:
                    p = float(closes[-1])
                    if base * 0.5 < p < base * 2.0:
                        return round(p, 5)
    except Exception:
        pass
    return base

# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════
def calc_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def calc_sma(s, n): return s.rolling(n).mean()

def calc_atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

# ═══════════════════════════════════════════════════════════════
# FIBONACCI RETRACEMENT / EXTENSION
# ═══════════════════════════════════════════════════════════════
def fibonacci_levels(df, lookback=100):
    """Swing-based Fibonacci retracement + extension levels for a timeframe."""
    if df.empty or len(df) < 10:
        return {}
    window = df.tail(min(lookback, len(df))).reset_index(drop=True)
    hi_idx = window["high"].idxmax()
    lo_idx = window["low"].idxmin()
    swing_high = float(window["high"].iloc[hi_idx])
    swing_low = float(window["low"].iloc[lo_idx])
    uptrend = lo_idx < hi_idx  # low formed first -> measuring an up-leg

    levels = {}
    span = swing_high - swing_low
    if span <= 0:
        return {}
    for r in FIB_RATIOS:
        price = (swing_high - span * r) if uptrend else (swing_low + span * r)
        levels[f"{r*100:.1f}%"] = round(price, 5)
    for r in FIB_EXT:
        price = (swing_low - span * (r - 1)) if uptrend else (swing_high + span * (r - 1))
        levels[f"Ext {r*100:.1f}%"] = round(price, 5)
    levels["_direction"] = "Up-leg" if uptrend else "Down-leg"
    levels["_swing_high"] = round(swing_high, 5)
    levels["_swing_low"] = round(swing_low, 5)
    return levels

# ═══════════════════════════════════════════════════════════════
# SUPPORT / RESISTANCE (pivot-based, clustered)
# ═══════════════════════════════════════════════════════════════
def support_resistance(df, left=3, right=3, cluster_pct=0.0012, top_n=4):
    """Detects swing pivots, clusters nearby levels, returns S/R closest to price."""
    if df.empty or len(df) < left + right + 5:
        return [], []
    highs, lows, price = df["high"].values, df["low"].values, float(df["close"].iloc[-1])
    piv_hi, piv_lo = [], []
    for i in range(left, len(df) - right):
        if highs[i] == max(highs[i-left:i+right+1]):
            piv_hi.append(highs[i])
        if lows[i] == min(lows[i-left:i+right+1]):
            piv_lo.append(lows[i])

    def cluster(levels):
        levels = sorted(levels)
        clustered = []
        for lvl in levels:
            if clustered and abs(lvl - clustered[-1][-1]) / lvl < cluster_pct:
                clustered[-1].append(lvl)
            else:
                clustered.append([lvl])
        return [round(float(np.mean(c)), 5) for c in clustered]

    res_levels = [l for l in cluster(piv_hi) if l > price]
    sup_levels = [l for l in cluster(piv_lo) if l < price]
    resistance = sorted(res_levels)[:top_n]
    support = sorted(sup_levels, reverse=True)[:top_n]
    return support, resistance

# ═══════════════════════════════════════════════════════════════
# TREND / CONFLUENCE SCORE (replaces old smoothness/VP scoring)
# ═══════════════════════════════════════════════════════════════
def trend_score(df, fib, support, resistance):
    """0-3 score: trend direction, confluence with a Fib/S-R level, volatility state."""
    if df.empty or len(df) < 55:
        return 0, []
    checks = []
    score = 0
    last = float(df["close"].iloc[-1])
    e20, e50 = calc_ema(df["close"], 20).iloc[-1], calc_ema(df["close"], 50).iloc[-1]
    atr_n = float(calc_atr(df).iloc[-1])
    atr_m = float(calc_atr(df).rolling(20).mean().iloc[-1]) if len(df) >= 34 else atr_n

    bullish = e20 > e50
    checks.append(("✅" if bullish else "✅", "Trend",
                    "EMA20 > EMA50 — bullish bias" if bullish else "EMA20 < EMA50 — bearish bias"))
    score += 1

    tol = atr_n * 0.4 if atr_n else last * 0.001
    near_levels = [v for k, v in fib.items() if not k.startswith("_")] + support + resistance
    at_level = any(abs(last - lv) < tol for lv in near_levels)
    if at_level:
        score += 1
        checks.append(("✅", "Confluence", "Price sitting at a Fibonacci/S-R level"))
    else:
        checks.append(("❌", "Confluence", "Price between levels — no confluence"))

    if atr_m and atr_n < atr_m * 0.9:
        score += 1
        checks.append(("✅", "Volatility", "ATR contracting — coiled range"))
    else:
        checks.append(("❌", "Volatility", "ATR expanded — move may be extended"))

    return score, checks

# ═══════════════════════════════════════════════════════════════
# USD STRENGTH DASHBOARD
# ═══════════════════════════════════════════════════════════════
def usd_strength(candles_by_pair, tf):
    """% change per currency vs USD over the selected timeframe's last completed bar."""
    rows = []
    usd_contribs = []
    for pair in PAIRS:
        df = candles_by_pair[pair]
        if df.empty or len(df) < 2:
            continue
        prev_close = float(df["close"].iloc[-2])
        last_close = float(df["close"].iloc[-1])
        pct = (last_close - prev_close) / prev_close * 100
        if USD_IS_BASE[pair]:
            other_pct = -pct
            usd_contribs.append(pct)
        else:
            other_pct = pct
            usd_contribs.append(-pct)
        rows.append({"Currency": OTHER_CCY[pair], "Pair": pair, "Change %": round(other_pct, 3)})

    usd_pct = round(float(np.mean(usd_contribs)), 3) if usd_contribs else 0.0
    rows.append({"Currency": "USD", "Pair": "(basket avg)", "Change %": usd_pct})
    out = pd.DataFrame(rows).sort_values("Change %", ascending=False).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out[["Rank", "Currency", "Pair", "Change %"]]

def render_strength_bar(df):
    max_abs = max(df["Change %"].abs().max(), 0.01)
    rows_html = ""
    for _, r in df.iterrows():
        pct = r["Change %"]
        color = "#1a7a4a" if pct > 0 else ("#b5281c" if pct < 0 else "#666")
        width = min(abs(pct) / max_abs * 100, 100)
        side = "left" if pct >= 0 else "right"
        bold = "font-weight:800;" if r["Currency"] == "USD" else ""
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<div style="width:46px;{bold}color:#fff;">{r["Currency"]}</div>'
            f'<div style="flex:1;background:#1b1b2e;border-radius:4px;height:16px;position:relative;">'
            f'<div style="position:absolute;{side}:0;top:0;height:16px;width:{width:.1f}%;'
            f'background:{color};border-radius:4px;"></div></div>'
            f'<div style="width:70px;text-align:right;color:{color};font-family:monospace;'
            f'font-weight:700;">{pct:+.3f}%</div>'
            f'</div>')
    st.markdown(rows_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# OPTIONAL AI MARKET NOTE (Claude — separate from removed sentiment engine)
# ═══════════════════════════════════════════════════════════════
def ai_market_note(pair, fib, support, resistance, score, claude_key):
    if not (ANTHROPIC_AVAILABLE and claude_key):
        return None
    try:
        client = anthropic.Anthropic(api_key=claude_key)
        prompt = (f"In 2 short sentences, comment on {pair} given a trend/confluence "
                  f"score of {score}/3, nearby support {support[:2]}, resistance {resistance[:2]}. "
                  f"Be neutral and factual, no trade advice.")
        msg = client.messages.create(model="claude-sonnet-5", max_tokens=120,
                                      messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    except Exception as e:
        return f"(AI note unavailable: {e})"

# ═══════════════════════════════════════════════════════════════
# CACHED DATA LOAD
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=55, show_spinner=False)
def load_all(td_key, _bust):
    out = {tf: {} for tf in TIMEFRAMES}
    prices = {}
    for pair in PAIRS:
        prices[pair] = get_live_price(pair, td_key)
        for tf in TIMEFRAMES:
            out[tf][pair] = get_candles(pair, tf, td_key, limit=300)
    return out, prices

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")
    td_key = st.text_input("Twelve Data API Key", type="password",
                            placeholder="Free at twelvedata.com")
    claude_key = st.text_input("Claude API Key (optional)", type="password",
                                placeholder="For a short AI market note")
    st.caption("Twelve Data = live OHLC. No key falls back to Yahoo Finance, then simulated data.")

    st.divider()
    st.markdown("**Timeframes shown:** M15 · H4 · D1 · W1 · MN1")
    fib_lookback = st.slider("Fibonacci swing lookback (bars)", 30, 250, 100, 10)
    sr_cluster = st.slider("S/R cluster tolerance (%)", 0.05, 0.30, 0.12, 0.01) / 100

    st.divider()
    if td_key:
        st.success("🟢 Twelve Data connected")
    else:
        st.info("📡 Yahoo Finance fallback")
    if claude_key:
        st.success("🤖 Claude AI note — enabled")
    else:
        st.caption("🤖 Claude AI note — add a key to enable")

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.title("💱 USDINTEL — FX Strength & Key Levels")

tc1, tc2 = st.columns([5, 1])
with tc2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.session_state.cache_buster = int(time.time())
        st.rerun()

with st.spinner("Fetching data..."):
    candles, prices = load_all(td_key, st.session_state.cache_buster)

# ═══════════════════════════════════════════════════════════════
# USD STRENGTH DASHBOARD
# ═══════════════════════════════════════════════════════════════
st.markdown("### 🏆 USD Strength Dashboard")
strength_tf = st.selectbox("Ranking timeframe", TIMEFRAMES, index=2, key="strength_tf")
strength_df = usd_strength(candles[strength_tf], strength_tf)

col_bar, col_table = st.columns([2, 1])
with col_bar:
    render_strength_bar(strength_df)
with col_table:
    st.dataframe(strength_df, use_container_width=True, hide_index=True)

gainer = strength_df.iloc[0]
loser = strength_df.iloc[-1]
st.caption(f"Strongest vs USD ({strength_tf}): **{gainer['Currency']}** {gainer['Change %']:+.3f}% "
           f"· Weakest: **{loser['Currency']}** {loser['Change %']:+.3f}%")

st.divider()

# ═══════════════════════════════════════════════════════════════
# PER-PAIR: FIBONACCI + SUPPORT/RESISTANCE ACROSS TIMEFRAMES
# ═══════════════════════════════════════════════════════════════
st.markdown("### 📐 Key Levels — Fibonacci & Support/Resistance")
st.caption("Computed independently on each of the five timeframes: M15, H4, D1, W1, MN1.")

tabs = st.tabs(PAIRS)
for i, pair in enumerate(PAIRS)