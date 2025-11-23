# config.py
import os
from typing import Optional

# ---- secret helper (prefer userdata.get in Colab) ----
def get_secret(k: str) -> Optional[str]:
    try:
        import userdata
        v = userdata.get(k)
        if v is not None:
            return v
    except Exception:
        pass
    return os.environ.get(k)

# ---- SECRETS (fill in userdata or os.environ before enabling live) ----
BITGET_KEY    = get_secret("BITGET_KEY")       # keep empty to avoid accidental live
BITGET_SECRET = get_secret("BITGET_SECRET")
BITGET_PASSPH = get_secret("BITGET_PASSPH")

TELEGRAM_TOKEN = get_secret("TELEGRAM_TOKEN")
TELEGRAM_CHAT  = get_secret("TELEGRAM_CHAT")   # numeric chat id

# ---- PAIRS & TFs ----
PAIR_HYPE  = "HYPEUSDT"
PAIR_ASTER = "ASTERUSDT"
TIMEFRAMES = [1,3,5,15,60]   # minutes (1m,3m,5m,15m,1h)

# ---- RISK & MM ----
TP_PCT = 1.2/100   # 1.2%
SL_PCT = 0.6/100   # 0.6%

def margin_percent_for_balance(b: float) -> float:
    if b < 10: return 0.35
    if b < 40: return 0.28
    if b < 200: return 0.22
    return 0.12

# hedging
HEDGE_MIN_PCT = 0.05
HEDGE_MAX_PCT = 0.5
HEDGE_INCREASE_TRIGGER = -0.006
HEDGE_MAX_LAPS = 2

# filters
VOLUME_SPIKE_MULT = 1.6
OB_IMBALANCE_THRESH = 0.18
CONF_THRESHOLD = 4
CONF_THRESHOLD_CONSOL = 3

# operational
MAX_ENTRIES_PER_DAY = 25
STATE_FILE = "bot_state_final.json"
LOOP_SLEEP_SECONDS = 12   # cadence

# Bitget public endpoints (candles & depth)
BITGET_CANDLES_URL = "https://api.bitget.com/api/spot/v1/market/candles?symbol={symbol}&granularity={granularity}&limit={limit}"
BITGET_DEPTH_URL   = "https://api.bitget.com/api/spot/v1/market/depth?symbol={symbol}&limit={limit}"

DEBUG = True

print("config.py loaded. TELEGRAM set:", bool(TELEGRAM_TOKEN))
# indicators.py
import math, requests, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from config import BITGET_CANDLES_URL, BITGET_DEPTH_URL, VOLUME_SPIKE_MULT, OB_IMBALANCE_THRESH, DEBUG

# ---------- network helpers ----------
def fetch_klines(symbol: str, granularity_seconds: int, limit: int = 200) -> pd.DataFrame:
    """
    Returns DataFrame with columns: time, open, high, low, close, volume
    granularity_seconds = e.g. 60 for 1m
    """
    try:
        url = BITGET_CANDLES_URL.format(symbol=symbol, granularity=granularity_seconds, limit=limit)
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        rows = j.get("data") if isinstance(j, dict) else j
        df = pd.DataFrame(rows)
        # many Bitget endpoints return reversed order; try to normalize
        if df.shape[1] >= 6:
            df = df.iloc[:, :6]
            df.columns = ["time","open","high","low","close","volume"]
        else:
            df = df[["time","open","high","low","close","volume"]]
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        # try parse time
        try:
            df['time'] = pd.to_datetime(df['time'], unit='ms')
        except:
            try:
                df['time'] = pd.to_datetime(df['time'])
            except:
                pass
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        if DEBUG: print("fetch_klines err", e)
        return None

def fetch_depth(symbol: str, limit: int = 50):
    try:
        url = BITGET_DEPTH_URL.format(symbol=symbol, limit=limit)
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") if isinstance(j, dict) else j
        # expect bids/asks list of [price, size]
        return data
    except Exception as e:
        if DEBUG: print("fetch_depth err", e)
        return {"bids": [], "asks": []}

# ---------- indicators ----------
def ema(series: pd.Series, span:int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length:int=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=(length-1), adjust=False).mean()
    ma_down = down.ewm(com=(length-1), adjust=False).mean()
    rs = ma_up/ma_down
    return 100 - (100/(1+rs))

def vwap(df: pd.DataFrame, window:int=50):
    pv = (df["close"] * df["volume"]).rolling(window).sum()
    v = df["volume"].rolling(window).sum()
    return pv / v

def parallel_channel(df: pd.DataFrame, window:int=30):
    if df is None or len(df) < window: return None
    sub = df.tail(window).reset_index(drop=True)
    X = np.arange(len(sub)).reshape(-1,1)
    lr_high = LinearRegression().fit(X, sub['high'].values)
    lr_low  = LinearRegression().fit(X, sub['low'].values)
    last_i = np.array([[len(sub)-1]])
    top = float(lr_high.predict(last_i)[0])
    bot = float(lr_low.predict(last_i)[0])
    return {"top": top, "bot": bot, "slope_high": float(lr_high.coef_[0]), "slope_low": float(lr_low.coef_[0])}

def pivots(df: pd.DataFrame, n:int=3):
    highs, lows = [], []
    close = df['close']
    for i in range(n, len(close)-n):
        w = close[i-n:i+n+1]
        c = close.iat[i]
        if c == w.max(): highs.append((i,c))
        if c == w.min(): lows.append((i,c))
    return {"highs": highs, "lows": lows}

def orderbook_imbalance(symbol: str) -> float:
    d = fetch_depth(symbol, limit=50)
    bids = d.get('bids', []) if isinstance(d, dict) else []
    asks = d.get('asks', []) if isinstance(d, dict) else []
    try:
        top_bids = sum(float(b[1]) for b in bids[:5]) if bids else 0.0
        top_asks = sum(float(a[1]) for a in asks[:5]) if asks else 0.0
        if (top_bids + top_asks) == 0: return 0.0
        return (top_bids - top_asks) / (top_bids + top_asks)
    except Exception:
        return 0.0

def volume_checks(df: pd.DataFrame):
    if df is None or len(df) < 10:
        return {"spike": False, "momentum": 0.0, "vol_last": 0.0, "vol_avg": 0.0}
    vol_last = float(df['volume'].iloc[-1])
    vol_avg = float(df['volume'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(df['volume'].mean())
    spike = vol_last > vol_avg * VOLUME_SPIKE_MULT if vol_avg > 0 else False
    signed = ((df['close'] - df['open'])/df['open']) * df['volume']
    momentum = float(signed.tail(10).sum()) if len(signed) >= 1 else 0.0
    return {"spike": spike, "momentum": momentum, "vol_last": vol_last, "vol_avg": vol_avg}

# quick sanity print
print("indicators.py loaded.")
# strategy.py
import math
from indicators import fetch_klines, volume_checks, orderbook_imbalance, parallel_channel, pivots, vwap
from indicators import ema, rsi
from config import TIMEFRAMES, PAIR_HYPE, PAIR_ASTER, CONF_THRESHOLD, CONF_THRESHOLD_CONSOL, HEDGE_MIN_PCT, HEDGE_MAX_PCT, margin_percent_for_balance

# weights by timeframe (can tune)
TF_WEIGHTS = {1:0.8, 3:0.9, 5:1.0, 15:1.2, 60:1.4}

def score_signals_for_df(df, symbol):
    if df is None or len(df) < 20:
        return {"signal":"NONE","score_long":0,"score_short":0,"reasons":["insufficient"], "df": df}
    sl = 0; ss = 0; reasons=[]
    # trend EMA20/50
    try:
        ema20 = ema(df['close'],20).iloc[-1]
        ema50 = ema(df['close'],50).iloc[-1]
        if df['close'].iloc[-1] > ema20 and ema20 > ema50:
            sl += 1; reasons.append("trend_up")
        if df['close'].iloc[-1] < ema20 and ema20 < ema50:
            ss += 1; reasons.append("trend_dn")
    except: pass

    # scalp EMA5/13 + RSI
    try:
        ema5 = ema(df['close'],5).iloc[-1]; ema13 = ema(df['close'],13).iloc[-1]
        rr = rsi(df['close'],14).iloc[-1]
        if ema5 > ema13 and rr > 45: sl += 1; reasons.append("scalp_long")
        if ema5 < ema13 and rr < 55: ss += 1; reasons.append("scalp_short")
    except: pass

    # volume spike
    vol = volume_checks(df)
    if vol["spike"]:
        if df['close'].iloc[-1] > df['open'].iloc[-1]:
            sl += 2; reasons.append("vol_spike_long")
        else:
            ss += 2; reasons.append("vol_spike_short")

    # breakout
    try:
        prev_high = df['high'].iloc[-2]; prev_low = df['low'].iloc[-2]; last = df['close'].iloc[-1]
        if last > prev_high and vol["spike"]:
            sl += 2; reasons.append("breakout")
        if last < prev_low and vol["spike"]:
            ss += 2; reasons.append("breakdown")
    except: pass

    # parallel channel bounce
    pc = parallel_channel(df, window=30)
    if pc:
        if df['low'].iloc[-1] <= pc['bot'] * 1.002 and df['close'].iloc[-1] > df['open'].iloc[-1]:
            sl += 1; reasons.append("pc_bounce_long")
        if df['high'].iloc[-1] >= pc['top'] * 0.998 and df['close'].iloc[-1] < df['open'].iloc[-1]:
            ss += 1; reasons.append("pc_bounce_short")

    # pivots SNR
    pv = pivots(df, n=3)
    if pv['lows']:
        lvl = pv['lows'][-1][1]
        if df['low'].iloc[-1] <= lvl * 1.002 and df['close'].iloc[-1] > df['open'].iloc[-1]:
            sl += 1; reasons.append("snr_long")
    if pv['highs']:
        lvl = pv['highs'][-1][1]
        if df['high'].iloc[-1] >= lvl * 0.998 and df['close'].iloc[-1] < df['open'].iloc[-1]:
            ss += 1; reasons.append("snr_short")

    # VWAP bias
    try:
        vv = vwap(df, window=50).iloc[-1]
        if df['close'].iloc[-1] > vv: sl += 1; reasons.append("vwap_above")
        else: ss += 1; reasons.append("vwap_below")
    except: pass

    # orderbook imbalance (approx)
    try:
        ob = orderbook_imbalance(symbol)
        if ob > 0.18: sl += 1; reasons.append("ob_bid_heavy")
        if ob < -0.18: ss += 1; reasons.append("ob_ask_heavy")
    except: pass

    final = "NONE"
    if sl >= 2 and sl > ss: final = "LONG"
    if ss >= 2 and ss > sl: final = "SHORT"
    return {"signal": final, "score_long": sl, "score_short": ss, "reasons": reasons, "df": df}

def compute_tf_signal(symbol: str, tf_minutes: int, balance: float):
    gran = tf_minutes * 60
    df = fetch_klines(symbol, gran, limit=200)
    res = score_signals_for_df(df, symbol)
    res["weight"] = TF_WEIGHTS.get(tf_minutes, 1.0)
    return res

def aggregate_multitf(symbol: str, balance: float):
    tf_results = {}
    total_long = 0.0; total_short = 0.0; reasons=[]
    for tf in TIMEFRAMES:
        res = compute_tf_signal(symbol, tf, balance)
        tf_results[tf] = res
        total_long += res["score_long"] * res.get("weight",1.0)
        total_short += res["score_short"] * res.get("weight",1.0)
        reasons += res["reasons"][:3]
    threshold = CONF_THRESHOLD
    if all(r["signal"]=="NONE" for r in tf_results.values()):
        threshold = CONF_THRESHOLD_CONSOL
    final = "NO_SIGNAL"
    if total_long >= threshold and total_long > total_short: final = "LONG"
    elif total_short >= threshold and total_short > total_long: final = "SHORT"
    return {"final": final, "conf_long": total_long, "conf_short": total_short, "tf_results": tf_results, "threshold": threshold, "reasons": list(set(reasons))}

def decide_main_and_hedge(hype_payload, aster_payload, balance: float):
    hype_conf = hype_payload['conf_long'] if hype_payload['final']=="LONG" else hype_payload['conf_short']
    aster_conf = aster_payload['conf_long'] if aster_payload['final']=="LONG" else aster_payload['conf_short']
    if hype_conf >= aster_conf:
        main_name, main = PAIR_HYPE, hype_payload
        counter_name, counter = PAIR_ASTER, aster_payload
    else:
        main_name, main = PAIR_ASTER, aster_payload
        counter_name, counter = PAIR_HYPE, hype_payload

    if main['final'] in ("NO_SIGNAL","NONE"):
        return {"action":"NO_TRADE"}

    conf_main = main['conf_long'] if main['final']=="LONG" else main['conf_short']
    conf_counter = counter['conf_long'] if counter['final']=="LONG" else counter['conf_short']

    # hedge decision
    hedge_needed = (conf_counter >= 3 and conf_counter >= 0.6 * (conf_main + 1))
    base_margin = margin_percent_for_balance(balance) * balance
    leverage = 20 if balance < 3 else 15
    if not hedge_needed:
        return {"action":"OPEN_MAIN", "symbol": main_name, "side": main['final'], "conf": conf_main, "order_params": {"margin_usdt": base_margin, "leverage": leverage}}
    hedge_pct = max(HEDGE_MIN_PCT, min((conf_counter/(conf_main+1))*0.5, HEDGE_MAX_PCT))
    main_params = {"margin_usdt": base_margin, "leverage": leverage}
    hedge_params = {"side": "SHORT" if main['final']=="LONG" else "LONG", "margin_usdt": base_margin * hedge_pct, "leverage": max(5, leverage-3)}
    return {"action":"OPEN_MAIN_AND_HEDGE", "symbol": main_name, "side": main['final'], "conf": conf_main, "hedge_against": counter_name, "main_params": main_params, "hedge_params": hedge_params}
# main.py
!pip install --quiet pandas scikit-learn python-telegram-bot requests

import time, json, math, os
from strategy import aggregate_multitf, decide_main_and_hedge
from config import PAIR_HYPE, PAIR_ASTER, STATE_FILE, MAX_ENTRIES_PER_DAY, LOOP_SLEEP_SECONDS, TP_PCT, SL_PCT, DEBUG, BITGET_KEY, BITGET_SECRET, BITGET_PASSPH
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT
import requests

# Telegram init
bot_enabled = TELEGRAM_TOKEN is not None and TELEGRAM_CHAT is not None
if bot_enabled:
    TG_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
def send_tele(msg: str):
    print("[TG]", msg)
    if bot_enabled:
        try:
            requests.post(TG_URL, json={"chat_id": TELEGRAM_CHAT, "text": msg}, timeout=5)
        except Exception as e:
            print("tg err", e)

# state persistence
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except:
            pass
    s = {"trades_today":0, "last_action":None, "balance":1.3, "history":[]}
    json.dump(s, open(STATE_FILE,"w"))
    return s
def save_state(s): json.dump(s, open(STATE_FILE,"w"))

# ---- helper to check min order size (basic / heuristic) ----
def fetch_market_min_order_usdt(symbol: str) -> float:
    # Attempt to get market info from public endpoints — fallback 1.0
    # NOTE: exchange-specific endpoints vary. User should verify on Bitget app.
    return 1.0

# ====== EXECUTE ORDER STUB (SIMULATE) ======
def execute_order(order: dict):
    """
    SAFETY: This function DOES NOT send live orders by default.
    It prints the exact payload and returns a simulated response.
    To enable live trading:
      - Replace the body with official SDK calls or signed REST per Bitget docs.
      - Test thoroughly on TESTNET.
      - Ensure minimum order size and size rounding are handled.
    Example commented code (bitget-python) is below — copy & enable yourself if you accept risk.
    """
    print("=== EXECUTE ORDER (SIMULATION) ===")
    print(json.dumps(order, indent=2))
    # Simulated response (mimic exchange)
    resp = {"status":"SIMULATED", "order_id":"SIM-"+str(int(time.time())), "payload": order}
    return resp

# Example (commented) how to implement using bitget-python (TESTNET first)
# ----------------------------------------------------------------------------
# from bitget_python import MixApi
# client = MixApi(BITGET_KEY, BITGET_SECRET, BITGET_PASSPH)
# # set leverage
# client.set_leverage(symbol=order['symbol'], marginCoin="USDT", leverage=order['leverage'])
# # compute qty
# ticker = client.get_ticker(order['symbol'])
# price = float(ticker['data']['last'])
# qty = round(order['margin_usdt'] / price, 6)  # adjust decimals & min lot
# # place order
# resp = client.place_order(symbol=order['symbol'], side="open_long" if order['side']=="LONG" else "open_short", size=str(qty), orderType="market")
# ----------------------------------------------------------------------------

# ====== MAIN LOOP ======
def main_loop():
    state = load_state()
    balance = state.get("balance", 1.3)
    send_tele(f"BOT START — Balance {balance:.2f} USDT — TP {TP_PCT*100:.2f}% / SL {SL_PCT*100:.2f}%")

    while True:
        try:
            hype = aggregate_multitf(PAIR_HYPE, balance)
            aster = aggregate_multitf(PAIR_ASTER, balance)
            action = decide_main_and_hedge(hype, aster, balance)
            send_tele(f"Decision: {action['action']} | HYPE conf {hype['conf_long']:.1f}/{hype['conf_short']:.1f} | ASTER conf {aster['conf_long']:.1f}/{aster['conf_short']:.1f}")

            if action['action'] == "OPEN_MAIN":
                order = {
                    "symbol": action["symbol"],
                    "side": action["side"],
                    "margin_usdt": float(action["order_params"]["margin_usdt"]),
                    "leverage": int(action["order_params"]["leverage"]),
                    "tp_pct": TP_PCT, "sl_pct": SL_PCT
                }
                # check min
                min_usdt = fetch_market_min_order_usdt(order['symbol'])
                if order['margin_usdt'] < min_usdt:
                    send_tele(f"Order {order['margin_usdt']:.2f} USDT < min {min_usdt} USDT — skipped")
                else:
                    res = execute_order(order)
                    state['last_action'] = {"type":"MAIN","order":order,"res":res}
                    state['trades_today'] += 1
                    state['history'].append({"t": time.time(), "order": order, "res": res})
                    save_state(state)

            elif action['action'] == "OPEN_MAIN_AND_HEDGE":
                mainp = action['main_params']
                hedgep = action['hedge_params']
                order_main = {"symbol": action['symbol'], "side": action['side'], "margin_usdt": float(mainp['margin_usdt']), "leverage": int(mainp['leverage']), "tp_pct": TP_PCT, "sl_pct": SL_PCT}
                order_hedge = {"symbol": action['hedge_against'], "side": hedgep['side'], "margin_usdt": float(hedgep['margin_usdt']), "leverage": int(hedgep['leverage']), "tp_pct": TP_PCT, "sl_pct": SL_PCT}
                # min checks
                if order_main['margin_usdt'] < fetch_market_min_order_usdt(order_main['symbol']):
                    send_tele("Main size too small — skipped main")
                else:
                    r1 = execute_order(order_main)
                if order_hedge['margin_usdt'] < fetch_market_min_order_usdt(order_hedge['symbol']):
                    send_tele("Hedge size too small — skipped hedge")
                else:
                    r2 = execute_order(order_hedge)
                state['trades_today'] += 2
                save_state(state)

            else:
                if DEBUG:
                    print("No trade this loop.")

            if state['trades_today'] >= MAX_ENTRIES_PER_DAY:
                send_tele("MAX trades/day reached — sleeping 10 minutes")
                time.sleep(600)

            time.sleep(LOOP_SLEEP_SECONDS)
        except Exception as e:
            send_tele(f"Engine error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main_loop()
