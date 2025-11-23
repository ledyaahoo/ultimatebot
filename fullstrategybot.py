# core.py
# Bitget Futures hybrid bot (full: data fetch -> strategy -> risk -> execution)
# Default LIVE=False (simulation). Set LIVE=True yourself only after verification.
# Uses userdata.get(...) first, fallback to os.environ
# Requires: pandas, numpy, requests

import os, time, json, math, traceback, requests
from datetime import datetime
from typing import Optional
import hmac, hashlib, base64
import numpy as np
import pandas as pd

# --------------------
# SETTINGS (edit only if you know what you do)
# --------------------
LIVE = True   # <<-- CHANGE TO True YOURSELF WHEN READY FOR LIVE ORDERS
USE_TESTNET = False
LOOP_SLEEP_SECONDS = 6
DEBUG = True
MAX_ENTRIES_PER_DAY = 25

# Correct futures symbols (Bitget USDT-M perp)
CONTRACT_PAIRS = ["HYPEUSDT_UMCBL", "ASTERUSDT_UMCBL", "UNIUSDT_UMCBL"]

TIMEFRAMES_MIN = [1,3,5,15,60]
TP_PCT = 0.012
SL_PCT = 0.006
LEVERAGE_SCALP = 15
LEVERAGE_RET = 17
MIN_ORDER_USDT_DEFAULT = 1.0

BASE_URL = "https://testnet.bitget.com" if USE_TESTNET else "https://api.bitget.com"
CANDLES_ENDPOINT = BASE_URL + "/api/mix/v1/market/candles"
TICKER_ENDPOINT = BASE_URL + "/api/mix/v1/market/ticker"
CONTRACTS_ENDPOINT = BASE_URL + "/api/mix/v1/market/contracts"
PLACE_ORDER_ENDPOINT = BASE_URL + "/api/mix/v1/order/placeOrder"
DEPTH_ENDPOINT_SPOT = BASE_URL + "/api/spot/v1/market/depth"

STATE_FILE = "bot_state_core.json"

# --------------------
# SECRETS loader (userdata preferred)
# --------------------
def get_secret(k: str) -> Optional[str]:
    try:
        import userdata
        v = userdata.get(k)
        if v is not None:
            return v
    except Exception:
        pass
    return os.environ.get(k)

API_KEY = get_secret("BITGET_API_KEY") or get_secret("API")
API_SECRET = get_secret("BITGET_API_SECRET") or get_secret("BITGET_SECRET_KEY") or get_secret("SECRET")
API_PASSPHRASE = get_secret("BITGET_PASSPHRASE") or get_secret("BITGET_PASSPHRASE_KEY")
TELEGRAM_TOKEN = get_secret("TELEGRAM_TOKEN")
TELEGRAM_CHAT = get_secret("TELEGRAM_CHAT")

# --------------------
# Telegram bridge (status only)
# --------------------
# telegram_status.py must be in same folder (we call tg_notify)
try:
    from telegram_status import tg_notify
except Exception:
    # fallback simple printer if telegram_status unavailable
    def tg_notify(m: str):
        print("[TG (local)]", m)

# --------------------
# State persistence
# --------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except:
            pass
    s = {"balance": 1.3, "trades_today": 0, "open_positions": [], "history": []}
    json.dump(s, open(STATE_FILE, "w"))
    return s

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)

STATE = load_state()

# --------------------
# Logging helpers
# --------------------
def nowstr():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def log(s):
    print(f"[{nowstr()}] {s}")

def tg_log(s):
    log(s)
    try:
        tg_notify(s)
    except Exception as e:
        log("tg_notify err: " + str(e))

# --------------------
# Bitget signing & headers
# --------------------
def _ts_ms():
    return str(int(time.time() * 1000))

def create_signature(timestamp: str, method: str, request_path: str, body: str = "") -> str:
    if not API_SECRET:
        return ""
    message = timestamp + method.upper() + request_path + (body or "")
    mac = hmac.new(API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def auth_headers(method: str, request_path: str, body: str=""):
    ts = _ts_ms()
    sig = create_signature(ts, method, request_path, body)
    return {
        "Content-Type": "application/json",
        "ACCESS-KEY": API_KEY or "",
        "ACCESS-SIGN": sig or "",
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": API_PASSPHRASE or ""
    }

# --------------------
# Market data helpers (Futures)
# --------------------
def fetch_klines_futures(symbol: str, gran_seconds: int, limit: int = 250):
    try:
        r = requests.get(CANDLES_ENDPOINT, params={"symbol": symbol, "granularity": gran_seconds, "limit": limit}, timeout=8)
        r.raise_for_status()
        j = r.json()
        return j.get("data") if isinstance(j, dict) and j.get("data") is not None else j
    except Exception as e:
        if DEBUG: log("fetch_klines err: " + str(e))
        return None

def fetch_ticker_futures(symbol: str):
    try:
        r = requests.get(TICKER_ENDPOINT, params={"symbol": symbol}, timeout=6)
        r.raise_for_status()
        j = r.json()
        return j.get("data") if isinstance(j, dict) and j.get("data") else j
    except Exception as e:
        if DEBUG: log("fetch_ticker err: " + str(e))
        return None

def fetch_depth_spot(symbol: str, limit: int = 50):
    try:
        r = requests.get(DEPTH_ENDPOINT_SPOT, params={"symbol": symbol.replace("_UMCBL",""), "limit": limit}, timeout=6)
        r.raise_for_status()
        j = r.json()
        return j.get("data") if isinstance(j, dict) and j.get("data") else j
    except Exception as e:
        if DEBUG: log("fetch_depth err: " + str(e))
        return {"bids": [], "asks": []}

def fetch_contracts():
    try:
        r = requests.get(CONTRACTS_ENDPOINT, timeout=6)
        r.raise_for_status()
        j = r.json()
        return j.get("data") if isinstance(j, dict) and j.get("data") else []
    except Exception as e:
        if DEBUG: log("fetch_contracts err: " + str(e))
        return []

def fetch_contract_info(symbol: str):
    try:
        contracts = fetch_contracts()
        for c in contracts:
            if c.get("symbol") == symbol:
                return {
                    "min_notional": float(c.get("minTrade", 0)) if c.get("minTrade") else None,
                    "size_precision": int(c.get("sizePrecision", 6)) if c.get("sizePrecision") else 6,
                    "price_precision": int(c.get("pricePrecision", 6)) if c.get("pricePrecision") else 6,
                    "contract_value": float(c.get("contractSize", 1)) if c.get("contractSize") else 1
                }
    except Exception:
        pass
    return {"min_notional": MIN_ORDER_USDT_DEFAULT, "size_precision": 6, "price_precision": 6, "contract_value": 1}

# --------------------
# Indicators (same as earlier multi-tech)
# --------------------
def rows_to_df(rows):
    if rows is None:
        return None
    df = pd.DataFrame(rows)
    if df.shape[1] >= 6:
        df = df.iloc[:, :6]
        df.columns = ["time","open","high","low","close","volume"]
    try:
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    except Exception:
        pass
    return df

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    d = series.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    ma_up = up.ewm(com=(length-1), adjust=False).mean()
    ma_down = down.ewm(com=(length-1), adjust=False).mean()
    rs = ma_up/ma_down
    return 100 - (100/(1+rs))

def vwap(df, window=50):
    pv = (df["close"] * df["volume"]).rolling(window).sum()
    v = df["volume"].rolling(window).sum()
    return pv / v

def parallel_channel(df, window=30):
    if df is None or len(df) < window: return None
    sub = df.tail(window).reset_index(drop=True)
    X = np.arange(len(sub))
    aH = np.polyfit(X, sub['high'].values, 1)
    aL = np.polyfit(X, sub['low'].values, 1)
    top = aH[0]*(len(sub)-1) + aH[1]
    bot = aL[0]*(len(sub)-1) + aL[1]
    return {"top": float(top), "bot": float(bot), "slope_high": float(aH[0]), "slope_low": float(aL[0])}

def pivots(df, n=3):
    highs, lows = [], []
    if df is None or len(df) < n*2+1: return {"highs": [], "lows": []}
    close = df['close']
    for i in range(n, len(close)-n):
        w = close[i-n:i+n+1]
        c = close.iat[i]
        if c == w.max(): highs.append((i,c))
        if c == w.min(): lows.append((i,c))
    return {"highs": highs, "lows": lows}

def orderbook_imbalance(symbol: str):
    d = fetch_depth_spot(symbol, limit=50)
    bids = d.get('bids', []) if isinstance(d, dict) else []
    asks = d.get('asks', []) if isinstance(d, dict) else []
    try:
        top_bids = sum(float(b[1]) for b in bids[:5]) if bids else 0.0
        top_asks = sum(float(a[1]) for a in asks[:5]) if asks else 0.0
        if (top_bids + top_asks) == 0: return 0.0
        return (top_bids - top_asks) / (top_bids + top_asks)
    except Exception:
        return 0.0

def volume_checks(df, lookback=20, mult=1.6):
    if df is None or len(df) < 10:
        return {"spike": False, "momentum": 0.0, "vol_last": 0.0, "vol_avg": 0.0}
    vol_last = float(df['volume'].iloc[-1])
    vol_avg = float(df['volume'].rolling(lookback).mean().iloc[-1]) if len(df) >= lookback else float(df['volume'].mean())
    spike = vol_last > vol_avg * mult if vol_avg > 0 else False
    signed = ((df['close'] - df['open'])/df['open']) * df['volume']
    momentum = float(signed.tail(10).sum()) if len(signed) >= 1 else 0.0
    return {"spike": spike, "momentum": momentum, "vol_last": vol_last, "vol_avg": vol_avg}

# --------------------
# Strategy scoring & multi-TF aggregation
# --------------------
TF_WEIGHTS = {1:0.8,3:0.9,5:1.0,15:1.2,60:1.4}
CONF_THRESHOLD = 4.0
CONF_THRESHOLD_CONSOL = 3.0

def score_signals_for_df(df, symbol):
    if df is None or len(df) < 30:
        return {"signal":"NONE","score_long":0,"score_short":0,"reasons":["insufficient"], "df": df}
    s_long = 0; s_short = 0; reasons=[]
    try:
        ema20 = ema(df['close'],20).iloc[-1]; ema50 = ema(df['close'],50).iloc[-1]
        if df['close'].iloc[-1] > ema20 and ema20 > ema50: s_long += 1; reasons.append("trend_up")
        if df['close'].iloc[-1] < ema20 and ema20 < ema50: s_short += 1; reasons.append("trend_dn")
    except: pass
    try:
        ema5 = ema(df['close'],5).iloc[-1]; ema13 = ema(df['close'],13).iloc[-1]
        rr = rsi(df['close'],14).iloc[-1]
        if ema5 > ema13 and rr > 45: s_long += 1; reasons.append("scalp_long")
        if ema5 < ema13 and rr < 55: s_short += 1; reasons.append("scalp_short")
    except: pass
    vol = volume_checks(df)
    if vol["spike"]:
        if df['close'].iloc[-1] > df['open'].iloc[-1]:
            s_long += 2; reasons.append("vol_spike_long")
        else:
            s_short += 2; reasons.append("vol_spike_short")
    try:
        prev_high = df['high'].iloc[-2]; prev_low = df['low'].iloc[-2]; last = df['close'].iloc[-1]
        if last > prev_high and vol["spike"]: s_long += 2; reasons.append("breakout")
        if last < prev_low and vol["spike"]: s_short += 2; reasons.append("breakdown")
    except: pass
    pc = parallel_channel(df, window=30)
    if pc:
        if df['low'].iloc[-1] <= pc['bot'] * 1.002 and df['close'].iloc[-1] > df['open'].iloc[-1]:
            s_long += 1; reasons.append("pc_bounce_long")
        if df['high'].iloc[-1] >= pc['top'] * 0.998 and df['close'].iloc[-1] < df['open'].iloc[-1]:
            s_short += 1; reasons.append("pc_bounce_short")
    pv = pivots(df, n=3)
    if pv['lows']:
        lvl = pv['lows'][-1][1]
        if df['low'].iloc[-1] <= lvl * 1.002 and df['close'].iloc[-1] > df['open'].iloc[-1]:
            s_long += 1; reasons.append("snr_long")
    if pv['highs']:
        lvl = pv['highs'][-1][1]
        if df['high'].iloc[-1] >= lvl * 0.998 and df['close'].iloc[-1] < df['open'].iloc[-1]:
            s_short += 1; reasons.append("snr_short")
    try:
        vv = vwap(df, window=50).iloc[-1]
        if df['close'].iloc[-1] > vv: s_long += 1; reasons.append("vwap_above")
        else: s_short += 1; reasons.append("vwap_below")
    except: pass
    try:
        ob = orderbook_imbalance(symbol)
        if ob > 0.18: s_long += 1; reasons.append("ob_bid_heavy")
        if ob < -0.18: s_short += 1; reasons.append("ob_ask_heavy")
    except: pass
    final = "NONE"
    if s_long >= 2 and s_long > s_short: final = "LONG"
    if s_short >= 2 and s_short > s_long: final = "SHORT"
    return {"signal": final, "score_long": s_long, "score_short": s_short, "reasons": reasons, "df": df}

def compute_tf_signal(symbol: str, tf_minutes: int, balance: float):
    gran = tf_minutes * 60
    rows = fetch_klines_futures(symbol, gran, limit=250)
    df = rows_to_df(rows)
    res = score_signals_for_df(df, symbol)
    res["weight"] = TF_WEIGHTS.get(tf_minutes, 1.0)
    return res

def aggregate_multitf(symbol: str, balance: float):
    tf_results = {}
    total_long = 0.0; total_short = 0.0; reasons=[]
    for tf in TIMEFRAMES_MIN:
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
    return {"final": final, "conf_long": total_long, "conf_short": total_short, "tf_results": tf_results, "reasons": list(set(reasons))}

def decide_main_and_hedge(hype_payload, aster_payload, balance: float):
    hype_conf = hype_payload['conf_long'] if hype_payload['final']=="LONG" else hype_payload['conf_short']
    aster_conf = aster_payload['conf_long'] if aster_payload['final']=="LONG" else aster_payload['conf_short']
    if hype_conf >= aster_conf:
        main_name, main = CONTRACT_PAIRS[0], hype_payload
        counter_name, counter = CONTRACT_PAIRS[1], aster_payload
    else:
        main_name, main = CONTRACT_PAIRS[1], aster_payload
        counter_name, counter = CONTRACT_PAIRS[0], hype_payload
    if main['final'] in ("NO_SIGNAL","NONE"):
        return {"action":"NO_TRADE"}
    conf_main = main['conf_long'] if main['final']=="LONG" else main['conf_short']
    conf_counter = counter['conf_long'] if counter['final']=="LONG" else counter['conf_short']
    base_margin = max(0.9, (0.35 if STATE.get("balance",1.3) < 3 else 0.30 if STATE.get("balance",1.3) < 10 else 0.25 if STATE.get("balance",1.3) < 40 else 0.18) * STATE.get("balance",1.3))
    leverage = LEVERAGE_RET if conf_main > 6 else LEVERAGE_SCALP
    hedge_needed = (conf_counter >= 0.6 * (conf_main + 1) and conf_counter >= 3)
    if not hedge_needed:
        return {"action":"OPEN_MAIN", "symbol": main_name, "side": main['final'], "conf": conf_main, "order_params": {"margin_usdt": base_margin, "leverage": leverage}}
    else:
        hedge_pct = max(0.05, min((conf_counter/(conf_main+1))*0.5, 0.5))
        main_params = {"margin_usdt": base_margin, "leverage": leverage}
        hedge_params = {"side": "SHORT" if main['final']=="LONG" else "LONG", "margin_usdt": base_margin * hedge_pct, "leverage": max(5, leverage-3)}
        return {"action":"OPEN_MAIN_AND_HEDGE", "symbol": main_name, "side": main['final'], "main_params": main_params, "hedge_params": hedge_params, "hedge_against": counter_name}

# --------------------
# Order execution helpers
# --------------------
def compute_qty_from_margin(symbol: str, margin_usdt: float):
    cinfo = fetch_contract_info(symbol)
    ticker = fetch_ticker_futures(symbol)
    price = None
    if isinstance(ticker, dict):
        price = float(ticker.get("last") or ticker.get("lastPrice") or ticker.get("last") or 0)
    if not price or price == 0:
        raise Exception("Cannot fetch price")
    contract_value = cinfo.get("contract_value", 1) or 1
    prec = cinfo.get("size_precision", 6) or 6
    qty = (margin_usdt / price) / contract_value
    qty = math.floor(qty * (10**prec)) / (10**prec)
    if qty <= 0:
        raise Exception("qty <= 0")
    return qty, price, cinfo

def place_market_order_live(symbol: str, side: str, qty: float):
    body = {
        "symbol": symbol,
        "marginCoin": "USDT",
        "size": str(qty),
        "side": "open_long" if side == "LONG" else "open_short",
        "orderType": "market"
    }
    b = json.dumps(body, separators=(',',':'))
    headers = auth_headers("POST", "/api/mix/v1/order/placeOrder", b)
    try:
        resp = requests.post(PLACE_ORDER_ENDPOINT, data=b, headers=headers, timeout=10)
        return resp.json()
    except Exception as e:
        return {"code":"ERR","msg": str(e)}

def place_close_order_live(symbol: str, side: str, qty: float):
    body = {
        "symbol": symbol,
        "marginCoin": "USDT",
        "size": str(qty),
        "side": "close_long" if side == "LONG" else "close_short",
        "orderType": "market",
        "reduceOnly": True
    }
    b = json.dumps(body, separators=(',',':'))
    headers = auth_headers("POST", "/api/mix/v1/order/placeOrder", b)
    try:
        resp = requests.post(PLACE_ORDER_ENDPOINT, data=b, headers=headers, timeout=10)
        return resp.json()
    except Exception as e:
        return {"code":"ERR","msg": str(e)}

def execute_order_payload(payload: dict):
    symbol = payload["symbol"]; side = payload["side"]; margin_usdt = float(payload["margin_usdt"]); leverage = int(payload["leverage"])
    cinfo = fetch_contract_info(symbol)
    min_notional = cinfo.get("min_notional") or MIN_ORDER_USDT_DEFAULT
    if margin_usdt < min_notional:
        tg_log(f"SKIP order: margin {margin_usdt:.3f} < min_notional {min_notional}")
        return {"status":"SKIPPED_MIN"}
    try:
        qty, price, _ = compute_qty_from_margin(symbol, margin_usdt)
    except Exception as e:
        tg_log("Qty compute error: " + str(e))
        return {"status":"ERR_QTY","err": str(e)}
    if LIVE:
        res = place_market_order_live(symbol, side, qty)
        tg_log(f"LIVE ORDER {symbol} {side} qty {qty} -> {res.get('code')}")
    else:
        res = {"code":"SIM","payload":{"symbol":symbol,"side":side,"qty":qty,"price_est":price,"margin":margin_usdt}}
        tg_log(f"SIM ORDER {symbol} {side} qty {qty} price {price} margin {margin_usdt}")
    STATE["open_positions"].append({"symbol": symbol, "side": side, "qty": qty, "entry": price, "leverage": leverage, "ts": time.time()})
    STATE["trades_today"] = STATE.get("trades_today",0) + 1
    STATE["history"].append({"t": time.time(), "payload": payload, "resp": res})
    save_state(STATE)
    return res

def close_position(pos):
    symbol = pos["symbol"]; side = pos["side"]; qty = pos["qty"]
    if LIVE:
        res = place_close_order_live(symbol, side, qty)
        tg_log(f"LIVE CLOSE {symbol} {side} qty {qty} -> {res.get('code')}")
    else:
        res = {"code":"SIM_CLOSE","payload":{"symbol":symbol,"side":side,"qty":qty}}
        tg_log(f"SIM CLOSE {symbol} {side} qty {qty}")
    STATE["open_positions"] = [p for p in STATE["open_positions"] if not (p["symbol"]==symbol and p["ts"]==pos["ts"])]
    STATE["history"].append({"t": time.time(), "action":"close", "pos": pos, "resp": res})
    save_state(STATE)
    return res

# --------------------
# Position manager (TP/SL)
# --------------------
def manage_positions():
    for pos in list(STATE.get("open_positions", [])):
        try:
            t = fetch_ticker_futures(pos["symbol"])
            price = float(t.get("last") or t.get("lastPrice") or t.get("last") or 0)
            entry = float(pos["entry"])
            side = pos["side"]
            pnl_pct = (price - entry)/entry if side=="LONG" else (entry - price)/entry
            if pnl_pct >= TP_PCT:
                tg_log(f"TP HIT {pos['symbol']} {side} pnl {pnl_pct*100:.2f}% -> closing")
                close_position(pos)
            elif pnl_pct <= -SL_PCT:
                tg_log(f"SL HIT {pos['symbol']} {side} pnl {pnl_pct*100:.2f}% -> closing")
                close_position(pos)
        except Exception as e:
            tg_log("manage_positions err: " + str(e))

# --------------------
# Main loop
# --------------------
def main_loop():
    tg_log(f"BOT START (LIVE={LIVE}) Pairs: {','.join(CONTRACT_PAIRS)}")
    last_day = datetime.utcnow().date()
    while True:
        try:
            if datetime.utcnow().date() != last_day:
                STATE["trades_today"] = 0
                last_day = datetime.utcnow().date()
                save_state(STATE)
            manage_positions()
            if STATE.get("trades_today",0) >= MAX_ENTRIES_PER_DAY:
                tg_log("Daily max reached - sleeping 10min")
                time.sleep(600)
                continue
            balance = STATE.get("balance", 1.3)
            hype = aggregate_multitf(CONTRACT_PAIRS[0], balance)
            aster = aggregate_multitf(CONTRACT_PAIRS[1], balance)
            decision = decide_main_and_hedge(hype, aster, balance)
            tg_log(f"Decision: {decision.get('action')} | HYPE {hype.get('final')} conf {hype.get('conf_long'):.2f}/{hype.get('conf_short'):.2f} | ASTER {aster.get('final')} conf {aster.get('conf_long'):.2f}/{aster.get('conf_short'):.2f}")
            if decision.get("action") == "OPEN_MAIN":
                op = decision.get("order_params") or decision.get("main_params")
                if op:
                    payload = {"symbol": decision["symbol"], "side": decision["side"], "margin_usdt": op["margin_usdt"], "leverage": op["leverage"]}
                    execute_order_payload(payload)
            elif decision.get("action") == "OPEN_MAIN_AND_HEDGE":
                mainp = decision.get("main_params"); hedgep = decision.get("hedge_params")
                if mainp:
                    p1 = {"symbol": decision["symbol"], "side": decision["side"], "margin_usdt": mainp["margin_usdt"], "leverage": mainp["leverage"]}
                    execute_order_payload(p1)
                if hedgep:
                    p2 = {"symbol": decision.get("hedge_against"), "side": hedgep["side"], "margin_usdt": hedgep["margin_usdt"], "leverage": hedgep["leverage"]}
                    execute_order_payload(p2)
            else:
                if DEBUG: log("No trade this loop.")
            time.sleep(LOOP_SLEEP_SECONDS)
        except KeyboardInterrupt:
            tg_log("Stopped by user.")
            break
        except Exception as e:
            tg_log("Main loop error: " + str(e))
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    tg_log("Core loaded. Starting main loop (SIMULATION unless LIVE=True).")
    main_loop()
