# === BAGIAN 1: SETUP & KONEKSI EXCHANGE ===
print("🔄 MEMULAI BAGIAN 1: SETUP & KONEKSI EXCHANGE...")

# Install required packages
!pip install ccxt ta pandas numpy requests python-telegram-bot plotly > /dev/null 2>&1

# Import semua library
from google.colab import userdata
import ccxt
import pandas as pd
import numpy as np
import time
import requests
import json
from datetime import datetime, timedelta
import ta
import warnings
from typing import Dict, List, Tuple, Optional
from collections import deque, Counter
import statistics

warnings.filterwarnings('ignore')

print("✅ Semua library berhasil diimport")

# Test Colab Secrets
try:
    # BitGet credentials
    BITGET_API_KEY = userdata.get('BITGET_API_KEY')
    BITGET_SECRET_KEY = userdata.get('BITGET_SECRET_KEY') 
    BITGET_PASSPHRASE = userdata.get('BITGET_PASSPHRASE')
    
    # Telegram credentials
    TELEGRAM_TOKEN = userdata.get('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = userdata.get('TELEGRAM_CHAT_ID')
    
    # Buat variabel kompatibel
    TELEGRAM_BOT_TOKEN = TELEGRAM_TOKEN
    TELEGRAM_BOT_CHAT_ID = TELEGRAM_CHAT_ID
    
    # Check credentials
    bitget_creds = [BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE]
    
    if all(bitget_creds):
        print("✅ BitGet Secrets BERHASIL diakses")
    else:
        print("❌ BitGet Secrets TIDAK LENGKAP")
        missing = [name for name, value in zip(['API_KEY', 'SECRET_KEY', 'PASSPHRASE'], bitget_creds) if not value]
        print(f"   Missing: {', '.join(missing)}")
    
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        print("✅ Telegram Secrets BERHASIL diakses")
    else:
        print("❌ Telegram Secrets TIDAK LENGKAP")
        
except Exception as e:
    print(f"❌ Error mengakses Colab Secrets: {e}")

# Test Exchange Connection
def connect_bitget(max_retries=3):
    for attempt in range(max_retries):
        try:
            exchange = ccxt.bitget({
                'apiKey': BITGET_API_KEY,
                'secret': BITGET_SECRET_KEY,
                'password': BITGET_PASSPHRASE,
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Test connectivity
            markets = exchange.load_markets()
            print(f"✅ Koneksi BitGet BERHASIL - {len(markets)} markets loaded")
            
            # Test balance
            balance = exchange.fetch_balance()
            usdt_balance = balance['USDT']['free'] if 'USDT' in balance else 0
            print(f"💰 Balance USDT: ${usdt_balance:.2f}")
            
            return exchange
            
        except ccxt.AuthenticationError as e:
            print(f"❌ Authentication Error: {e}")
            break
            
        except ccxt.ExchangeError as e:
            if "Invalid IP" in str(e):
                print(f"❌ Attempt {attempt + 1}: Invalid IP Error")
                if attempt < max_retries - 1:
                    time.sleep(10)
                else:
                    print("💡 Buat API Key BARU dengan IP Restriction DISABLED")
            else:
                print(f"❌ Exchange Error: {e}")
                break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
    return None

# Jalankan koneksi
exchange = connect_bitget()

if exchange is None:
    print("\n🚨 TINDAKAN DARURAT:")
    print("1. Buat API Key BARU dengan Futures Trading permissions")
    print("2. IP Restriction: DISABLED")
    print("3. Update secrets di Colab")
    print("4. Jalankan ulang script")
else:
    print("✅ Koneksi BitGet SIAP digunakan!")

# ✅ UPDATE: Daftar simbol dengan Aster & Hype, hapus MATICUSDT
ALTCOINS = [
    'LTCUSDT', 'SOLUSDT', 'AVAXUSDT', 'UNIUSDT', 
    'ADAUSDT', 'XRPUSDT', 'ATOMUSDT', 'ALGOUSDT',
    'ASTERUSDT', 'HYPEUSDT'  # ✅ Tambahkan Aster & Hype
]

print(f"🎯 Target Altcoins: {', '.join(ALTCOINS)}")
print(f"📊 Total {len(ALTCOINS)} symbols termasuk Aster & Hype")

print("✅ BAGIAN 1 SELESAI: Setup berhasil!")

# === BAGIAN 2: RISK MANAGEMENT & STRATEGY ===
print("\n🔄 MEMULAI BAGIAN 2: RISK MANAGEMENT & STRATEGY...")

class ProgressiveRiskManager:
    def __init__(self, initial_balance=2.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.total_profit = 0.0
        self.daily_order_count = 0
        self.max_daily_orders = 40
        self.last_reset = datetime.now()
        self.conservative_mode = False
        self.hourly_trade_count = 0
        self.last_hour_reset = datetime.now()
        
        print(f"✅ Progressive Risk Manager initialized - Balance: ${initial_balance}")
    
    def get_trading_tier(self):
        if self.current_balance < 5:
            return "micro"
        elif self.current_balance < 10:
            return "small"
        elif self.current_balance < 15:
            return "medium"
        else:
            return "advanced"
    
    def calculate_position_size(self, confidence, strategy_type="scalping", symbol=None):
        """✅ UPDATE: Special handling untuk meme coins"""
        tier = self.get_trading_tier()
        
        # ✅ SPECIAL RULES UNTUK MEME COINS
        if symbol in ['ASTERUSDT', 'HYPEUSDT']:
            base_size = 0.5  # Fixed $0.5 untuk meme coins
            leverage = 8 if tier == "micro" else 12
            max_trades = 2
            print(f"🎭 MEME COIN RULES: {symbol} - Size: ${base_size}, Leverage: {leverage}x")
        else:
            # Normal rules untuk coins lain
            if tier == "micro":
                base_size = 1.0
                leverage = 15
                max_trades = 2
            elif tier == "small":
                base_size = 1.5
                leverage = 15
                max_trades = 3
            elif tier == "medium":
                base_size = self.current_balance * 0.3 * confidence
                leverage = 20
                max_trades = 3
            else:
                base_size = self.current_balance * 0.4 * confidence
                leverage = 25
                max_trades = 4
        
        # Adjust berdasarkan confidence
        position_size = base_size * confidence
        
        # Batasan minimum & maximum
        min_size = 0.3
        max_size = self.current_balance * 0.5
        
        position_size = max(min_size, min(position_size, max_size))
        
        print(f"💰 Tier: {tier.upper()} | Size: ${position_size:.2f} | Leverage: {leverage}x")
        return position_size, leverage, max_trades
    
    def update_balance(self, new_balance):
        profit = new_balance - self.current_balance
        self.current_balance = new_balance
        self.total_profit += profit
        
        print(f"📈 Balance Update: ${self.current_balance:.2f} | Profit: ${profit:.2f}")
        
        # Reset counters
        if datetime.now() - self.last_reset > timedelta(hours=24):
            self.daily_order_count = 0
            self.conservative_mode = False
            self.last_reset = datetime.now()
        
        if datetime.now() - self.last_hour_reset > timedelta(hours=1):
            self.hourly_trade_count = 0
            self.last_hour_reset = datetime.now()
    
    def can_trade(self):
        if self.daily_order_count >= 30 and not self.conservative_mode:
            self.conservative_mode = True
            print("🎯 ACTIVATING CONSERVATIVE MODE (30+ trades today)")
        
        if self.daily_order_count >= self.max_daily_orders:
            print(f"⚠️ Daily order limit reached ({self.max_daily_orders} trades)")
            return False
        
        hourly_limit = 8
        if self.hourly_trade_count >= hourly_limit:
            print(f"⚠️ Hourly trade limit reached ({hourly_limit} trades)")
            return False
            
        return True
    
    def record_trade(self):
        self.daily_order_count += 1
        self.hourly_trade_count += 1
        print(f"📝 Trade recorded: {self.daily_order_count}/{self.max_daily_orders} daily")

class MarketAnalyzer:
    def __init__(self, exchange):
        self.exchange = exchange
        print("✅ Market Analyzer initialized")
    
    def get_ohlcv_data(self, symbol: str, timeframe: str = '5m', limit: int = 100):
        try:
            # ✅ UPDATE: Handle symbol tidak tersedia
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                print(f"⚠️ No data for {symbol}")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame):
        try:
            # Price Action
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            
            # Volume Analysis
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_spike'] = df['volume_ratio'] > 1.5
            
            # Momentum Indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['macd_hist'] = ta.trend.MACD(df['close']).macd_diff()
            
            # Trend Indicators
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            
            return df
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return df

print("✅ BAGIAN 2 SELESAI: Risk management berhasil!")

# === BAGIAN 3: UNIVERSAL CHANNEL STRATEGY ===
print("\n🔄 MEMULAI BAGIAN 3: UNIVERSAL CHANNEL STRATEGY...")

class UniversalChannelStrategy:
    def __init__(self, exchange):
        self.exchange = exchange
        self.analyzer = MarketAnalyzer(exchange)
        print("✅ Universal Channel Strategy initialized")
    
    def calculate_parallel_channels(self, df: pd.DataFrame, period: int = 20):
        try:
            df['highest_high'] = df['high'].rolling(window=period).max()
            df['lowest_low'] = df['low'].rolling(window=period).min()
            df['channel_middle'] = (df['highest_high'] + df['lowest_low']) / 2
            
            channel_width = (df['highest_high'] - df['lowest_low']) * 0.1
            df['upper_channel'] = df['highest_high'] - channel_width
            df['lower_channel'] = df['lowest_low'] + channel_width
            
            df['channel_direction'] = 'sideways'
            df.loc[df['highest_high'] > df['highest_high'].shift(5), 'channel_direction'] = 'up'
            df.loc[df['lowest_low'] < df['lowest_low'].shift(5), 'channel_direction'] = 'down'
            
            current_price = df['close'].iloc[-1]
            price_position = (current_price - df['lower_channel'].iloc[-1]) / (df['upper_channel'].iloc[-1] - df['lower_channel'].iloc[-1])
            
            channel_info = {
                'upper': df['upper_channel'].iloc[-1],
                'lower': df['lower_channel'].iloc[-1],
                'middle': df['channel_middle'].iloc[-1],
                'direction': df['channel_direction'].iloc[-1],
                'price_position': price_position,
                'width': (df['upper_channel'].iloc[-1] - df['lower_channel'].iloc[-1]) / df['channel_middle'].iloc[-1]
            }
            
            return df, channel_info
            
        except Exception as e:
            print(f"❌ Error calculating channels: {e}")
            return df, {}
    
    def generate_scalping_signals(self, df: pd.DataFrame, symbol: str, conservative_mode=False):
        """✅ UPDATE: Special handling untuk meme coins"""
        try:
            df, channel = self.calculate_parallel_channels(df)
            if not channel:
                return []
            
            current_price = df['close'].iloc[-1]
            signals = []
            
            # ✅ SPECIAL RULES UNTUK MEME COINS
            if symbol in ['ASTERUSDT', 'HYPEUSDT']:
                return self.generate_meme_coin_signals(df, symbol, channel, conservative_mode)
            
            # Normal rules untuk coins lain
            min_conditions = 6 if conservative_mode else 5
            min_confidence = 0.75 if conservative_mode else 0.65
            
            # SCALPING LONG Signals
            long_conditions = [
                current_price <= channel['lower'] * 1.008,
                df['rsi'].iloc[-1] < 65,
                df['volume_ratio'].iloc[-1] > 1.3,
                df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2],
                current_price > df['ema_12'].iloc[-1],
                channel['direction'] in ['up', 'sideways'],
                df['close'].iloc[-1] > df['open'].iloc[-1]
            ]
            
            long_score = sum(long_conditions)
            if long_score >= min_conditions:
                confidence = long_score / 7
                if confidence >= min_confidence:
                    signals.append({
                        'strategy': 'scalping_channel_long',
                        'direction': 'long',
                        'confidence': confidence,
                        'timeframe': '5m',
                        'entry_type': 'channel_bounce',
                        'conservative_approved': conservative_mode,
                        'channel_info': channel
                    })
                    mode_tag = " 🎯CONSERVATIVE" if conservative_mode else ""
                    print(f"✅ {symbol} SCALPING LONG - Score: {long_score}/7{mode_tag}")
            
            # SCALPING SHORT Signals
            short_conditions = [
                current_price >= channel['upper'] * 0.992,
                df['rsi'].iloc[-1] > 35,
                df['volume_ratio'].iloc[-1] > 1.3,
                df['macd_hist'].iloc[-1] < df['macd_hist'].iloc[-2],
                current_price < df['ema_12'].iloc[-1],
                channel['direction'] in ['down', 'sideways'],
                df['close'].iloc[-1] < df['open'].iloc[-1]
            ]
            
            short_score = sum(short_conditions)
            if short_score >= min_conditions:
                confidence = short_score / 7
                if confidence >= min_confidence:
                    signals.append({
                        'strategy': 'scalping_channel_short',
                        'direction': 'short',
                        'confidence': confidence,
                        'timeframe': '5m',
                        'entry_type': 'channel_rejection',
                        'conservative_approved': conservative_mode,
                        'channel_info': channel
                    })
                    mode_tag = " 🎯CONSERVATIVE" if conservative_mode else ""
                    print(f"✅ {symbol} SCALPING SHORT - Score: {short_score}/7{mode_tag}")
            
            return signals
            
        except Exception as e:
            print(f"❌ Error generating scalping signals for {symbol}: {e}")
            return []
    
    def generate_meme_coin_signals(self, df: pd.DataFrame, symbol: str, channel, conservative_mode=False):
        """✅ NEW: Strategy khusus untuk meme coins"""
        try:
            current_price = df['close'].iloc[-1]
            signals = []
            
            # Lebih strict conditions untuk meme coins
            min_conditions = 5
            min_confidence = 0.75
            
            # Hanya LONG signals untuk meme coins (kurang predictable untuk short)
            meme_conditions = [
                df['volume_ratio'].iloc[-1] > 2.0,  # Volume sangat tinggi
                df['price_change'].iloc[-1] > 0.01,  # Minimum 1% movement
                df['rsi'].iloc[-1] < 70,             # Tidak overbought
                df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2],
                current_price > df['ema_12'].iloc[-1],
                channel['width'] > 0.08,             # Cukup volatile
                df['close'].iloc[-1] > df['open'].iloc[-1]  # Bullish candle
            ]
            
            meme_score = sum(meme_conditions)
            if meme_score >= min_conditions:
                confidence = meme_score / 7
                if confidence >= min_confidence:
                    signals.append({
                        'strategy': 'meme_sniper_long',
                        'direction': 'long',  # Hanya long untuk meme coins
                        'confidence': confidence,
                        'timeframe': '5m',
                        'entry_type': 'meme_breakout',
                        'conservative_approved': conservative_mode,
                        'channel_info': channel
                    })
                    mode_tag = " 🎯CONSERVATIVE" if conservative_mode else ""
                    print(f"🎭 {symbol} MEME SNIPER - Score: {meme_score}/7{mode_tag}")
            
            return signals
            
        except Exception as e:
            print(f"❌ Error generating meme signals for {symbol}: {e}")
            return []

print("✅ BAGIAN 3 SELESAI: Universal channel strategy berhasil!")

# === BAGIAN 4: TRADING BOT ===
print("\n🔄 MEMULAI BAGIAN 4: TRADING BOT...")

# Fix Telegram variables
try:
    telegram_available = TELEGRAM_TOKEN is not None and TELEGRAM_CHAT_ID is not None
    if telegram_available:
        print("✅ Telegram integration READY")
    else:
        print("⚠️ Telegram integration DISABLED")
except NameError:
    TELEGRAM_TOKEN = None
    TELEGRAM_CHAT_ID = None
    telegram_available = False
    print("⚠️ Telegram integration DISABLED")

class GrowthTradingBot:
    def __init__(self, exchange, initial_balance=2.0):
        self.exchange = exchange
        self.risk_manager = ProgressiveRiskManager(initial_balance)
        self.channel_strategy = UniversalChannelStrategy(exchange)
        self.analyzer = MarketAnalyzer(exchange)
        
        self.real_trading_enabled = False
        self.telegram_enabled = telegram_available
        self.active_trades = []
        self.cycle_count = 0
        self.total_signals_generated = 0
        self.total_trades_executed = 0
        
        print("✅ Growth Trading Bot initialized")
        print(f"📱 Telegram: {'ENABLED' if self.telegram_enabled else 'DISABLED'}")
    
    def enable_real_trading(self):
        print("🚨 REAL TRADING MODE ACTIVATION")
        print(f"💰 Initial Balance: ${self.risk_manager.current_balance:.2f}")
        print("⚠️ HIGH RISK! ONLY USE READY-TO-LOSE FUNDS!")
        
        confirm1 = input("Enable REAL TRADING? (YES/NO): ")
        if confirm1.upper() != 'YES':
            print("❌ Real trading not enabled")
            return False
            
        confirm2 = input("Confirm understand 100% loss risk? (YES/NO): ")
        if confirm2.upper() == 'YES':
            self.real_trading_enabled = True
            print("✅ REAL TRADING MODE ENABLED")
            self.send_telegram_message(f"🚀 REAL TRADING STARTED - Balance: ${self.risk_manager.current_balance:.2f}")
            return True
            
        print("❌ Real trading cancelled")
        return False
    
    def send_telegram_message(self, message: str):
        if not self.telegram_enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            print(f"❌ Telegram error: {e}")
    
    def execute_trade(self, symbol: str, signal: dict):
        if not self.risk_manager.can_trade():
            return None
        
        try:
            # ✅ UPDATE: Gunakan symbol-specific position sizing
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            strategy_type = signal['strategy'].split('_')[0]
            position_size, leverage, max_trades = self.risk_manager.calculate_position_size(
                signal['confidence'], strategy_type, symbol  # ✅ Pass symbol untuk special handling
            )
            
            if len(self.active_trades) >= max_trades:
                print(f"⚠️ Max trades reached ({max_trades}), skipping...")
                return None
            
            amount = position_size / current_price
            
            trade_info = (
                f"🎯 TRADE SIGNAL:\n"
                f"Symbol: {symbol}\n"
                f"Strategy: {signal['strategy']}\n"
                f"Direction: {signal['direction'].upper()}\n"
                f"Amount: ${position_size:.2f}\n"
                f"Leverage: {leverage}x\n"
                f"Confidence: {signal['confidence']:.2%}\n"
                f"Price: ${current_price:.6f}"
            )
            
            print(trade_info)
            self.send_telegram_message(trade_info)
            
            if self.real_trading_enabled:
                confirm = input("🚨 EXECUTE REAL ORDER? (YES/NO): ")
                if confirm.upper() == 'YES':
                    # ✅ UPDATE: Futures trading parameters
                    self.exchange.set_leverage(leverage, symbol)
                    
                    order = self.exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=signal['direction'],
                        amount=amount,
                        params={'productType': 'USDT-FUTURES'}  # ✅ Futures trading
                    )
                    
                    trade_record = {
                        'order_id': order['id'],
                        'symbol': symbol,
                        'direction': signal['direction'],
                        'amount': amount,
                        'entry_price': current_price,
                        'position_size': position_size,
                        'leverage': leverage,
                        'timestamp': datetime.now()
                    }
                    self.active_trades.append(trade_record)
                    self.risk_manager.record_trade()
                    self.total_trades_executed += 1
                    
                    success_msg = f"✅ ORDER FILLED: {order['id']}"
                    print(success_msg)
                    self.send_telegram_message(success_msg)
                    
                    return order
                else:
                    print("❌ Order cancelled")
                    return None
            else:
                # Paper trading
                print("📝 PAPER TRADE EXECUTED")
                paper_trade = {
                    'order_id': f"paper_{datetime.now().strftime('%H%M%S')}",
                    'symbol': symbol,
                    'direction': signal['direction'],
                    'amount': amount,
                    'entry_price': current_price,
                    'position_size': position_size,
                    'leverage': leverage,
                    'timestamp': datetime.now(),
                    'paper': True
                }
                self.active_trades.append(paper_trade)
                self.risk_manager.record_trade()
                self.total_trades_executed += 1
                return paper_trade
                
        except Exception as e:
            error_msg = f"❌ Trade execution error: {str(e)}"
            print(error_msg)
            self.send_telegram_message(error_msg)
            return None
    
    def analyze_symbols(self, symbols):
        print(f"\n🔍 ANALYZING {len(symbols)} SYMBOLS...")
        
        all_signals = []
        conservative_mode = self.risk_manager.conservative_mode
        
        for symbol in symbols:
            try:
                # ✅ UPDATE: Skip symbols yang error
                df_5m = self.analyzer.get_ohlcv_data(symbol, '5m', 50)
                if df_5m is not None:
                    df_5m = self.analyzer.calculate_technical_indicators(df_5m)
                    signals = self.channel_strategy.generate_scalping_signals(df_5m, symbol, conservative_mode)
                    all_signals.extend(signals)
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
        
        for signal in all_signals:
            signal['symbol'] = symbol
        
        self.total_signals_generated += len(all_signals)
        print(f"📊 Total signals: {len(all_signals)} | Conservative: {conservative_mode}")
        return all_signals
    
    def run_trading_cycle(self, symbols):
        print(f"\n🔄 TRADING CYCLE STARTED - {datetime.now().strftime('%H:%M:%S')}")
        
        signals = self.analyze_symbols(symbols)
        
        if not signals:
            print("📊 No trading signals this cycle")
            return
        
        conservative_mode = self.risk_manager.conservative_mode
        if conservative_mode:
            filtered_signals = [s for s in signals if s.get('conservative_approved', False) and s['confidence'] >= 0.75]
            print(f"🎯 Conservative Mode: {len(filtered_signals)} high-quality signals")
        else:
            filtered_signals = [s for s in signals if s['confidence'] >= 0.65]
            print(f"⚡ Normal Mode: {len(filtered_signals)} signals")
        
        filtered_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        current_tier = self.risk_manager.get_trading_tier()
        max_executions = 1 if conservative_mode else (1 if current_tier == "micro" else 3)
        
        executed = 0
        for signal in filtered_signals:
            if executed >= max_executions:
                break
            result = self.execute_trade(signal['symbol'], signal)
            if result:
                executed += 1
                time.sleep(2)
        
        print(f"✅ Cycle completed: {executed} trades executed")
    
    def start_growth_bot(self, symbols=None, cycle_interval=180):
        if symbols is None:
            symbols = ALTCOINS
        
        print(f"🚀 STARTING GROWTH TRADING BOT")
        print(f"💰 Initial Balance: ${self.risk_manager.current_balance:.2f}")
        print(f"📊 Symbols: {', '.join(symbols)}")
        print(f"⏱ Cycle Interval: {cycle_interval}s")
        
        if not self.real_trading_enabled:
            print("💵 PAPER TRADING MODE - No real orders")
        
        self.cycle_count = 0
        try:
            while True:
                self.cycle_count += 1
                print(f"\n" + "="*50)
                print(f"🎯 CYCLE #{self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"💰 Balance: ${self.risk_manager.current_balance:.2f}")
                print(f"🔢 Daily Orders: {self.risk_manager.daily_order_count}/40")
                print(f"🎯 Mode: {'CONSERVATIVE' if self.risk_manager.conservative_mode else 'NORMAL'}")
                print("="*50)
                
                self.run_trading_cycle(symbols)
                
                # Simulate balance update untuk paper trading
                if not self.real_trading_enabled and self.cycle_count % 5 == 0:
                    simulated_change = np.random.uniform(-0.2, 0.5)
                    new_balance = max(0.1, self.risk_manager.current_balance + simulated_change)
                    self.risk_manager.update_balance(new_balance)
                
                print(f"💤 Waiting {cycle_interval} seconds...")
                time.sleep(cycle_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            self.send_telegram_message("🛑 Trading bot stopped manually")
        except Exception as e:
            error_msg = f"❌ Bot error: {str(e)}"
            print(error_msg)

# FINAL BOT SETUP
print("\n🎯 FINAL BOT SETUP...")

try:
    growth_bot = GrowthTradingBot(exchange, initial_balance=2.0)
    
    print("🤖 GROWTH TRADING BOT READY!")
    print("📋 Available commands:")
    print("   - growth_bot.enable_real_trading()  # Enable real trading")
    print("   - growth_bot.start_growth_bot()     # Start the bot")
    print("   - growth_bot.run_trading_cycle(['ASTERUSDT'])  # Test Aster")
    
    print("\n🆕 ENHANCED FEATURES:")
    print("   ✅ Aster & Hype integration dengan special rules")
    print("   ✅ 40 trades/hari maximum")
    print("   ✅ Conservative mode auto-aktif")
    print("   ✅ Futures trading dengan permissions fix")
    
except Exception as e:
    print(f"❌ Final bot setup failed: {e}")

print("✅ BAGIAN 4 SELESAI: Trading bot siap!")
print("\n🎉 BOT FINAL++ SIAP DENGAN SEMUA UPDATE!")
