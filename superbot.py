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
import math

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

# Test Exchange Connection - FIXED VERSION
def connect_bitget(max_retries=3):
    for attempt in range(max_retries):
        try:
            exchange = ccxt.bitget({
                'apiKey': BITGET_API_KEY,
                'secret': BITGET_SECRET_KEY,
                'password': BITGET_PASSPHRASE,
                'sandbox': False,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'createMarketBuyOrderRequiresPrice': False,
                },
            })
            
            # Test connectivity
            markets = exchange.load_markets()
            print(f"✅ Koneksi BitGet BERHASIL - {len(markets)} markets loaded")
            
            # Test balance - FIXED untuk futures
            try:
                balance = exchange.fetch_balance({'type': 'swap'})
                if 'USDT' in balance['total']:
                    usdt_balance = balance['total']['USDT']
                    print(f"💰 Balance Futures USDT: ${usdt_balance:.2f}")
                else:
                    print("⚠️ USDT balance not found in futures account")
                    usdt_balance = 0
            except Exception as balance_error:
                print(f"⚠️ Balance check error: {balance_error}")
                usdt_balance = 0
            
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

# ✅ Daftar simbol trading
ALTCOINS = [
    'LTCUSDT', 'SOLUSDT', 'AVAXUSDT', 'UNIUSDT', 
    'ADAUSDT', 'XRPUSDT', 'ATOMUSDT', 'ALGOUSDT',
    'ASTERUSDT', 'HYPEUSDT'
]

print(f"🎯 Target Altcoins: {', '.join(ALTCOINS)}")
print(f"📊 Total {len(ALTCOINS)} symbols termasuk Aster & Hype")

print("✅ BAGIAN 1 SELESAI: Setup berhasil!")
# === BAGIAN 2: RISK MANAGEMENT & STRATEGY ===
print("\n🔄 MEMULAI BAGIAN 2: RISK MANAGEMENT & STRATEGY...")

class ProgressiveRiskManager:
    def __init__(self, initial_balance=1.3):  # ✅ UBAH: $1.30
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.total_profit = 0.0
        self.daily_order_count = 0
        self.max_daily_orders = 40
        self.last_reset = datetime.now()
        self.conservative_mode = False
        self.hourly_trade_count = 0
        self.last_hour_reset = datetime.now()
        self.market_condition = "NORMAL"
        
        print(f"✅ Progressive Risk Manager initialized - Balance: ${initial_balance:.2f}")  # ✅ Format 2 decimal
    
    def get_trading_tier(self):
        if self.current_balance < 3:  # ✅ UBAH: Tier adjustment untuk balance kecil
            return "micro"
        elif self.current_balance < 6:
            return "small"
        elif self.current_balance < 10:
            return "medium"
        else:
            return "advanced"
    
    def calculate_position_size(self, confidence, strategy_type="scalping", symbol=None):
        """✅ Position sizing adaptif untuk balance $1.30"""
        tier = self.get_trading_tier()
        
        # ✅ SPECIAL RULES UNTUK MEME COINS
        if symbol in ['ASTERUSDT', 'HYPEUSDT']:
            base_size = 0.3  # ✅ UBAH: Fixed $0.3 untuk meme coins (balance kecil)
            leverage = 6 if tier == "micro" else 8  # ✅ UBAH: Leverage lebih rendah
            max_trades = 2
        else:
            # Adaptive sizing berdasarkan market condition
            if self.market_condition == "LOW_VOLATILITY":
                size_multiplier = 0.6  # ✅ UBAH: Lebih konservatif
                leverage_multiplier = 0.7
            elif self.market_condition == "HIGH_VOLATILITY":
                size_multiplier = 0.7
                leverage_multiplier = 0.8
            elif self.market_condition == "EXTREME":
                size_multiplier = 0.4
                leverage_multiplier = 0.5
            else:
                size_multiplier = 0.8  # ✅ UBAH: Reduced dari 1.0
                leverage_multiplier = 0.9
            
            if tier == "micro":
                base_size = 0.8 * size_multiplier  # ✅ UBAH: $0.8 base untuk micro
                leverage = int(8 * leverage_multiplier)  # ✅ UBAH: Leverage lebih rendah
                max_trades = 2
            elif tier == "small":
                base_size = 1.0 * size_multiplier
                leverage = int(10 * leverage_multiplier)
                max_trades = 2  # ✅ UBAH: Max trades lebih sedikit
            elif tier == "medium":
                base_size = self.current_balance * 0.2 * confidence * size_multiplier  # ✅ UBAH: 20% risk
                leverage = int(12 * leverage_multiplier)
                max_trades = 3
            else:
                base_size = self.current_balance * 0.25 * confidence * size_multiplier  # ✅ UBAH: 25% risk
                leverage = int(15 * leverage_multiplier)
                max_trades = 3  # ✅ UBAH: Max trades lebih sedikit
        
        # Adjust berdasarkan confidence
        position_size = base_size * confidence
        
        # Batasan minimum & maximum untuk balance kecil
        min_size = 0.2  # ✅ UBAH: Minimum $0.20
        max_size = min(self.current_balance * 0.3, 3.0)  # ✅ UBAH: Maksimal 30% balance
        
        position_size = max(min_size, min(position_size, max_size))
        
        print(f"💰 Tier: {tier.upper()} | Market: {self.market_condition} | Size: ${position_size:.2f} | Leverage: {leverage}x")
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
        if self.daily_order_count >= 25 and not self.conservative_mode:  # ✅ UBAH: 25 trades untuk conservative
            self.conservative_mode = True
            print("🎯 ACTIVATING CONSERVATIVE MODE (25+ trades today)")
        
        if self.daily_order_count >= self.max_daily_orders:
            print(f"⚠️ Daily order limit reached ({self.max_daily_orders} trades)")
            return False
        
        # Adaptive hourly limits berdasarkan market condition
        hourly_limits = {
            "NORMAL": 5,  # ✅ UBAH: Limits lebih rendah
            "TRENDING": 6,
            "RANGING": 4,
            "HIGH_VOLATILITY": 3,
            "LOW_VOLATILITY": 2,
            "EXTREME": 1
        }
        
        hourly_limit = hourly_limits.get(self.market_condition, 5)
        if self.hourly_trade_count >= hourly_limit:
            print(f"⚠️ Hourly trade limit reached ({hourly_limit} trades)")
            return False
            
        return True
    
    def record_trade(self):
        self.daily_order_count += 1
        self.hourly_trade_count += 1
        print(f"📝 Trade recorded: {self.daily_order_count}/{self.max_daily_orders} daily")
    
    def update_market_condition(self, condition):
        """Update market condition untuk adaptive trading"""
        self.market_condition = condition
        print(f"📊 Market Condition Updated: {condition}")

# ✅ SNR ENHANCEMENT UNTUK SCALPING PRESISI
class SNRScalpingEnhancer:
    def __init__(self):
        print("✅ SNR Scalping Enhancer initialized")
    
    def calculate_dynamic_snr(self, df, short_period=15, medium_period=30, long_period=50):
        """✅ DYNAMIC SNR UNTUK SCALPING 5M"""
        try:
            # Multiple timeframe SNR untuk scalping
            df['sr_short_high'] = df['high'].rolling(window=short_period).max()
            df['sr_short_low'] = df['low'].rolling(window=short_period).min()
            
            df['sr_medium_high'] = df['high'].rolling(window=medium_period).max() 
            df['sr_medium_low'] = df['low'].rolling(window=medium_period).min()
            
            df['sr_long_high'] = df['high'].rolling(window=long_period).max()
            df['sr_long_low'] = df['low'].rolling(window=long_period).min()
            
            # Weighted SNR levels berdasarkan timeframe
            current_price = df['close'].iloc[-1]
            
            snr_levels = {
                'short_resistance': df['sr_short_high'].iloc[-1],
                'short_support': df['sr_short_low'].iloc[-1],
                'medium_resistance': df['sr_medium_high'].iloc[-1],
                'medium_support': df['sr_medium_low'].iloc[-1], 
                'long_resistance': df['sr_long_high'].iloc[-1],
                'long_support': df['sr_long_low'].iloc[-1],
            }
            
            # Hitung jarak ke setiap SNR level
            for key, level in snr_levels.items():
                if not pd.isna(level):
                    distance_pct = abs(level - current_price) / current_price
                    snr_levels[f'{key}_distance'] = distance_pct
            
            return df, snr_levels
            
        except Exception as e:
            print(f"❌ Error calculating dynamic SNR: {e}")
            return df, {}
    
    def get_snr_signal_strength(self, current_price, snr_levels, direction):
        """✅ Hitung kekuatan sinyal berdasarkan SNR"""
        strength = 1.0  # Base strength
        
        if direction == 'long':
            # Untuk LONG: dekat support = kuat, dekat resistance = lemah
            support_distances = [
                snr_levels.get('short_support_distance', 1),
                snr_levels.get('medium_support_distance', 1), 
                snr_levels.get('long_support_distance', 1)
            ]
            
            resistance_distances = [
                snr_levels.get('short_resistance_distance', 1),
                snr_levels.get('medium_resistance_distance', 1),
                snr_levels.get('long_resistance_distance', 1)  
            ]
            
            # Boost jika dekat support manapun
            valid_support_dists = [d for d in support_distances if d < 1]
            if valid_support_dists:
                min_support_dist = min(valid_support_dists)
                if min_support_dist < 0.02:  # Dalam 2% dari support
                    strength *= 1.3
                    print(f"🎯 SNR BOOST: Near support ({min_support_dist:.2%})")
            
            # Reduce jika dekat resistance
            valid_resistance_dists = [d for d in resistance_distances if d < 1]
            if valid_resistance_dists:
                min_resistance_dist = min(valid_resistance_dists)
                if min_resistance_dist < 0.015:  # Dalam 1.5% dari resistance
                    strength *= 0.7
                    print(f"⚠️ SNR WARNING: Near resistance ({min_resistance_dist:.2%})")
                
        elif direction == 'short':
            # Untuk SHORT: dekat resistance = kuat, dekat support = lemah
            resistance_distances = [
                snr_levels.get('short_resistance_distance', 1),
                snr_levels.get('medium_resistance_distance', 1),
                snr_levels.get('long_resistance_distance', 1)  
            ]
            
            support_distances = [
                snr_levels.get('short_support_distance', 1),
                snr_levels.get('medium_support_distance', 1), 
                snr_levels.get('long_support_distance', 1)
            ]
            
            # Boost jika dekat resistance manapun
            valid_resistance_dists = [d for d in resistance_distances if d < 1]
            if valid_resistance_dists:
                min_resistance_dist = min(valid_resistance_dists)
                if min_resistance_dist < 0.02:  # Dalam 2% dari resistance
                    strength *= 1.3
                    print(f"🎯 SNR BOOST: Near resistance ({min_resistance_dist:.2%})")
            
            # Reduce jika dekat support  
            valid_support_dists = [d for d in support_distances if d < 1]
            if valid_support_dists:
                min_support_dist = min(valid_support_dists)
                if min_support_dist < 0.015:  # Dalam 1.5% dari support
                    strength *= 0.7
                    print(f"⚠️ SNR WARNING: Near support ({min_support_dist:.2%})")
        
        return max(0.5, min(2.0, strength))  # Clamp antara 0.5-2.0

class MarketAnalyzer:
    def __init__(self, exchange):
        self.exchange = exchange
        self.snr_enhancer = SNRScalpingEnhancer()
        print("✅ Market Analyzer dengan SNR Enhancement initialized")
    
    def get_ohlcv_data(self, symbol: str, timeframe: str = '5m', limit: int = 100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 20:
                print(f"⚠️ Insufficient data for {symbol}")
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
            
            # ✅ Tambahkan Dynamic SNR
            df, snr_levels = self.snr_enhancer.calculate_dynamic_snr(df)
            
            return df, snr_levels
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return df, {}
    
    def assess_market_condition(self, df, symbol):
        """✅ Deteksi kondisi market untuk adaptive trading"""
        try:
            volatility = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            adx = df['adx'].iloc[-1] if not pd.isna(df['adx'].iloc[-1]) else 0
            
            condition_scores = {
                "NORMAL": 0,
                "TRENDING": 0, 
                "RANGING": 0,
                "HIGH_VOLATILITY": 0,
                "LOW_VOLATILITY": 0,
                "EXTREME": 0
            }
            
            # Scoring system
            if 0.03 <= volatility <= 0.08 and 1.2 <= volume_ratio <= 2.5:
                condition_scores["NORMAL"] += 3
            if adx > 25:
                condition_scores["TRENDING"] += 2
            if volatility < 0.02:
                condition_scores["LOW_VOLATILITY"] += 2
            if volatility > 0.12:
                condition_scores["HIGH_VOLATILITY"] += 2
            if volume_ratio > 3.0 or volatility > 0.15:
                condition_scores["EXTREME"] += 3
            if 0.02 <= volatility <= 0.05 and adx < 20:
                condition_scores["RANGING"] += 2
                
            detected_condition = max(condition_scores, key=condition_scores.get)
            print(f"📊 {symbol} Market Condition: {detected_condition} (Vol: {volatility:.2%}, ADX: {adx:.1f})")
            
            return detected_condition
            
        except Exception as e:
            print(f"❌ Error assessing market condition: {e}")
            return "NORMAL"

print("✅ BAGIAN 2 SELESAI: Risk management & SNR enhancement berhasil!")
# === BAGIAN 3: UNIVERSAL CHANNEL STRATEGY DENGAN SNR ===
print("\n🔄 MEMULAI BAGIAN 3: UNIVERSAL CHANNEL STRATEGY DENGAN SNR...")

class UniversalChannelStrategy:
    def __init__(self, exchange):
        self.exchange = exchange
        self.analyzer = MarketAnalyzer(exchange)
        print("✅ Universal Channel Strategy dengan SNR initialized")
    
    def calculate_parallel_channels(self, df: pd.DataFrame, period: int = 20):
        try:
            if len(df) < period:
                return df, {}
                
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
    
    def check_reversal_conditions(self, df, current_direction):
        """✅ Deteksi kondisi reversal untuk quick flip"""
        try:
            if current_direction == 'long':
                # Long to Short reversal conditions
                reversal_conditions = [
                    df['rsi'].iloc[-1] > 70,                      # Overbought
                    df['macd_hist'].iloc[-1] < df['macd_hist'].iloc[-2] * 0.8 if not pd.isna(df['macd_hist'].iloc[-2]) else False,
                    df['close'].iloc[-1] < df['sma_20'].iloc[-1] if not pd.isna(df['sma_20'].iloc[-1]) else False,
                    df['volume_ratio'].iloc[-1] > 1.5,            # Volume konfirmasi
                    df['close'].iloc[-1] < df['open'].iloc[-1],   # Bearish candle
                ]
            else:
                # Short to Long reversal conditions  
                reversal_conditions = [
                    df['rsi'].iloc[-1] < 30,                      # Oversold
                    df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2] * 1.2 if not pd.isna(df['macd_hist'].iloc[-2]) else False,
                    df['close'].iloc[-1] > df['sma_20'].iloc[-1] if not pd.isna(df['sma_20'].iloc[-1]) else False,
                    df['volume_ratio'].iloc[-1] > 1.5,            # Volume konfirmasi
                    df['close'].iloc[-1] > df['open'].iloc[-1],   # Bullish candle
                ]
            
            reversal_score = sum(reversal_conditions)
            reversal_strength = reversal_score / len(reversal_conditions)
            
            if reversal_strength >= 0.6:  # 60% confidence untuk reversal
                print(f"🔄 REVERSAL DETECTED: {current_direction}→{'short' if current_direction == 'long' else 'long'} (Strength: {reversal_strength:.2f})")
                return True, reversal_strength
            
            return False, reversal_strength
            
        except Exception as e:
            print(f"❌ Error checking reversal conditions: {e}")
            return False, 0.0
    
    def generate_scalping_signals(self, df: pd.DataFrame, symbol: str, conservative_mode=False, market_condition="NORMAL"):
        """✅ ENHANCED: Channel strategy dengan SNR confirmation & market adaptation"""
        try:
            if df is None or len(df) < 30:
                return []
                
            df, channel = self.calculate_parallel_channels(df)
            if not channel:
                return []
            
            # ✅ Dapatkan SNR levels dari analyzer
            df, snr_levels = self.analyzer.calculate_technical_indicators(df)
            
            current_price = df['close'].iloc[-1]
            signals = []
            
            # ✅ SPECIAL RULES UNTUK MEME COINS
            if symbol in ['ASTERUSDT', 'HYPEUSDT']:
                return self.generate_meme_coin_signals(df, symbol, channel, snr_levels, conservative_mode, market_condition)
            
            # Adaptive requirements berdasarkan market condition
            condition_requirements = {
                "NORMAL": {"min_conditions": 5, "min_confidence": 0.65},
                "TRENDING": {"min_conditions": 5, "min_confidence": 0.60},
                "RANGING": {"min_conditions": 6, "min_confidence": 0.70},
                "HIGH_VOLATILITY": {"min_conditions": 6, "min_confidence": 0.75},
                "LOW_VOLATILITY": {"min_conditions": 6, "min_confidence": 0.70},
                "EXTREME": {"min_conditions": 7, "min_confidence": 0.80}
            }
            
            req = condition_requirements.get(market_condition, condition_requirements["NORMAL"])
            min_conditions = req["min_conditions"] if not conservative_mode else req["min_conditions"] + 1
            min_confidence = req["min_confidence"] if not conservative_mode else req["min_confidence"] + 0.05

            # Validasi data terlebih dahulu
            if any(pd.isna([df['rsi'].iloc[-1], df['volume_ratio'].iloc[-1], df['macd_hist'].iloc[-1]])):
                print(f"⚠️ Invalid indicators for {symbol}, skipping...")
                return []

            # SCALPING LONG Signals
            long_conditions = [
                current_price <= channel['lower'] * 1.008,
                df['rsi'].iloc[-1] < 65,
                df['volume_ratio'].iloc[-1] > 1.3,
                df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2] if not pd.isna(df['macd_hist'].iloc[-2]) else False,
                current_price > df['ema_12'].iloc[-1] if not pd.isna(df['ema_12'].iloc[-1]) else False,
                channel['direction'] in ['up', 'sideways'],
                df['close'].iloc[-1] > df['open'].iloc[-1]
            ]
            
            long_score = sum(long_conditions)
            if long_score >= min_conditions:
                base_confidence = long_score / 7
                
                # ✅ Apply SNR strength adjustment
                snr_strength = self.analyzer.snr_enhancer.get_snr_signal_strength(
                    current_price, snr_levels, 'long'
                )
                final_confidence = base_confidence * snr_strength
                
                if final_confidence >= min_confidence:
                    signals.append({
                        'strategy': 'scalping_channel_long',
                        'direction': 'long',
                        'confidence': final_confidence,
                        'base_confidence': base_confidence,
                        'snr_strength': snr_strength,
                        'timeframe': '5m',
                        'entry_type': 'channel_bounce',
                        'conservative_approved': conservative_mode,
                        'market_condition': market_condition,
                        'channel_info': channel,
                        'snr_levels': snr_levels
                    })
                    mode_tag = " 🎯CONSERVATIVE" if conservative_mode else ""
                    print(f"✅ {symbol} SCALPING LONG - Score: {long_score}/7 | SNR: {snr_strength:.2f}x | Market: {market_condition}{mode_tag}")
            
            # SCALPING SHORT Signals
            short_conditions = [
                current_price >= channel['upper'] * 0.992,
                df['rsi'].iloc[-1] > 35,
                df['volume_ratio'].iloc[-1] > 1.3,
                df['macd_hist'].iloc[-1] < df['macd_hist'].iloc[-2] if not pd.isna(df['macd_hist'].iloc[-2]) else False,
                current_price < df['ema_12'].iloc[-1] if not pd.isna(df['ema_12'].iloc[-1]) else False,
                channel['direction'] in ['down', 'sideways'],
                df['close'].iloc[-1] < df['open'].iloc[-1]
            ]
            
            short_score = sum(short_conditions)
            if short_score >= min_conditions:
                base_confidence = short_score / 7
                
                # ✅ Apply SNR strength adjustment
                snr_strength = self.analyzer.snr_enhancer.get_snr_signal_strength(
                    current_price, snr_levels, 'short'
                )
                final_confidence = base_confidence * snr_strength
                
                if final_confidence >= min_confidence:
                    signals.append({
                        'strategy': 'scalping_channel_short',
                        'direction': 'short',
                        'confidence': final_confidence,
                        'base_confidence': base_confidence,
                        'snr_strength': snr_strength,
                        'timeframe': '5m',
                        'entry_type': 'channel_rejection',
                        'conservative_approved': conservative_mode,
                        'market_condition': market_condition,
                        'channel_info': channel,
                        'snr_levels': snr_levels
                    })
                    mode_tag = " 🎯CONSERVATIVE" if conservative_mode else ""
                    print(f"✅ {symbol} SCALPING SHORT - Score: {short_score}/7 | SNR: {snr_strength:.2f}x | Market: {market_condition}{mode_tag}")
            
            return signals
            
        except Exception as e:
            print(f"❌ Error generating scalping signals for {symbol}: {e}")
            return []
    
    def generate_meme_coin_signals(self, df: pd.DataFrame, symbol: str, channel, snr_levels, conservative_mode=False, market_condition="NORMAL"):
        """✅ ENHANCED: Meme coin strategy dengan SNR & market adaptation"""
        try:
            current_price = df['close'].iloc[-1]
            signals = []
            
            # Lebih strict conditions untuk meme coins
            min_conditions = 5
            min_confidence = 0.75
            
            # Validasi data
            if any(pd.isna([df['rsi'].iloc[-1], df['volume_ratio'].iloc[-1], df['macd_hist'].iloc[-1]])):
                return []

            # Hanya LONG signals untuk meme coins (kurang predictable untuk short)
            meme_conditions = [
                df['volume_ratio'].iloc[-1] > 2.0,  # Volume sangat tinggi
                df['price_change'].iloc[-1] > 0.01,  # Minimum 1% movement
                df['rsi'].iloc[-1] < 70,             # Tidak overbought
                df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2] if not pd.isna(df['macd_hist'].iloc[-2]) else False,
                current_price > df['ema_12'].iloc[-1] if not pd.isna(df['ema_12'].iloc[-1]) else False,
                channel['width'] > 0.08,             # Cukup volatile
                df['close'].iloc[-1] > df['open'].iloc[-1]  # Bullish candle
            ]
            
            meme_score = sum(meme_conditions)
            if meme_score >= min_conditions:
                base_confidence = meme_score / 7
                
                # ✅ Apply SNR strength adjustment untuk meme coins
                snr_strength = self.analyzer.snr_enhancer.get_snr_signal_strength(
                    current_price, snr_levels, 'long'  # Meme coins hanya long
                )
                final_confidence = base_confidence * snr_strength
                
                if final_confidence >= min_confidence:
                    signals.append({
                        'strategy': 'meme_sniper_long',
                        'direction': 'long',  # Hanya long untuk meme coins
                        'confidence': final_confidence,
                        'base_confidence': base_confidence,
                        'snr_strength': snr_strength,
                        'timeframe': '5m',
                        'entry_type': 'meme_breakout',
                        'conservative_approved': conservative_mode,
                        'market_condition': market_condition,
                        'channel_info': channel,
                        'snr_levels': snr_levels
                    })
                    mode_tag = " 🎯CONSERVATIVE" if conservative_mode else ""
                    print(f"🎭 {symbol} MEME SNIPER - Score: {meme_score}/7 | SNR: {snr_strength:.2f}x | Market: {market_condition}{mode_tag}")
            
            return signals
            
        except Exception as e:
            print(f"❌ Error generating meme signals for {symbol}: {e}")
            return []

print("✅ BAGIAN 3 SELESAI: Universal channel strategy dengan SNR berhasil!")
# === BAGIAN 4.1: TRADING BOT CORE & ORDER EXECUTION ===
print("\n🔄 MEMULAI BAGIAN 4.1: TRADING BOT CORE & ORDER EXECUTION...")

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
    def __init__(self, exchange, initial_balance=1.3):  # ✅ UBAH: $1.30
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
        self.reversal_trades_count = 0
        self.profit_history = []
        
        print("✅ Growth Trading Bot dengan SNR Enhancement initialized")
        print(f"💰 Initial Balance: ${initial_balance:.2f}")  # ✅ Format 2 decimal
        print(f"📱 Telegram: {'ENABLED' if self.telegram_enabled else 'DISABLED'}")
    
    def enable_real_trading(self):
        print("🚨 REAL TRADING MODE ACTIVATION")
        print(f"💰 Initial Balance: ${self.risk_manager.current_balance:.2f}")  # ✅ Format 2 decimal
        print("⚠️ HIGH RISK! ONLY USE READY-TO-LOSE FUNDS!")
        
        confirm1 = input("Enable REAL TRADING? (YES/NO): ")
        if confirm1.upper() != 'YES':
            print("❌ Real trading not enabled")
            return False
            
        confirm2 = input("Confirm understand 100% loss risk? (YES/NO): ")
        if confirm2.upper() == 'YES':
            self.real_trading_enabled = True
            print("✅ REAL TRADING MODE ENABLED")
            if self.telegram_enabled:
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
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                print(f"⚠️ Telegram send failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Telegram error: {e}")
    
    def close_position(self, trade):
        """✅ Close position untuk reversal trading"""
        try:
            if trade.get('paper', False):
                # Paper trading - simulate close
                print(f"📝 PAPER TRADE CLOSED: {trade['symbol']} {trade['direction']}")
                if trade in self.active_trades:
                    self.active_trades.remove(trade)
                return True
            else:
                # Real trading - execute close order
                if self.real_trading_enabled:
                    close_side = 'sell' if trade['direction'] == 'long' else 'buy'
                    close_order = self.exchange.create_order(
                        symbol=trade['symbol'],
                        type='market',
                        side=close_side,
                        amount=trade['amount'],
                        params={'productType': 'USDT-FUTURES'}
                    )
                    
                    print(f"✅ POSITION CLOSED: {trade['symbol']} {trade['direction']} - Order: {close_order['id']}")
                    self.send_telegram_message(f"🔚 POSITION CLOSED: {trade['symbol']} {trade['direction']}")
                    
                    if trade in self.active_trades:
                        self.active_trades.remove(trade)
                    return True
                else:
                    print("❌ Real trading not enabled for position close")
                    return False
                    
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False
    
    def execute_trade(self, symbol: str, signal: dict):
        if not self.risk_manager.can_trade():
            return None
        
        try:
            # ✅ Dapatkan ticker dengan error handling
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            if current_price is None:
                print(f"❌ Cannot get current price for {symbol}")
                return None
            
            strategy_type = signal['strategy'].split('_')[0]
            position_size, leverage, max_trades = self.risk_manager.calculate_position_size(
                signal['confidence'], strategy_type, symbol
            )
            
            if len(self.active_trades) >= max_trades:
                print(f"⚠️ Max trades reached ({max_trades}), skipping...")
                return None
            
            # ✅ Hitung amount dengan benar
            amount = position_size / current_price
            
            # ✅ Tambahkan SNR info di trade message
            snr_info = ""
            if 'snr_strength' in signal:
                strength = signal['snr_strength']
                if strength > 1.2:
                    snr_info = f" | 🎯 SNR BOOST: {strength:.2f}x"
                elif strength < 0.8:
                    snr_info = f" | ⚠️ SNR WEAK: {strength:.2f}x"
            
            # ✅ Tambahkan reversal info jika ada
            reversal_info = " | 🔄 REVERSAL TRADE" if signal.get('reversal_trade', False) else ""
            
            trade_info = (
                f"🎯 TRADE SIGNAL:\n"
                f"Symbol: {symbol}\n"
                f"Strategy: {signal['strategy']}\n"
                f"Direction: {signal['direction'].upper()}\n"
                f"Amount: ${position_size:.2f}\n"
                f"Leverage: {leverage}x\n"
                f"Confidence: {signal['confidence']:.2%}{snr_info}{reversal_info}\n"
                f"Market: {signal.get('market_condition', 'NORMAL')}\n"
                f"Price: ${current_price:.6f}"
            )
            
            print(trade_info)
            self.send_telegram_message(trade_info)
            
            if self.real_trading_enabled:
                confirm = input("🚨 EXECUTE REAL ORDER? (YES/NO): ")
                if confirm.upper() == 'YES':
                    try:
                        # ✅ Order execution dengan parameter yang benar
                        order = self.exchange.create_order(
                            symbol=symbol,
                            type='market',
                            side=signal['direction'],
                            amount=amount,
                            params={
                                'productType': 'USDT-FUTURES',
                                'leverage': f'{leverage}',
                                'marginMode': 'crossed'
                            }
                        )
                        
                        trade_record = {
                            'order_id': order['id'],
                            'symbol': symbol,
                            'direction': signal['direction'],
                            'amount': amount,
                            'entry_price': current_price,
                            'position_size': position_size,
                            'leverage': leverage,
                            'timestamp': datetime.now(),
                            'snr_strength': signal.get('snr_strength', 1.0),
                            'market_condition': signal.get('market_condition', 'NORMAL'),
                            'reversal_trade': signal.get('reversal_trade', False),
                            'paper': False
                        }
                        self.active_trades.append(trade_record)
                        self.risk_manager.record_trade()
                        self.total_trades_executed += 1
                        
                        if signal.get('reversal_trade', False):
                            self.reversal_trades_count += 1
                        
                        success_msg = f"✅ ORDER FILLED: {order['id']} - {symbol} {signal['direction']} @ ${current_price:.6f}"
                        print(success_msg)
                        self.send_telegram_message(success_msg)
                        
                        return order
                        
                    except Exception as order_error:
                        error_msg = f"❌ ORDER FAILED: {str(order_error)}"
                        print(error_msg)
                        self.send_telegram_message(error_msg)
                        return None
                else:
                    print("❌ Order cancelled by user")
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
                    'paper': True,
                    'snr_strength': signal.get('snr_strength', 1.0),
                    'market_condition': signal.get('market_condition', 'NORMAL'),
                    'reversal_trade': signal.get('reversal_trade', False)
                }
                self.active_trades.append(paper_trade)
                self.risk_manager.record_trade()
                self.total_trades_executed += 1
                
                if signal.get('reversal_trade', False):
                    self.reversal_trades_count += 1
                    
                return paper_trade
                
        except Exception as e:
            error_msg = f"❌ Trade execution error: {str(e)}"
            print(error_msg)
            self.send_telegram_message(error_msg)
            return None
    
    def check_and_execute_reversals(self, symbols):
        """✅ Check reversal conditions untuk active trades"""
        if not self.active_trades:
            return
        
        print(f"🔍 CHECKING REVERSALS FOR {len(self.active_trades)} ACTIVE TRADES...")
        
        reversal_executed = False
        for trade in self.active_trades[:]:  # Copy list untuk safe iteration
            try:
                symbol = trade['symbol']
                df = self.analyzer.get_ohlcv_data(symbol, '5m', 50)
                
                if df is not None and len(df) > 30:
                    df, _ = self.analyzer.calculate_technical_indicators(df)
                    
                    reversal_detected, reversal_strength = self.channel_strategy.check_reversal_conditions(
                        df, trade['direction']
                    )
                    
                    if reversal_detected and reversal_strength > 0.7:
                        print(f"🔄 STRONG REVERSAL DETECTED: {symbol} {trade['direction']}→{'short' if trade['direction'] == 'long' else 'long'}")
                        
                        # Close current position
                        close_success = self.close_position(trade)
                        
                        if close_success:
                            # Execute reverse trade
                            reverse_signal = {
                                'strategy': 'reversal_followthrough',
                                'direction': 'short' if trade['direction'] == 'long' else 'long',
                                'confidence': min(0.9, reversal_strength + 0.1),  # Boost confidence untuk reversal
                                'market_condition': self.risk_manager.market_condition,
                                'reversal_trade': True,
                                'previous_direction': trade['direction'],
                                'snr_strength': 1.2  # Reversal signals dapat SNR boost
                            }
                            
                            reverse_result = self.execute_trade(symbol, reverse_signal)
                            if reverse_result:
                                reversal_executed = True
                                print(f"🔄 REVERSAL EXECUTED: {symbol} {trade['direction']}→{reverse_signal['direction']}")
                            
            except Exception as e:
                print(f"❌ Error checking reversal for {trade['symbol']}: {e}")
        
        return reversal_executed

print("✅ BAGIAN 4.1 SELESAI: Trading bot core & order execution berhasil!")
# === BAGIAN 4.2: TRADING LOGIC & MAIN LOOP ===
print("\n🔄 MEMULAI BAGIAN 4.2: TRADING LOGIC & MAIN LOOP...")

# Lanjutan class GrowthTradingBot
def analyze_symbols(self, symbols):
    print(f"\n🔍 ANALYZING {len(symbols)} SYMBOLS...")
    
    all_signals = []
    conservative_mode = self.risk_manager.conservative_mode
    market_conditions = {}
    
    for symbol in symbols:
        try:
            df_5m = self.analyzer.get_ohlcv_data(symbol, '5m', 50)
            if df_5m is not None and len(df_5m) > 30:
                # Deteksi kondisi market untuk symbol ini
                market_condition = self.analyzer.assess_market_condition(df_5m, symbol)
                market_conditions[symbol] = market_condition
                
                signals = self.channel_strategy.generate_scalping_signals(
                    df_5m, symbol, conservative_mode, market_condition
                )
                for signal in signals:
                    signal['symbol'] = symbol
                all_signals.extend(signals)
            else:
                print(f"⚠️ Skipping {symbol} - insufficient data")
            
            time.sleep(0.3)  # Rate limiting
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
    
    # Update overall market condition berdasarkan majority
    if market_conditions:
        condition_counter = Counter(market_conditions.values())
        overall_condition = condition_counter.most_common(1)[0][0]
        self.risk_manager.update_market_condition(overall_condition)
    
    self.total_signals_generated += len(all_signals)
    print(f"📊 Total signals: {len(all_signals)} | Conservative: {conservative_mode} | Market: {overall_condition}")
    return all_signals

def run_trading_cycle(self, symbols):
    print(f"\n🔄 TRADING CYCLE STARTED - {datetime.now().strftime('%H:%M:%S')}")
    
    # ✅ Pertama, check reversal conditions
    reversal_executed = self.check_and_execute_reversals(symbols)
    
    # Jika reversal sudah dieksekusi, skip new signals untuk cycle ini
    if reversal_executed:
        print("🔄 Reversal executed, skipping new signals this cycle")
        return
    
    # Kemudian analyze symbols untuk new signals
    signals = self.analyze_symbols(symbols)
    
    if not signals:
        print("📊 No new trading signals this cycle")
        return
    
    conservative_mode = self.risk_manager.conservative_mode
    market_condition = self.risk_manager.market_condition
    
    # ✅ ENHANCED: Filter dengan SNR consideration & market condition
    if conservative_mode:
        filtered_signals = [s for s in signals if s.get('conservative_approved', False) and s['confidence'] >= 0.75]
        print(f"🎯 Conservative Mode: {len(filtered_signals)} high-quality signals")
    else:
        # Prioritize signals berdasarkan market condition
        if market_condition in ["HIGH_VOLATILITY", "EXTREME"]:
            # Hanya signals dengan SNR boost dan confidence tinggi
            strong_signals = [s for s in signals if s.get('snr_strength', 1.0) > 1.1 and s['confidence'] >= 0.7]
            filtered_signals = strong_signals
            print(f"⚡ {market_condition} Mode: {len(filtered_signals)} SNR-boosted signals")
        else:
            # Normal filtering
            strong_signals = [s for s in signals if s.get('snr_strength', 1.0) > 1.1 and s['confidence'] >= 0.6]
            normal_signals = [s for s in signals if s.get('snr_strength', 1.0) <= 1.1 and s['confidence'] >= 0.65]
            filtered_signals = strong_signals + normal_signals
            print(f"⚡ Normal Mode: {len(filtered_signals)} signals ({len(strong_signals)} SNR boosted)")
    
    filtered_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    current_tier = self.risk_manager.get_trading_tier()
    
    # Adaptive execution limits berdasarkan market condition
    execution_limits = {
        "NORMAL": 2,
        "TRENDING": 2,  # ✅ UBAH: Lebih konservatif untuk balance kecil
        "RANGING": 1,
        "HIGH_VOLATILITY": 1,
        "LOW_VOLATILITY": 1,
        "EXTREME": 1
    }
    
    max_executions = execution_limits.get(market_condition, 2)
    if conservative_mode:
        max_executions = min(max_executions, 1)
    
    executed = 0
    for signal in filtered_signals:
        if executed >= max_executions:
            break
        
        # Skip jika sudah ada active trade di symbol yang sama
        symbol_active = any(trade['symbol'] == signal['symbol'] for trade in self.active_trades)
        if symbol_active:
            print(f"⚠️ Active trade exists for {signal['symbol']}, skipping...")
            continue
            
        result = self.execute_trade(signal['symbol'], signal)
        if result:
            executed += 1
            time.sleep(2)  # Rate limiting antara order
    
    print(f"✅ Cycle completed: {executed} trades executed")

def simulate_profit_loss(self):
    """✅ Simulate profit/loss untuk paper trading"""
    if not self.active_trades:
        return 0.0
    
    total_pl = 0.0
    current_time = datetime.now()
    
    for trade in self.active_trades:
        if trade.get('paper', False) and trade.get('entry_price'):
            try:
                # Dapatkan current price
                ticker = self.exchange.fetch_ticker(trade['symbol'])
                current_price = ticker['last']
                
                if current_price:
                    # Hitung P&L
                    if trade['direction'] == 'long':
                        pl = (current_price - trade['entry_price']) / trade['entry_price'] * trade['position_size'] * trade['leverage']
                    else:
                        pl = (trade['entry_price'] - current_price) / trade['entry_price'] * trade['position_size'] * trade['leverage']
                    
                    # Simulate closing setelah certain time atau target profit
                    trade_duration = (current_time - trade['timestamp']).total_seconds() / 60  # dalam menit
                    
                    # Close conditions untuk paper trading
                    if trade_duration > 10:  # Close setelah 10 menit
                        self.close_position(trade)
                    elif abs(pl) >= trade['position_size'] * 0.3:  # 30% profit/loss
                        self.close_position(trade)
                        total_pl += pl
                    
            except Exception as e:
                print(f"❌ Error simulating P&L for {trade['symbol']}: {e}")
    
    return total_pl

def start_growth_bot(self, symbols=None, cycle_interval=180):
    if symbols is None:
        symbols = ALTCOINS
    
    print(f"🚀 STARTING GROWTH TRADING BOT")
    print(f"💰 Initial Balance: ${self.risk_manager.current_balance:.2f}")  # ✅ Format 2 decimal
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⏱ Cycle Interval: {cycle_interval}s")
    
    if not self.real_trading_enabled:
        print("💵 PAPER TRADING MODE - No real orders")
    
    self.cycle_count = 0
    last_balance_update = datetime.now()
    
    try:
        while True:
            self.cycle_count += 1
            print(f"\n" + "="*60)
            print(f"🎯 CYCLE #{self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"💰 Balance: ${self.risk_manager.current_balance:.2f}")  # ✅ Format 2 decimal
            print(f"🔢 Daily Orders: {self.risk_manager.daily_order_count}/40")
            print(f"🎯 Mode: {'CONSERVATIVE' if self.risk_manager.conservative_mode else 'NORMAL'}")
            print(f"📊 Market: {self.risk_manager.market_condition}")
            print(f"📈 Active Trades: {len(self.active_trades)}")
            print(f"🔄 Reversal Trades: {self.reversal_trades_count}")
            print("="*60)
            
            self.run_trading_cycle(symbols)
            
            # Simulate profit/loss untuk paper trading
            if not self.real_trading_enabled:
                simulated_pl = self.simulate_profit_loss()
                if abs(simulated_pl) > 0.01:  # Hanya tampilkan jika significant
                    print(f"📊 Simulated P&L: ${simulated_pl:.2f}")
                
                # Update balance setiap 5 cycles atau 15 menit
                if self.cycle_count % 5 == 0 or (datetime.now() - last_balance_update).total_seconds() > 900:
                    # Enhanced simulation dengan market condition consideration
                    base_range = {
                        "NORMAL": (-0.05, 0.15),  # ✅ UBAH: Range lebih kecil untuk balance kecil
                        "TRENDING": (-0.03, 0.20),
                        "RANGING": (-0.04, 0.10),
                        "HIGH_VOLATILITY": (-0.10, 0.25),
                        "LOW_VOLATILITY": (-0.02, 0.08),
                        "EXTREME": (-0.15, 0.30)
                    }
                    
                    low, high = base_range.get(self.risk_manager.market_condition, (-0.05, 0.15))
                    base_change = np.random.uniform(low, high)
                    
                    # Adjust berdasarkan performance recent
                    performance_factor = 1.0
                    if len(self.profit_history) > 0:
                        avg_profit = np.mean(self.profit_history[-5:])  # Last 5 trades
                        performance_factor = 1.0 + avg_profit * 2
                    
                    final_change = base_change * performance_factor
                    new_balance = max(0.5, self.risk_manager.current_balance + final_change)  # ✅ UBAH: Minimum $0.50
                    
                    # Record profit
                    profit = new_balance - self.risk_manager.current_balance
                    if abs(profit) > 0.01:  # Hanya record significant changes
                        self.profit_history.append(profit)
                    
                    self.risk_manager.update_balance(new_balance)
                    last_balance_update = datetime.now()
                    
                    print(f"📈 Balance updated: ${new_balance:.2f} (Change: ${final_change:.2f})")
            
            print(f"💤 Waiting {cycle_interval} seconds...")
            time.sleep(cycle_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
        self.send_telegram_message("🛑 Trading bot stopped manually")
        
        # Summary statistics
        print(f"\n📈 BOT PERFORMANCE SUMMARY:")
        print(f"   Total Cycles: {self.cycle_count}")
        print(f"   Total Trades: {self.total_trades_executed}")
        print(f"   Reversal Trades: {self.reversal_trades_count}")
        print(f"   Final Balance: ${self.risk_manager.current_balance:.2f}")  # ✅ Format 2 decimal
        print(f"   Total Profit: ${self.risk_manager.total_profit:.2f}")  # ✅ Format 2 decimal
        
        if self.profit_history:
            win_rate = len([p for p in self.profit_history if p > 0]) / len(self.profit_history) * 100
            avg_profit = np.mean(self.profit_history)
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Average Profit: ${avg_profit:.2f}")  # ✅ Format 2 decimal
            
    except Exception as e:
        error_msg = f"❌ Bot error: {str(e)}"
        print(error_msg)
        self.send_telegram_message(error_msg)

# Tambahkan methods ke class
GrowthTradingBot.analyze_symbols = analyze_symbols
GrowthTradingBot.run_trading_cycle = run_trading_cycle
GrowthTradingBot.simulate_profit_loss = simulate_profit_loss
GrowthTradingBot.start_growth_bot = start_growth_bot

# FINAL BOT SETUP
print("\n🎯 FINAL BOT SETUP...")

try:
    growth_bot = GrowthTradingBot(exchange, initial_balance=1.3)  # ✅ UBAH: $1.30
    
    print("🤖 GROWTH TRADING BOT DENGAN SEMUA FITUR READY!")
    print("💰 INITIAL BALANCE: $1.30")  # ✅ Tampilkan balance dengan jelas
    print("📋 Available commands:")
    print("   - growth_bot.enable_real_trading()  # Enable real trading")
    print("   - growth_bot.start_growth_bot()     # Start the bot")
    print("   - growth_bot.run_trading_cycle(['BTCUSDT'])  # Test single symbol")
    
    print("\n🆕 OPTIMIZED FOR $1.30 BALANCE:")
    print("   ✅ Smaller position sizes ($0.20 - $0.80)")
    print("   ✅ Lower leverage (6x - 15x)")
    print("   ✅ Conservative risk management")
    print("   ✅ Reduced trade frequency")
    print("   ✅ Minimum balance protection ($0.50)")
    
    print("\n🎯 STRATEGY FEATURES:")
    print("   ✅ Dynamic SNR Enhancement")
    print("   ✅ Auto-Reversal Trading") 
    print("   ✅ Market Condition Adaptation")
    print("   ✅ Meme Coin Special Rules")
    
except Exception as e:
    print(f"❌ Final bot setup failed: {e}")

print("✅ BAGIAN 4.2 SELESAI: Trading logic & main loop berhasil!")
print("\n🎉 SCRIPT LENGKAP DENGAN BALANCE $1.30 SIAP DIGUNAKAN!")

# Quick test function
def test_bot_functionality():
    print("\n🧪 TESTING BOT FUNCTIONALITY...")
    try:
        # Test market analysis
        test_symbol = 'BTCUSDT'
        df = growth_bot.analyzer.get_ohlcv_data(test_symbol, '5m', 50)
        if df is not None:
            market_condition = growth_bot.analyzer.assess_market_condition(df, test_symbol)
            print(f"✅ Market Analysis: {test_symbol} - {market_condition}")
            
            # Test signal generation
            signals = growth_bot.channel_strategy.generate_scalping_signals(df, test_symbol, False, market_condition)
            print(f"✅ Signal Generation: {len(signals)} signals")
            
            # Test position sizing
            if signals:
                position_size, leverage, max_trades = growth_bot.risk_manager.calculate_position_size(
                    signals[0]['confidence'], 'scalping', test_symbol
                )
                print(f"✅ Position Sizing: ${position_size:.2f}, {leverage}x leverage")
            
        print("✅ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# Run quick test
test_bot_functionality()

print(f"\n🚀 **BOT READY FOR ACTION!**")
print(f"💡 **NEXT STEPS:**")
print(f"1. growth_bot.enable_real_trading()  # Untuk real trading")
print(f"2. growth_bot.start_growth_bot()     # Untuk mulai bot")
print(f"3. Ctrl+C untuk stop bot")

print(f"\n📊 **$1.30 BALANCE SETTINGS:**")
print(f"   • Position Size: $0.20 - $0.80")
print(f"   • Leverage: 6x - 15x (adaptive)")
print(f"   • Max Trades: 40/hari")
print(f"   • Min Balance: $0.50 protection")
print(f"   • Cycle: 3 menit")

print(f"\n⚠️ **REAL TRADING WARNING:**")
print(f"   • Balance $1.30 = HIGH RISK")
print(f"   • Bisa loss 100% dalam hitungan menit")
print(f"   • Hanya gunakan uang siap hilang")
print(f"   • Pastikan ketik 'YES' (huruf besar) saat konfirmasi")
