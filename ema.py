import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import threading
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class AdvancedForexBot:
    def __init__(self, account_size, risk_per_trade=0.01):
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.positions = {}
        
        # Initialize strategies dictionary after all methods are defined
        self.strategies = {}
        
        self.timeframes = {
            'primary': mt5.TIMEFRAME_H1,      # Main decision timeframe
            'confirmation': mt5.TIMEFRAME_M15 # Confirmation timeframe
        }
        
        # Strategy parameters for all pairs
        self.params = {
            'USDJPYm': {
                'atr_period': 14,
                'rsi_period': 14,
                'ema_fast': 8,
                'ema_slow': 21,
                'bollinger_period': 20,
                'stoch_period': 14
            },
            'EURUSDm': {
                'atr_period': 14,
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'williams_period': 14
            },
            'GBPUSDm': {
                'atr_period': 14,
                'rsi_period': 14,
                'ema_fast': 5,
                'ema_medium': 13,
                'ema_slow': 21,
                'stoch_period': 14
            },
            'USDCHFm': {
                'atr_period': 14,
                'rsi_period': 14,
                'macd_fast': 8,
                'macd_slow': 21,
                'macd_signal': 5,
                'bollinger_period': 20
            },
            'AUDUSDm': {
                'atr_period': 14,
                'rsi_period': 14,
                'sma_fast': 10,
                'sma_slow': 30,
                'parabolic_sar': 0.02,
                'stoch_period': 14
            },
            'USDCADm': {
                'atr_period': 14,
                'rsi_period': 14,
                'ema_fast': 9,
                'ema_slow': 18,
                'bollinger_period': 20,
                'williams_period': 14
            },
            'NZDUSDm': {
                'atr_period': 14,
                'rsi_period': 14,
                'sma_fast': 7,
                'sma_slow': 25,
                'macd_fast': 6,
                'macd_slow': 13,
                'macd_signal': 5
            },
            'BTCUSDm': {
                'atr_period': 14,
                'rsi_period': 14,
                'ema_short': 9,
                'ema_medium': 21,
                'ema_long': 50,
                'volume_sma': 20,
                'volatility_threshold': 2.0,
                'momentum_period': 10
            },
            'ETHUSDm': {
                'atr_period': 14,
                'rsi_period': 14,
                'sma_fast': 12,
                'sma_slow': 26,
                'volume_ema': 20,
                'stoch_period': 14,
                'volatility_threshold': 2.5,
                'trend_strength_period': 25
            }
        }
        
        self.trading_hours = {
            'Asian': (22, 6),      # 10 PM - 6 AM GMT
            'European': (7, 16),    # 7 AM - 4 PM GMT
            'US': (13, 22),         # 1 PM - 10 PM GMT
            'Crypto': (0, 24)       # Crypto trades 24/7
        }
        
        # Pair-specific trading sessions
        self.pair_sessions = {
            'USDJPYm': ['Asian', 'US'],
            'EURUSDm': ['European', 'US'],
            'GBPUSDm': ['European', 'US'],
            'USDCHFm': ['European', 'US'],
            'AUDUSDm': ['Asian', 'European'],
            'USDCADm': ['US', 'European'],
            'NZDUSDm': ['Asian', 'European'],
            'BTCUSDm': ['Crypto'],  # Crypto trades 24/7
            'ETHUSDm': ['Crypto']   # Crypto trades 24/7
        }
        
        # Crypto-specific risk management
        self.crypto_risk_params = {
            'BTCUSDm': {
                'max_position_size': 0.5,  # Maximum lot size for BTC
                'volatility_multiplier': 2.5,
                'min_sl_pips': 150,       # Larger SL for crypto volatility
                'max_sl_pips': 500,
                'risk_multiplier': 0.8    # Lower risk for crypto
            },
            'ETHUSDm': {
                'max_position_size': 1.0,  # Maximum lot size for ETH
                'volatility_multiplier': 3.0,
                'min_sl_pips': 100,
                'max_sl_pips': 400,
                'risk_multiplier': 0.7    # Even lower risk for ETH
            }
        }
        
        self.setup_mt5()
        # Initialize strategies after all methods are defined
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize strategies dictionary after all methods are defined"""
        self.strategies = {
            'USDJPYm': self.usdjpy_strategy,
            'EURUSDm': self.eurusd_strategy,
            'GBPUSDm': self.gbpusd_strategy,
            'USDCHFm': self.usdchf_strategy,
            'AUDUSDm': self.audusd_strategy,
            'USDCADm': self.usdcad_strategy,
            'NZDUSDm': self.nzdusd_strategy,
            'BTCUSDm': self.btcusd_strategy,
            'ETHUSDm': self.ethusd_strategy
        }

    def setup_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print("MT5 initialization failed")
            return False
        
        # Verify connection
        if not mt5.terminal_info():
            print("MT5 terminal not connected")
            return False
            
        print("MT5 initialized successfully")
        
        # Enable symbol trading and subscribe to symbols
        symbols = ['USDJPYm', 'EURUSDm', 'GBPUSDm', 'USDCHFm', 'AUDUSDm', 'USDCADm', 'NZDUSDm', 'BTCUSDm', 'ETHUSDm']
        for symbol in symbols:
            # Select the symbol
            selected = mt5.symbol_select(symbol, True)
            if selected:
                print(f"Symbol {symbol} enabled successfully")
            else:
                print(f"Failed to enable {symbol}, error: {mt5.last_error()}")
            
        return True

    def calculate_position_size(self, symbol, stop_loss_pips):
        """Calculate position size based on risk management"""
        # Special handling for cryptocurrencies
        if symbol in ['BTCUSDm', 'ETHUSDm']:
            return self.calculate_crypto_position_size(symbol, stop_loss_pips)
            
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"Symbol info not available for {symbol}")
            return 0.01  # Default minimum lot size
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"Tick data not available for {symbol}")
            return 0.01
            
        # Calculate pip value
        pip_size = 0.01 if 'JPY' in symbol else 0.0001
        pip_value = (0.0001 / tick.ask) * 100000 if 'JPY' not in symbol else (0.01 / tick.ask) * 100000
        
        # Risk calculation
        risk_amount = self.account_size * self.risk_per_trade
        stop_loss_dollars = stop_loss_pips * pip_value
        
        if stop_loss_dollars <= 0:
            return 0.01
            
        lot_size = risk_amount / stop_loss_dollars
        
        # Ensure lot size is within broker limits
        lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))
        lot_size = round(lot_size, 2)
        
        print(f"Position size calculated: {lot_size} lots for {symbol}, SL: {stop_loss_pips} pips")
        return lot_size

    def calculate_crypto_position_size(self, symbol, stop_loss_pips):
        """Calculate position size for cryptocurrencies with special risk management"""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"Symbol info not available for {symbol}")
            return 0.01
            
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"Tick data not available for {symbol}")
            return 0.01
            
        # Crypto-specific risk parameters
        crypto_params = self.crypto_risk_params[symbol]
        
        # Adjusted risk for crypto (lower due to higher volatility)
        adjusted_risk = self.risk_per_trade * crypto_params['risk_multiplier']
        risk_amount = self.account_size * adjusted_risk
        
        # Calculate pip value for crypto (different calculation)
        # For BTCUSDm and ETHUSDm, 1 pip = 1.0 (price move of $1)
        pip_value = 1.0  # For crypto, 1 pip = $1 price movement
        
        stop_loss_dollars = stop_loss_pips * pip_value
        
        if stop_loss_dollars <= 0:
            return 0.01
            
        # For crypto, we need to adjust lot size calculation
        # Standard lot in crypto might be different
        lot_size = risk_amount / stop_loss_dollars
        
        # Apply crypto-specific limits
        max_lot_size = crypto_params['max_position_size']
        lot_size = max(symbol_info.volume_min, min(max_lot_size, lot_size))
        lot_size = round(lot_size, 3)  # More precision for crypto
        
        print(f"Crypto position size: {lot_size} lots for {symbol}, SL: {stop_loss_pips} pips, Adjusted Risk: {adjusted_risk*100:.1f}%")
        return lot_size

    def get_technical_indicators(self, symbol, timeframe, periods=100):
        """Calculate technical indicators for the given symbol"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, periods)
            if rates is None or len(rates) == 0:
                print(f"No data received for {symbol} on timeframe {timeframe}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Common indicators for all pairs
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['atr'] = self.calculate_atr(df, 14)
            
            # Volume-based indicators (especially important for crypto)
            df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            
            # Pair-specific indicators
            if symbol == 'USDJPYm':
                df['rsi'] = self.calculate_rsi(df['close'], 14)
                df['ema_fast'] = df['close'].ewm(span=8).mean()
                df['ema_slow'] = df['close'].ewm(span=21).mean()
                df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df, 14)
                
            elif symbol == 'EURUSDm':
                df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
                df['williams_r'] = self.calculate_williams_r(df, 14)
                df['bollinger_upper'], df['bollinger_lower'] = self.calculate_bollinger_bands(df['close'], 20)
            
            elif symbol == 'GBPUSDm':
                df['rsi'] = self.calculate_rsi(df['close'], 14)
                df['ema_fast'] = df['close'].ewm(span=5).mean()
                df['ema_medium'] = df['close'].ewm(span=13).mean()
                df['ema_slow'] = df['close'].ewm(span=21).mean()
                df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df, 14)
                
            elif symbol == 'USDCHFm':
                df['rsi'] = self.calculate_rsi(df['close'], 14)
                df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd_custom(df['close'], 8, 21, 5)
                df['bollinger_upper'], df['bollinger_lower'] = self.calculate_bollinger_bands(df['close'], 20)
                
            elif symbol == 'AUDUSDm':
                df['rsi'] = self.calculate_rsi(df['close'], 14)
                df['sma_fast'] = df['close'].rolling(window=10).mean()
                df['sma_slow'] = df['close'].rolling(window=30).mean()
                df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df, 14)
                df['sar'] = self.calculate_parabolic_sar(df)
                
            elif symbol == 'USDCADm':
                df['rsi'] = self.calculate_rsi(df['close'], 14)
                df['ema_fast'] = df['close'].ewm(span=9).mean()
                df['ema_slow'] = df['close'].ewm(span=18).mean()
                df['bollinger_upper'], df['bollinger_lower'] = self.calculate_bollinger_bands(df['close'], 20)
                df['williams_r'] = self.calculate_williams_r(df, 14)
                
            elif symbol == 'NZDUSDm':
                df['rsi'] = self.calculate_rsi(df['close'], 14)
                df['sma_fast'] = df['close'].rolling(window=7).mean()
                df['sma_slow'] = df['close'].rolling(window=25).mean()
                df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd_custom(df['close'], 6, 13, 5)
            
            # Crypto-specific indicators
            elif symbol == 'BTCUSDm':
                df['rsi'] = self.calculate_rsi(df['close'], 14)
                df['ema_short'] = df['close'].ewm(span=9).mean()
                df['ema_medium'] = df['close'].ewm(span=21).mean()
                df['ema_long'] = df['close'].ewm(span=50).mean()
                df['momentum'] = df['close'] - df['close'].shift(10)
                df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
                df['volume_ema'] = df['tick_volume'].ewm(span=20).mean()
                
            elif symbol == 'ETHUSDm':
                df['rsi'] = self.calculate_rsi(df['close'], 14)
                df['sma_fast'] = df['close'].rolling(window=12).mean()
                df['sma_slow'] = df['close'].rolling(window=26).mean()
                df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df, 14)
                df['volume_ema'] = df['tick_volume'].ewm(span=20).mean()
                df['trend_strength'] = self.calculate_trend_strength(df, 25)
                df['volatility_ratio'] = self.calculate_volatility_ratio(df, 14)
            
            return df
            
        except Exception as e:
            print(f"Error getting technical indicators for {symbol}: {e}")
            return None

    def calculate_trend_strength(self, df, period):
        """Calculate trend strength using linear regression slope"""
        if len(df) < period:
            return np.nan
            
        x = np.arange(period)
        y = df['close'].tail(period).values
        
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by price to get percentage slope
        trend_strength = (slope / df['close'].iloc[-1]) * 100 * period
        
        return trend_strength

    def calculate_volatility_ratio(self, df, period):
        """Calculate volatility ratio (current volatility vs average)"""
        if len(df) < period:
            return np.nan
            
        current_volatility = df['high'].tail(period) - df['low'].tail(period)
        avg_volatility = current_volatility.rolling(window=period).mean()
        
        volatility_ratio = current_volatility.iloc[-1] / avg_volatility.iloc[-1] if avg_volatility.iloc[-1] != 0 else 1
        
        return volatility_ratio

    # Existing indicator methods
    def calculate_atr(self, df, period):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        return true_range.rolling(window=period).mean()

    def calculate_rsi(self, prices, period):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices):
        """Calculate MACD with standard parameters"""
        return self.calculate_macd_custom(prices, 12, 26, 9)

    def calculate_macd_custom(self, prices, fast, slow, signal):
        """Calculate MACD with custom parameters"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def calculate_stochastic(self, df, period):
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=3).mean()
        return stoch_k, stoch_d

    def calculate_williams_r(self, df, period):
        """Calculate Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
        return williams_r

    def calculate_bollinger_bands(self, prices, period):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band

    def calculate_parabolic_sar(self, df, acceleration=0.02, maximum=0.2):
        """Calculate Parabolic SAR"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        sar = np.zeros(len(high))
        ep = np.zeros(len(high))
        af = np.zeros(len(high))
        
        # Initial values
        sar[0] = high[0] if close[0] < close[1] else low[0]
        ep[0] = high[0] if close[0] > close[1] else low[0]
        af[0] = acceleration
        
        for i in range(1, len(high)):
            # Update SAR
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Check for SAR reversal
            if close[i] > close[i-1]:
                # Uptrend
                if sar[i] > low[i]:
                    sar[i] = min(low[i-1], low[i])
                    af[i] = acceleration
                    ep[i] = high[i]
                else:
                    af[i] = min(af[i-1] + acceleration, maximum)
                    ep[i] = max(ep[i-1], high[i])
            else:
                # Downtrend
                if sar[i] < high[i]:
                    sar[i] = max(high[i-1], high[i])
                    af[i] = acceleration
                    ep[i] = low[i]
                else:
                    af[i] = min(af[i-1] + acceleration, maximum)
                    ep[i] = min(ep[i-1], low[i])
        
        return sar

    # Strategy Definitions
    def usdjpy_strategy(self, df):
        """USDJPY Trend Following Strategy"""
        if df is None or len(df) < 2:
            return 0, 30, 45
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if any(pd.isna([current['ema_fast'], current['ema_slow'], current['rsi'], 
                       current['stoch_k'], current['stoch_d']])):
            return 0, 30, 45
        
        ema_trend = current['ema_fast'] > current['ema_slow']
        ema_prev_trend = prev['ema_fast'] > prev['ema_slow']
        
        buy_signal = (
            ema_trend and not ema_prev_trend and
            current['rsi'] < 70 and
            current['stoch_k'] > current['stoch_d'] and
            current['stoch_k'] < 80
        )
        
        sell_signal = (
            not ema_trend and ema_prev_trend and
            current['rsi'] > 30 and
            current['stoch_k'] < current['stoch_d'] and
            current['stoch_k'] > 20
        )
        
        stop_loss_pips = current['atr'] * 2 * 100
        take_profit_pips = stop_loss_pips * 1.5
        
        stop_loss_pips = max(15, min(100, stop_loss_pips))
        take_profit_pips = max(20, min(150, take_profit_pips))
        
        signal = 0
        if buy_signal:
            signal = 1
        elif sell_signal:
            signal = -1
            
        return signal, stop_loss_pips, take_profit_pips

    def eurusd_strategy(self, df):
        """EURUSD Mean Reversion Strategy"""
        if df is None or len(df) < 1:
            return 0, 20, 30
            
        current = df.iloc[-1]
        
        if any(pd.isna([current['bollinger_upper'], current['bollinger_lower'], 
                       current['macd_hist'], current['williams_r']])):
            return 0, 20, 30
        
        price_ratio = (current['close'] - current['bollinger_lower']) / (current['bollinger_upper'] - current['bollinger_lower'])
        
        buy_signal = (
            current['close'] <= current['bollinger_lower'] and
            current['macd_hist'] > 0 and
            current['williams_r'] < -80 and
            price_ratio < 0.2
        )
        
        sell_signal = (
            current['close'] >= current['bollinger_upper'] and
            current['macd_hist'] < 0 and
            current['williams_r'] > -20 and
            price_ratio > 0.8
        )
        
        stop_loss_pips = (current['bollinger_upper'] - current['bollinger_lower']) * 10000
        take_profit_pips = stop_loss_pips * 1.2
        
        stop_loss_pips = max(10, min(80, stop_loss_pips))
        take_profit_pips = max(15, min(120, take_profit_pips))
        
        signal = 0
        if buy_signal:
            signal = 1
        elif sell_signal:
            signal = -1
            
        return signal, stop_loss_pips, take_profit_pips

    def gbpusd_strategy(self, df):
        """GBPUSD Momentum Breakout Strategy"""
        if df is None or len(df) < 3:
            return 0, 25, 40
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if any(pd.isna([current['ema_fast'], current['ema_medium'], current['ema_slow'],
                       current['rsi'], current['stoch_k'], current['stoch_d']])):
            return 0, 25, 40
        
        # Triple EMA alignment for strong momentum
        ema_alignment = current['ema_fast'] > current['ema_medium'] > current['ema_slow']
        prev_ema_alignment = prev['ema_fast'] > prev['ema_medium'] > prev['ema_slow']
        
        buy_signal = (
            ema_alignment and not prev_ema_alignment and
            current['rsi'] > 50 and
            current['stoch_k'] > 50 and
            current['stoch_k'] > current['stoch_d']
        )
        
        sell_signal = (
            not ema_alignment and prev_ema_alignment and
            current['rsi'] < 50 and
            current['stoch_k'] < 50 and
            current['stoch_k'] < current['stoch_d']
        )
        
        stop_loss_pips = current['atr'] * 1.8 * 100
        take_profit_pips = stop_loss_pips * 1.6
        
        stop_loss_pips = max(20, min(120, stop_loss_pips))
        take_profit_pips = max(25, min(180, take_profit_pips))
        
        signal = 0
        if buy_signal:
            signal = 1
        elif sell_signal:
            signal = -1
            
        return signal, stop_loss_pips, take_profit_pips

    def usdchf_strategy(self, df):
        """USDCHF Safe Haven Strategy"""
        if df is None or len(df) < 2:
            return 0, 18, 30
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if any(pd.isna([current['rsi'], current['macd_hist'], current['bollinger_upper'], current['bollinger_lower']])):
            return 0, 18, 30
        
        # CHF often moves inversely to risk sentiment
        buy_signal = (
            current['close'] < current['bollinger_lower'] and
            current['rsi'] < 35 and
            current['macd_hist'] > prev['macd_hist'] and  # MACD improving
            current['macd_hist'] > 0
        )
        
        sell_signal = (
            current['close'] > current['bollinger_upper'] and
            current['rsi'] > 65 and
            current['macd_hist'] < prev['macd_hist'] and  # MACD deteriorating
            current['macd_hist'] < 0
        )
        
        stop_loss_pips = current['atr'] * 1.5 * 100
        take_profit_pips = stop_loss_pips * 1.8
        
        stop_loss_pips = max(12, min(60, stop_loss_pips))
        take_profit_pips = max(18, min(100, take_profit_pips))
        
        signal = 0
        if buy_signal:
            signal = 1
        elif sell_signal:
            signal = -1
            
        return signal, stop_loss_pips, take_profit_pips

    def audusd_strategy(self, df):
        """AUDUSD Commodity Correlation Strategy"""
        if df is None or len(df) < 2:
            return 0, 22, 35
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if any(pd.isna([current['sma_fast'], current['sma_slow'], current['rsi'], 
                       current['stoch_k'], current['stoch_d'], current['sar']])):
            return 0, 22, 35
        
        # SMA crossover with SAR confirmation
        sma_crossover = current['sma_fast'] > current['sma_slow'] and prev['sma_fast'] <= prev['sma_slow']
        sma_crossunder = current['sma_fast'] < current['sma_slow'] and prev['sma_fast'] >= prev['sma_slow']
        
        buy_signal = (
            sma_crossover and
            current['close'] > current['sar'] and  # Price above SAR (uptrend)
            current['rsi'] > 45 and
            current['stoch_k'] > current['stoch_d']
        )
        
        sell_signal = (
            sma_crossunder and
            current['close'] < current['sar'] and  # Price below SAR (downtrend)
            current['rsi'] < 55 and
            current['stoch_k'] < current['stoch_d']
        )
        
        stop_loss_pips = current['atr'] * 2.2 * 100
        take_profit_pips = stop_loss_pips * 1.4
        
        stop_loss_pips = max(15, min(90, stop_loss_pips))
        take_profit_pips = max(20, min(130, take_profit_pips))
        
        signal = 0
        if buy_signal:
            signal = 1
        elif sell_signal:
            signal = -1
            
        return signal, stop_loss_pips, take_profit_pips

    def usdcad_strategy(self, df):
        """USDCAD Oil Correlation Strategy"""
        if df is None or len(df) < 2:
            return 0, 25, 40
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if any(pd.isna([current['ema_fast'], current['ema_slow'], current['rsi'],
                       current['williams_r'], current['bollinger_upper'], current['bollinger_lower']])):
            return 0, 25, 40
        
        # EMA crossover with multiple confirmations
        ema_crossover = current['ema_fast'] > current['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']
        ema_crossunder = current['ema_fast'] < current['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']
        
        buy_signal = (
            ema_crossover and
            current['rsi'] < 70 and
            current['williams_r'] < -20 and  # Not overbought
            current['close'] > current['bollinger_lower']
        )
        
        sell_signal = (
            ema_crossunder and
            current['rsi'] > 30 and
            current['williams_r'] > -80 and  # Not oversold
            current['close'] < current['bollinger_upper']
        )
        
        stop_loss_pips = current['atr'] * 2 * 100
        take_profit_pips = stop_loss_pips * 1.5
        
        stop_loss_pips = max(18, min(100, stop_loss_pips))
        take_profit_pips = max(22, min(150, take_profit_pips))
        
        signal = 0
        if buy_signal:
            signal = 1
        elif sell_signal:
            signal = -1
            
        return signal, stop_loss_pips, take_profit_pips

    def nzdusd_strategy(self, df):
        """NZDUSD Carry Trade Strategy"""
        if df is None or len(df) < 2:
            return 0, 20, 35
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if any(pd.isna([current['sma_fast'], current['sma_slow'], current['rsi'],
                       current['macd_hist']])):
            return 0, 20, 35
        
        # SMA crossover with MACD confirmation
        sma_crossover = current['sma_fast'] > current['sma_slow'] and prev['sma_fast'] <= prev['sma_slow']
        sma_crossunder = current['sma_fast'] < current['sma_slow'] and prev['sma_fast'] >= prev['sma_slow']
        
        buy_signal = (
            sma_crossover and
            current['macd_hist'] > 0 and
            current['macd_hist'] > prev['macd_hist'] and  # MACD strengthening
            current['rsi'] < 65
        )
        
        sell_signal = (
            sma_crossunder and
            current['macd_hist'] < 0 and
            current['macd_hist'] < prev['macd_hist'] and  # MACD weakening
            current['rsi'] > 35
        )
        
        stop_loss_pips = current['atr'] * 1.8 * 100
        take_profit_pips = stop_loss_pips * 1.7
        
        stop_loss_pips = max(15, min(80, stop_loss_pips))
        take_profit_pips = max(20, min(120, take_profit_pips))
        
        signal = 0
        if buy_signal:
            signal = 1
        elif sell_signal:
            signal = -1
            
        return signal, stop_loss_pips, take_profit_pips

    def btcusd_strategy(self, df):
        """BTCUSD Momentum & Trend Strategy with Volume Confirmation"""
        if df is None or len(df) < 3:
            return 0, 200, 350
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if any(pd.isna([current['ema_short'], current['ema_medium'], current['ema_long'],
                       current['rsi'], current['volume_ratio'], current['volatility']])):
            return 0, 200, 350
        
        # Multi-timeframe EMA alignment for strong trend
        ema_alignment = current['ema_short'] > current['ema_medium'] > current['ema_long']
        prev_ema_alignment = prev['ema_short'] > prev['ema_medium'] > prev['ema_long']
        
        # Volume confirmation
        volume_confirm = current['volume_ratio'] > 1.2
        
        # Momentum confirmation
        momentum_positive = current['momentum'] > 0
        
        # Volatility filter (avoid extremely high volatility periods)
        volatility_ok = current['volatility'] < 0.05  # 5% volatility threshold
        
        buy_signal = (
            ema_alignment and not prev_ema_alignment and
            current['rsi'] > 45 and current['rsi'] < 75 and
            volume_confirm and
            momentum_positive and
            volatility_ok
        )
        
        sell_signal = (
            not ema_alignment and prev_ema_alignment and
            current['rsi'] < 55 and current['rsi'] > 25 and
            volume_confirm and
            not momentum_positive and
            volatility_ok
        )
        
        # Crypto-specific stop loss and take profit
        crypto_params = self.crypto_risk_params['BTCUSDm']
        base_sl = current['atr'] * crypto_params['volatility_multiplier'] * 100
        stop_loss_pips = max(crypto_params['min_sl_pips'], 
                           min(crypto_params['max_sl_pips'], base_sl))
        
        take_profit_pips = stop_loss_pips * 1.8  # Higher reward ratio for crypto
        
        signal = 0
        if buy_signal:
            signal = 1
        elif sell_signal:
            signal = -1
            
        return signal, stop_loss_pips, take_profit_pips

    def ethusd_strategy(self, df):
        """ETHUSD Trend Reversal with Stochastic Confirmation"""
        if df is None or len(df) < 3:
            return 0, 150, 280
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if any(pd.isna([current['sma_fast'], current['sma_slow'], current['rsi'],
                       current['stoch_k'], current['stoch_d'], current['volume_ema'],
                       current['trend_strength'], current['volatility_ratio']])):
            return 0, 150, 280
        
        # SMA crossover
        sma_crossover = current['sma_fast'] > current['sma_slow'] and prev['sma_fast'] <= prev['sma_slow']
        sma_crossunder = current['sma_fast'] < current['sma_slow'] and prev['sma_fast'] >= prev['sma_slow']
        
        # Stochastic conditions
        stoch_bullish = current['stoch_k'] > current['stoch_d'] and current['stoch_k'] < 80
        stoch_bearish = current['stoch_k'] < current['stoch_d'] and current['stoch_k'] > 20
        
        # Volume and trend strength
        volume_ok = current['tick_volume'] > current['volume_ema']
        trend_strength_ok = abs(current['trend_strength']) > 0.5  # Minimum trend strength
        
        # Volatility filter
        volatility_ok = current['volatility_ratio'] < 2.0
        
        buy_signal = (
            sma_crossover and
            stoch_bullish and
            current['rsi'] > 40 and current['rsi'] < 70 and
            volume_ok and
            trend_strength_ok and
            volatility_ok
        )
        
        sell_signal = (
            sma_crossunder and
            stoch_bearish and
            current['rsi'] < 60 and current['rsi'] > 30 and
            volume_ok and
            trend_strength_ok and
            volatility_ok
        )
        
        # ETH-specific risk management
        crypto_params = self.crypto_risk_params['ETHUSDm']
        base_sl = current['atr'] * crypto_params['volatility_multiplier'] * 100
        stop_loss_pips = max(crypto_params['min_sl_pips'], 
                           min(crypto_params['max_sl_pips'], base_sl))
        
        take_profit_pips = stop_loss_pips * 2.0  # Even higher reward ratio for ETH
        
        signal = 0
        if buy_signal:
            signal = 1
        elif sell_signal:
            signal = -1
            
        return signal, stop_loss_pips, take_profit_pips

    def is_optimal_trading_time(self, symbol):
        """Check if current time is optimal for trading specific pair"""
        # Crypto trades 24/7
        if symbol in ['BTCUSDm', 'ETHUSDm']:
            return True
            
        current_hour = datetime.now().hour
        current_weekday = datetime.now().weekday()
        
        # Avoid weekends for forex
        if current_weekday >= 5:
            return False
            
        # Check pair-specific trading sessions
        if symbol in self.pair_sessions:
            for session in self.pair_sessions[symbol]:
                start, end = self.trading_hours[session]
                if start <= current_hour < end:
                    return True
                    
        return False

    def execute_trade(self, symbol, signal, stop_loss_pips, take_profit_pips):
        """Execute trade with proper risk management"""
        if not self.is_optimal_trading_time(symbol):
            print(f"Not optimal trading time for {symbol}")
            return False
            
        # Check if position already exists
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            print(f"Position already exists for {symbol}")
            return False
            
        # Calculate position size
        lot_size = self.calculate_position_size(symbol, stop_loss_pips)
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"No tick data for {symbol}")
            return False
            
        # Calculate stop loss and take profit
        point = mt5.symbol_info(symbol).point
        deviation = 20
        
        if signal == 1:  # Buy
            price = tick.ask
            sl = price - stop_loss_pips * point * 10
            tp = price + take_profit_pips * point * 10
            order_type = mt5.ORDER_TYPE_BUY
        else:  # Sell
            price = tick.bid
            sl = price + stop_loss_pips * point * 10
            tp = price - take_profit_pips * point * 10
            order_type = mt5.ORDER_TYPE_SELL
            
        # Create request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": 12345,
            "comment": "Python Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Trade execution failed for {symbol}: {result.retcode}")
            return False
            
        print(f"Trade executed: {symbol} {'BUY' if signal == 1 else 'SELL'} {lot_size} lots, SL: {stop_loss_pips} pips, TP: {take_profit_pips} pips")
        return True

    def run_bot(self):
        """Main bot execution loop"""
        print("Starting Advanced Forex & Crypto Trading Bot...")
        symbols = list(self.strategies.keys())
        print(f"Trading pairs: {', '.join(symbols)}")
        
        while True:
            try:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n--- Checking signals at {current_time} ---")
                
                for symbol in symbols:
                    print(f"Analyzing {symbol}...")
                    
                    # Get technical data
                    df = self.get_technical_indicators(symbol, self.timeframes['primary'])
                    if df is None:
                        print(f"No data for {symbol}, skipping...")
                        continue
                    
                    # Get trading signal
                    signal, stop_loss_pips, take_profit_pips = self.strategies[symbol](df)
                    
                    if signal != 0:
                        print(f"{symbol} Signal: {'BUY' if signal == 1 else 'SELL'} | SL: {stop_loss_pips:.1f} pips | TP: {take_profit_pips:.1f} pips")
                        self.execute_trade(symbol, signal, stop_loss_pips, take_profit_pips)
                    else:
                        print(f"{symbol}: No clear signal")
                
                # Wait before next iteration
                print("Waiting 60 seconds for next analysis...")
                time.sleep(60)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)

    def close_all_positions(self):
        """Close all open positions"""
        positions = mt5.positions_get()
        if positions:
            print(f"Closing {len(positions)} positions...")
            for position in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": position.ticket,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
                    "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 1 else mt5.symbol_info_tick(position.symbol).ask,
                    "deviation": 20,
                    "magic": 12345,
                    "comment": "Close All",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(close_request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Closed position: {position.symbol}")
                else:
                    print(f"Failed to close position: {result.retcode}")
        else:
            print("No positions to close")

    def get_account_info(self):
        """Get current account information"""
        account_info = mt5.account_info()
        if account_info:
            print(f"Account Balance: ${account_info.balance:.2f}")
            print(f"Account Equity: ${account_info.equity:.2f}")
            print(f"Account Margin: ${account_info.margin:.2f}")
            print(f"Free Margin: ${account_info.margin_free:.2f}")
        return account_info

# Configuration and Execution
if __name__ == "__main__":
    # Configuration
    ACCOUNT_SIZE = 10000  # Adjust based on your account size
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    
    # Create and run bot
    bot = AdvancedForexBot(ACCOUNT_SIZE, RISK_PER_TRADE)
    
    try:
        # Display account info
        bot.get_account_info()
        
        # Run the bot
        bot.run_bot()
    except KeyboardInterrupt:
        print("\nStopping bot...")
        bot.close_all_positions()
        mt5.shutdown()
    except Exception as e:
        print(f"Fatal error: {e}")
        bot.close_all_positions()
        mt5.shutdown()
