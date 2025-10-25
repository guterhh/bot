import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hft_bot.log'),
        logging.StreamHandler()
    ]
)

class TradingPair(Enum):
    BTCUSDm = "BTCUSDm"
    ETHUSDm = "ETHUSDm" 
    XAUUSDm = "XAUUSDm"

class OrderSide(Enum):
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL

@dataclass
class TradeSignal:
    pair: TradingPair
    side: OrderSide
    entry_price: float
    timestamp: float
    signal_strength: float
    volume: float

@dataclass
class PositionData:
    ticket: int
    symbol: str
    type: int
    volume: float
    entry_price: float
    current_price: float
    profit: float
    open_time: datetime

@dataclass
class BotConfig:
    # Trading Parameters
    profit_target: float = 0.0015      # 0.15%
    stop_loss: float = 0.0010          # 0.10%
    max_position_size: float = 0.1     # Default position size
    max_positions: int = 3             # Max concurrent positions
    max_daily_trades: int = 200        # Daily trade limit
    
    # Strategy Parameters
    momentum_period: int = 5
    rsi_period: int = 14
    volatility_period: int = 10
    min_volatility: float = 0.0002
    max_volatility: float = 0.005
    
    # Risk Management
    max_drawdown: float = 0.05         # 5% max drawdown
    risk_per_trade: float = 0.01       # 1% risk per trade
    equity_protection: float = 0.8     # Stop if equity < 80% balance
    
    # Execution
    update_interval: float = 0.05      # 20Hz update
    data_points: int = 100             # Data points to keep

class MT5HFTBot:
    def __init__(self, account: int, password: str, server: str):
        self.account = account
        self.password = password
        self.server = server
        self.config = BotConfig()
        
        # Initialize state
        self.is_running = False
        self.trading_active = True
        self.last_reset = datetime.now().date()
        
        # Data storage
        self.price_data: Dict[str, List[float]] = {}
        self.volume_data: Dict[str, List[float]] = {}
        self.last_update: Dict[str, float] = {}
        
        # Trading state
        self.active_pairs: Dict[TradingPair, bool] = {
            TradingPair.BTCUSDm: True,
            TradingPair.ETHUSDm: True,
            TradingPair.XAUUSDm: True
        }
        
        # Performance tracking
        self.daily_trades = 0
        self.total_trades = 0
        self.total_profit = 0.0
        self.closed_profit = 0.0
        self.open_profit = 0.0
        
        # Threading
        self.trading_thread = None
        self.monitor_thread = None
        
        # Initialize MT5
        self._initialize_mt5()
        
        # Load configuration
        self.load_config()
        
        logging.info("MT5 HFT Bot initialized")

    def _initialize_mt5(self) -> bool:
        """Initialize connection to MT5"""
        try:
            if not mt5.initialize():
                logging.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            if not mt5.login(self.account, password=self.password, server=self.server):
                logging.error(f"MT5 login failed: {mt5.last_error()}")
                return False
                
            logging.info(f"Connected to MT5 Account: {self.account}")
            return True
            
        except Exception as e:
            logging.error(f"MT5 connection error: {e}")
            return False

    def shutdown(self):
        """Shutdown bot gracefully"""
        logging.info("Shutting down HFT bot...")
        self.is_running = False
        self.trading_active = False
        
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        mt5.shutdown()
        logging.info("HFT bot shutdown complete")

    # ==================== PAIR MANAGEMENT ====================
    def activate_pair(self, pair: TradingPair):
        """Activate trading for pair"""
        self.active_pairs[pair] = True
        logging.info(f"Activated {pair.value}")

    def deactivate_pair(self, pair: TradingPair):
        """Deactivate trading for pair"""
        self.active_pairs[pair] = False
        self._close_pair_positions(pair.value)
        logging.info(f"Deactivated {pair.value}")

    def _close_pair_positions(self, symbol: str):
        """Close all positions for symbol"""
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for position in positions:
                self.close_position(position.ticket)

    # ==================== MARKET DATA ====================
    def update_market_data(self):
        """Update market data for active pairs"""
        current_time = time.time()
        
        for pair in self.active_pairs:
            if not self.active_pairs[pair]:
                continue
                
            symbol = pair.value
            
            # Throttle updates
            if symbol in self.last_update and current_time - self.last_update[symbol] < 0.1:
                continue
                
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
                
            # Initialize data structures
            if symbol not in self.price_data:
                self.price_data[symbol] = []
                self.volume_data[symbol] = []
                
            # Add new data
            self.price_data[symbol].append(tick.bid)
            self.volume_data[symbol].append(tick.volume)
            
            # Maintain data length
            if len(self.price_data[symbol]) > self.config.data_points:
                self.price_data[symbol] = self.price_data[symbol][-self.config.data_points:]
            if len(self.volume_data[symbol]) > self.config.data_points:
                self.volume_data[symbol] = self.volume_data[symbol][-self.config.data_points:]
                
            self.last_update[symbol] = current_time

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current bid price"""
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid if tick else None

    # ==================== INDICATORS ====================
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate momentum indicator"""
        prices = self.price_data.get(symbol, [])
        if len(prices) < self.config.momentum_period:
            return 0.0
            
        recent = prices[-self.config.momentum_period:]
        return (recent[-1] - recent[0]) / recent[0]

    def calculate_volatility(self, symbol: str) -> float:
        """Calculate price volatility"""
        prices = self.price_data.get(symbol, [])
        if len(prices) < self.config.volatility_period:
            return 0.0
            
        returns = np.diff(prices[-self.config.volatility_period:]) / prices[-self.config.volatility_period:-1]
        return np.std(returns)

    def calculate_rsi(self, symbol: str) -> float:
        """Calculate RSI indicator"""
        prices = self.price_data.get(symbol, [])
        if len(prices) < self.config.rsi_period + 1:
            return 50.0
            
        deltas = np.diff(prices[-self.config.rsi_period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # ==================== SIGNAL GENERATION ====================
    def generate_signal(self, pair: TradingPair) -> Optional[TradeSignal]:
        """Generate trading signal"""
        symbol = pair.value
        
        # Check data availability
        if symbol not in self.price_data or len(self.price_data[symbol]) < 20:
            return None
            
        # Calculate indicators
        momentum = self.calculate_momentum(symbol)
        volatility = self.calculate_volatility(symbol)
        rsi = self.calculate_rsi(symbol)
        
        # Adaptive threshold based on volatility
        threshold = 0.0008 + (volatility * 0.3)
        
        # Check volatility boundaries
        if volatility < self.config.min_volatility or volatility > self.config.max_volatility:
            return None
            
        # Generate signals
        if momentum > threshold and rsi < 65:  # Buy signal
            volume = self.calculate_position_size(symbol)
            current_price = self.get_current_price(symbol)
            
            if current_price:
                return TradeSignal(
                    pair=pair,
                    side=OrderSide.BUY,
                    entry_price=current_price,
                    timestamp=time.time(),
                    signal_strength=momentum,
                    volume=volume
                )
                
        elif momentum < -threshold and rsi > 35:  # Sell signal
            volume = self.calculate_position_size(symbol)
            current_price = self.get_current_price(symbol)
            
            if current_price:
                return TradeSignal(
                    pair=pair,
                    side=OrderSide.SELL,
                    entry_price=current_price,
                    timestamp=time.time(),
                    signal_strength=abs(momentum),
                    volume=volume
                )
                
        return None

    # ==================== POSITION MANAGEMENT ====================
    def calculate_position_size(self, symbol: str) -> float:
        """Calculate position size based on risk management"""
        account_info = mt5.account_info()
        if not account_info:
            return self.config.max_position_size
            
        balance = account_info.balance
        risk_amount = balance * self.config.risk_per_trade
        
        # Get symbol info for point value
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return self.config.max_position_size
            
        # Calculate position size based on stop loss
        point_value = symbol_info.trade_tick_value
        if point_value > 0:
            position_size = risk_amount / (self.config.stop_loss * point_value * 100)
        else:
            position_size = self.config.max_position_size
            
        return min(position_size, self.config.max_position_size)

    def get_open_positions(self, symbol: str = None) -> List[PositionData]:
        """Get all open positions"""
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions:
            return []
            
        result = []
        for position in positions:
            current_price = self.get_current_price(position.symbol)
            result.append(PositionData(
                ticket=position.ticket,
                symbol=position.symbol,
                type=position.type,
                volume=position.volume,
                entry_price=position.price_open,
                current_price=current_price or position.price_open,
                profit=position.profit,
                open_time=position.time
            ))
            
        return result

    def has_open_position(self, symbol: str) -> bool:
        """Check if symbol has open position"""
        positions = self.get_open_positions(symbol)
        return len(positions) > 0

    def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute trade order"""
        try:
            symbol = signal.pair.value
            order_type = signal.side.value
            volume = signal.volume
            
            # Get current price
            if signal.side == OrderSide.BUY:
                price = mt5.symbol_info_tick(symbol).ask
                tp_price = price * (1 + self.config.profit_target)
                sl_price = price * (1 - self.config.stop_loss)
            else:
                price = mt5.symbol_info_tick(symbol).bid
                tp_price = price * (1 - self.config.profit_target)
                sl_price = price * (1 + self.config.stop_loss)
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 2024001,
                "comment": "HFT_BOT",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"EXECUTED {signal.side.name} {symbol} | "
                           f"Price: {price:.2f} | Volume: {volume:.4f} | "
                           f"Ticket: {result.order}")
                self.daily_trades += 1
                self.total_trades += 1
                return True
            else:
                logging.error(f"Order failed: {result.retcode} - {result.comment}")
                return False
                
        except Exception as e:
            logging.error(f"Trade execution error: {e}")
            return False

    def close_position(self, ticket: int) -> bool:
        """Close specific position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
                
            position = position[0]
            symbol = position.symbol
            volume = position.volume
            
            # Determine close type
            if position.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 2024001,
                "comment": "HFT_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"CLOSED position {ticket} for {symbol}")
                return True
            else:
                logging.error(f"Close position failed: {result.retcode}")
                return False
                
        except Exception as e:
            logging.error(f"Close position error: {e}")
            return False

    def close_all_positions(self):
        """Close all open positions"""
        positions = self.get_open_positions()
        for position in positions:
            self.close_position(position.ticket)

    # ==================== RISK MANAGEMENT ====================
    def check_risk_limits(self) -> bool:
        """Check if trading is within risk limits"""
        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            self.daily_trades = 0
            self.last_reset = current_date
        
        # Check daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            if self.trading_active:
                logging.warning("Daily trade limit reached - pausing trading")
                self.trading_active = False
            return False
            
        # Check account equity
        account_info = mt5.account_info()
        if not account_info:
            return False
            
        equity = account_info.equity
        balance = account_info.balance
        
        # Check equity protection
        if equity < balance * self.config.equity_protection:
            logging.error(f"Equity protection triggered: {equity:.2f} < {balance * self.config.equity_protection:.2f}")
            self.trading_active = False
            return False
            
        # Check drawdown
        if balance > 0:
            drawdown = (balance - equity) / balance
            if drawdown > self.config.max_drawdown:
                logging.error(f"Max drawdown exceeded: {drawdown:.2%}")
                self.trading_active = False
                return False
                
        return True

    def get_open_positions_count(self) -> int:
        """Get count of open positions"""
        positions = mt5.positions_get()
        return len(positions) if positions else 0

    # ==================== TRADING ENGINE ====================
    def trading_loop(self):
        """Main trading loop"""
        logging.info("Starting HFT trading engine...")
        self.is_running = True
        
        while self.is_running:
            try:
                # Update market data
                self.update_market_data()
                
                # Check risk management
                risk_ok = self.check_risk_limits()
                
                # Generate and execute signals if trading is active
                if self.trading_active and risk_ok:
                    current_positions = self.get_open_positions_count()
                    
                    for pair in self.active_pairs:
                        if not self.active_pairs[pair]:
                            continue
                            
                        # Check position limit
                        if current_positions >= self.config.max_positions:
                            break
                            
                        # Check if pair already has position
                        if self.has_open_position(pair.value):
                            continue
                            
                        # Generate signal
                        signal = self.generate_signal(pair)
                        if signal:
                            if self.execute_trade(signal):
                                current_positions += 1
                                # Small delay between executions
                                time.sleep(0.01)
                
                # Update performance metrics
                self.update_performance()
                
                # Sleep for next iteration
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                logging.error(f"Trading loop error: {e}")
                time.sleep(1)

    def update_performance(self):
        """Update performance metrics"""
        positions = self.get_open_positions()
        self.open_profit = sum(position.profit for position in positions)
        
        # Update total profit (this would need to track closed trades properly)
        account_info = mt5.account_info()
        if account_info:
            self.total_profit = account_info.profit

    # ==================== CONFIGURATION MANAGEMENT ====================
    def save_config(self, filename: str = "hft_config.json"):
        """Save configuration to file"""
        config_data = {
            'bot_config': asdict(self.config),
            'active_pairs': {p.value: active for p, active in self.active_pairs.items()}
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=4)
            logging.info(f"Configuration saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")

    def load_config(self, filename: str = "hft_config.json"):
        """Load configuration from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    config_data = json.load(f)
                
                # Update bot config
                if 'bot_config' in config_data:
                    for key, value in config_data['bot_config'].items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                
                # Update active pairs
                if 'active_pairs' in config_data:
                    for pair_str, active in config_data['active_pairs'].items():
                        pair = TradingPair(pair_str)
                        self.active_pairs[pair] = active
                
                logging.info(f"Configuration loaded from {filename}")
                
        except Exception as e:
            logging.error(f"Failed to load config: {e}")

    # ==================== BOT CONTROL ====================
    def start_trading(self):
        """Start the trading bot"""
        if self.trading_thread and self.trading_thread.is_alive():
            logging.warning("Trading bot is already running")
            return
            
        self.trading_active = True
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()
        logging.info("HFT trading bot started")

    def stop_trading(self):
        """Stop the trading bot"""
        self.trading_active = False
        self.is_running = False
        logging.info("HFT trading bot stopped")

    def get_status(self) -> Dict:
        """Get bot status"""
        account_info = mt5.account_info()
        positions = self.get_open_positions()
        
        return {
            'trading_active': self.trading_active,
            'bot_running': self.is_running,
            'account_equity': account_info.equity if account_info else 0,
            'account_balance': account_info.balance if account_info else 0,
            'open_positions': len(positions),
            'daily_trades': self.daily_trades,
            'total_trades': self.total_trades,
            'open_profit': self.open_profit,
            'total_profit': self.total_profit,
            'active_pairs': [p.value for p, active in self.active_pairs.items() if active]
        }

# ==================== CONTROL INTERFACE ====================
class HFTControlPanel:
    def __init__(self, bot: MT5HFTBot):
        self.bot = bot
        self.running = True
        
    def display_menu(self):
        print("\n" + "="*60)
        print("           MT5 HFT TRADING BOT CONTROL PANEL")
        print("="*60)
        print("1.  Start Trading")
        print("2.  Stop Trading")
        print("3.  Bot Status")
        print("4.  Pair Management")
        print("5.  Risk Management")
        print("6.  Close All Positions")
        print("7.  Save Configuration")
        print("8.  Update Trading Parameters")
        print("9.  Exit")
        print("="*60)
        
    def handle_choice(self, choice: str):
        if choice == "1":
            self.bot.start_trading()
            print("Trading started")
            
        elif choice == "2":
            self.bot.stop_trading()
            print("Trading stopped")
            
        elif choice == "3":
            self.show_status()
            
        elif choice == "4":
            self.manage_pairs()
            
        elif choice == "5":
            self.manage_risk()
            
        elif choice == "6":
            confirm = input("Close ALL positions? (y/n): ").lower()
            if confirm == 'y':
                self.bot.close_all_positions()
                print("All positions closed")
                
        elif choice == "7":
            self.bot.save_config()
            
        elif choice == "8":
            self.update_parameters()
            
        elif choice == "9":
            self.bot.stop_trading()
            self.bot.shutdown()
            self.running = False
            print("Exiting...")
            
        else:
            print("Invalid choice")
            
    def show_status(self):
        status = self.bot.get_status()
        print("\n" + "-"*40)
        print("BOT STATUS")
        print("-"*40)
        for key, value in status.items():
            if key == 'active_pairs':
                print(f"{key}: {', '.join(value)}")
            elif isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
                
    def manage_pairs(self):
        print("\nPair Management:")
        pairs = list(TradingPair)
        for i, pair in enumerate(pairs, 1):
            status = "ACTIVE" if self.bot.active_pairs[pair] else "INACTIVE"
            print(f"{i}. {pair.value} - {status}")
            
        choice = input("Select pair to toggle (1-3) or 0 to return: ")
        if choice in ["1", "2", "3"]:
            pair = pairs[int(choice) - 1]
            if self.bot.active_pairs[pair]:
                self.bot.deactivate_pair(pair)
            else:
                self.bot.activate_pair(pair)
                
    def manage_risk(self):
        print("\nCurrent Risk Parameters:")
        config = self.bot.config
        print(f"1. Profit Target: {config.profit_target:.4f}")
        print(f"2. Stop Loss: {config.stop_loss:.4f}")
        print(f"3. Max Position Size: {config.max_position_size}")
        print(f"4. Max Daily Trades: {config.max_daily_trades}")
        print(f"5. Max Drawdown: {config.max_drawdown:.2%}")
        
        choice = input("Select parameter to update (1-5) or 0 to return: ")
        if choice in ["1", "2", "3", "4", "5"]:
            new_value = input("Enter new value: ")
            try:
                if choice in ["1", "2"]:
                    new_value = float(new_value)
                elif choice in ["3", "4"]:
                    new_value = int(new_value)
                elif choice == "5":
                    new_value = float(new_value)
                    
                param_map = {
                    "1": "profit_target", "2": "stop_loss", 
                    "3": "max_position_size", "4": "max_daily_trades",
                    "5": "max_drawdown"
                }
                
                setattr(config, param_map[choice], new_value)
                print("Parameter updated")
                
            except ValueError:
                print("Invalid value")
                
    def update_parameters(self):
        print("\nTrading Parameters:")
        config = self.bot.config
        print(f"1. Momentum Period: {config.momentum_period}")
        print(f"2. RSI Period: {config.rsi_period}")
        print(f"3. Update Interval: {config.update_interval}")
        
        choice = input("Select parameter to update (1-3) or 0 to return: ")
        if choice in ["1", "2", "3"]:
            new_value = input("Enter new value: ")
            try:
                if choice == "3":
                    new_value = float(new_value)
                else:
                    new_value = int(new_value)
                    
                param_map = {"1": "momentum_period", "2": "rsi_period", "3": "update_interval"}
                setattr(config, param_map[choice], new_value)
                print("Parameter updated")
                
            except ValueError:
                print("Invalid value")
                
    def run(self):
        """Run control panel"""
        print("HFT Bot Control Panel Started")
        
        while self.running:
            try:
                self.display_menu()
                choice = input("Enter your choice (1-9): ").strip()
                self.handle_choice(choice)
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\nShutting down...")
                self.bot.stop_trading()
                self.bot.shutdown()
                break
            except Exception as e:
                print(f"Error: {e}")

# ==================== MAIN EXECUTION ====================
def main():
    # MT5 Connection Configuration
    ACCOUNT = 12345678      # Replace with your MT5 account number
    PASSWORD = "your_password"  # Replace with your MT5 password
    SERVER = "your_broker_server"  # Replace with your broker server
    
    # Initialize bot
    bot = MT5HFTBot(account=ACCOUNT, password=PASSWORD, server=SERVER)
    
    # Start control panel
    control_panel = HFTControlPanel(bot)
    control_panel.run()

if __name__ == "__main__":
    main(
