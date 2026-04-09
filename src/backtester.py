import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_capital=10000, commission=0.001, risk_per_trade=0.01):
        """
        risk_per_trade: 0.01 หมายถึง ยอมเสี่ยงแค่ 1% ของพอร์ตต่อการเทรด 1 ครั้ง
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_per_trade = risk_per_trade

    def run(self, df1, df2, signals):
        results = signals.copy()
        

        ratio = df1['Close'] / df2['Close']
        results['ratio_ret'] = ratio.pct_change()
        
        results['volatility'] = results['ratio_ret'].rolling(window=20).std()
        
        results['pos_size'] = (self.initial_capital * self.risk_per_trade) / results['volatility']
        
        results['pos_size'] = results['pos_size'].clip(upper=self.initial_capital)

        results['strat_pnl'] = results['signal'].shift(1) * results['pos_size'].shift(1) * results['ratio_ret']
        
        trades = results['signal'].diff().fillna(0) != 0
        results['costs'] = trades * results['pos_size'] * self.commission
        results['net_pnl'] = results['strat_pnl'] - results['costs']
        
        results['equity_curve'] = self.initial_capital + results['net_pnl'].fillna(0).cumsum()
        results['cum_ret'] = results['equity_curve'] / self.initial_capital
        
        return results

    def show_metrics(self, results):
        total_ret = ((results['equity_curve'].iloc[-1] / self.initial_capital) - 1) * 100
        rolling_max = results['equity_curve'].cummax()
        drawdown = (results['equity_curve'] - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        daily_pnl = results['net_pnl'].dropna()
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() != 0 else 0

        print("\n🏆 --- PERFORMANCE METRICS (With Position Sizing) ---")
        print(f"💰 Total Return: {total_ret:.2f}%")
        print(f"📉 Max Drawdown: {max_dd:.2f}%")
        print(f"📊 Sharpe Ratio: {sharpe:.2f}")