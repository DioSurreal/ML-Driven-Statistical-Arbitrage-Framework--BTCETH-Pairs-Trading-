import joblib
import os
from src.data_loader import DataHandler
from src.market_regime import RegimeDetector
from src.strategy import PairsStrategy
from src.backtester import Backtester

if __name__ == "__main__":
    dh = DataHandler()
    target_tickers = ["BTC-USD", "ETH-USD"]
    dh.download_data(target_tickers, start_date="2018-01-01", end_date="2026-01-01")
    
    btc_df = dh.load_local_data("BTC-USD")
    eth_df = dh.load_local_data("ETH-USD")

    if btc_df is not None and eth_df is not None:
        # --- TRAIN MODEL ---
        split_idx = int(len(btc_df) * 0.8)
        detector = RegimeDetector(n_regimes=3)
        detector.fit(btc_df.iloc[:split_idx])

        # (Sideways / Bull / Bear)
        target_label = "Sideways" 
        target_class = detector.get_class_by_label(target_label)
        
        full_regimes = detector.predict(btc_df)

        # --- GENERATE SIGNALS ---
        print(f"\n⚡ Strategy: Pairs Trading | Target: {target_label} (Class {target_class})")
        strategy = PairsStrategy(z_entry=3.0, z_exit=0.0)
        common_index = btc_df.index.intersection(eth_df.index).intersection(full_regimes.index)
        
        signals = strategy.generate_signals(
            btc_df.loc[common_index], 
            eth_df.loc[common_index], 
            full_regimes.loc[common_index],
            target_regime=target_class
        )

        # --- BACKTEST ---
        print(f"\n📉 Backtesting on {target_label} Regime...")
        bt = Backtester(initial_capital=10000, commission=0.001, risk_per_trade=0.01)
        backtest_results = bt.run(btc_df.loc[common_index], eth_df.loc[common_index], signals)
        
        bt.show_metrics(backtest_results)
        backtest_results.to_csv("data/backtest_results.csv")
        print(f"\n✅ Done! Performance for {target_label} saved.")