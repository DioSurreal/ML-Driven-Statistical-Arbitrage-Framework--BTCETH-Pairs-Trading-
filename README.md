# ML-Driven Statistical Arbitrage Framework (BTC/ETH Pairs Trading)

This project implements an automated **pairs trading** system that combines **statistical arbitrage** with **machine learning-based market regime detection**.  
Its objective is to improve trade selection, reduce drawdown risk, and capture relative value opportunities between BTC and ETH.

---

## 1. Strategy Overview

We use a **Pairs Trading (Statistical Arbitrage)** approach:

- **Core idea:** BTC and ETH often exhibit a strong relationship that tends to revert toward a long-term equilibrium (**mean reversion**).
- **Execution logic:** When the spread deviates significantly from its mean (for example, `Z-Score > 3.0`), the system opens a long position in the undervalued asset and a short position in the overvalued asset.
- **ML filter:** A **Gaussian Mixture Model (GMM)** classifies market regimes (Bull, Bear, Sideways). The strategy can then trade only in favorable regimes and avoid periods associated with poor risk-adjusted performance.

---

## 2. Tech Stack & Infrastructure

- **Language:** Python 3.9+
- **Data source:** Yahoo Finance API (`yfinance`)
- **Analytics & ML:** Pandas, NumPy, Scikit-learn (GMM), Statsmodels (cointegration)
- **Infrastructure:** Docker and Docker Compose for portable, reproducible execution
- **Development workflow:** Jupyter Notebook support inside the container for EDA and experimentation

---

## 3. Project Architecture

1. **`data_loader.py`**  
   Downloads market data and manages local CSV storage.
2. **`market_regime.py`**  
   Extracts market features and clusters data into three regimes, with auto-labeling based on volatility and momentum characteristics.
3. **`strategy.py`**  
   Computes spread and Z-Score, then generates trading signals conditioned on the ML regime filter.
4. **`backtester.py`**  
   Simulates realistic execution, including transaction costs and volatility-adjusted position sizing.

---

## 4. Key Evaluation Metrics

We evaluate performance using:

- **Z-Score:** Measures how far the spread is from its mean in standard deviation units.
- **Cointegration:** Tests whether the BTC/ETH pair has a statistically meaningful long-run relationship.
- **Sharpe Ratio:** Measures risk-adjusted returns (higher is better).
- **Max Drawdown (MDD):** Captures the largest peak-to-trough portfolio decline.
- **ATR-based Position Sizing:** Scales exposure based on market volatility to control risk.

---

## 5. Backtesting Results (2018-2026)

| Market Regime | Total Return | Max Drawdown | Sharpe Ratio |
| :--- | :--- | :--- | :--- |
| **Bull Market** | **+7.14%** | -2.22% | **0.37** |
| **Bear Market** | +1.61% | **-0.03%** | 0.29 |
| **Sideways Market** | -11.76% | -13.05% | -0.51 |

**Conclusion:** The strategy performs best in **Bull** conditions and remains most defensive in **Bear** conditions.  
Given weak results in **Sideways** markets, the regime filter is configured to avoid trading during those periods.

---

## 6. How to Run

1. Clone this repository.
2. Build and run with Docker:

```bash
docker-compose up --build
```

3. Review outputs:
- Backtest records are written to `data/backtest_results.csv`
- Summary metrics are printed in the terminal

---

## 7. Key Takeaways

- Market regime awareness is critical for crypto pairs trading.
- Machine learning can act as an effective risk filter, similar to a circuit breaker.
- Robust risk management is more important than maximizing nominal returns.
