# Volatility Alpha Engine (VAE)

**Volatility Alpha Engine** is a research-grade options/stock trading project that turns historical price data into a full reinforcement learning (RL) laboratory.

The goal:  
> Use engineered volatility + edge features to train an RL agent that systematically outperforms simple baseline rules on next-day returns.

This repo is the **research backend** for a future Streamlit dashboard and RL options trading UI.

---

## Project Structure

```text
volatility-alpha-engine/
├── data/
│   └── volatility_alpha.duckdb      # DuckDB file with all engineered tables
├── notebooks/
│   ├── 00_backfill.ipynb
│   ├── 01_eda_volatility_alpha.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_backtesting_signals.ipynb
│   ├── 04_rl_environment.ipynb
│   ├── 05_baseline_policies.ipynb
│   ├── 06_rl_training_qlearning.ipynb
│   └── 07_diagnostics_interpretation.ipynb
├── src/
├── tests/
├── .env
├── requirements.txt
└── README.md