# Volatility Alpha Engine (VAE) — Volatility Regimes & Decision Discipline
🔗 Live App

https://vae-screener-10109427624.us-central1.run.app/

## Problem

Volatility is not stationary. Strategies fail when regimes shift and risk assumptions stay fixed.

## Why This Problem Matters

Volatility governs:

- Risk

- Option pricing

- Strategy survival

Ignoring regime structure leads to fragile systems.

## Data Used

- Daily OHLCV equity data via Polygon.io

- Stored in a DuckDB feature store

- Derived volatility and regime features

## Approach

- Volatility feature engineering

- Regime segmentation experiments

- Rule-based backtests

- Exploratory RL via a custom Gym environment

- Walk-forward validation

## Evaluation & Findings

- Volatility regimes materially alter return expectancy

- Regime-aware filters reduce drawdowns

- RL gains come from selective participation

## Limitations

- No transaction cost modeling

- Discrete RL state space

- Equity-only scope

## Planned Next Steps

- Transaction cost modeling

- Expanded regime inference

- Regime-shift alerts in the UI

## Reproducibility — Run Locally
```bash
git clone https://github.com/btheard3/volatility-alpha-engine
cd volatility-alpha-engine
pip install -r requirements.txt
```


Run notebooks sequentially (`00` → `07`) or launch the app:
```bash
streamlit run app.py
```

Docker + Cloud Run configs are included.

## Portfolio Context

**Risk and regime backbone** — supports every other project.

## Author

Brandon Theard
Data Scientist | Decision-Support Systems

GitHub: https://github.com/btheard3

LinkedIn: https://www.linkedin.com/in/brandon-theard-811b38131/