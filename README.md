# Volatility Alpha Engine (VAE)

## Overview

**Volatility Alpha Engine (VAE)** is a research-driven quantitative trading system designed to study how volatility regimes affect decision quality, signal reliability, and expected returns.

The project is built around a simple but often ignored premise:

> Volatility is not opportunity — it is uncertainty.
And uncertainty should change behavior.

Rather than predicting price direction, VAE focuses on determining when market participation is statistically justified and when restraint dominates.

---

## Why This Is a Problem

Most trading and decision systems implicitly assume:

- More activity = more opportunity

- Volatility increases edge

- Models should always act

In reality, volatility often signals **lower information quality**, unstable regimes, and degraded signal reliability.

The true failure mode is not bad prediction — it is overconfidence under uncertainty.

This problem generalizes beyond trading to:

- Product experimentation

- Risk modeling

- Forecasting

- Automated decision systems

VAE treats *abstention* as a valid and sometimes optimal decision.

---

## Research Question

VAE investigates three core questions:

1. Do volatility regimes materially change return expectancy?

2. Can simple, interpretable filters reduce drawdowns before ML is applied?

3. Can a learning agent discover when not to trade without being explicitly told?

The objective is not alpha maximization, but decision discipline.

---

## Dataset

**Market Data**

- Source: Polygon.io

- Data: Daily OHLCV bars (U.S. equities)

- Usage:

    - Return computation

    - Realized volatility estimation

    - Regime labeling

    - RL reward calculation

    - Live screener metrics

**Storage**

- All cleaned and engineered data is cached locally in DuckDB:
```bash
data/volatility_alpha.duckdb
```
- DuckDB serves as a lightweight, reproducible analytical feature store
- Models operate only on derived features, never raw prices

---

## Methodology

1. **Exploratory Analysis**

Initial analysis evaluates:

- Return distributions across volatility regimes

- Volatility clustering and persistence

- Liquidity effects

- Regime-dependent instability

This establishes that volatility meaningfully alters decision context, not just return magnitude.

2. **Feature Engineering**

Core features include:

- 20-day and 60-day realized volatility

- Volatility regime labels

- Liquidity filters

- Edge score (VAE’s primary decision feature)

All features are explicitly designed to be interpretable and regime-aware.

3. **Signal Backtesting**

Baseline strategies are tested before ML is introduced:

- Edge-threshold rules

- Regime-conditioned entries

- Momentum vs mean-reversion variants

Evaluation includes:

- Equity curves

- Sharpe ratios

- Maximum drawdown

- Regime-level performance breakdowns

These baselines answer a prerequisite question:

*Is there any edge to learn from at all?*

4. **Reinforcement Learning Setup**

A custom Gym-style environment is implemented:
```python

class VAETradingEnv(gym.Env):
    ...
```
- **State**: (volatility regime, edge bucket)

- **Action**: flat vs long

- **Reward**: next-day return

- **Algorithm**: tabular Q-learning

The agent’s task is not prediction accuracy, but selective participation.

5. **Baselines vs RL**

RL performance is benchmarked against:

- Random policy

- Always-in policy

- Simple regime heuristics

The RL agent is only considered useful if it outperforms these controls by trading less, not more.

## Results

Key findings:

- Volatility regimes materially alter return expectancy

- Many high-volatility regimes are net-negative for participation

- Simple volatility-aware filters already reduce drawdowns

- The RL agent learns to avoid unstable regimes

- Performance gains come primarily from inaction, not directional forecasting

The dominant edge is knowing when uncertainty is too high to act.

## Failure Modes & Limitations

This project explicitly documents its boundaries:

- No transaction cost modeling

- No leverage or position sizing

- Discrete state space limits expressiveness

- Q-learning chosen for interpretability, not peak performance

- Equity-only scope (options planned, not implied)

VAE is framed as a research system, not a production trading bot.

## Live Screener

A lightweight Streamlit application exposes VAE’s volatility and edge metrics in real time.

- Deployment: Docker + GCP Cloud Run

- URL: https://vae-screener-10109427624.us-central1.run.app

- Purpose: Regime inspection and discretionary decision support

The screener mirrors the research pipeline exactly — no hidden logic.

## Reproducibility

- Deterministic notebook execution

- Local DuckDB feature store

- No proprietary infrastructure required

- Runs fully on a local machine or via Docker

Notebooks are intended to be run sequentially from `00` → `07`.

Repository Structure
```text
volatility-alpha-engine/
├── notebooks/        # Research pipeline
├── data/             # DuckDB feature store
├── dashboards/       # Live Streamlit screener
├── src/              # Data + feature utilities
├── Dockerfile
├── cloudbuild.yaml
└── requirements.txt
```

## What This Project Demonstrates

- Decision-making under uncertainty

- Volatility-aware feature engineering

- Baseline-first ML evaluation

- Reinforcement learning used conservatively

- Research-to-deployment ownership

- Clear articulation of limitations and failure modes

## Future Work

- Options-aware state space (Greeks, IV rank)

- Continuous state embeddings

- Actor-Critic agents

- Trade journal feedback loop

- Paper trading integration

## Author

Brandon Theard
GitHub: https://github.com/btheard3

LinkedIn: https://www.linkedin.com/in/brandon-theard-811b38131


