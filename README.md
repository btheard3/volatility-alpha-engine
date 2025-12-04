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
```
---

**Notebooks Overview**

**00 – Backfill & Data Ingest**
Loads raw OHLCV data into DuckDB and builds a clean, gap-free daily dataset per ticker.

**01 – Volatility & EDA**
Explores returns, volatility regimes, and liquidity. Sanity checks that the universe and date ranges are tradeable.

**02 – Feature Engineering**
Builds the core edge score and volatility/liquidity features that later drive both baselines and RL.
Outputs `screener_features` and related tables in DuckDB.

**03 – Backtesting Signals**
Turns the edge score into simple rule-based trading signals and equity curves to confirm the features have economic signal.

**04 – RL Environment Skeleton**
Converts engineered features into an RL-ready dataset and implements a minimal `VAETradingEnv` with `reset()` and `step()`.

**05 – Baseline Policies**
Benchmarks three strategies on the same dataset:

- Random policy

- Edge-threshold policy

- Volatility-regime policy

Produces equity curves and performance metrics (total return, Sharpe, max drawdown).

**06 – RL Agent Training (Q-Learning Prototype)**
Trains a tabular Q-learning agent on a discrete state space:

- State = (volatility regime, edge bucket)

- Action = flat vs long

- Reward = next-day return when long

Backtests the learned policy vs a random baseline on an unseen test period.

**07 – Diagnostics & Interpretation (Production Version)**
Does not train new models; instead it explains what the RL agent is doing:

- Trading frequency by volatility regime

- Trading frequency by edge bucket

- Average RL returns by regime and bucket

- RL equity curve vs flat-cash baseline

This is the “storytelling” notebook that turns the math into an interpretable trading narrative.

Tech Stack

- **Python 3.11+**

- **DuckDB** for analytics storage

- **Pandas / NumPy** for data wrangling

- **Matplotlib** for research plots

- **Tabular Q-Learning** for RL prototype

Planned extensions:

- **Streamlit** front-end (“Tradezilla-style” sleek UI)

- **Docker** for reproducible deployment

- Optional **React UI** for a full RL options trading console

**How to Run the Project**

**1. Create and activate a virtual environment**

```
python -m venv .venv
source .venv/bin/activate         # Linux/WSL
# .venv\Scripts\activate          # Windows
```

**2. Install dependencies**

pip install -r requirements.txt

**3. Ensure DuckDB file is present**

Place the DuckDB database at:

data/volatility_alpha.duckdb
(Notebook paths assume this location.)

Run the notebooks in order

Recommended order:

1. `00_backfill.ipynb`

2. `01_eda_volatility_alpha.ipynb`

3. `02_feature_engineering.ipynb`

4. `03_backtesting_signals.ipynb`

5. `04_rl_environment.ipynb`

6. `05_baseline_policies.ipynb`

7. `06_rl_training_qlearning.ipynb`

8. `07_diagnostics_interpretation.ipynb`

By the end, you will have:

- Engineered features & edge scores stored in DuckDB

- Baseline strategy benchmarks

- A trained Q-learning policy

- Diagnostics explaining when and why the RL agent trades