# Volatility Alpha Engine (VAE)

**Volatility Alpha Engine** is a research-grade quantitative trading framework designed to analyze volatility regimes, engineer predictive features, backtest rule-based signals, and train a reinforcement-learning (RL) agent on next-day return dynamics.

VAE powers two major surfaces:

1. The research pipeline (DuckDB + notebooks)

2. The live volatility screener (Streamlit + Polygon + Docker + GCP Cloud Run)

**What VAE Does in One Sentence**

> VAE transforms raw OHLCV data into volatility/edge features, evaluates them through signal backtests, and uses them to train an interpretable RL agent designed to outperform naïve trading policies.
---

## Project Structure

```text
volatility-alpha-engine/
├── dashboards/
│   └── option_screener_v1/streamlit_app.py     # Live deployed screener (Cloud Run + Docker)
├── data/
│   └── volatility_alpha.duckdb                 # Engineered feature store in DuckDB
├── notebooks/
│   ├── 00_backfill.ipynb
│   ├── 01_eda_volatility_alpha.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_backtesting_signals.ipynb
│   ├── 04_rl_environment.ipynb
│   ├── 05_baseline_policies.ipynb
│   ├── 06_rl_training_qlearning.ipynb
│   └── 07_diagnostics_interpretation.ipynb
├── src/                                        # Polygon client, DB utilities, feature builders
├── tests/
├── Dockerfile
├── cloudbuild.yaml                             # CI/CD: GitHub → Cloud Build → Cloud Run
├── requirements.txt
└── README.md
```
---

**Notebooks Overview**

**00 – Backfill & Data Ingest**
Pulls historical price data, normalizes timestamps, and builds a clean OHLCV dataset in DuckDB.

**01 – Volatility & EDA**
Explores returns, realized volatility regimes, gaps, and liquidity effects.

**02 – Feature Engineering**
Creates volatility-aware predictive features, including:

- /60d realized volatility

- Edge score (VAE’s core feature)

- Liquidity filters

- Regime labels

**03 – Backtesting Signals**
Evaluates simple benchmark strategies:

- Edge-threshold

- Regime-based

- Momentum/reversal flavor tests

Produces equity curves and Sharpe/max-drawdown diagnostics

**04 – RL Environment Skeleton**
Implements the first production-ready version of:
```
class VAETradingEnv(gym.Env)
```
Used for step-by-step RL training

**05 – Baseline Policies**
Backtests random, simple heuristics, and regime policies to benchmark RL performance.

**06 – RL Agent Training (Q-Learning Prototype)**
Trains a tabular agent on:

- State = (volatility regime, edge bucket)

- Action = (flat vs long)

- Reward = next-day return

**07 – Diagnostics & Interpretation (Production Version)**
Turns the RL results into a trading narrative:

- When the agent trades

- Which regimes it avoids

- When edges behave predictably

- RL vs baseline curves

---

# Data Sources & API Strategy

1. **Historical Market Data**

- **Polygon.io**

    - Realized volatility calculations

    - OHLCV bars for backtesting

    - Live RV/price metrics for the Streamlit screener

- **Local cache via DuckDB**

    - All cleaned and engineered features live in `data/volatility_alpha.duckdb`

    - Ensures fast, reproducible research runs

    - Minimizes API calls and avoids rate-limit issues

2. API Key Strategy

To avoid Polygon rate-limit issues across multiple projects (VAE, Sentinel, RL trader):

✔ Use multiple free-tier keys only for dev
✔ Use one production key, stored in:

- `.env` locally

- Cloud Run service variables (secure & encrypted)

✔ Long-term plan:
Move to **Polygon Paid Tier** or **Tiingo** for 50,000–100,000 requests/day.

3. **Live Screener (GCP Cloud Run)**

- Containerized with Docker

- Public HTTPS endpoint

- Auto-scales to zero → free until hit

- CI/CD via Cloud Build triggers → push to GitHub = redeploy automatically

4. Research Compute

- Local/WSL DuckDB computations

- Optional migration to BigQuery for scalable signal testing (stretch goal)

---

# Trade Journal Module (Upcoming)

This is the missing piece that turns VAE from “research project” into a “real trading system.”
It will integrate with:

- The VAE screener output

- Future RL decisions

- User-entered trades

**Planned Features**

✔ Log trades (direction, size, rationale, screenshot of chart)
✔ Auto-tag by volatility regime + edge bucket
✔ Compute P&L, win rate, expectancy
✔ Learn which trades perform best for the user
✔ Feed back into ML/RL training (“meta-learning lite”)

> The system doesn’t just generate signals — it learns from the trader’s behavior and P&L to refine model features and RL reward shaping.

---

# ML & RL Expansion Roadmap

**Phase 1 — Already Implemented**

- Realized volatility features

- Daily edge score

- Q-learning prototype

- Backtests + diagnostics

**Phase 2 — Near-Term Additions**

- Logistic regression edge classifier

- Gradient boosting model (LightGBM / XGBoost)

- Custom reward shaping for RL

- Multi-action RL (flat, long, short, reduce size)

 **Phase 3 — Full RL Options Trading Engine**

- Gym-compatible environment with Greeks (Δ, Γ, Θ, Vega)

- State embedding:

    - 20d/60d RV

    - IV rank (from Polygon options API)

    - Edge score

    - Market regime estimates

- Actor-Critic agent

- Position sizing algorithm

**Phase 4 — Broker Integration**

- Paper trading via Webull / Tastytrade / Alpaca

- Real order execution (long horizon goal)

---

Tech Stack

- **Python 3.11+**

- **DuckDB** for analytics storage

- **Pandas / NumPy** for data wrangling

- **Matplotlib** for research plots

- **Tabular Q-Learning** for RL prototype

**How to Run the Project**

**1. Create and activate a virtual environment**

```
python -m venv .venv
source .venv/bin/activate         # Linux/WSL
# .venv\Scripts\activate          # Windows
```

**2. Install dependencies**
```
pip install -r requirements.txt
```

**3. Ensure DuckDB file is present**

Place the DuckDB database at:
```
data/volatility_alpha.duckdb
```
(Notebook paths assume this location.)

Run the notebooks in order

**Recommended order**:

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

---

# Contributing

Future contributors can:

- Add new volatility features

- Build new baselines

- Improve RL reward shaping

- Help with React dashboard integration

---

Author

🔹 Brandon Theard

GitHub → https://github.com/btheard3

LinkedIn → https://www.linkedin.com/in/brandon-theard-811b38131