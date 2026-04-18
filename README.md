# Strategy-Lab-2


 ## Strategy Lab #2 — Mean Reversion in FX Pairs with Reinforcement Learning
=========================================================================
Algorithmic Token · ENTER Invest

Implements a three-phase FX pairs trading framework:
    Phase 1 — Pair identification via cointegration testing
    Phase 2 — Spread construction via Empirical Mean Reversion Time (EMRT)
    Phase 3a — Classical z-score baseline trading rule
    Phase 3b — Q-learning RL agent (Ning & Lee, 2024)

Primary reference:
    Ning, B. & Lee, K. (2024) — "Advanced Statistical Arbitrage with
    Reinforcement Learning" — Purdue University · arXiv:2403.12180

Classical baseline references:
    Gatev, Goetzmann & Rouwenhorst (2006) — "Pairs Trading: Performance
    of a Relative-Value Arbitrage Rule" — Review of Financial Studies
    Leung & Li (2016) — "Optimal Mean Reversion Trading: Mathematical
    Analysis and Practical Applications" — World Scientific

This is an experimental prototype. See risk disclosure at the bottom
of this file and in the accompanying Strategy Lab #2 article at
Algorithmic Token: https://algorithmictoken.substack.com

Dependencies: numpy, pandas, yfinance, statsmodels


import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ---------------------------------------------------------------------------
# Phase 1 — Pair identification via cointegration
# ---------------------------------------------------------------------------

def test_cointegration(series1: pd.Series,
                       series2: pd.Series,
                       significance: float = 0.05) -> dict:
    """
    Test for cointegration between two price series.

    Uses the Engle-Granger two-step method via statsmodels coint().
    Returns cointegration result, hedge ratio, spread, and half-life
    of mean reversion estimated via AR(1) regression on the spread.

    Parameters
    ----------
    series1      : pd.Series — daily close prices, asset 1
    series2      : pd.Series — daily close prices, asset 2
    significance : float     — p-value threshold (default 0.05 = 95% confidence)

    Returns
    -------
    dict with keys:
        cointegrated : bool
        p_value      : float
        beta         : float  — hedge ratio (series1 = beta * series2 + spread)
        spread       : pd.Series
        half_life    : float  — in trading days; np.inf if spread is not mean-reverting
    """
    score, pvalue, _ = coint(series1, series2)

    # Hedge ratio via OLS regression
    model  = OLS(series1, add_constant(series2)).fit()
    beta   = model.params.iloc[1]
    spread = series1 - beta * series2

    # Half-life via AR(1): delta_spread = lambda * spread_lagged + noise
    spread_lag   = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()
    ar_model     = OLS(delta_spread, add_constant(spread_lag)).fit()
    lam          = ar_model.params.iloc[1]
    half_life    = -np.log(2) / lam if lam < 0 else np.inf

    return {
        "cointegrated": pvalue < significance,
        "p_value":      round(pvalue, 6),
        "beta":         round(float(beta), 6),
        "spread":       spread,
        "half_life":    round(half_life, 2),
    }


# ---------------------------------------------------------------------------
# Phase 2 — Spread construction via EMRT (Ning & Lee, 2024)
# ---------------------------------------------------------------------------

def compute_emrt(spread: pd.Series, C: float = 2.0) -> float:
    """
    Compute the Empirical Mean Reversion Time (EMRT) of a spread series.

    EMRT measures the average time for a spread to revert to its mean
    after a significant local extreme. Lower EMRT = faster mean reversion
    = better spread candidate for pairs trading.

    Based on Definition 3.1 in Ning & Lee (2024), arXiv:2403.12180,
    and the local extreme criterion from Fink & Gandhi (2007).

    Parameters
    ----------
    spread : pd.Series — spread time series
    C      : float     — threshold multiplier; a local extreme requires
                         price to move C * std(spread) away from the extreme
                         on both sides (default 2.0)

    Returns
    -------
    float — mean reversion time in bars (trading days for daily data);
            np.inf if insufficient extremes are found
    """
    s      = spread.std()
    mu_hat = spread.mean()
    values = spread.values
    n      = len(values)
    window = 20  # bars to look either side of candidate extreme

    extremes = []
    for m in range(window, n - window):
        left_segment  = values[m - window:m]
        right_segment = values[m + 1:m + window + 1]

        # Local minimum: neighbours are at least C * s higher
        if (max(left_segment)  - values[m] >= C * s and
                max(right_segment) - values[m] >= C * s):
            extremes.append(m)

        # Local maximum: neighbours are at least C * s lower
        elif (values[m] - min(left_segment)  >= C * s and
                values[m] - min(right_segment) >= C * s):
            extremes.append(m)

    if len(extremes) < 2:
        return np.inf

    # For each extreme, find time to next mean crossing
    reversion_times = []
    for pos in extremes:
        for future in range(pos + 1, n - 1):
            crossed = ((values[future - 1] < mu_hat <= values[future]) or
                       (values[future - 1] > mu_hat >= values[future]))
            if crossed:
                reversion_times.append(future - pos)
                break

    return float(np.mean(reversion_times)) if reversion_times else np.inf


def find_optimal_hedge_ratio(fx1: pd.Series,
                              fx2: pd.Series,
                              beta_min: float = -3.0,
                              beta_max: float = 3.0,
                              step: float = 0.05,
                              max_variance_multiplier: float = 10.0) -> dict:
    """
    Find the hedge ratio B that minimises EMRT of (FX1 - B * FX2).

    This is the model-free spread construction method from Ning & Lee (2024),
    replacing the OU maximum-likelihood estimator used in classical pairs
    trading. The variance cap prevents degenerate solutions.

    Parameters
    ----------
    fx1                      : pd.Series — price series, asset 1
    fx2                      : pd.Series — price series, asset 2
    beta_min / beta_max      : float     — search range for hedge ratio
    step                     : float     — grid search step size (default 0.05)
    max_variance_multiplier  : float     — variance cap as multiple of fx1 variance

    Returns
    -------
    dict with keys:
        beta        : float     — optimal hedge ratio
        emrt        : float     — EMRT at optimal beta (trading days)
        spread      : pd.Series — optimal spread series
        spread_mean : float
        spread_std  : float
    """
    max_var   = fx1.var() * max_variance_multiplier
    best_beta = 1.0
    best_emrt = np.inf

    for beta in np.arange(beta_min, beta_max + step, step):
        spread = fx1 - beta * fx2
        if spread.var() > max_var:
            continue
        emrt = compute_emrt(spread)
        if emrt < best_emrt:
            best_emrt = emrt
            best_beta = float(beta)

    optimal_spread = fx1 - best_beta * fx2
    return {
        "beta":        round(best_beta, 4),
        "emrt":        round(best_emrt, 2),
        "spread":      optimal_spread,
        "spread_mean": float(optimal_spread.mean()),
        "spread_std":  float(optimal_spread.std()),
    }


# ---------------------------------------------------------------------------
# Phase 3a — Classical z-score trading rule (baseline)
# ---------------------------------------------------------------------------

def classical_zscore_strategy(spread: pd.Series,
                               formation_mean: float,
                               formation_std: float,
                               entry_threshold: float = 2.0,
                               exit_threshold: float = 0.5,
                               cost_bps: float = 5.0) -> pd.Series:
    """
    Classical fixed-threshold mean reversion strategy on z-score.

    Enters long when z-score falls below -entry_threshold (spread cheap),
    enters short when z-score rises above +entry_threshold (spread rich),
    exits when z-score reverts toward zero past exit_threshold.

    Parameters
    ----------
    spread           : pd.Series — spread time series (trading period)
    formation_mean   : float     — mean estimated on formation period
    formation_std    : float     — std  estimated on formation period
    entry_threshold  : float     — z-score entry level (default 2.0)
    exit_threshold   : float     — z-score exit level (default 0.5)
    cost_bps         : float     — round-trip transaction cost in basis points

    Returns
    -------
    pd.Series — daily P&L
    """
    cost     = cost_bps / 10_000
    z        = (spread - formation_mean) / formation_std
    position = 0
    pnl      = []

    for i in range(1, len(spread)):
        daily_pnl = position * (spread.iloc[i] - spread.iloc[i - 1])

        if position == 0:
            if z.iloc[i] < -entry_threshold:
                position   = 1
                daily_pnl -= cost * abs(spread.iloc[i])
            elif z.iloc[i] > entry_threshold:
                position   = -1
                daily_pnl -= cost * abs(spread.iloc[i])
        elif position == 1 and z.iloc[i] > -exit_threshold:
            daily_pnl -= cost * abs(spread.iloc[i])
            position   = 0
        elif position == -1 and z.iloc[i] < exit_threshold:
            daily_pnl -= cost * abs(spread.iloc[i])
            position   = 0

        pnl.append(daily_pnl)

    return pd.Series(pnl, index=spread.index[1:])


# ---------------------------------------------------------------------------
# Phase 3b — Tabular Q-learning RL agent (Ning & Lee, 2024)
# ---------------------------------------------------------------------------

class MeanReversionRLAgent:
    """
    Tabular Q-learning agent for mean reversion pairs trading.

    The agent encodes recent spread return directions as a discrete state,
    selects entry/hold/exit actions, and learns via the Bellman update.
    Pre-training on simulated OU paths is essential before live deployment —
    real data alone provides insufficient episodes for Q-table convergence.

    Reference: Ning & Lee (2024), Section 4, arXiv:2403.12180

    Parameters
    ----------
    lookback : int   — state window length (default 4; total states = 4^lookback)
    k        : float — return threshold for state encoding (default 0.03 = 3%)
    alpha    : float — learning rate (default 0.1)
    gamma    : float — discount factor (default 0.99)
    epsilon  : float — exploration rate during training (default 0.1)
    """

    def __init__(self, lookback: int = 4, k: float = 0.03,
                 alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.1):
        self.lookback = lookback
        self.k        = k
        self.alpha    = alpha
        self.gamma    = gamma
        self.epsilon  = epsilon
        # Q[state, position, action_index]
        # position: 0 = flat, 1 = long
        # action_index: 0 = sell/close, 1 = hold, 2 = buy/open
        self.Q          = np.zeros((4 ** lookback, 2, 3))
        self.action_map = {0: -1, 1: 0, 2: 1}

    def _encode_state(self, returns: np.ndarray) -> int:
        """
        Encode recent spread returns as a base-4 integer state index.

        Each return is discretised into four levels:
            +2 (index 3) : return > +k
            +1 (index 2) : 0 < return <= +k
            -1 (index 1) : -k <= return < 0
            -2 (index 0) : return < -k
        """
        encoded = 0
        for r in returns[-self.lookback:]:
            if r > self.k:
                d = 3
            elif r > 0:
                d = 2
            elif r > -self.k:
                d = 1
            else:
                d = 0
            encoded = encoded * 4 + d
        return encoded

    def _valid_actions(self, position: int) -> list:
        """Return valid action indices for the current position."""
        return [1, 2] if position == 0 else [0, 1]

    def select_action(self, state: int, position: int,
                      training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        valid = self._valid_actions(position)
        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid)
        masked = np.full(3, -np.inf)
        for a in valid:
            masked[a] = self.Q[state, position, a]
        return int(np.argmax(masked))

    def update(self, state: int, position: int, action_idx: int,
               reward: float, next_state: int, next_position: int):
        """Q-learning Bellman update."""
        valid_next = self._valid_actions(next_position)
        max_q_next = max(self.Q[next_state, next_position, a]
                         for a in valid_next)
        target = reward + self.gamma * max_q_next
        self.Q[state, position, action_idx] += (
            self.alpha * (target - self.Q[state, position, action_idx])
        )

    def train_on_simulated_paths(self,
                                  n_paths: int = 2000,
                                  T: int = 252,
                                  mu: float = 5.0,
                                  theta: float = 0.0,
                                  sigma: float = 0.1,
                                  cost: float = 0.001):
        """
        Pre-train the Q-table on simulated Ornstein-Uhlenbeck paths.

        This step is critical — Ning & Lee (2024) demonstrate that real
        market data alone provides far too few training episodes for the
        Q-table to converge to a useful policy. Simulated OU paths provide
        the breadth of experience the agent needs before deployment.

        Parameters
        ----------
        n_paths : int   — number of simulated paths (default 2000)
        T       : int   — path length in bars (default 252 = 1 trading year)
        mu      : float — OU mean reversion speed (higher = faster reversion)
        theta   : float — OU long-term mean
        sigma   : float — OU volatility
        cost    : float — per-trade transaction cost in price units
        """
        dt = 1.0 / T
        for _ in range(n_paths):
            # Simulate OU path
            X    = np.zeros(T)
            X[0] = theta
            for t in range(1, T):
                X[t] = (X[t - 1]
                        + mu * (theta - X[t - 1]) * dt
                        + sigma * np.random.randn() * np.sqrt(dt))

            returns  = np.diff(X) / (np.abs(X[:-1]) + 1e-8)
            position = 0

            for t in range(self.lookback, T - 1):
                state      = self._encode_state(returns[t - self.lookback:t])
                action_idx = self.select_action(state, position, training=True)
                action     = self.action_map[action_idx]

                if action == 1 and position == 0:
                    position = 1
                elif action == -1 and position == 1:
                    position = 0

                reward     = action * (theta - X[t]) - cost * abs(action)
                next_rets  = returns[t - self.lookback + 1:t + 1]
                next_state = self._encode_state(next_rets)
                self.update(state, position, action_idx,
                            reward, next_state, position)

    def trade(self, spread: pd.Series,
              theta: float,
              cost: float = 0.001) -> pd.Series:
        """
        Deploy trained agent on real spread data (epsilon = 0, pure exploitation).

        Parameters
        ----------
        spread : pd.Series — spread time series (trading period)
        theta  : float     — long-term mean of the spread
        cost   : float     — per-trade transaction cost in price units

        Returns
        -------
        pd.Series — daily P&L
        """
        values   = spread.values
        returns  = np.diff(values) / (np.abs(values[:-1]) + 1e-8)
        position = 0
        pnl      = []

        for t in range(self.lookback, len(returns)):
            state      = self._encode_state(returns[t - self.lookback:t])
            action_idx = self.select_action(state, position, training=False)
            action     = self.action_map[action_idx]

            if action == 1 and position == 0:
                position = 1
            elif action == -1 and position == 1:
                position = 0

            daily_pnl  = position * (values[t + 1] - values[t])
            daily_pnl -= cost * abs(action) * abs(values[t])
            pnl.append(daily_pnl)

        return pd.Series(pnl)


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

def performance_summary(pnl: pd.Series,
                         label: str = "Strategy") -> pd.DataFrame:
    """
    Compute standard performance metrics for a daily P&L series.

    Returns a single-row DataFrame for easy comparison across strategies.
    """
    r        = pnl.dropna()
    ann_ret  = r.mean() * 252
    ann_vol  = r.std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum      = (1 + r).cumprod()
    max_dd   = float((cum / cum.cummax() - 1).min())
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    n_trades = int((r != 0).sum())

    return pd.DataFrame([{
        "Label":           label,
        "Ann. Return":     f"{ann_ret:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Max Drawdown":    f"{max_dd:.2%}",
        "Calmar Ratio":    f"{calmar:.3f}",
        "Approx. Trades":  n_trades,
    }]).set_index("Label")


# ---------------------------------------------------------------------------
# Full pipeline — end-to-end FX pairs backtest
# ---------------------------------------------------------------------------

def run_fx_pairs_backtest(ticker1: str          = "AUDUSD=X",
                           ticker2: str          = "NZDUSD=X",
                           formation_start: str  = "2022-01-01",
                           formation_end: str    = "2022-12-31",
                           trading_start: str    = "2023-01-01",
                           trading_end: str      = "2023-12-31",
                           target_half_life_min: int = 5,
                           target_half_life_max: int = 60,
                           entry_threshold: float = 2.0,
                           exit_threshold: float  = 0.5,
                           cost_bps: float         = 5.0,
                           rl_n_paths: int         = 2000,
                           verbose: bool           = True) -> dict:
    """
    End-to-end FX pairs trading backtest.

    Runs all three phases: cointegration testing, EMRT-based spread
    construction, classical z-score baseline, and RL agent comparison.
    Downloads data automatically via yfinance.

    Common Yahoo Finance FX tickers:
        AUDUSD=X, NZDUSD=X, EURUSD=X, GBPUSD=X, USDCAD=X, USDNOK=X

    Parameters
    ----------
    ticker1 / ticker2        : str   — Yahoo Finance ticker symbols
    formation_start/end      : str   — formation period (YYYY-MM-DD)
    trading_start/end        : str   — trading period (YYYY-MM-DD)
    target_half_life_min/max : int   — half-life filter in trading days
    entry_threshold          : float — z-score entry level
    exit_threshold           : float — z-score exit level
    cost_bps                 : float — round-trip transaction cost (bps)
    rl_n_paths               : int   — RL pre-training simulation paths
    verbose                  : bool  — print progress and results

    Returns
    -------
    dict with keys: coint_result, emrt_result, classical_pnl, rl_pnl,
                    trading_spread, performance
    """
    # --- Data download ---
    raw1 = yf.download(ticker1, start=formation_start, end=trading_end,
                       auto_adjust=True, progress=False)
    raw2 = yf.download(ticker2, start=formation_start, end=trading_end,
                       auto_adjust=True, progress=False)

    p1, p2 = raw1["Close"].squeeze(), raw2["Close"].squeeze()
    p1, p2 = p1.align(p2, join="inner")

    form1 = p1[formation_start:formation_end]
    form2 = p2[formation_start:formation_end]
    trad1 = p1[trading_start:trading_end]
    trad2 = p2[trading_start:trading_end]

    # --- Phase 1: Cointegration ---
    coint_result = test_cointegration(form1, form2)
    if verbose:
        print(f"\nPhase 1 — Cointegration")
        print(f"  p-value      : {coint_result['p_value']}")
        print(f"  Cointegrated : {coint_result['cointegrated']}")
        print(f"  Half-life    : {coint_result['half_life']} days")

        hl = coint_result["half_life"]
        if not coint_result["cointegrated"]:
            print("  WARNING: Pair not cointegrated at 95% — proceed with caution.")
        elif hl < target_half_life_min or hl > target_half_life_max:
            print(f"  WARNING: Half-life {hl}d outside target range "
                  f"[{target_half_life_min}, {target_half_life_max}].")

    # --- Phase 2: EMRT spread construction ---
    if verbose:
        print(f"\nPhase 2 — EMRT Spread Construction")
    emrt_result    = find_optimal_hedge_ratio(form1, form2)
    beta           = emrt_result["beta"]
    formation_mean = emrt_result["spread_mean"]
    formation_std  = emrt_result["spread_std"]
    trading_spread = trad1 - beta * trad2

    if verbose:
        print(f"  Optimal beta : {beta}")
        print(f"  EMRT         : {emrt_result['emrt']} days")

    # --- Phase 3a: Classical z-score baseline ---
    if verbose:
        print(f"\nPhase 3a — Classical Z-Score Strategy")
    classical_pnl = classical_zscore_strategy(
        trading_spread, formation_mean, formation_std,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        cost_bps=cost_bps,
    )

    # --- Phase 3b: RL agent ---
    if verbose:
        print(f"\nPhase 3b — RL Agent (training on {rl_n_paths} simulated paths...)")
    agent            = MeanReversionRLAgent(lookback=4, k=0.03)
    formation_spread = form1 - beta * form2
    theta            = float(formation_spread.mean())
    agent.train_on_simulated_paths(n_paths=rl_n_paths, mu=5.0, theta=theta)
    rl_pnl = agent.trade(trading_spread, theta=theta,
                         cost=cost_bps / 10_000 * abs(trading_spread.mean()))

    # --- Performance summary ---
    perf = pd.concat([
        performance_summary(classical_pnl, label="Classical Z-Score"),
        performance_summary(rl_pnl,        label="RL Agent (Q-Learning)"),
    ])

    if verbose:
        print(f"\n--- Performance Summary ---")
        print(perf.to_string())

    return {
        "coint_result":   coint_result,
        "emrt_result":    emrt_result,
        "classical_pnl":  classical_pnl,
        "rl_pnl":         rl_pnl,
        "trading_spread": trading_spread,
        "performance":    perf,
    }


# ---------------------------------------------------------------------------
# Entry point — quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Strategy Lab #2 — FX Pairs Mean Reversion + RL")
    print("Algorithmic Token · ENTER Invest")
    print("=" * 60)
    print()
    print("Pair: AUD/USD — NZD/USD")
    print("Formation: 2022  |  Trading: 2023")

    results = run_fx_pairs_backtest(
        ticker1         = "AUDUSD=X",
        ticker2         = "NZDUSD=X",
        formation_start = "2022-01-01",
        formation_end   = "2022-12-31",
        trading_start   = "2023-01-01",
        trading_end     = "2023-12-31",
        cost_bps        = 5.0,
        rl_n_paths      = 2000,
        verbose         = True,
    )


# ---------------------------------------------------------------------------
# Risk Disclosure
# ---------------------------------------------------------------------------
# The strategies and implementations in this file are experimental and
# provided for educational and research purposes only. Past performance
# is not indicative of future results. All algorithmic trading carries
# significant financial risk, including the potential total loss of capital.
# Nothing here constitutes financial advice. ENTER Invest does not manage
# client funds based on strategies described here unless explicitly contracted.
# ---------------------------------------------------------------------------
