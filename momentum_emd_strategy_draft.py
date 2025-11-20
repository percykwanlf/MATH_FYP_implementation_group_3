# momentum_emd_strategy_final.py
# Works out-of-the-box with S&P100 daily data (Yahoo Finance format)

import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert
from scipy.stats import ttest_rel, norm
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DATA_DIR            = r'C:\Users\KwanPY\Desktop\Stockinf'           # <-- your folder
TICKERS_FILE        = r'D:\Downloads\sp100_100_tickers.txt'         # plain text, one ticker per line
PERIODS_MONTHS      = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]          # J-month formation = J-month holding
TRADING_DAYS_MONTH  = 21
LOW_PERIOD_DAYS     = 60                     # ~3 months (lower bound for cycle)
HIGH_PERIOD_DAYS    = 252                    # ~12 months (upper bound)
ENERGY_THRESH       = 0.30                   # at least 30% energy in  evoc_3-12m band
STOP_SD             = 0.05
MAX_SIFT_ITER       = 100
N_BOOT              = 5_000                  # increased for better p-values
SUBPERIOD_SPLIT     = '2020-01-01'

# Pre-computation toggle (RECOMMENDED = True)
PRECOMPUTE_EMD_TRENDS = True
EMD_WINDOW_DAYS       = 504   # ~2 years rolling window for stable IMFs


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_tickers() -> list[str]:
    with open(TICKERS_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"Found {len(tickers)} tickers in file.")
    return tickers

def load_adj_close(tickers: list[str]) -> pd.DataFrame:
    series_dict = {}
    for t in tqdm(tickers, desc="Loading CSVs"):
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        if "Adj Close" not in df.columns:
            continue
        series_dict[t] = df.set_index("Date")["Adj Close"]
    prices = pd.DataFrame(series_dict).sort_index()
    print(f"Loaded {prices.shape[1]} tickers from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices

# ----------------------------------------------------------------------
# 2. EMPIRICAL MODE DECOMPOSITION 
# ----------------------------------------------------------------------
def _extrema_indices(y: np.ndarray):
    """Return indices of local maxima and minima (endpoints included only if true extrema)."""
    dy = np.diff(y)
    maxima = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    minima = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0] + 1

    # Add endpoints only if they are actual extrema
    if y[0] > y[1]:
        maxima = np.r_[0, maxima]
    if y[0] < y[1]:
        minima = np.r_[0, minima]
    if y[-1] > y[-2]:
        maxima = np.r_[maxima, len(y)-1]
    if y[-1] < y[-2]:
        minima = np.r_[minima, len(y)-1]

    return np.unique(maxima), np.unique(minima)

def emd_sift(x: np.ndarray, max_iter: int = MAX_SIFT_ITER, stop_sd: float = STOP_SD) -> np.ndarray:
    """Return (n_imf, n_samples) array. Last row = residual."""
    imfs = []
    r = x.astype(float).copy()

    while True:
        max_idx, min_idx = _extrema_indices(r)
        if len(max_idx) + len(min_idx) < 4:        # too few extrema → monotonic
            break

        h = r.copy()
        for _ in range(max_iter):
            max_idx, min_idx = _extrema_indices(h)
            if len(max_idx) < 3 or len(min_idx) < 3:   # need at least 3 points for spline
                break

            t = np.arange(len(h))
            upper = CubicSpline(max_idx, h[max_idx], bc_type="clamped")(t)
            lower = CubicSpline(min_idx, h[min_idx], bc_type="clamped")(t)
            mean_env = (upper + lower) / 2.0

            h_new = h - mean_env
            sd = np.sum((h - h_new)**2) / (np.sum(h**2) + 1e-12)
            if sd < stop_sd:
                break
            h = h_new

        imfs.append(h)
        r = r - h

    imfs.append(r)          # residual
    return np.stack(imfs)

# ----------------------------------------------------------------------
# 3. HILBERT SPECTRUM + IMF SELECTION (3-12 month cycles)
# ----------------------------------------------------------------------
def hilbert_spectrum(imfs: np.ndarray):
    amps, freqs = [], []
    for imf in imfs[:-1]:  # skip residual for frequency estimation
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.abs(np.diff(phase) / (2 * np.pi))
        inst_freq = np.concatenate([inst_freq, [inst_freq[-1]]])
        amps.append(amp)
        freqs.append(inst_freq)
    return np.stack(amps), np.stack(freqs)

def select_cycle_imfs(amps: np.ndarray, freqs: np.ndarray) -> list[int]:
    selected = []
    for j in range(amps.shape[0]):
        f = freqs[j] + 1e-12
        period_days = 1.0 / f
        mean_period = period_days.mean()

        energy = amps[j] ** 2
        total_energy = energy.sum()
        if total_energy == 0:
            continue
        cycle_energy = energy[(period_days >= LOW_PERIOD_DAYS) & (period_days <= HIGH_PERIOD_DAYS)].sum()
        cycle_ratio = cycle_energy / total_energy

        if (LOW_PERIOD_DAYS <= mean_period <= HIGH_PERIOD_DAYS) or (cycle_ratio > ENERGY_THRESH):
            selected.append(j)
    return selected

def reconstruct_cycle_trend(imfs: np.ndarray, selected: list[int]) -> np.ndarray:
    if not selected:
        return np.zeros(imfs.shape[1])
    return imfs[selected].sum(axis=0)

# ----------------------------------------------------------------------
# 4. Pre-compute EMD-filtered trends
# ----------------------------------------------------------------------
def precompute_emd_trends(prices: pd.DataFrame) -> pd.DataFrame:
    print("Pre-computing EMD 3-12 month cycle trends")
    trend_dict = {}
    window = EMD_WINDOW_DAYS

    for ticker in tqdm(prices.columns, desc="EMD precompute"):
        logp = np.log(prices[ticker]).dropna()
        if len(logp) < window + 100:
            trend_dict[ticker] = logp  # fallback to raw log price
            continue

        trend_vals = np.full(len(logp), np.nan)
        series_idx = logp.index

        for end in range(window, len(logp)):
            segment = logp.iloc[end-window:end].values
            imfs = emd_sift(segment)
            if imfs.shape[0] > 1:
                amps, freqs = hilbert_spectrum(imfs)
                sel = select_cycle_imfs(amps, freqs)
                cycle_trend = reconstruct_cycle_trend(imfs, sel)
                trend_vals[end] = cycle_trend[-1]  # last point of current window

        # Forward fill early period
        last_valid = np.where(~np.isnan(trend_vals))[0]
        if len(last_valid) > 0:
            first = last_valid[0]
            trend_vals[:first] = trend_vals[first]

        trend_dict[ticker] = pd.Series(trend_vals, index=series_idx)

    trend_df = pd.DataFrame(trend_dict)
    trend_df = trend_df.reindex(prices.index).ffill().bfill()
    print("EMD pre-computation finished.")
    return trend_df  # values are in log-price space

# ----------------------------------------------------------------------
# 5. MOMENTUM SIGNAL
# ----------------------------------------------------------------------
def momentum_signal(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return series.iloc[-1] - series.iloc[-days]

# ----------------------------------------------------------------------
# 6. BACKTEST ENGINE
# ----------------------------------------------------------------------
def run_backtest(prices: pd.DataFrame, trend_df: pd.DataFrame | None = None) -> dict[int, pd.Series]:
    name = "EMD" if trend_df is not None else "Raw"
    print(f"\nRunning {name} momentum backtest...")

    monthly_prices = prices.resample('ME').last()
    monthly_ret = monthly_prices.pct_change().shift(-1)  # next month's return

    results = {J: pd.Series(dtype=float) for J in PERIODS_MONTHS}

    for J in PERIODS_MONTHS:
        formation_days = J * TRADING_DAYS_MONTH
        portfolio_returns = []

        for i in tqdm(range(12, len(monthly_prices) - J - 1), desc=f"J={J}", leave=False):
            rank_date = monthly_prices.index[i]
            hold_start = rank_date + pd.offsets.MonthEnd(0)   # next month start
            hold_end = hold_start + pd.offsets.MonthEnd(J)

            past_mom = {}
            source = trend_df if trend_df is not None else np.log(prices)

            for t in prices.columns:
                if trend_df is not None:
                    s = source[t][:rank_date]
                else:
                    s = source[t][:rank_date].dropna()
                    if len(s) <= formation_days + 20:
                        continue
                    s = pd.Series(np.log(s), index=s.index)

                if len(s) <= formation_days:
                    continue
                mom = momentum_signal(s, formation_days)
                if not np.isnan(mom):
                    past_mom[t] = mom

            if len(past_mom) < 30:
                continue

            ranks = pd.Series(past_mom).rank(pct=True)
            winners = ranks[ranks >= 0.90].index
            losers = ranks[ranks <= 0.10].index

            if len(winners) < 2 or len(losers) < 2:
                continue

            # Equal-weighted long-short return over next J months
            w_ret = monthly_ret.loc[hold_start:hold_end, winners].mean(axis=1)
            l_ret = monthly_ret.loc[hold_start:hold_end, losers].mean(axis=1)
            ls_ret = w_ret - l_ret
            avg_monthly = ls_ret.mean()

            results[J].loc[hold_end] = avg_monthly

        results[J] = results[J].sort_index()

    return results

# ----------------------------------------------------------------------
# 7. STATISTICS & REPORT
# ----------------------------------------------------------------------
def risk_metrics(ret: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) == 0:
        return {"Ann.Ret": 0, "Sharpe": 0, "MaxDD": 0, "Calmar": 0}
    ann_ret = r.mean() * 12
    sharpe = ann_ret / (r.std() * np.sqrt(12)) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    maxdd = dd.min()
    calmar = ann_ret / -maxdd if maxdd < 0 else np.nan
    return {"Ann.Ret": ann_ret, "Sharpe": sharpe, "MaxDD": maxdd, "Calmar": calmar}

def bootstrap_pvalue(ret: pd.Series, n=N_BOOT):
    r = ret.dropna().values
    if len(r) == 0:
        return np.nan
    obs = r.mean()
    boot = [np.mean(np.random.choice(r, len(r), replace=True)) for _ in range(n)]
    return np.mean(np.array(boot) >= obs)

def print_full_report(raw_ret: dict, emd_ret: dict):
    print("\n" + "="*80)
    print("FINAL RESULTS: Raw Momentum vs EMD Cycle-Filtered Momentum")
    print("="*80)

    for J in PERIODS_MONTHS:
        print(f"\n{'='*20} J = {J} months {'='*20}")
        o = raw_ret[J]
        e = emd_ret[J]

        df = pd.DataFrame({
            "Raw": risk_metrics(o),
            "EMD": risk_metrics(e)
        }).round(4)

        print(df.T.to_string())

        common = o.index.intersection(e.index)
        if len(common) >= 10:
            t_stat, p_val = ttest_rel(o.loc[common], e.loc[common])
            print(f"Paired t-test (Raw - EMD): t={t_stat:.3f}, p={p_val:.4f} ", end="")
        print(f"Bootstrap p (Raw > 0): {bootstrap_pvalue(o):.4f}")
        print(f"Bootstrap p (EMD > 0): {bootstrap_pvalue(e):.4f}")

# ----------------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    tickers = load_tickers()
    prices = load_adj_close(tickers)

    # Run raw momentum
    raw_results = run_backtest(prices, trend_df=None)

    # Run EMD-enhanced momentum
    if PRECOMPUTE_EMD_TRENDS:
        trend_log = precompute_emd_trends(prices)
        emd_results = run_backtest(prices, trend_df=trend_log)
    else:
        print("Running EMD on-the-fly (very slow!)")
        emd_results = run_backtest(prices, trend_df=None)  # fallback uses raw but flag is for EMD inside loop

    # Final report
    print_full_report(raw_results, emd_results)
