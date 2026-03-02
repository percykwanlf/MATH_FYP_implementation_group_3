### EMD-Enhanced Momentum Strategy: S&P 100 Backtest

This repository contains the Python implementation and final report for a **Mathematics Final Year Project (MATH4900C)**. The study investigates whether **Empirical Mode Decomposition (EMD)** can improve classical momentum trading strategies by filtering out high-frequency market noise.

---

## 📈 Project Overview

The project evaluates the **Jegadeesh & Titman (1993)** Momentum Strategy and the **George & Hwang (2004)** 52-Week High Strategy within the S&P 100 universe over a 10-year period.

### Core Methodology

* 
**EMD Filtering**: Daily log prices are decomposed into Intrinsic Mode Functions (IMFs). Short-term noise (IMF1 and IMF2) is removed to reconstruct a "cleaner" trend for signal generation.


* 
**Hilbert Spectrum**: Used to calculate instantaneous frequencies and identify cycles within the 3–12 month range.


* 
**Zero-Cost Portfolios**: The strategy ranks stocks by past returns, buying the top 10% ("Winners") and selling the bottom 10% ("Losers").



---

## 📊 Performance: Long-Short Strategy

The primary experiment tested a **Long-Short (Winner minus Loser)** approach across various holding periods ($J=K=3$ to 12 months).

### Key Statistical Results (Raw vs. EMD)

The following data represents the performance of the zero-cost (long-short) momentum portfolio:

| Period (J) | Strategy | APY | Sharpe Ratio | Max Drawdown | Calmar Ratio |
| --- | --- | --- | --- | --- | --- |
| **3 Month** | Raw | 9.30% | 0.8725 | -17.92% | 0.5189 |
|  | EMD | 6.32% | 0.6920 | -24.49% | 0.2581 |
| **6 Month** | Raw | 1.19% | 0.1363 | -37.65% | 0.0315 |
|  | EMD | 2.67% | 0.3462 | -31.90% | 0.0838 |
| **12 Month** | Raw | -2.46% | -0.3695 | -41.07% | -0.0600 |
|  | EMD | -0.51% | -0.0935 | -34.20% | -0.0148 |

### Conclusions on Long-Short Approach

* 
**No Significant Superiority**: Paired t-tests showed no statistically meaningful difference between Raw and EMD-filtered long-short approaches.


* 
**High Risk Exposure**: Both long-short implementations experienced significant drawdowns (up to 43%) and poor risk-adjusted returns.


* 
**Market Context**: The strategy struggled because the S&P 100 constituents performed exceptionally well over the last 10 years, making shorting the "Loser" portfolio disadvantageous.



---

## 🛠 Features

* 
**Adaptive Decomposition**: Custom sifting process using cubic splines for envelope interpolation.


* 
**Rolling Window Analysis**: EMD trends are precomputed using a 504-day (~2 year) rolling window to ensure stability.


* 
**Statistical Validation**: Includes automated paired t-tests and bootstrap p-value calculations to determine significance.



---

## 📂 Repository Structure

* 
`main.py`: The core script for data loading, EMD decomposition, and backtesting.


* 
`data/`: Directory for S&P 100 ticker CSV files (Yahoo Finance format).


* 
`final_report.pdf`: Detailed mathematical analysis and methodology.



---

## 📝 References


1. Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng, Q., ... & Liu, H. H. (1998). The empirical
mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. Proceedings
of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences, 454(1971), 903-995.

2. Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market
e”ciency. The Journal of Finance, 48(1), 65-91.

4. George, T. J., & Hwang, C.-Y. (2004). The 52-week high and momentum investing. The Journal of Finance,
59(5), 2145-2176. 
