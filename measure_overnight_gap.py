"""
Mide cuántos días tuvieron gap overnight relevante (close[t] → open[t+1])
relativo al ATR del día. Sirve para evaluar el riesgo de que el bot decida
una señal a las 15:30 ET y la orden se ejecute en la apertura del día siguiente
con un precio significativamente distinto.

Output por ticker:
  - Total días observados
  - % días con |gap| > 0.5 ATR
  - % días con |gap| > 1.0 ATR
  - % días con |gap| > 2.0 ATR
  - Gap promedio absoluto en ATRs
"""

from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO',
    'NU', 'MELI', 'BABA',
    'SPY', 'QQQ', 'IWM',
    'GLD', 'SLV', 'XLE',
    'VIST',
]

YEARS = 2


def atr(df, n=14):
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def analyze(ticker):
    end = datetime.now()
    start = end - timedelta(days=YEARS * 365 + 30)
    df = yf.download(ticker, start=start, end=end, interval='1d',
                     progress=False, auto_adjust=False)
    if df.empty or len(df) < 30:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=['Open', 'Close', 'High', 'Low']).copy()
    df['atr14'] = atr(df, 14)
    df['gap_abs'] = (df['Open'] - df['Close'].shift(1)).abs()
    df['gap_atr'] = df['gap_abs'] / df['atr14'].shift(1)
    df = df.dropna(subset=['gap_atr'])

    total = len(df)
    if total == 0:
        return None

    pct_05 = (df['gap_atr'] > 0.5).mean() * 100
    pct_10 = (df['gap_atr'] > 1.0).mean() * 100
    pct_20 = (df['gap_atr'] > 2.0).mean() * 100
    pct_30 = (df['gap_atr'] > 3.0).mean() * 100
    mean_gap = df['gap_atr'].mean()
    p95 = df['gap_atr'].quantile(0.95)
    p99 = df['gap_atr'].quantile(0.99)
    max_gap = df['gap_atr'].max()

    return dict(
        ticker=ticker, total=total,
        pct_05=pct_05, pct_10=pct_10, pct_20=pct_20, pct_30=pct_30,
        mean=mean_gap, p95=p95, p99=p99, max=max_gap,
    )


def main():
    results = []
    for t in TICKERS:
        try:
            r = analyze(t)
            if r:
                results.append(r)
                print(f"{t:6s}  n={r['total']:>4}  >0.5ATR={r['pct_05']:5.1f}%  "
                      f">1ATR={r['pct_10']:5.1f}%  >2ATR={r['pct_20']:4.1f}%  "
                      f">3ATR={r['pct_30']:4.1f}%  mean={r['mean']:.2f}  "
                      f"p95={r['p95']:.2f}  p99={r['p99']:.2f}  max={r['max']:.1f}")
        except Exception as e:
            print(f"{t}: error {e}")

    if not results:
        return

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("AGREGADO (canasta de 24 tickers, ~2 años):")
    print("=" * 80)
    n_total = df['total'].sum()
    print(f"Total días-ticker observados: {n_total}")
    # promedio ponderado por # observaciones
    for col, label, threshold in [
        ('pct_05', 'gap > 0.5 ATR', 0.5),
        ('pct_10', 'gap > 1.0 ATR', 1.0),
        ('pct_20', 'gap > 2.0 ATR', 2.0),
        ('pct_30', 'gap > 3.0 ATR', 3.0),
    ]:
        weighted = (df[col] * df['total']).sum() / n_total
        print(f"  {label:18s}  {weighted:5.2f}% de los días "
              f"(unos {weighted/100*n_total:.0f} eventos sobre {n_total})")
    print(f"  Gap medio absoluto:  {(df['mean'] * df['total']).sum() / n_total:.3f} ATRs")
    print(f"\nWorst tickers por % de días con gap > 1 ATR:")
    print(df.sort_values('pct_10', ascending=False)[['ticker', 'pct_10', 'pct_20', 'p99', 'max']].head(8).to_string(index=False))
    print(f"\nBest tickers (más estables overnight):")
    print(df.sort_values('pct_10')[['ticker', 'pct_10', 'pct_20', 'p99', 'max']].head(5).to_string(index=False))


if __name__ == '__main__':
    main()
