"""
Backtest de mean reversion en 1h sobre la canasta.
Probamos varios sets de parámetros para encontrar edge real.
"""

from datetime import datetime, timedelta
from backtester import backtest, print_summary_table, portfolio_stats

TICKERS = ['NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT',
           'AMZN', 'GLD', 'SPY', 'QQQ', 'VIST']

GRID = {
    'baseline':           dict(rsi_low=30, rsi_high=70, bb_std=2.0, trend_filter=True,  vol_max=1.5),
    'estricto':           dict(rsi_low=25, rsi_high=75, bb_std=2.0, trend_filter=True,  vol_max=1.5),
    'muy_estricto':       dict(rsi_low=20, rsi_high=80, bb_std=2.5, trend_filter=True,  vol_max=1.5),
    'sin_trend_filter':   dict(rsi_low=30, rsi_high=70, bb_std=2.0, trend_filter=False, vol_max=1.5),
    'bb_2_5':             dict(rsi_low=30, rsi_high=70, bb_std=2.5, trend_filter=True,  vol_max=1.5),
    'sin_vol_filter':     dict(rsi_low=30, rsi_high=70, bb_std=2.0, trend_filter=True,  vol_max=99),
}

# Mean reversion necesita target más cortos
ATR_MULT_SL = 1.5   # stop un poco más lejos (el precio puede seguir extendiendo el extremo)
RR_RATIO    = 1.0   # target 1:1 — la reversión suele ser corta


def run(label, start, end, params):
    print(f"\n╔══ MEAN REVERSION 1h — {label}  |  {start} → {end} ══╗")
    print(f"   params: {params}")
    print(f"   atr_mult_sl={ATR_MULT_SL}  rr={RR_RATIO}")

    results = []
    for t in TICKERS:
        r = backtest(t, start=start, end=end,
                     strategy='mean_reversion',
                     timeframe='1h',
                     signal_params=params,
                     atr_stop_mult=ATR_MULT_SL,
                     rr_ratio=RR_RATIO)
        if r:
            results.append(r)
        else:
            print(f"   {t}: sin data suficiente")

    if not results:
        return None

    print_summary_table(results)
    portfolio_stats(results, ann_factor=1638)
    return results


def main():
    end_d   = datetime.now().strftime('%Y-%m-%d')
    start_d = (datetime.now() - timedelta(days=715)).strftime('%Y-%m-%d')

    for label, params in GRID.items():
        run(label, start_d, end_d, params)


if __name__ == '__main__':
    main()
