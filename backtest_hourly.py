"""
Backtest exploratorio de la estrategia swing en timeframe 1h.
Compara con el resultado diario para decidir si vale la pena migrar.

yfinance limita a 730 días de data 1h, así que arrancamos 2 años atrás.
"""

from datetime import datetime, timedelta
from backtester import backtest, print_summary_table, portfolio_stats

TICKERS = ['NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT',
           'AMZN', 'GLD', 'SPY', 'QQQ', 'VIST']

# Parámetros actuales del bot (optimizados para diario)
DAILY_PARAMS = dict(periods=8, vol_spike=1.1, adx_min=20,
                    rsi_long_max=70, rsi_short_min=35)

# Sets de prueba para 1h (ajustados proporcionalmente)
HOURLY_GRID = {
    'params_daily_clone':  dict(periods=8,  vol_spike=1.1, adx_min=20, rsi_long_max=70, rsi_short_min=35),
    'params_6h_lookback':  dict(periods=6,  vol_spike=1.1, adx_min=20, rsi_long_max=70, rsi_short_min=35),
    'params_16h_lookback': dict(periods=16, vol_spike=1.1, adx_min=20, rsi_long_max=70, rsi_short_min=35),
    'params_loose_vol':    dict(periods=8,  vol_spike=1.0, adx_min=20, rsi_long_max=70, rsi_short_min=35),
    'params_loose_adx':    dict(periods=8,  vol_spike=1.1, adx_min=15, rsi_long_max=70, rsi_short_min=35),
}


def run(label, start, end, timeframe, params, atr_mult=1.0, rr=1.5):
    print(f"\n╔══ {label}  |  tf={timeframe}  |  {start} → {end} ══╗")
    print(f"   params: {params}")
    print(f"   atr_mult={atr_mult}  rr={rr}")

    results = []
    for t in TICKERS:
        r = backtest(t, start=start, end=end,
                     strategy='swing',
                     timeframe=timeframe,
                     signal_params=params,
                     atr_stop_mult=atr_mult,
                     rr_ratio=rr)
        if r:
            results.append(r)
        else:
            print(f"   {t}: sin data suficiente")

    if not results:
        print("   Sin resultados.")
        return None

    print_summary_table(results)
    ann = {'1d': 252, '1h': 1638, '60m': 1638, '1wk': 52}.get(timeframe, 252)
    portfolio_stats(results, ann_factor=ann)
    return results


def main():
    end_d = datetime.now().strftime('%Y-%m-%d')

    # yfinance limita 1h a 730 días, usamos 715 para margen
    start_1h = (datetime.now() - timedelta(days=715)).strftime('%Y-%m-%d')

    # Baseline diario en la misma ventana temporal para comparar apples-to-apples
    run('BASELINE DIARIO (mismo período que 1h)', start_1h, end_d, '1d', DAILY_PARAMS)

    # 1h — probamos varios sets de parámetros
    for label, params in HOURLY_GRID.items():
        run(f'1H — {label}', start_1h, end_d, '1h', params)


if __name__ == '__main__':
    main()
