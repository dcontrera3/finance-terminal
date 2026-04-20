"""
Backtest de pullback diario en la canasta ampliada.
Estrategia: LONG cuando precio en tendencia alcista (sobre EMA200)
retrocede a la zona entre EMA 20 y EMA 50, con RSI medio (40-65).
Captura continuación de tendencia sin pelear contra ella.
"""

from datetime import datetime, timedelta
from backtester import backtest, print_summary_table, portfolio_stats

TICKERS = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO',
    'NU', 'MELI', 'BABA',
    'SPY', 'QQQ', 'IWM',
    'GLD', 'SLV', 'XLE',
    'VIST',
]

GRID = {
    'baseline':       dict(rsi_min=40, rsi_max=65, pullback_pct=1.02, vol_min=0.8),
    'rsi_amplio':     dict(rsi_min=35, rsi_max=70, pullback_pct=1.02, vol_min=0.8),
    'rsi_estricto':   dict(rsi_min=45, rsi_max=60, pullback_pct=1.02, vol_min=1.0),
    'pullback_prof':  dict(rsi_min=40, rsi_max=65, pullback_pct=1.05, vol_min=0.8),
    'vol_alto':       dict(rsi_min=40, rsi_max=65, pullback_pct=1.02, vol_min=1.2),
}

ATR_MULT_SL = 1.5
RR_RATIO    = 2.0


def run(label, start, end, params):
    print(f"\n╔══ PULLBACK DIARIO — {label}  |  {start} → {end} ══╗")
    print(f"   params: {params}")
    print(f"   atr_mult_sl={ATR_MULT_SL}  rr={RR_RATIO}")

    results = []
    for t in TICKERS:
        r = backtest(t, start=start, end=end,
                     strategy='pullback',
                     timeframe='1d',
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
    portfolio_stats(results, ann_factor=252)
    return results


def main():
    end_d   = datetime.now().strftime('%Y-%m-%d')
    # 5 años para muestra amplia
    start_d = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

    for label, params in GRID.items():
        run(label, start_d, end_d, params)


if __name__ == '__main__':
    main()
