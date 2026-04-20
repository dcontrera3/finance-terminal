"""
Diagnóstico rápido: para cada ticker, mira los últimos 60 días y cuenta
cuántas barras tocaron cada filtro de la estrategia swing.
Identifica cuál filtro está ahogando la señal.
"""

from datetime import datetime, timedelta
from backtester import fetch, add_indicators

TICKERS = ['NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT',
           'AMZN', 'GLD', 'SPY', 'QQQ', 'VIST']

PARAMS = dict(periods=8, vol_spike=1.1, adx_min=20,
              rsi_long_max=70, rsi_short_min=35)

LOOKBACK_DAYS = 60


def diagnose(ticker):
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    df = fetch(ticker, start, end, interval='1d')
    if df.empty or len(df) < 50:
        return None

    df = add_indicators(df)
    df = df.dropna().tail(LOOKBACK_DAYS)

    c = df['Close']
    h = df['High']
    l = df['Low']

    high_n = h.rolling(PARAMS['periods']).max().shift(1)
    low_n  = l.rolling(PARAMS['periods']).min().shift(1)

    breakout_up   = c > high_n
    breakout_down = c < low_n
    vol_ok        = df['vol_ratio'] > PARAMS['vol_spike']
    adx_ok        = df['adx']       > PARAMS['adx_min']
    rsi_long_ok   = df['rsi']       < PARAMS['rsi_long_max']
    rsi_short_ok  = df['rsi']       > PARAMS['rsi_short_min']

    long_sig  = (breakout_up   & vol_ok & adx_ok & rsi_long_ok).sum()
    short_sig = (breakout_down & vol_ok & adx_ok & rsi_short_ok).sum()

    return {
        'price':   float(c.iloc[-1]),
        'adx':     float(df['adx'].iloc[-1]),
        'rsi':     float(df['rsi'].iloc[-1]),
        'vol_r':   float(df['vol_ratio'].iloc[-1]),
        # porcentaje de barras que pasaron cada filtro por separado
        'bkout_up_pct':   breakout_up.mean()   * 100,
        'bkout_dn_pct':   breakout_down.mean() * 100,
        'vol_ok_pct':     vol_ok.mean()        * 100,
        'adx_ok_pct':     adx_ok.mean()        * 100,
        # señales completas generadas
        'long_signals':   int(long_sig),
        'short_signals':  int(short_sig),
    }


def main():
    print(f"\n{'Ticker':<7} {'Precio':>8} {'ADX':>6} {'RSI':>6} {'VolR':>6}"
          f" {'BkUp%':>7} {'BkDn%':>7} {'Vol>1.1%':>9} {'ADX>20%':>9}"
          f" {'L-sig':>6} {'S-sig':>6}")
    print("─" * 95)

    totals = {'long': 0, 'short': 0}
    for t in TICKERS:
        r = diagnose(t)
        if not r:
            print(f"{t:<7}  sin datos")
            continue
        print(f"{t:<7} {r['price']:>8.2f} {r['adx']:>6.1f} {r['rsi']:>6.1f} "
              f"{r['vol_r']:>6.2f} {r['bkout_up_pct']:>7.1f} {r['bkout_dn_pct']:>7.1f} "
              f"{r['vol_ok_pct']:>9.1f} {r['adx_ok_pct']:>9.1f} "
              f"{r['long_signals']:>6} {r['short_signals']:>6}")
        totals['long']  += r['long_signals']
        totals['short'] += r['short_signals']

    print("─" * 95)
    print(f"Últimos {LOOKBACK_DAYS} días de trading | TOTAL señales LONG={totals['long']}  SHORT={totals['short']}")
    print("\nLectura:")
    print("  BkUp/BkDn%: % de días que el precio cerró sobre/bajo el rango de 8 días")
    print("  Vol>1.1%:   % de días con volumen 1.1x su promedio de 20 días")
    print("  ADX>20%:    % de días con tendencia confirmada")
    print("  Si todos los filtros están altos pero las señales son bajas,")
    print("  es porque los filtros NO se superponen (no coinciden en el mismo día).\n")


if __name__ == '__main__':
    main()
