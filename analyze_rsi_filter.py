"""
Analiza el costo/beneficio del filtro RSI en la estrategia swing.

Compara dos configuraciones sobre 5 años y 24 tickers:
  - A: swing actual (rsi_long_max=70, rsi_short_min=35)
  - B: swing sin filtro RSI (rsi_long_max=100, rsi_short_min=0)

Identifica los trades que B toma y A bloquea, y mide:
  - Cuántos son winners vs losers
  - PnL agregado de los trades bloqueados
  - Correlación entre ADX alto (>45) y outcome positivo
  - Si vale la pena relajar condicionalmente el filtro cuando ADX > 45

Uso: python3.10 analyze_rsi_filter.py
"""
from datetime import datetime, timedelta
import pandas as pd

from backtester import backtest, fetch, add_indicators

TICKERS = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO',
    'NU', 'MELI', 'BABA',
    'SPY', 'QQQ', 'IWM',
    'GLD', 'SLV', 'XLE',
    'VIST',
]

COMMON = dict(
    strategy='swing', timeframe='1d',
    atr_stop_mult=1.0, trailing_atr=2.0,
    risk_per_trade=0.0075,
    dd_pause_threshold=0.10,
    initial_capital=1_000_000,
)

PARAMS_ACTUAL = dict(periods=8, vol_spike=1.1, adx_min=20,
                     rsi_long_max=70, rsi_short_min=35)
PARAMS_RELAXED = dict(periods=8, vol_spike=1.1, adx_min=20,
                      rsi_long_max=100, rsi_short_min=0)

end = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

print(f"Período: {start} → {end}")
print(f"Tickers: {len(TICKERS)}\n")


def trade_key(t):
    """Identifica un trade por ticker + fecha de entrada + dirección."""
    return (t.get('entry_date'), t.get('dir'))


# Acumular trades de los 24 tickers en ambas configuraciones
all_actual = {}
all_relaxed = {}

for i, ticker in enumerate(TICKERS, 1):
    print(f"  [{i:2d}/{len(TICKERS)}] {ticker:6s}", end='  ')
    r_actual = backtest(ticker, start=start, end=end,
                        signal_params=PARAMS_ACTUAL, **COMMON)
    r_relax = backtest(ticker, start=start, end=end,
                       signal_params=PARAMS_RELAXED, **COMMON)
    if not r_actual or not r_relax:
        print("(sin datos)")
        continue
    t_a = r_actual.get('trades', [])
    t_r = r_relax.get('trades', [])
    for t in t_a:
        all_actual[(ticker, trade_key(t))] = t
    for t in t_r:
        all_relaxed[(ticker, trade_key(t))] = t
    print(f"actual: {len(t_a):3d} trades  relax: {len(t_r):3d} trades  "
          f"extra: {len(t_r) - len(t_a):+d}")

# Trades que existen en relaxed pero no en actual = bloqueados por RSI
blocked = []
for k, t in all_relaxed.items():
    if k not in all_actual:
        ticker, key = k
        t = {**t, '_ticker': ticker}
        blocked.append(t)

# Para cada trade bloqueado, sacar el ADX y RSI del momento de entrada
def enrich_with_indicators(blocked_trades):
    """Agrega ADX y RSI del día de entrada de cada trade."""
    by_ticker = {}
    for t in blocked_trades:
        by_ticker.setdefault(t['_ticker'], []).append(t)
    enriched = []
    for ticker, trades in by_ticker.items():
        daily = fetch(ticker, start, end, interval='1d')
        if daily.empty:
            continue
        daily = add_indicators(daily.copy())
        for t in trades:
            ent_date = pd.Timestamp(t['entry_date'])
            if ent_date.tz is not None and daily.index.tz is None:
                ent_date = ent_date.tz_localize(None)
            elif ent_date.tz is None and daily.index.tz is not None:
                ent_date = ent_date.tz_localize(daily.index.tz)
            # encontrar la barra más cercana
            try:
                if ent_date in daily.index:
                    bar = daily.loc[ent_date]
                else:
                    idx = daily.index.searchsorted(ent_date)
                    if idx >= len(daily):
                        idx = len(daily) - 1
                    bar = daily.iloc[idx]
                t['adx_at_entry'] = float(bar['adx'])
                t['rsi_at_entry'] = float(bar['rsi'])
            except Exception:
                t['adx_at_entry'] = None
                t['rsi_at_entry'] = None
            enriched.append(t)
    return enriched


blocked = enrich_with_indicators(blocked)

# Filtrar solo trades cuyo motivo de bloqueo fue RSI (no ADX, vol, etc.)
# Un trade está bloqueado por RSI long si rsi >= 70 y direction == 'LONG'
# Bloqueado por RSI short si rsi <= 35 y direction == 'SHORT'
truly_rsi_blocked = []
for t in blocked:
    rsi = t.get('rsi_at_entry')
    if rsi is None:
        continue
    if t['dir'] == 'LONG' and rsi >= 70:
        t['block_reason'] = f'RSI {rsi:.1f} > 70'
        truly_rsi_blocked.append(t)
    elif t['dir'] == 'SHORT' and rsi <= 35:
        t['block_reason'] = f'RSI {rsi:.1f} < 35'
        truly_rsi_blocked.append(t)

# Stats agregadas
print("\n" + "=" * 70)
print("TRADES BLOQUEADOS POR EL FILTRO RSI EN 5 AÑOS")
print("=" * 70)
print(f"Total bloqueados por RSI: {len(truly_rsi_blocked)}")
if not truly_rsi_blocked:
    print("(no hay trades bloqueados por RSI)")
    exit()

df_b = pd.DataFrame(truly_rsi_blocked)
pnls = df_b['pnl']
winners = df_b[pnls > 0]
losers = df_b[pnls <= 0]
total_pnl = pnls.sum()
win_rate = len(winners) / len(df_b) * 100

print(f"\nResumen agregado:")
print(f"  Winners:    {len(winners):4d}  ({win_rate:.1f}%)")
print(f"  Losers:     {len(losers):4d}  ({100-win_rate:.1f}%)")
print(f"  PnL total:  ${total_pnl:>12,.0f}")
print(f"  PnL avg:    ${pnls.mean():>12,.0f}")
print(f"  PnL median: ${pnls.median():>12,.0f}")
if len(winners):
    print(f"  Avg win:    ${winners['pnl'].mean():>12,.0f}")
if len(losers):
    print(f"  Avg loss:   ${losers['pnl'].mean():>12,.0f}")
print(f"  Profit factor: {winners['pnl'].sum() / abs(losers['pnl'].sum()) if len(losers) and losers['pnl'].sum() != 0 else float('inf'):.2f}")

# Distribución por ADX
print(f"\nSegmentado por ADX al momento de entrada:")
print(f"  {'ADX bucket':25s} {'N':>5s} {'Win%':>7s} {'PnL total':>15s} {'PnL avg':>12s}")
for lo, hi, label in [(0, 30, 'ADX 20-30'), (30, 40, 'ADX 30-40'),
                       (40, 50, 'ADX 40-50'), (50, 100, 'ADX >50')]:
    sub = df_b[(df_b['adx_at_entry'] >= lo) & (df_b['adx_at_entry'] < hi)]
    if len(sub) == 0:
        continue
    wr = (sub['pnl'] > 0).mean() * 100
    print(f"  {label:25s} {len(sub):>5d} {wr:>6.1f}% ${sub['pnl'].sum():>13,.0f} ${sub['pnl'].mean():>10,.0f}")

# Top 10 bloqueados con mayor PnL
print(f"\nTop 10 trades bloqueados (mayor ganancia potencial perdida):")
top = df_b.nlargest(10, 'pnl')[['_ticker', 'entry_date', 'dir', 'rsi_at_entry', 'adx_at_entry', 'pnl']]
print(top.to_string(index=False))

print(f"\nBottom 10 (mayores pérdidas evitadas):")
bot = df_b.nsmallest(10, 'pnl')[['_ticker', 'entry_date', 'dir', 'rsi_at_entry', 'adx_at_entry', 'pnl']]
print(bot.to_string(index=False))

# Test: variante condicional ADX > 45
print("\n" + "=" * 70)
print("VARIANTE CONDICIONAL: relajar RSI cuando ADX > 45")
print("=" * 70)
adx_high = df_b[df_b['adx_at_entry'] > 45]
adx_mid = df_b[(df_b['adx_at_entry'] >= 30) & (df_b['adx_at_entry'] <= 45)]
adx_low = df_b[df_b['adx_at_entry'] < 30]

print(f"\nSi solo se tomaran los bloqueados con ADX > 45:")
print(f"  N={len(adx_high)}  Win%={(adx_high['pnl']>0).mean()*100:.1f}%  "
      f"PnL total=${adx_high['pnl'].sum():,.0f}")

print(f"\nSi se tomaran TODOS los bloqueados (sin filtro RSI):")
print(f"  N={len(df_b)}  Win%={(df_b['pnl']>0).mean()*100:.1f}%  PnL total=${total_pnl:,.0f}")

# Comparativa de portfolio total con/sin filtro
print("\n" + "=" * 70)
print("PORTFOLIO TOTAL: con filtro vs sin filtro")
print("=" * 70)

total_actual = sum(t.get('pnl', 0) for t in all_actual.values())
total_relax = sum(t.get('pnl', 0) for t in all_relaxed.values())
print(f"  PnL agregado swing actual  (rsi 35-70): ${total_actual:>14,.0f}  "
      f"({len(all_actual)} trades)")
print(f"  PnL agregado swing relaxed (sin filtro):  ${total_relax:>14,.0f}  "
      f"({len(all_relaxed)} trades)")
print(f"  Diferencia:                             ${total_relax - total_actual:>14,.0f}")
