"""
Análisis del filtro RSI swing — versión TRAIN ONLY para Fase 2.4.

Idéntico a analyze_rsi_filter.py pero restringido a 2021-2023 (train).
Identifica el umbral de ADX a partir del cual relajar el filtro RSI
SIN tocar datos del test set (2024-2026). Eso preserva la validez
del out-of-sample posterior.

Uso: python3.10 analyze_rsi_filter_train.py
"""
from datetime import datetime
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

PARAMS_ACTUAL  = dict(periods=8, vol_spike=1.1, adx_min=20,
                      rsi_long_max=70, rsi_short_min=35)
PARAMS_RELAXED = dict(periods=8, vol_spike=1.1, adx_min=20,
                      rsi_long_max=100, rsi_short_min=0)

# TRAIN ONLY — NO TOCAR 2024+
START = '2021-05-01'
END   = '2023-12-31'

print(f"Período TRAIN: {START} → {END}")
print(f"Tickers: {len(TICKERS)}\n")


def trade_key(t):
    return (t.get('entry_date'), t.get('dir'))


all_actual = {}
all_relaxed = {}

for i, ticker in enumerate(TICKERS, 1):
    r_actual = backtest(ticker, start=START, end=END, signal_params=PARAMS_ACTUAL, **COMMON)
    r_relax  = backtest(ticker, start=START, end=END, signal_params=PARAMS_RELAXED, **COMMON)
    if not r_actual or not r_relax:
        continue
    for t in r_actual.get('trades', []):
        all_actual[(ticker, trade_key(t))] = t
    for t in r_relax.get('trades', []):
        all_relaxed[(ticker, trade_key(t))] = t

# Trades bloqueados por RSI (en relaxed pero no en actual)
blocked = []
for k, t in all_relaxed.items():
    if k not in all_actual:
        ticker, key = k
        blocked.append({**t, '_ticker': ticker})


def enrich_with_indicators(blocked_trades):
    by_ticker = {}
    for t in blocked_trades:
        by_ticker.setdefault(t['_ticker'], []).append(t)
    enriched = []
    for ticker, trades in by_ticker.items():
        daily = fetch(ticker, START, END, interval='1d')
        if daily.empty:
            continue
        daily = add_indicators(daily.copy())
        for t in trades:
            ent_date = pd.Timestamp(t['entry_date'])
            if ent_date.tz is not None and daily.index.tz is None:
                ent_date = ent_date.tz_localize(None)
            elif ent_date.tz is None and daily.index.tz is not None:
                ent_date = ent_date.tz_localize(daily.index.tz)
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
truly_rsi_blocked = []
for t in blocked:
    rsi = t.get('rsi_at_entry')
    if rsi is None:
        continue
    if t['dir'] == 'LONG' and rsi >= 70:
        truly_rsi_blocked.append(t)
    elif t['dir'] == 'SHORT' and rsi <= 35:
        truly_rsi_blocked.append(t)

if not truly_rsi_blocked:
    print("Sin trades bloqueados por RSI en train. Algo raro.")
    exit()

df = pd.DataFrame(truly_rsi_blocked)
print(f"Trades bloqueados por RSI en TRAIN: {len(df)}")
print(f"  Win rate: {(df['pnl'] > 0).mean()*100:.1f}%")
print(f"  PnL total: ${df['pnl'].sum():,.0f}")
print(f"  PnL median: ${df['pnl'].median():,.0f}")
print()

# Buscar umbral óptimo: probar ADX desde 25 hasta 65 en pasos de 5,
# medir PnL agregado del subset, win rate, profit factor.
print("Búsqueda de umbral ADX (subset = bloqueados con ADX > threshold):")
print(f"  {'ADX>':<6} {'N':>5} {'Win%':>6} {'PnL total':>14} {'Avg':>10} {'PF':>6}")
print(f"  {'─'*6} {'─'*5} {'─'*6} {'─'*14} {'─'*10} {'─'*6}")
best_threshold = None
best_score = -float('inf')
for thr in [25, 30, 35, 40, 42, 45, 48, 50, 55, 60, 65]:
    sub = df[df['adx_at_entry'] > thr]
    if len(sub) < 5:
        continue
    pnl_tot = sub['pnl'].sum()
    win     = sub[sub['pnl'] > 0]
    loss    = sub[sub['pnl'] <= 0]
    wr      = len(win) / len(sub) * 100
    gw      = win['pnl'].sum() if len(win) else 0
    gl      = abs(loss['pnl'].sum()) if len(loss) else 0
    pf      = gw / gl if gl > 0 else float('inf')
    avg     = pnl_tot / len(sub)
    # Score combinado: PnL total con penalty por sample muy chico
    score = pnl_tot * (1 if len(sub) >= 30 else len(sub) / 30)
    pf_str = f"{pf:.2f}" if pf != float('inf') else " inf"
    print(f"  {thr:<6} {len(sub):>5} {wr:>5.1f}% ${pnl_tot:>12,.0f} ${avg:>8,.0f} {pf_str:>6}")
    if score > best_score:
        best_score = score
        best_threshold = thr

print()
print(f"Umbral con mejor score (PnL ponderado por sample): ADX > {best_threshold}")
print()

# Reporte por bucket clásico también
print("Buckets clásicos por ADX (para comparar con análisis full anterior):")
print(f"  {'Bucket':<10} {'N':>5} {'Win%':>6} {'PnL total':>14}")
for lo, hi in [(20, 30), (30, 40), (40, 50), (50, 999)]:
    sub = df[(df['adx_at_entry'] >= lo) & (df['adx_at_entry'] < hi)]
    if len(sub) == 0:
        continue
    wr = (sub['pnl'] > 0).mean() * 100
    label = f"{lo}-{hi}" if hi < 999 else f">{lo}"
    print(f"  {label:<10} {len(sub):>5} {wr:>5.1f}% ${sub['pnl'].sum():>12,.0f}")
