"""Diagnóstico AMD: ¿hubo señales y por qué el bot no abrió?"""
from datetime import datetime, timedelta
import pandas as pd

from backtester import (
    fetch, add_indicators,
    generate_signals_swing,
    generate_signals_pullback,
    generate_signals_weekly_trend,
)

SWING_PARAMS = dict(periods=8, vol_spike=1.1, adx_min=20,
                    rsi_long_max=70, rsi_short_min=35)
PULLBACK_PARAMS = dict(rsi_min=40, rsi_max=65, pullback_pct=1.02, vol_min=1.2)
WEEKLY_PARAMS = dict(ema_fast=10, ema_slow=20, adx_min=20,
                     rsi_long_max=75, rsi_short_min=25)

end = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

print(f"Fetching AMD {start} → {end}\n")

# Diaria
daily = fetch('AMD', start, end, interval='1d')
print(f"Daily bars: {len(daily)}")
print(f"Última barra: {daily.index[-1].date()} close=${daily['Close'].iloc[-1]:.2f}")
print(f"Hace 30 dias: {daily.index[-21].date()} close=${daily['Close'].iloc[-21]:.2f}")
print(f"Hace 15 dias: {daily.index[-11].date()} close=${daily['Close'].iloc[-11]:.2f}")
print(f"Hace  7 dias: {daily.index[-6].date()} close=${daily['Close'].iloc[-6]:.2f}")
print(f"Hace  3 dias: {daily.index[-3].date()} close=${daily['Close'].iloc[-3]:.2f}")

# Movimiento
hi = daily['High'].iloc[-30:].max()
lo = daily['Low'].iloc[-30:].min()
last = daily['Close'].iloc[-1]
print(f"\nÚltimos 30 dias: low ${lo:.2f}  high ${hi:.2f}  last ${last:.2f}")
print(f"Movimiento desde el low: {(last/lo - 1)*100:+.1f}%")
print(f"Movimiento desde el high: {(last/hi - 1)*100:+.1f}%")

# Indicadores en la última barra
daily = add_indicators(daily.copy())
last_bar = daily.iloc[-1]
print(f"\nIndicadores última barra ({daily.index[-1].date()}):")
print(f"  close:     ${last_bar['Close']:.2f}")
print(f"  ema20:     ${last_bar['ema20']:.2f}")
print(f"  ema50:     ${last_bar['ema50']:.2f}")
print(f"  ema200:    ${last_bar['ema200']:.2f}")
print(f"  rsi:       {last_bar['rsi']:.2f}")
print(f"  adx:       {last_bar['adx']:.2f}")
print(f"  atr:       {last_bar['atr']:.2f}")
print(f"  vol_ratio: {last_bar['vol_ratio']:.2f}")

# Señales swing y pullback en últimos 30 días
print("\n" + "=" * 70)
print("SEÑALES SWING últimos 30 días")
print("=" * 70)
swing_df = generate_signals_swing(daily.copy(), **SWING_PARAMS)
sw_signals = swing_df[swing_df['signal'] != 0].iloc[-10:]
if len(sw_signals) == 0:
    print("  Ninguna señal swing en últimos 30 días")
else:
    for idx, row in sw_signals.iterrows():
        d = 'LONG ' if row['signal'] == 1 else 'SHORT'
        print(f"  {idx.date()}  {d}  close=${row['Close']:.2f}  rsi={row['rsi']:.1f}  adx={row['adx']:.1f}  vol={row['vol_ratio']:.2f}")

# Diagnóstico de por qué no dispara swing
print("\nFiltros swing en las últimas 10 barras:")
h = daily['High']
l = daily['Low']
c = daily['Close']
high_8 = h.rolling(8).max().shift(1)
low_8  = l.rolling(8).min().shift(1)
for i in range(-10, 0):
    idx = daily.index[i]
    row = daily.iloc[i]
    bk_up   = row['Close'] > high_8.iloc[i]
    bk_dn   = row['Close'] < low_8.iloc[i]
    vol_ok  = row['vol_ratio'] > 1.1
    adx_ok  = row['adx'] > 20
    rsi_lng = row['rsi'] < 70
    rsi_sht = row['rsi'] > 35
    print(f"  {idx.date()} cls=${row['Close']:6.2f}  high8=${high_8.iloc[i]:6.2f}  "
          f"bkUp={bk_up}  vol={row['vol_ratio']:.2f}({vol_ok})  "
          f"adx={row['adx']:.1f}({adx_ok})  rsi={row['rsi']:.1f}(L:{rsi_lng}/S:{rsi_sht})")

# Pullback
print("\n" + "=" * 70)
print("SEÑALES PULLBACK últimos 30 días")
print("=" * 70)
pull_df = generate_signals_pullback(daily.copy(), **PULLBACK_PARAMS)
pu_signals = pull_df[pull_df['signal'] != 0].iloc[-10:]
if len(pu_signals) == 0:
    print("  Ninguna señal pullback en últimos 30 días")
else:
    for idx, row in pu_signals.iterrows():
        print(f"  {idx.date()}  LONG  close=${row['Close']:.2f}  rsi={row['rsi']:.1f}")

# Weekly
print("\n" + "=" * 70)
print("SEÑALES WEEKLY_TREND últimas 10 semanas")
print("=" * 70)
weekly_raw = fetch('AMD', start, end, interval='1wk')
weekly = add_indicators(weekly_raw.copy())
wk_df = generate_signals_weekly_trend(weekly.copy(), **WEEKLY_PARAMS)
wk_signals = wk_df[wk_df['signal'] != 0].iloc[-5:]
if len(wk_signals) == 0:
    print("  Ninguna señal weekly_trend en últimas 10 semanas")
else:
    for idx, row in wk_signals.iterrows():
        d = 'LONG ' if row['signal'] == 1 else 'SHORT'
        print(f"  {idx.date()}  {d}  close=${row['Close']:.2f}")

print("\nÚltimas 5 barras semanales con EMAs:")
for i in range(-5, 0):
    idx = weekly.index[i]
    row = weekly.iloc[i]
    cross = "ema_fast>slow" if row.get('ema_fast', 0) > row.get('ema_slow', 0) else "ema_fast<slow"
    # Calcular cross manualmente
    ef = weekly['Close'].ewm(span=10, adjust=False).mean().iloc[i]
    es = weekly['Close'].ewm(span=20, adjust=False).mean().iloc[i]
    print(f"  {idx.date()}  cls=${row['Close']:6.2f}  ema10=${ef:.2f}  ema20=${es:.2f}  adx={row['adx']:.1f}")
