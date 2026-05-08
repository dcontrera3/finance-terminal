"""
Analiza contradicciones históricas entre las 3 estrategias del bot.

Una "contradicción" es: dos estrategias generan señales en direcciones
OPUESTAS sobre el mismo ticker dentro de una ventana de N días (proxy de
coexistencia de posiciones, ya que los stops trail mantienen posiciones
abiertas durante varios días).

Compara contra "confluencias" (señales en la MISMA dirección) para tener
la base.

Uso: python3 analyze_contradictions.py
"""

from datetime import datetime, timedelta
import pandas as pd

from backtester import (
    fetch, add_indicators,
    generate_signals_swing,
    generate_signals_pullback,
    generate_signals_weekly_trend,
)

TICKERS = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO',
    'NU', 'MELI', 'BABA',
    'SPY', 'QQQ', 'IWM',
    'GLD', 'SLV', 'XLE',
    'VIST',
]

SWING_PARAMS = dict(periods=8, vol_spike=1.1, adx_min=20,
                    rsi_long_max=70, rsi_short_min=35)
PULLBACK_PARAMS = dict(rsi_min=40, rsi_max=65, pullback_pct=1.02, vol_min=1.2)
WEEKLY_PARAMS = dict(ema_fast=10, ema_slow=20, adx_min=20,
                     rsi_long_max=75, rsi_short_min=25)

# Ventana de coexistencia: si una estrategia dio señal hace <= WINDOW dias
# y la otra dispara hoy, asumimos que la primera todavía tiene posición
# abierta (los trails típicos duran 5-30 dias).
WINDOW_DAYS = 10


def get_signals(ticker, start, end):
    """Devuelve dict con las señales de las 3 estrategias.
    Cada valor es lista de (date, dir) donde dir es +1 (LONG) o -1 (SHORT)."""
    out = {'swing': [], 'pullback': [], 'weekly_trend': []}

    # Diaria (swing + pullback)
    daily = fetch(ticker, start, end, interval='1d')
    if daily.empty or len(daily) < 200:
        return out
    daily = add_indicators(daily.copy())

    swing_df = generate_signals_swing(daily.copy(), **SWING_PARAMS)
    for date, sig in swing_df['signal'].items():
        if sig != 0:
            out['swing'].append((pd.Timestamp(date), int(sig)))

    pull_df = generate_signals_pullback(daily.copy(), **PULLBACK_PARAMS)
    for date, sig in pull_df['signal'].items():
        if sig != 0:
            out['pullback'].append((pd.Timestamp(date), int(sig)))

    # Semanal (weekly_trend)
    weekly = fetch(ticker, start, end, interval='1wk')
    if not weekly.empty and len(weekly) >= 50:
        weekly = add_indicators(weekly.copy())
        wk_df = generate_signals_weekly_trend(weekly.copy(), **WEEKLY_PARAMS)
        for date, sig in wk_df['signal'].items():
            if sig != 0:
                out['weekly_trend'].append((pd.Timestamp(date), int(sig)))

    return out


def find_overlaps(sigs_a, sigs_b, window_days):
    """Para cada señal en A, busca señales en B dentro de +/- window_days.
    Devuelve (confluencias, contradicciones) — listas de tuplas
    (date_a, dir_a, date_b, dir_b)."""
    confluences = []
    contradictions = []
    if not sigs_a or not sigs_b:
        return confluences, contradictions

    for date_a, dir_a in sigs_a:
        for date_b, dir_b in sigs_b:
            delta_days = abs((date_a - date_b).days)
            if delta_days <= window_days:
                if dir_a == dir_b:
                    confluences.append((date_a, dir_a, date_b, dir_b))
                else:
                    contradictions.append((date_a, dir_a, date_b, dir_b))
    return confluences, contradictions


def main():
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

    print(f"Analizando {len(TICKERS)} tickers, {start} → {end}, ventana ±{WINDOW_DAYS} dias\n")

    pairs = [
        ('swing', 'pullback'),
        ('swing', 'weekly_trend'),
        ('pullback', 'weekly_trend'),
    ]

    totals = {p: {'conf': 0, 'contra': 0} for p in pairs}
    total_signals = {'swing': 0, 'pullback': 0, 'weekly_trend': 0}
    contradiction_examples = []  # primeros 10 ejemplos para inspeccionar

    per_ticker = {}

    for i, ticker in enumerate(TICKERS, 1):
        sigs = get_signals(ticker, start, end)
        for s, lst in sigs.items():
            total_signals[s] += len(lst)

        ticker_row = {'ticker': ticker}
        for a, b in pairs:
            conf, contra = find_overlaps(sigs[a], sigs[b], WINDOW_DAYS)
            totals[(a, b)]['conf']   += len(conf)
            totals[(a, b)]['contra'] += len(contra)
            ticker_row[f'{a}_v_{b}_conf']   = len(conf)
            ticker_row[f'{a}_v_{b}_contra'] = len(contra)

            for date_a, dir_a, date_b, dir_b in contra[:3]:
                if len(contradiction_examples) < 15:
                    contradiction_examples.append(
                        (ticker, a, date_a.date(), dir_a, b, date_b.date(), dir_b)
                    )

        per_ticker[ticker] = ticker_row
        n_sigs = sum(len(l) for l in sigs.values())
        print(f"  [{i:2d}/{len(TICKERS)}] {ticker:6s} señales: "
              f"swing={len(sigs['swing']):3d}  pull={len(sigs['pullback']):3d}  "
              f"weekly={len(sigs['weekly_trend']):2d}")

    print("\n" + "=" * 70)
    print("TOTAL DE SEÑALES POR ESTRATEGIA (5 años, 24 tickers)")
    print("=" * 70)
    for s, n in total_signals.items():
        print(f"  {s:15s} {n:5d} señales")

    print("\n" + "=" * 70)
    print(f"SOLAPAMIENTOS POR PAR (ventana ±{WINDOW_DAYS} dias)")
    print("=" * 70)
    print(f"{'Par':30s} {'Confluencias':>14s} {'Contradicciones':>16s} {'%Contra':>10s}")
    print("-" * 70)
    grand_conf = 0
    grand_contra = 0
    for (a, b), counts in totals.items():
        c = counts['conf']
        x = counts['contra']
        total = c + x
        pct = (x / total * 100) if total else 0
        grand_conf += c
        grand_contra += x
        print(f"{a + ' vs ' + b:30s} {c:>14d} {x:>16d} {pct:>9.1f}%")
    print("-" * 70)
    grand_total = grand_conf + grand_contra
    grand_pct = (grand_contra / grand_total * 100) if grand_total else 0
    print(f"{'TOTAL':30s} {grand_conf:>14d} {grand_contra:>16d} {grand_pct:>9.1f}%")

    print("\n" + "=" * 70)
    print("TOP 10 TICKERS CON MAS CONTRADICCIONES")
    print("=" * 70)
    df = pd.DataFrame(per_ticker.values())
    contra_cols = [c for c in df.columns if c.endswith('_contra')]
    df['total_contra'] = df[contra_cols].sum(axis=1)
    df = df.sort_values('total_contra', ascending=False)
    print(df[['ticker', 'total_contra'] + contra_cols].head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("EJEMPLOS DE CONTRADICCIONES")
    print("=" * 70)
    for t, a, da, dira, b, db, dirb in contradiction_examples:
        sa = 'LONG ' if dira == 1 else 'SHORT'
        sb = 'LONG ' if dirb == 1 else 'SHORT'
        print(f"  {t:6s}  {a:12s} {sa} {da}   vs   {b:12s} {sb} {db}")

    print()
    print("=" * 70)
    print("INTERPRETACION RAPIDA")
    print("=" * 70)
    print(f"  - Total de pares de señales solapadas: {grand_total}")
    print(f"  - {grand_pct:.1f}% son contradicciones (lados opuestos)")
    print(f"  - {100-grand_pct:.1f}% son confluencias (mismo lado)")


if __name__ == '__main__':
    main()
