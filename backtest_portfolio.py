"""
backtest_portfolio.py — Backtest DE CARTERA del bot completo.

A diferencia de backtester.py (que corre cada ticker AISLADO con su propio
capital), este simula el sistema real: las 3 estrategias × N tickers
compartiendo UN capital, con múltiples posiciones simultáneas y los mismos
filtros del bot (conflicto entre estrategias FCFS, drawdown stop, tope por
posición) más un tope de exposición bruta parametrizable.

Objetivo: medir riesgo/retorno del CONJUNTO y barrer MAX_GROSS_EXPOSURE para
encontrar el techo óptimo — lo que el backtest por-ticker nunca pudo responder.

Aproximaciones (documentadas para honestidad):
  - Todo se simula en resolución DIARIA con capital compartido.
  - Equity = realizado + no realizado (mark-to-market al Close) → drawdown real.
  - Entradas al Open del día en que la señal (de la barra previa) está vigente.
  - weekly_trend: señal y ATR vienen de la vela SEMANAL cerrada anterior
    (shift+ffill, sin lookahead); la gestión (trailing/SL) usa precio diario.
  - Sin comisiones ni slippage (igual que backtester.py → comparables).
  - El tope de exposición usa entry×size (igual que la baranda del bot real).

Uso:  python3.10 backtest_portfolio.py [start] [end]
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from backtester import fetch, add_indicators
from bot import TICKERS, STRATEGIES, MAX_POS_PCT, DD_PAUSE_THRESHOLD

START = sys.argv[1] if len(sys.argv) > 1 else '2021-01-01'
END   = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
INITIAL_CAPITAL = 1_000_000
CAPS  = [None, 0.6, 0.8, 1.0, 1.3]   # None = sin tope (comportamiento viejo)


def _norm(df):
    """Normaliza el índice a fecha tz-naive (medianoche) para alinear timeframes."""
    idx = df.index
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    out = df.copy()
    out.index = pd.to_datetime(idx).normalize()
    out = out[~out.index.duplicated(keep='last')]
    return out


def load_data():
    """Devuelve (daily, sig_series, atr_series).

    daily[t]            : OHLC diario normalizado (precios de gestión).
    sig_series[(s,t)]   : señal vigente por día (barra previa, ffill).
    atr_series[(s,t)]   : ATR de la estrategia vigente por día (barra previa).
    """
    print(f"Descargando datos {START} → {END} ...", flush=True)
    daily, sig_series, atr_series = {}, {}, {}

    for t in TICKERS:
        d = fetch(t, START, END, '1d')
        if d is None or d.empty:
            print(f"  ⚠ {t}: sin datos diarios, lo salto", flush=True)
            continue
        daily[t] = _norm(add_indicators(d))

    cal = sorted(set().union(*[df.index for df in daily.values()]))
    cal = pd.DatetimeIndex(cal)

    for strat, cfg in STRATEGIES.items():
        tf, fn, params = cfg['timeframe'], cfg['signal_fn'], cfg['params']
        for t in TICKERS:
            if t not in daily:
                continue
            if tf == '1d':
                base = _norm(add_indicators(fetch(t, START, END, '1d')))
            else:
                raw = fetch(t, START, END, tf)
                if raw is None or raw.empty:
                    continue
                base = _norm(add_indicators(raw))
            s = fn(base.copy(), **params).dropna(subset=['signal'])
            # Barra PREVIA cerrada (shift 1 = sin lookahead), luego a diario.
            shifted = s[['signal', 'atr']].shift(1)
            sig_series[(strat, t)] = shifted['signal'].reindex(cal, method='ffill')
            atr_series[(strat, t)] = shifted['atr'].reindex(cal, method='ffill')

    print(f"  {len(daily)} tickers, {len(cal)} días hábiles", flush=True)
    return daily, sig_series, atr_series, cal


def run(gross_cap, daily, sig_series, atr_series, cal):
    """Simula la cartera con un tope de exposición bruta dado (None = sin tope)."""
    cap        = INITIAL_CAPITAL
    peak       = cap
    positions  = {}            # (strat,ticker) -> dict
    eq_curve   = []
    gross_curve = []
    trades     = []
    capped     = 0             # señales rechazadas por el tope de exposición

    for d in cal:
        # ── 1. Gestionar posiciones abiertas (trailing + SL + señal opuesta) ──
        for key in list(positions.keys()):
            strat, t = key
            p = positions[key]
            day = daily.get(t)
            if day is None or d not in day.index:
                continue
            row  = day.loc[d]
            high, low, close = float(row['High']), float(row['Low']), float(row['Close'])
            atr = atr_series[key].get(d, np.nan)
            tr  = STRATEGIES[strat]['trailing_atr']

            # Trailing stop con ATR de la estrategia y high/low del día
            if pd.notna(atr) and atr > 0:
                if p['dir'] == 'LONG':
                    p['stop'] = max(p['stop'], round(high - tr * atr, 2))
                else:
                    p['stop'] = min(p['stop'], round(low + tr * atr, 2))

            sl_hit = (low <= p['stop']) if p['dir'] == 'LONG' else (high >= p['stop'])
            # Señal opuesta vigente → el bot también cierra por eso
            sig_now = sig_series[key].get(d, 0)
            opp = (p['dir'] == 'LONG' and sig_now == -1) or (p['dir'] == 'SHORT' and sig_now == 1)

            if sl_hit or opp:
                exit_px = p['stop'] if sl_hit else close
                pnl = (exit_px - p['entry']) * p['size'] * (1 if p['dir'] == 'LONG' else -1)
                cap += pnl
                trades.append({'key': key, 'dir': p['dir'], 'pnl': pnl,
                               'exit': 'TRAIL' if sl_hit else 'OPP', 'date': d})
                del positions[key]

        # ── 2. Drawdown stop (sobre capital realizado, igual que el bot) ──
        peak = max(peak, cap)
        dd_paused = (cap - peak) / peak < -DD_PAUSE_THRESHOLD if peak > 0 else False

        # ── 3. Evaluar señales nuevas ──
        if not dd_paused:
            gross_open = sum(abs(q['entry'] * q['size']) for q in positions.values())
            for strat in STRATEGIES:
                cfg = STRATEGIES[strat]
                for t in TICKERS:
                    key = (strat, t)
                    if key in positions or key not in sig_series:
                        continue
                    day = daily.get(t)
                    if day is None or d not in day.index:
                        continue
                    sig = sig_series[key].get(d, 0)
                    if sig == 0 or pd.isna(sig):
                        continue
                    new_dir = 'LONG' if sig == 1 else 'SHORT'
                    if not cfg['allow_short'] and new_dir == 'SHORT':
                        continue

                    # Conflicto entre estrategias (FCFS): otra estrategia con
                    # posición opuesta en el mismo ticker bloquea la apertura.
                    opp_dir = 'SHORT' if new_dir == 'LONG' else 'LONG'
                    if any(q['ticker'] == t and q['dir'] == opp_dir
                           for q in positions.values()):
                        continue

                    entry = float(day.loc[d]['Open'])
                    atr   = atr_series[key].get(d, np.nan)
                    if pd.isna(atr) or atr <= 0 or entry <= 0:
                        continue
                    stop_dist = atr * cfg['atr_stop_mult']
                    risk_usd  = cap * cfg['risk_per_trade']
                    size      = risk_usd / stop_dist
                    size      = min(size, (cap * MAX_POS_PCT) / entry)
                    if size < 1:
                        continue

                    # Tope de exposición bruta (la baranda parametrizable)
                    notional = entry * size
                    if gross_cap is not None and gross_open + notional > cap * gross_cap:
                        capped += 1
                        continue

                    stop = entry - stop_dist if new_dir == 'LONG' else entry + stop_dist
                    positions[key] = {'ticker': t, 'dir': new_dir, 'entry': entry,
                                      'stop': round(stop, 2), 'size': size}
                    gross_open += notional

        # ── 4. Mark-to-market: equity = realizado + no realizado ──
        unreal = 0.0
        gross_now = 0.0
        for key, p in positions.items():
            t = p['ticker']
            day = daily.get(t)
            if day is None or d not in day.index:
                px = p['entry']
            else:
                px = float(day.loc[d]['Close'])
            unreal += (px - p['entry']) * p['size'] * (1 if p['dir'] == 'LONG' else -1)
            gross_now += abs(px * p['size'])
        equity = cap + unreal
        eq_curve.append(equity)
        gross_curve.append(gross_now / equity if equity > 0 else 0)

    # ── Métricas ──
    eq = pd.Series(eq_curve, index=cal)
    ret = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)
    total_ret = (eq.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    cagr_years = max((cal[-1] - cal[0]).days / 365.25, 1e-9)
    cagr = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1 / cagr_years) - 1) * 100
    gross_s = pd.Series(gross_curve)
    wins = sum(1 for x in trades if x['pnl'] > 0)

    return {
        'cap':        'sin tope' if gross_cap is None else f'{gross_cap*100:.0f}%',
        'final':      eq.iloc[-1],
        'total_ret':  total_ret,
        'cagr':       cagr,
        'sharpe':     sharpe,
        'maxdd':      maxdd,
        'gross_avg':  gross_s.mean() * 100,
        'gross_peak': gross_s.max() * 100,
        'n_trades':   len(trades),
        'win_rate':   wins / len(trades) * 100 if trades else 0,
        'capped':     capped,
        'eq':         eq,
    }


def main():
    daily, sig_series, atr_series, cal = load_data()
    if not daily:
        print("Sin datos. Abortando.")
        return

    # Buy & hold de SPY como referencia
    bh = None
    if 'SPY' in daily:
        spy = daily['SPY']
        bh = (spy['Close'].iloc[-1] / spy['Open'].iloc[0] - 1) * 100

    print(f"\nBacktest de cartera | {START} → {END} | capital ${INITIAL_CAPITAL:,.0f}")
    print(f"{'Tope':>9} {'Final':>14} {'Ret%':>8} {'CAGR%':>7} {'Sharpe':>7} "
          f"{'MaxDD%':>8} {'GrossAvg':>9} {'GrossPk':>8} {'Trades':>7} {'Win%':>6} {'Capped':>7}")
    print("─" * 110)
    results = []
    for c in CAPS:
        r = run(c, daily, sig_series, atr_series, cal)
        results.append(r)
        print(f"{r['cap']:>9} {r['final']:>14,.0f} {r['total_ret']:>8.1f} "
              f"{r['cagr']:>7.1f} {r['sharpe']:>7.2f} {r['maxdd']:>8.1f} "
              f"{r['gross_avg']:>8.0f}% {r['gross_peak']:>7.0f}% {r['n_trades']:>7} "
              f"{r['win_rate']:>5.0f}% {r['capped']:>7}")
    print("─" * 110)
    if bh is not None:
        print(f"Referencia: SPY buy & hold en el período = {bh:+.1f}%")

    # Mejor por Sharpe y por retorno ajustado (CAGR/|MaxDD|)
    best_sharpe = max(results, key=lambda r: r['sharpe'])
    best_calmar = max(results, key=lambda r: r['cagr'] / abs(r['maxdd']) if r['maxdd'] else 0)
    print(f"\nMejor Sharpe : tope {best_sharpe['cap']} (Sharpe {best_sharpe['sharpe']:.2f})")
    print(f"Mejor CAGR/DD: tope {best_calmar['cap']} "
          f"(CAGR {best_calmar['cagr']:.1f}% / MaxDD {best_calmar['maxdd']:.1f}%)")


if __name__ == '__main__':
    main()
