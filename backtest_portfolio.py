"""
backtest_portfolio.py — Backtest DE CARTERA del bot completo + ranking de señales.

A diferencia de backtester.py (cada ticker AISLADO), simula el sistema real:
las 3 estrategias × N tickers compartiendo UN capital, con múltiples posiciones
simultáneas y los filtros del bot (conflicto FCFS, drawdown stop, tope por
posición) más un tope de exposición bruta parametrizable.

RANKING DE SEÑALES: cuando hay más candidatas que cupo de exposición, se eligen
por CALIDAD (score), no por orden de llegada. Se barren varios scorers para que
el dato diga cuál sirve. Objetivo: ¿rankear por calidad + apalancamiento
moderado le gana a SPY con riesgo controlado?

Scorers probados:
  - fcfs    : orden del loop (baseline = comportamiento actual del bot)
  - adx     : fuerza de tendencia (ADX de la barra de señal)
  - vol     : confirmación de volumen (vol_ratio)
  - adx_vol : tendencia fuerte CON volumen (ADX × vol_ratio)

Aproximaciones (documentadas):
  - Resolución diaria, capital compartido, equity mark-to-market (drawdown real).
  - Entradas al Open del día en que la señal (barra previa) está vigente.
  - weekly_trend: señal/ATR/ADX de la vela semanal cerrada anterior (sin
    lookahead); gestión (trailing/SL) con precio diario.
  - Sin comisiones ni slippage (igual que backtester.py → comparables).

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
CAPS    = [0.6, 1.0, 1.5, 2.0]              # incluye apalancamiento moderado
SCORERS = ['fcfs', 'adx', 'vol', 'adx_vol']


def _norm(df):
    """Normaliza el índice a fecha tz-naive (medianoche) para alinear timeframes."""
    idx = df.index
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    out = df.copy()
    out.index = pd.to_datetime(idx).normalize()
    return out[~out.index.duplicated(keep='last')]


def load_data():
    """Devuelve (daily, feats, cal). feats[(strat,t)] = dict con dicts {date:val}
    de signal/atr/adx/vol vigentes (barra previa, ffill) para acceso rápido."""
    print(f"Descargando datos {START} → {END} ...", flush=True)
    daily, feats = {}, {}

    for t in TICKERS:
        d = fetch(t, START, END, '1d')
        if d is None or d.empty:
            print(f"  ⚠ {t}: sin datos diarios, lo salto", flush=True)
            continue
        dd = _norm(add_indicators(d))
        # dicts para gestión rápida (OHLC)
        daily[t] = {
            'O': dd['Open'].to_dict(), 'H': dd['High'].to_dict(),
            'L': dd['Low'].to_dict(),  'C': dd['Close'].to_dict(),
            'idx': set(dd.index),
        }

    cal = pd.DatetimeIndex(sorted(set().union(*[set(d['idx']) for d in daily.values()])))

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
            s = fn(base.copy(), **params)
            cols = s[['signal', 'atr', 'adx', 'vol_ratio']].shift(1)   # barra previa
            al = cols.reindex(cal, method='ffill')
            feats[(strat, t)] = {
                'sig': al['signal'].to_dict(), 'atr': al['atr'].to_dict(),
                'adx': al['adx'].to_dict(),    'vol': al['vol_ratio'].to_dict(),
            }

    print(f"  {len(daily)} tickers, {len(cal)} días hábiles", flush=True)
    return daily, feats, cal


def _score(scorer, adx, vol):
    adx = adx if pd.notna(adx) else 0.0
    vol = vol if pd.notna(vol) else 0.0
    if scorer == 'adx':     return adx
    if scorer == 'vol':     return vol
    if scorer == 'adx_vol': return adx * (vol if vol > 0 else 1.0)
    return 0.0   # fcfs → sin reordenar


def run(gross_cap, scorer, daily, feats, cal):
    cap, peak = INITIAL_CAPITAL, INITIAL_CAPITAL
    positions = {}
    eq_curve, gross_curve, trades = [], [], []
    capped = 0

    for d in cal:
        # ── 1. Gestionar posiciones abiertas (trailing + SL + señal opuesta) ──
        for key in list(positions.keys()):
            strat, t = key
            p, day, f = positions[key], daily.get(t), feats[key]
            if day is None or d not in day['idx']:
                continue
            high, low, close = day['H'][d], day['L'][d], day['C'][d]
            atr = f['atr'].get(d, np.nan)
            tr  = STRATEGIES[strat]['trailing_atr']
            if pd.notna(atr) and atr > 0:
                if p['dir'] == 'LONG':
                    p['stop'] = max(p['stop'], round(high - tr * atr, 2))
                else:
                    p['stop'] = min(p['stop'], round(low + tr * atr, 2))
            sl_hit = (low <= p['stop']) if p['dir'] == 'LONG' else (high >= p['stop'])
            sig_now = f['sig'].get(d, 0)
            opp = (p['dir'] == 'LONG' and sig_now == -1) or (p['dir'] == 'SHORT' and sig_now == 1)
            if sl_hit or opp:
                exit_px = p['stop'] if sl_hit else close
                pnl = (exit_px - p['entry']) * p['size'] * (1 if p['dir'] == 'LONG' else -1)
                cap += pnl
                trades.append({'pnl': pnl})
                del positions[key]

        # ── 2. Drawdown stop ──
        peak = max(peak, cap)
        dd_paused = (cap - peak) / peak < -DD_PAUSE_THRESHOLD if peak > 0 else False

        # ── 3. Recolectar candidatas, rankear por calidad, abrir según cupo ──
        if not dd_paused:
            cands = []
            for strat in STRATEGIES:
                cfg = STRATEGIES[strat]
                for t in TICKERS:
                    key = (strat, t)
                    if key in positions or key not in feats:
                        continue
                    day = daily.get(t)
                    if day is None or d not in day['idx']:
                        continue
                    f = feats[key]
                    sig = f['sig'].get(d, 0)
                    if sig == 0 or pd.isna(sig):
                        continue
                    new_dir = 'LONG' if sig == 1 else 'SHORT'
                    if not cfg['allow_short'] and new_dir == 'SHORT':
                        continue
                    entry = day['O'][d]
                    atr   = f['atr'].get(d, np.nan)
                    if pd.isna(atr) or atr <= 0 or entry <= 0:
                        continue
                    stop_dist = atr * cfg['atr_stop_mult']
                    size = min(cap * cfg['risk_per_trade'] / stop_dist,
                               cap * MAX_POS_PCT / entry)
                    if size < 1:
                        continue
                    cands.append({
                        'key': key, 'ticker': t, 'dir': new_dir, 'entry': entry,
                        'stop': round(entry - stop_dist if new_dir == 'LONG'
                                      else entry + stop_dist, 2),
                        'size': size, 'notional': entry * size,
                        'score': _score(scorer, f['adx'].get(d, np.nan), f['vol'].get(d, np.nan)),
                    })
            if scorer != 'fcfs':
                cands.sort(key=lambda c: c['score'], reverse=True)

            gross_open = sum(abs(q['entry'] * q['size']) for q in positions.values())
            limit = cap * gross_cap if gross_cap is not None else float('inf')
            for c in cands:
                # Conflicto entre estrategias: ticker ya tomado en dir opuesta
                opp_dir = 'SHORT' if c['dir'] == 'LONG' else 'LONG'
                if any(q['ticker'] == c['ticker'] and q['dir'] == opp_dir
                       for q in positions.values()):
                    continue
                if gross_open + c['notional'] > limit:
                    capped += 1
                    continue
                positions[c['key']] = {'ticker': c['ticker'], 'dir': c['dir'],
                                       'entry': c['entry'], 'stop': c['stop'], 'size': c['size']}
                gross_open += c['notional']

        # ── 4. Mark-to-market ──
        unreal = gross_now = 0.0
        for p in positions.values():
            day = daily.get(p['ticker'])
            px = day['C'][d] if (day and d in day['idx']) else p['entry']
            unreal += (px - p['entry']) * p['size'] * (1 if p['dir'] == 'LONG' else -1)
            gross_now += abs(px * p['size'])
        equity = cap + unreal
        eq_curve.append(equity)
        gross_curve.append(gross_now / equity if equity > 0 else 0)

    eq = pd.Series(eq_curve, index=cal)
    ret = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)
    total_ret = (eq.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    yrs = max((cal[-1] - cal[0]).days / 365.25, 1e-9)
    cagr = ((eq.iloc[-1] / INITIAL_CAPITAL) ** (1 / yrs) - 1) * 100
    gs = pd.Series(gross_curve)
    wins = sum(1 for x in trades if x['pnl'] > 0)
    return {'scorer': scorer, 'cap': f'{gross_cap*100:.0f}%' if gross_cap else 'sin tope',
            'final': eq.iloc[-1], 'total_ret': total_ret, 'cagr': cagr, 'sharpe': sharpe,
            'maxdd': maxdd, 'gross_avg': gs.mean() * 100, 'gross_peak': gs.max() * 100,
            'n_trades': len(trades), 'win_rate': wins / len(trades) * 100 if trades else 0}


def main():
    daily, feats, cal = load_data()
    if not daily:
        print("Sin datos. Abortando.")
        return
    bh = (daily['SPY']['C'][cal[-1]] / daily['SPY']['O'][cal[0]] - 1) * 100 if 'SPY' in daily else None

    print(f"\nBacktest de cartera + ranking | {START} → {END} | capital ${INITIAL_CAPITAL:,.0f}")
    print(f"{'Scorer':>8} {'Tope':>8} {'Final':>13} {'Ret%':>7} {'CAGR%':>7} {'Sharpe':>7} "
          f"{'MaxDD%':>8} {'GrAvg':>6} {'GrPk':>6} {'Trades':>7} {'Win%':>5}")
    print("─" * 100)
    results = []
    for scorer in SCORERS:
        for c in CAPS:
            r = run(c, scorer, daily, feats, cal)
            results.append(r)
            print(f"{r['scorer']:>8} {r['cap']:>8} {r['final']:>13,.0f} {r['total_ret']:>7.1f} "
                  f"{r['cagr']:>7.1f} {r['sharpe']:>7.2f} {r['maxdd']:>8.1f} "
                  f"{r['gross_avg']:>5.0f}% {r['gross_peak']:>5.0f}% {r['n_trades']:>7} {r['win_rate']:>4.0f}%")
        print("─" * 100)
    if bh is not None:
        print(f"Referencia: SPY buy & hold = {bh:+.1f}%  (CAGR {((1+bh/100)**(1/max((cal[-1]-cal[0]).days/365.25,1e-9))-1)*100:.1f}%)")

    beats = [r for r in results if bh is not None and r['total_ret'] > bh]
    print(f"\nConfigs que le ganan a SPY en retorno ({len(beats)}):")
    for r in sorted(beats, key=lambda r: r['sharpe'], reverse=True):
        print(f"  {r['scorer']:>8} @ {r['cap']:<8} Ret {r['total_ret']:>6.1f}% | "
              f"Sharpe {r['sharpe']:.2f} | MaxDD {r['maxdd']:.1f}% | GrossPk {r['gross_peak']:.0f}%")
    best = max(results, key=lambda r: r['sharpe'])
    print(f"\nMejor Sharpe global: {best['scorer']} @ {best['cap']} "
          f"(Sharpe {best['sharpe']:.2f}, Ret {best['total_ret']:.1f}%, MaxDD {best['maxdd']:.1f}%)")


if __name__ == '__main__':
    main()
