"""
Mide el costo histórico del drift de timing del bot.

Problema: el bot ejecuta a las 15:30 ET pero el signal está basado en el cierre
del día anterior (data congelada desde las 16:00 ET del día previo). El backtest
asume ejecución al OPEN del bar siguiente (~09:30 ET). El gap de 6 horas entre
backtest y producción puede generar slippage sistemático.

Este script:
  1. Toma todos los trades cerrados del bot (state.history).
  2. Para cada trade matched (open+close), compara:
     - PnL real (fills a 15:30 ET tanto en open como close).
     - PnL hipotético (fills al OPEN del día, alineado con backtest).
  3. Reporta el drift agregado: positivo significa que el fix de timing habría
     mejorado el PnL.

Uso: python3.10 analyze_timing_drift.py
"""

import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

PATH = '/home/futit/workspace/financial-terminal/bot_state.json'


def fetch_daily_opens(tickers, start_date, end_date):
    """Devuelve dict {ticker: {date: open_price}}."""
    out = {}
    for tk in tickers:
        try:
            df = yf.download(tk, start=start_date, end=end_date,
                             interval='1d', auto_adjust=True,
                             progress=False, timeout=30)
            if hasattr(df.columns, 'get_level_values'):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            out[tk] = {d.strftime('%Y-%m-%d'): float(o)
                       for d, o in df['Open'].items()}
        except Exception as e:
            print(f'  WARN: {tk} fetch fallido: {e}')
            out[tk] = {}
    return out


def main():
    with open(PATH) as f:
        s = json.load(f)
    history = sorted(s.get('history', []), key=lambda h: h.get('ts', ''))

    if not history:
        print('No hay history en bot_state.json. Salgo.')
        return

    first_ts = history[0]['ts'][:10]
    last_ts  = history[-1]['ts'][:10]
    tickers  = sorted(set(h['ticker'] for h in history))

    print(f"\n{'=' * 100}")
    print(f"  TIMING DRIFT ANALYSIS  ({first_ts} → {last_ts})  |  {len(tickers)} tickers")
    print(f"{'=' * 100}\n")

    # Buffer una semana antes/después para captura
    start_fetch = (datetime.strptime(first_ts, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
    end_fetch   = (datetime.strptime(last_ts, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')

    print(f"  Fetching OHLC diarios de {len(tickers)} tickers desde {start_fetch}...")
    opens = fetch_daily_opens(tickers, start_fetch, end_fetch)

    # Match open events con close events por (ticker, strategy, dir)
    open_pending = {}   # key (ticker, strategy, dir) → list of open events stacked
    matched = []        # (open_event, close_event)

    for h in history:
        key = (h['ticker'], h.get('strategy', '?'), h.get('dir'))
        if h['event'] == 'open':
            open_pending.setdefault(key, []).append(h)
        elif h['event'] == 'close':
            if open_pending.get(key):
                o = open_pending[key].pop(0)
                matched.append((o, h))
            else:
                # cierre sin open conocido (ej corrió antes del fix de history)
                pass

    print(f"  Trades matched (open + close): {len(matched)}")
    print(f"  Trades aún abiertos: {sum(len(v) for v in open_pending.values())}\n")

    rows = []
    skipped = 0
    for o, c in matched:
        tk    = o['ticker']
        date_o = o['ts'][:10]
        date_c = c['ts'][:10]
        real_entry = o.get('entry') or o.get('price')
        real_exit  = c.get('price') or c.get('exit')
        size       = o.get('size')
        direction  = 1 if o['dir'] == 'LONG' else -1
        real_pnl   = c.get('pnl')

        hyp_entry = opens.get(tk, {}).get(date_o)
        hyp_exit  = opens.get(tk, {}).get(date_c)

        if not all([real_entry, real_exit, size, hyp_entry, hyp_exit, real_pnl is not None]):
            skipped += 1
            continue

        hyp_pnl = (hyp_exit - hyp_entry) * size * direction
        drift   = hyp_pnl - real_pnl

        rows.append({
            'date_open':  date_o,
            'date_close': date_c,
            'ticker':     tk,
            'strategy':   o.get('strategy', '?'),
            'dir':        o['dir'],
            'size':       size,
            'real_entry': real_entry,
            'real_exit':  real_exit,
            'hyp_entry':  hyp_entry,
            'hyp_exit':   hyp_exit,
            'real_pnl':   real_pnl,
            'hyp_pnl':    hyp_pnl,
            'drift':      drift,
            'days':       (datetime.strptime(date_c, '%Y-%m-%d') -
                           datetime.strptime(date_o, '%Y-%m-%d')).days,
        })

    if skipped:
        print(f"  ({skipped} trades skippeados por datos faltantes.)\n")

    if not rows:
        print('  Sin trades con datos completos para analizar.')
        return

    df = pd.DataFrame(rows)

    # ── Reporte trade-by-trade ──
    print(f"  {'Open':<10} {'Close':<10} {'Tk':<5} {'Strat':<13} {'Dir':<5} "
          f"{'Real $':>9} {'Hyp $':>9} {'Drift':>8}")
    print(f"  {'-' * 92}")
    for r in df.sort_values('date_open').to_dict('records'):
        print(f"  {r['date_open']:<10} {r['date_close']:<10} {r['ticker']:<5} "
              f"{r['strategy']:<13} {r['dir']:<5} "
              f"{r['real_pnl']:>+9.2f} {r['hyp_pnl']:>+9.2f} {r['drift']:>+8.2f}")

    # ── Métricas agregadas ──
    print(f"\n{'─' * 100}")
    print(f"  AGREGADO")
    print(f"{'─' * 100}")

    n         = len(df)
    total_real = df['real_pnl'].sum()
    total_hyp  = df['hyp_pnl'].sum()
    total_dft  = df['drift'].sum()
    avg_drift  = df['drift'].mean()
    drift_pos  = (df['drift'] > 0).sum()
    drift_neg  = (df['drift'] < 0).sum()
    days_span  = (datetime.strptime(last_ts, '%Y-%m-%d') -
                  datetime.strptime(first_ts, '%Y-%m-%d')).days or 1

    print(f"  Trades analizados:        {n}")
    print(f"  Período cubierto:         {days_span} días corridos")
    print(f"  PnL REAL total:           ${total_real:>+12,.2f}")
    print(f"  PnL HIPOTÉTICO total:     ${total_hyp:>+12,.2f}")
    print(f"  DRIFT TOTAL (hyp-real):   ${total_dft:>+12,.2f}")
    print(f"  Drift por trade:          ${avg_drift:>+12,.2f}")
    print(f"  Drift por mes (extrap):   ${total_dft / days_span * 30:>+12,.2f}")
    print(f"  Trades donde hyp > real:  {drift_pos:>3} / {n}")
    print(f"  Trades donde hyp < real:  {drift_neg:>3} / {n}")

    # Breakdown por estrategia
    print(f"\n  Drift por estrategia:")
    for strat, sub in df.groupby('strategy'):
        s_total = sub['drift'].sum()
        s_avg   = sub['drift'].mean()
        print(f"    {strat:<14}  n={len(sub):>3}  total drift ${s_total:>+10,.2f}  "
              f"avg ${s_avg:>+8,.2f}")

    # Veredicto
    print(f"\n{'─' * 100}")
    print(f"  VEREDICTO")
    print(f"{'─' * 100}")
    monthly = total_dft / days_span * 30

    if total_dft > 0:
        direction = "A FAVOR del fix de timing (09:35 ET sería mejor)"
    else:
        direction = "CONTRA el fix (15:30 ET actual resultó MEJOR que hipotético 09:35 ET)"

    if abs(monthly) < 500:
        magnitud = "DRIFT MARGINAL — no justifica cambio."
    elif abs(monthly) < 2000:
        magnitud = "DRIFT MODERADO — evaluar."
    else:
        magnitud = "DRIFT MATERIAL — pero ojo a la dirección y al tamaño de muestra."

    print(f"  Drift mensual extrapolado: ${monthly:+,.2f}")
    print(f"  Dirección: {direction}")
    print(f"  Magnitud:  {magnitud}")
    print(f"  Muestra:   {n} trades / {days_span} días — chica, conclusiones tentativas.\n")


if __name__ == '__main__':
    main()
