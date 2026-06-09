"""
Mide tres tácticas de salida sobre los trades REALES cerrados del bot, usando
los mismos matched-trades del analizador de timing drift, pero descomponiendo
el efecto entrada vs cierre y agregando un ratchet de ganancia.

Escenarios (PnL por trade = (exit - entry) * size * dir):

  REAL      entry 15:30  /  exit 15:30        -> lo que pasó de verdad
  CIERRE_AM entry 15:30  /  exit open D_close -> IDEA 2: solo mover el cierre a la
                                                 mañana, entradas intactas
  FULL_AM   entry open   /  exit open         -> mover TODO el ciclo a la mañana
  RATCHET   entry 15:30  /  exit por ratchet  -> IDEA 3: si toca +UP% y retrocede a
                                                 +LOCK%, sale; si no, cierra como REAL

Descomposición:
  close_effect = CIERRE_AM - REAL   (puro timing de salida)
  entry_effect = FULL_AM   - CIERRE_AM (mover la entrada a la mañana)

Uso: python3.10 analyze_exit_tactics.py
"""

import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

PATH = '/home/futit/workspace/financial-terminal/bot_state.json'

# Parámetros del ratchet (idea 3)
RATCHET_UP   = 0.02   # arma el ratchet al tocar +2%
RATCHET_LOCK = 0.01   # cierra si retrocede a +1%


def fetch_ohlc(tickers, start_date, end_date):
    out = {}
    for tk in tickers:
        try:
            df = yf.download(tk, start=start_date, end=end_date,
                             interval='1d', auto_adjust=True,
                             progress=False, timeout=30)
            if hasattr(df.columns, 'get_level_values'):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            df.index = [d.strftime('%Y-%m-%d') for d in df.index]
            out[tk] = df
        except Exception as e:
            print(f'  WARN: {tk} fetch fallido: {e}')
            out[tk] = None
    return out


def open_on(df, date):
    if df is None or date not in df.index:
        return None
    return float(df.loc[date, 'Open'])


def ratchet_exit(df, date_o, date_c, entry, direction, up=RATCHET_UP, lock=RATCHET_LOCK):
    """Simula el ratchet sobre las barras diarias entre entrada y cierre real.
    Devuelve (exit_price, exit_reason). Entrada se asume al close de date_o
    (proxy del fill 15:30). Recorre días siguientes; usa high/low intradía para
    detectar el toque de +UP% y la retracción a +LOCK%.
    Conservador: si el mismo día toca UP y LOCK, asume que primero fue a favor.
    """
    if df is None:
        return None, 'nodata'
    dates = [d for d in df.index if date_o < d <= date_c]
    armed = False
    for d in dates:
        hi = float(df.loc[d, 'High'])
        lo = float(df.loc[d, 'Low'])
        op = float(df.loc[d, 'Open'])
        # retorno favorable segun direccion
        if direction == 1:   # LONG
            fav_extreme = (hi - entry) / entry           # mejor punto del dia
            adv_at_lock = entry * (1 + lock)      # nivel de lock
        else:                # SHORT
            fav_extreme = (entry - lo) / entry
            adv_at_lock = entry * (1 - lock)
        # arma el ratchet si tocó UP
        if not armed and fav_extreme >= up:
            armed = True
            # ¿mismo día retrocede al lock? si el open ya estaba >= lock y el
            # rango cubre el nivel, asumimos salida al lock ese mismo día.
            if direction == 1 and lo <= adv_at_lock:
                return adv_at_lock, f'ratchet@{date_c and d}'
            if direction == -1 and hi >= adv_at_lock:
                return adv_at_lock, f'ratchet@{d}'
            continue
        # ya armado: si la barra toca el lock, sale ahí
        if armed:
            if direction == 1 and lo <= adv_at_lock:
                return adv_at_lock, f'ratchet@{d}'
            if direction == -1 and hi >= adv_at_lock:
                return adv_at_lock, f'ratchet@{d}'
    return None, 'no_trigger'


def ratchet_exit_conservative(df, date_o, date_c, entry, direction, up, lock):
    """Versión pesimista: el ratchet solo dispara en un día ESTRICTAMENTE
    posterior al de armado, y la salida es al OPEN de ese día (no al nivel de
    lock idealizado). Elimina la magia intradía del mismo día."""
    if df is None:
        return None
    dates = [d for d in df.index if date_o < d <= date_c]
    armed = False
    for d in dates:
        hi = float(df.loc[d, 'High']); lo = float(df.loc[d, 'Low'])
        op = float(df.loc[d, 'Open'])
        if direction == 1:
            fav_extreme = (hi - entry) / entry
            cur_ret_open = (op - entry) / entry
        else:
            fav_extreme = (entry - lo) / entry
            cur_ret_open = (entry - op) / entry
        if armed and cur_ret_open <= lock:
            return op
        if not armed and fav_extreme >= up:
            armed = True
    return None


def main():
    with open(PATH) as f:
        s = json.load(f)
    history = sorted(s.get('history', []), key=lambda h: h.get('ts', ''))
    if not history:
        print('No hay history.'); return

    first_ts = history[0]['ts'][:10]
    last_ts  = history[-1]['ts'][:10]
    tickers  = sorted(set(h['ticker'] for h in history))

    start_fetch = (datetime.strptime(first_ts, '%Y-%m-%d') - timedelta(days=8)).strftime('%Y-%m-%d')
    end_fetch   = (datetime.strptime(last_ts, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
    print(f'  Fetching OHLC de {len(tickers)} tickers {start_fetch} → {end_fetch}...')
    ohlc = fetch_ohlc(tickers, start_fetch, end_fetch)

    # match open/close FIFO por (ticker, strategy, dir)
    pending, matched = {}, []
    for h in history:
        key = (h['ticker'], h.get('strategy', '?'), h.get('dir'))
        if h['event'] == 'open':
            pending.setdefault(key, []).append(h)
        elif h['event'] == 'close' and pending.get(key):
            matched.append((pending[key].pop(0), h))

    rows = []
    for o, c in matched:
        tk = o['ticker']
        d_o, d_c = o['ts'][:10], c['ts'][:10]
        entry = o.get('entry') or o.get('price')
        rexit = c.get('price') or c.get('exit')
        size  = o.get('size')
        dirn  = 1 if o['dir'] == 'LONG' else -1
        rpnl  = c.get('pnl')
        df    = ohlc.get(tk)

        hyp_entry = open_on(df, d_o)
        hyp_exit  = open_on(df, d_c)
        if not all([entry, rexit, size, hyp_entry, hyp_exit, rpnl is not None]):
            continue

        real    = (rexit - entry) * size * dirn
        cierre  = (hyp_exit - entry) * size * dirn     # idea 2
        fullam  = (hyp_exit - hyp_entry) * size * dirn

        rx, reason = ratchet_exit(df, d_o, d_c, entry, dirn)
        if rx is not None:
            ratchet = (rx - entry) * size * dirn
        else:
            ratchet = real   # nunca disparó: cierra como real
            reason  = 'no_trigger→real'

        rows.append(dict(d_o=d_o, d_c=d_c, tk=tk, strat=o.get('strategy', '?'),
                         dir=o['dir'], size=size,
                         real=real, cierre=cierre, fullam=fullam,
                         ratchet=ratchet, reason=reason))

    df = pd.DataFrame(rows)
    n = len(df)
    print(f'\n  Trades matched con data completa: {n}\n')

    print(f"  {'Open':<10}{'Tk':<6}{'Strat':<13}{'Dir':<6}"
          f"{'REAL':>10}{'CIERRE_AM':>11}{'FULL_AM':>10}{'RATCHET':>10}  reason")
    print('  ' + '-'*98)
    for r in df.sort_values('d_o').to_dict('records'):
        print(f"  {r['d_o']:<10}{r['tk']:<6}{r['strat']:<13}{r['dir']:<6}"
              f"{r['real']:>+10.0f}{r['cierre']:>+11.0f}{r['fullam']:>+10.0f}"
              f"{r['ratchet']:>+10.0f}  {r['reason']}")

    tot = {k: df[k].sum() for k in ['real', 'cierre', 'fullam', 'ratchet']}
    print('\n  ' + '='*70)
    print('  TOTALES (PnL agregado sobre los trades reales cerrados)')
    print('  ' + '='*70)
    print(f"    REAL (15:30/15:30)............ ${tot['real']:>+12,.0f}")
    print(f"    CIERRE_AM (idea 2)............ ${tot['cierre']:>+12,.0f}   vs real {tot['cierre']-tot['real']:>+10,.0f}")
    print(f"    FULL_AM (mover ciclo)......... ${tot['fullam']:>+12,.0f}   vs real {tot['fullam']-tot['real']:>+10,.0f}")
    print(f"    RATCHET +{RATCHET_UP:.0%}/+{RATCHET_LOCK:.0%} (idea 3).... ${tot['ratchet']:>+12,.0f}   vs real {tot['ratchet']-tot['real']:>+10,.0f}")

    print('\n  DESCOMPOSICIÓN timing:')
    print(f"    Efecto CIERRE a la mañana:  ${tot['cierre']-tot['real']:>+10,.0f}")
    print(f"    Efecto ENTRADA a la mañana: ${tot['fullam']-tot['cierre']:>+10,.0f}")

    rt = df[df['reason'].str.startswith('ratchet')]
    print(f"\n  Ratchet disparó en {len(rt)}/{n} trades.")
    if len(rt):
        print(f"    Suma ratchet en esos trades:     ${rt['ratchet'].sum():>+10,.0f}")
        print(f"    Suma real en esos mismos trades: ${rt['real'].sum():>+10,.0f}")

    # ── Robustez del ratchet: sweep de umbrales, optimista vs pesimista ──
    print('\n  ' + '='*70)
    print('  ROBUSTEZ DEL RATCHET (sweep de umbrales)')
    print('  ' + '='*70)
    print(f"    REAL baseline: ${tot['real']:>+12,.0f}\n")
    print(f"    {'UP/LOCK':<12}{'OPTIMISTA':>14}{'PESIMISTA':>14}{'#disparos':>11}")
    print('    ' + '-'*48)
    combos = [(0.015,0.005),(0.02,0.01),(0.025,0.01),(0.03,0.015),(0.04,0.02),(0.05,0.025)]
    for up, lock in combos:
        opt_tot = 0.0; pes_tot = 0.0; fires = 0
        for o, c in matched:
            tk = o['ticker']; d_o, d_c = o['ts'][:10], c['ts'][:10]
            entry = o.get('entry') or o.get('price'); rexit = c.get('price') or c.get('exit')
            size = o.get('size'); dirn = 1 if o['dir'] == 'LONG' else -1
            rpnl = c.get('pnl'); dff = ohlc.get(tk)
            if not all([entry, rexit, size, rpnl is not None]) or dff is None:
                continue
            real_pnl = (rexit - entry) * size * dirn
            # optimista (lock idealizado, mismo día permitido)
            rx, reason = ratchet_exit(dff, d_o, d_c, entry, dirn, up, lock)
            if rx is not None:
                opt_tot += (rx - entry) * size * dirn; fires += 1
            else:
                opt_tot += real_pnl
            # pesimista (salida al open del día posterior)
            px = ratchet_exit_conservative(dff, d_o, d_c, entry, dirn, up, lock)
            pes_tot += (px - entry) * size * dirn if px is not None else real_pnl
        print(f"    +{up:.1%}/+{lock:.1%}  {opt_tot:>+12,.0f}  {pes_tot:>+12,.0f}  {fires:>9}")
    print('\n    (OPTIMISTA = techo, asume orden intradía favorable;'
          ' PESIMISTA = piso, salida al open posterior.)')


if __name__ == '__main__':
    main()
