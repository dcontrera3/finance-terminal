"""
Reconcilia los PnL del state.history con los fills reales de IBKR.

Por qué existe: hasta el commit 05b4561, sync_positions_with_ibkr calculaba el
PnL con get_market_price() en el momento del sync, no con el avgFillPrice del
trail. Eso produjo PnLs ficticios para todas las posiciones cerradas via trail
stop nativo. Este script va a IBKR, busca las ejecuciones reales por orderId,
y reescribe los registros de cierre del history.

Uso:
  python3 reconcile_history.py --dry-run   # ver qué cambiaría
  python3 reconcile_history.py             # aplicar cambios (backup automático)

Limitación: IBKR retiene fills hasta ~7 días vía reqExecutions. Si un fill es
más viejo, no podemos recuperarlo y se queda como está (con un flag).
"""

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from ib_insync import IB, ExecutionFilter

import bot as botmod

STATE_PATH = Path(__file__).parent / 'bot_state.json'


def load_state():
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state, path=STATE_PATH):
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


def find_open_event(history, ticker, strategy, before_ts):
    """Busca el evento 'open' más reciente para (ticker, strategy) anterior a before_ts."""
    candidates = [
        h for h in history
        if h['event'] == 'open'
        and h['ticker'] == ticker
        and h.get('strategy') == strategy
        and h['ts'] < before_ts
    ]
    return candidates[-1] if candidates else None


def fetch_all_fills(ib, days_back=30):
    """Trae fills de IBKR usando ExecutionFilter con time = now - days_back."""
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    flt = ExecutionFilter()
    flt.time = cutoff.strftime('%Y%m%d %H:%M:%S')
    ib.reqExecutions(flt)
    ib.sleep(1.5)
    return list(ib.fills())


def aggregate_fill(fills, order_id):
    matches = [f for f in fills if f.execution.orderId == int(order_id)]
    if not matches:
        return None
    total = sum(float(f.execution.shares) for f in matches)
    if not total:
        return None
    avg = sum(float(f.execution.avgPrice) * float(f.execution.shares)
              for f in matches) / total
    last_ts = max(f.execution.time for f in matches)
    return {'avg_price': avg, 'shares': int(total), 'ts': last_ts.isoformat()}


def aggregate_by_symbol_after(fills, symbol, action, after_ts):
    """Fallback cuando no tenemos orderId guardado: matchea por símbolo/acción/fecha."""
    after = datetime.fromisoformat(after_ts) if isinstance(after_ts, str) else after_ts
    matches = []
    for f in fills:
        if f.contract.symbol != symbol:
            continue
        if f.execution.side != action:  # 'BOT' o 'SLD'
            continue
        ft = f.execution.time
        if ft.replace(tzinfo=None) < after:
            continue
        matches.append(f)
    if not matches:
        return None
    total = sum(float(f.execution.shares) for f in matches)
    avg = sum(float(f.execution.avgPrice) * float(f.execution.shares)
              for f in matches) / total
    last_ts = max(f.execution.time for f in matches)
    return {'avg_price': avg, 'shares': int(total), 'ts': last_ts.isoformat()}


def reconcile(dry_run=False, days_back=30):
    state = load_state()
    history = state.get('history', [])

    ib = IB()
    print(f"Conectando a IBKR ({botmod.IB_HOST}:{botmod.IB_PORT}, clientId={botmod.IB_CLIENT_ID})...")
    ib.connect(botmod.IB_HOST, botmod.IB_PORT, clientId=botmod.IB_CLIENT_ID, timeout=10)
    print("Conectado. Pidiendo ejecuciones de los últimos", days_back, "días...")

    fills = fetch_all_fills(ib, days_back=days_back)
    print(f"Fills recuperados: {len(fills)}")
    for f in fills:
        print(f"  {f.contract.symbol:6s} {f.execution.side} qty={f.execution.shares:>6} "
              f"price=${float(f.execution.avgPrice):>10.4f}  orderId={f.execution.orderId}  "
              f"ts={f.execution.time}")

    ib.disconnect()

    changes = []
    for i, h in enumerate(history):
        if h['event'] != 'close':
            continue
        ticker   = h['ticker']
        strategy = h.get('strategy')
        old_price = h.get('price')
        old_pnl   = h.get('pnl')
        entry     = h.get('entry')
        size      = h.get('size')
        direction = h.get('dir')

        open_evt = find_open_event(history, ticker, strategy, h['ts'])
        # El history viejo no guarda order_ids en open events, así que
        # matcheamos por símbolo + dirección + ventana de tiempo.
        close_action = 'SLD' if direction == 'LONG' else 'BOT'
        after = open_evt['ts'] if open_evt else h['ts']
        agg = aggregate_by_symbol_after(fills, ticker, close_action, after)

        if agg is None:
            print(f"  [{ticker} {strategy} cerrado {h['ts']}] sin fill recuperable, se mantiene")
            continue

        new_price = round(agg['avg_price'], 4)
        if entry and size and direction:
            mult = 1 if direction == 'LONG' else -1
            new_pnl = round((new_price - entry) * size * mult, 2)
        else:
            new_pnl = old_pnl

        diff = (new_pnl - old_pnl) if old_pnl is not None else None
        changes.append({
            'idx': i,
            'ticker': ticker,
            'strategy': strategy,
            'ts': h['ts'],
            'old_price': old_price,
            'new_price': new_price,
            'old_pnl': old_pnl,
            'new_pnl': new_pnl,
            'diff': diff,
            'fill_ts': agg['ts'],
        })

    print("\n=== Cambios propuestos ===")
    print(f"{'TICKER':6s} {'STRAT':14s} {'OLD$':>10s} {'NEW$':>10s} "
          f"{'OLD PNL':>12s} {'NEW PNL':>12s} {'DIFF':>12s}")
    total_diff = 0.0
    for c in changes:
        d = c['diff'] or 0
        total_diff += d
        print(f"{c['ticker']:6s} {c['strategy']:14s} "
              f"{c['old_price']:>10.4f} {c['new_price']:>10.4f} "
              f"{c['old_pnl']:>+12.2f} {c['new_pnl']:>+12.2f} {d:>+12.2f}")
    print(f"\nTotal ajuste PnL reportado: ${total_diff:+,.2f}")
    print("(El equity de IBKR no cambia. Esto solo corrige el history.)")

    if dry_run:
        print("\n--dry-run: no se aplicaron cambios.")
        return

    if not changes:
        print("\nSin cambios. State intacto.")
        return

    backup = STATE_PATH.with_suffix('.json.bak.' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    shutil.copy(STATE_PATH, backup)
    print(f"\nBackup: {backup}")

    for c in changes:
        h = history[c['idx']]
        h['price'] = c['new_price']
        h['pnl'] = c['new_pnl']
        h['reconciled'] = True
        h['fill_ts'] = c['fill_ts']

    save_state(state)
    print("State actualizado.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', help='Solo mostrar cambios')
    p.add_argument('--days-back', type=int, default=30,
                   help='Cuántos días hacia atrás pedir fills (default 30)')
    args = p.parse_args()
    reconcile(dry_run=args.dry_run, days_back=args.days_back)
