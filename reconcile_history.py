"""
Reconcilia los PnL del state.history con los fills reales de IBKR.

Por qué existe: hasta el commit 05b4561, sync_positions_with_ibkr calculaba el
PnL con get_market_price() en el momento del sync, no con el avgFillPrice del
trail. Eso produjo PnLs ficticios para todas las posiciones cerradas via trail
stop nativo. Este script reescribe los registros de cierre del history.

Dos modos:

  1) Live (default): consulta ib.fills() vía reqExecutions(). Limitado a fills
     del día corriente — útil solo el mismo día del cierre.

       python3 reconcile_history.py --dry-run
       python3 reconcile_history.py

  2) CSV: parsea un Activity Statement de IBKR Client Portal (Reports →
     Statements → Activity → CSV). Cubre cualquier período histórico.

       python3 reconcile_history.py --csv DUP587898_20260421_20260428.csv --dry-run
       python3 reconcile_history.py --csv DUP587898_20260421_20260428.csv
"""

import argparse
import csv as csvmod
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

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


# ─────────────────── Modo Live (IBKR API) ───────────────────

def fetch_all_fills_live(days_back=30):
    """Trae fills de IBKR usando ExecutionFilter con time = now - days_back."""
    from ib_insync import IB, ExecutionFilter
    import bot as botmod

    ib = IB()
    print(f"Conectando a IBKR ({botmod.IB_HOST}:{botmod.IB_PORT}, "
          f"clientId={botmod.IB_CLIENT_ID})...")
    ib.connect(botmod.IB_HOST, botmod.IB_PORT, clientId=botmod.IB_CLIENT_ID, timeout=10)
    print(f"Conectado. Pidiendo ejecuciones de los últimos {days_back} días...")

    cutoff = datetime.utcnow() - timedelta(days=days_back)
    flt = ExecutionFilter()
    flt.time = cutoff.strftime('%Y%m%d %H:%M:%S')
    ib.reqExecutions(flt)
    ib.sleep(1.5)

    raw = list(ib.fills())
    ib.disconnect()

    # Normalizamos al formato común que usa el resto del script
    fills = []
    for f in raw:
        side = 'SLD' if f.execution.side == 'SLD' else 'BOT'
        fills.append({
            'symbol': f.contract.symbol,
            'side':   side,
            'shares': float(f.execution.shares),
            'price':  float(f.execution.avgPrice),
            'ts':     f.execution.time.replace(tzinfo=None),
            'realized_pnl': None,
        })
    return fills


# ─────────────────── Modo CSV (Activity Statement) ───────────────────

def parse_activity_csv(path):
    """Parsea la sección 'Trades' de un Activity Statement de IBKR.

    Solo nos interesan filas con DataDiscriminator='Order' (las SubTotal/Total
    son agregados). Devuelve una lista de fills normalizados.
    """
    fills = []
    with open(path, encoding='utf-8-sig') as f:
        reader = csvmod.reader(f)
        for row in reader:
            if not row or row[0] != 'Trades' or len(row) < 14:
                continue
            if row[1] != 'Data' or row[2] != 'Order':
                continue
            # Header esperado:
            #   Trades, Data, DataDiscriminator, Asset Category, Currency,
            #   Account, Symbol, Date/Time, Quantity, T. Price, C. Price,
            #   Proceeds, Comm/Fee, Basis, Realized P/L, MTM P/L, Code
            symbol  = row[6]
            dt_str  = row[7].strip().replace('"', '')
            qty_str = row[8].replace(',', '').replace('"', '')
            price   = float(row[9])
            realized = row[14] if len(row) > 14 else ''

            try:
                qty = float(qty_str)
            except ValueError:
                continue
            try:
                ts = datetime.strptime(dt_str, '%Y-%m-%d, %H:%M:%S')
            except ValueError:
                continue

            side = 'BOT' if qty > 0 else 'SLD'
            fills.append({
                'symbol': symbol,
                'side':   side,
                'shares': abs(qty),
                'price':  price,
                'ts':     ts,
                'realized_pnl': float(realized) if realized else None,
            })
    return fills


# ─────────────────── Matching y aplicación ───────────────────

def aggregate_by_symbol_after(fills, symbol, action, after_ts):
    """Matchea fills por símbolo + dirección + ventana de tiempo."""
    after = (datetime.fromisoformat(after_ts) if isinstance(after_ts, str) else after_ts)
    matches = [f for f in fills
               if f['symbol'] == symbol
               and f['side'] == action
               and f['ts'] >= after]
    if not matches:
        return None

    total = sum(f['shares'] for f in matches)
    if not total:
        return None
    avg = sum(f['price'] * f['shares'] for f in matches) / total
    last_ts = max(f['ts'] for f in matches)
    realized = sum(f['realized_pnl'] for f in matches if f['realized_pnl'] is not None) or None
    return {
        'avg_price': avg,
        'shares':    int(total),
        'ts':        last_ts.isoformat(),
        'realized_pnl': realized,
    }


def reconcile(dry_run=False, csv_path=None, days_back=30):
    state = load_state()
    history = state.get('history', [])

    if csv_path:
        print(f"Modo CSV: parseando {csv_path}")
        fills = parse_activity_csv(csv_path)
    else:
        print("Modo Live: consultando IBKR API")
        fills = fetch_all_fills_live(days_back=days_back)

    print(f"Fills cargados: {len(fills)}")
    for f in fills:
        rp = f"  realized=${f['realized_pnl']:+.2f}" if f['realized_pnl'] is not None else ""
        print(f"  {f['symbol']:6s} {f['side']} qty={int(f['shares']):>6} "
              f"price=${f['price']:>10.4f}  ts={f['ts']}{rp}")

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
        # Matcheamos por símbolo + dirección + ventana desde el open.
        close_action = 'SLD' if direction == 'LONG' else 'BOT'
        after = open_evt['ts'] if open_evt else h['ts']
        agg = aggregate_by_symbol_after(fills, ticker, close_action, after)

        if agg is None:
            print(f"  [{ticker} {strategy} cerrado {h['ts']}] sin fill recuperable, se mantiene")
            continue

        new_price = round(agg['avg_price'], 4)
        # Preferir el realized_pnl reportado por IBKR (incluye comisiones).
        # Si no está, calcular a mano.
        if agg.get('realized_pnl') is not None:
            new_pnl = round(agg['realized_pnl'], 2)
        elif entry and size and direction:
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
    total_old = 0.0
    total_new = 0.0
    for c in changes:
        if c['old_pnl'] is not None:
            total_old += c['old_pnl']
        total_new += c['new_pnl'] or 0
        d = c['diff'] if c['diff'] is not None else 0
        print(f"{c['ticker']:6s} {c['strategy']:14s} "
              f"{c['old_price']:>10.4f} {c['new_price']:>10.4f} "
              f"{c['old_pnl']:>+12.2f} {c['new_pnl']:>+12.2f} {d:>+12.2f}")
    print(f"\nTotal PnL viejo (ficticio): ${total_old:+,.2f}")
    print(f"Total PnL nuevo (real):     ${total_new:+,.2f}")
    print(f"Diferencia neta:            ${total_new - total_old:+,.2f}")
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
    p.add_argument('--csv', type=str, default=None,
                   help='Path al CSV del Activity Statement (modo CSV)')
    p.add_argument('--days-back', type=int, default=30,
                   help='Cuántos días pedir fills (modo live, default 30)')
    args = p.parse_args()
    reconcile(dry_run=args.dry_run, csv_path=args.csv, days_back=args.days_back)
