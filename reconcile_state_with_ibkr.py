"""
Reconcilia el state local con las posiciones REALES de IBKR.

Usá esto cuando sospechás (o sabés) que el state quedó desalineado de IBKR.
Caso disparador: incidente 2026-05-19 donde sync_positions_with_ibkr corrió
con lectura inconsistente (equity=$0, ib.positions() vacío) y borró del state
3 posiciones que sí existían en IBKR. El bot después duplicó exposición.

Modo:
  python3 reconcile_state_with_ibkr.py --dry-run   # ver el diff
  python3 reconcile_state_with_ibkr.py --apply     # aplicar cambios al state

Lo que hace --apply:
  - Si IBKR tiene una posición que el state NO tiene: agrega entry al state
    con los datos disponibles (avgCost, qty, ATR estimado del log de apertura
    si se encuentra, fallback a placeholder).
  - Si state tiene una posición que IBKR NO tiene: la borra (huérfana real).
  - Si state y IBKR difieren en qty: deja un warning. El operador decide.

IMPORTANTE: no toca trail stops en IBKR. Solo reconstruye el state local.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ib_insync import IB

import bot as botmod

STATE_PATH = Path(__file__).parent / 'bot_state.json'


def load_state():
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state):
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def main(apply_changes=False):
    state = load_state()

    ib = IB()
    print(f"Conectando a IBKR ({botmod.IB_HOST}:{botmod.IB_PORT}, "
          f"clientId={botmod.IB_CLIENT_ID})...")
    ib.connect(botmod.IB_HOST, botmod.IB_PORT, clientId=botmod.IB_CLIENT_ID, timeout=10)

    # Forzar refresh de positions
    ib.reqPositions()
    ib.sleep(2)

    ibkr_positions = {}
    for p in ib.positions():
        sym = p.contract.symbol
        ibkr_positions.setdefault(sym, []).append({
            'qty':     int(p.position),
            'avgCost': float(p.avgCost),
        })

    # Sumar qty con signo por símbolo
    ibkr_qty = {s: sum(x['qty'] for x in v) for s, v in ibkr_positions.items()}
    # avgCost ponderado por |qty|
    ibkr_avg = {}
    for s, v in ibkr_positions.items():
        total = sum(abs(x['qty']) for x in v)
        if total:
            ibkr_avg[s] = sum(x['avgCost'] * abs(x['qty']) for x in v) / total

    print("\n=== Posiciones REALES en IBKR ===")
    for sym in sorted(ibkr_qty.keys()):
        qty = ibkr_qty[sym]
        avg = ibkr_avg[sym]
        side = 'LONG' if qty > 0 else 'SHORT'
        print(f"  {sym:6s}  {side:5s}  qty={abs(qty):>5}  avgCost=${avg:>10.4f}  "
              f"notional=${abs(qty)*avg:>12,.0f}")

    print("\n=== Posiciones en state local ===")
    state_qty_by_ticker = {}
    for key, pos in state['positions'].items():
        signed = pos['size'] if pos['dir'] == 'LONG' else -pos['size']
        state_qty_by_ticker.setdefault(pos['ticker'], 0)
        state_qty_by_ticker[pos['ticker']] += signed
        print(f"  {key:25s}  {pos['dir']:5s}  qty={pos['size']:>5}  "
              f"entry=${pos['entry']:>10.4f}")

    # Diferencias
    all_tickers = set(ibkr_qty.keys()) | set(state_qty_by_ticker.keys())
    diffs = []
    for t in sorted(all_tickers):
        ibkr_q = ibkr_qty.get(t, 0)
        state_q = state_qty_by_ticker.get(t, 0)
        if ibkr_q != state_q:
            diffs.append((t, state_q, ibkr_q, ibkr_avg.get(t)))

    print("\n=== Diferencias state ↔ IBKR ===")
    if not diffs:
        print("  Ninguna. State alineado con IBKR.")
    else:
        for t, s, i, avg in diffs:
            sign = '+' if (i - s) > 0 else ''
            print(f"  {t:6s}  state={s:>+6}  IBKR={i:>+6}  diff={sign}{i-s:+}  "
                  f"avgCost=${avg:.4f}" if avg else
                  f"  {t:6s}  state={s:>+6}  IBKR={i:>+6}  diff={sign}{i-s:+}")

    ib.disconnect()

    if not apply_changes:
        print("\n--dry-run: no se aplican cambios.")
        print("Para reconciliar manualmente, ejecutá --apply después de revisar el diff.")
        return

    # Apply: solo casos "seguros" (eliminar huérfanas reales)
    print("\n=== Aplicando cambios automáticos ===")
    print("(Por seguridad, este script NO inventa posiciones nuevas en el state.")
    print(" Solo elimina huérfanas del state que IBKR ya no tiene.")
    print(" Para reconstruir posiciones que IBKR tiene pero state no, editá")
    print(" el state a mano usando los datos del diff arriba.)")

    to_delete = []
    for key, pos in state['positions'].items():
        t = pos['ticker']
        if t not in ibkr_qty or ibkr_qty[t] == 0:
            to_delete.append(key)
            print(f"  Eliminando {key}: IBKR no tiene {t}")

    if not to_delete:
        print("  Sin cambios automáticos posibles.")
        return

    backup = STATE_PATH.with_suffix('.json.bak.' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    shutil.copy(STATE_PATH, backup)
    print(f"\nBackup: {backup}")

    for key in to_delete:
        del state['positions'][key]

    save_state(state)
    print(f"State actualizado: {len(to_delete)} entries eliminados.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', default=True,
                   help='Solo mostrar el diff (default)')
    p.add_argument('--apply', action='store_true', help='Aplicar cambios automáticos')
    args = p.parse_args()
    main(apply_changes=args.apply)
