"""
Limpia posiciones huérfanas de IBKR que el state no conoce.

Estrategia (Camino A del incidente 2026-05-19):
  Para cada símbolo, comparar qty signed en state vs IBKR. Si difieren:
    - Cancelar los openOrders activos del símbolo en IBKR (trails huérfanos).
    - Mandar una MarketOrder por la diferencia exacta para alinear posición.

Resultado: IBKR queda alineado con el state. El bot puede retomar operación
normal sin doble exposición.

Uso:
  python3 cleanup_orphan_positions.py --dry-run   # plan sin ejecutar
  python3 cleanup_orphan_positions.py --apply     # ejecuta a market
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from ib_insync import IB, Stock, MarketOrder

import bot as botmod

STATE_PATH = Path(__file__).parent / 'bot_state.json'


def load_state():
    with open(STATE_PATH) as f:
        return json.load(f)


def main(apply_changes=False):
    state = load_state()

    ib = IB()
    print(f"Conectando a IBKR ({botmod.IB_HOST}:{botmod.IB_PORT}, "
          f"clientId={botmod.IB_CLIENT_ID})...")
    ib.connect(botmod.IB_HOST, botmod.IB_PORT, clientId=botmod.IB_CLIENT_ID, timeout=10)

    ib.reqPositions()
    ib.sleep(2)

    ibkr_qty = {}
    contracts_by_sym = {}
    for p in ib.positions():
        sym = p.contract.symbol
        ibkr_qty[sym] = ibkr_qty.get(sym, 0) + int(p.position)
        contracts_by_sym[sym] = p.contract

    state_qty = {}
    for key, pos in state['positions'].items():
        sym = pos['ticker']
        signed = pos['size'] if pos['dir'] == 'LONG' else -pos['size']
        state_qty[sym] = state_qty.get(sym, 0) + signed

    # Open orders por símbolo (trails huérfanos a cancelar)
    open_orders_by_sym = {}
    for trade in ib.openTrades():
        sym = trade.contract.symbol
        open_orders_by_sym.setdefault(sym, []).append(trade)

    # Calcular plan
    all_syms = set(ibkr_qty.keys()) | set(state_qty.keys())
    actions = []
    for sym in sorted(all_syms):
        ib_q = ibkr_qty.get(sym, 0)
        st_q = state_qty.get(sym, 0)
        if ib_q == st_q:
            continue
        diff = ib_q - st_q  # cuánto sobra en IBKR (signo indica dirección)
        # Para alinear: si diff > 0 → IBKR tiene MÁS shares LONG (o menos SHORT)
        # que el state. Necesitamos SELL diff shares.
        # Si diff < 0 → IBKR tiene menos LONG (o más SHORT). Necesitamos BUY |diff|.
        if diff > 0:
            adjust_action = 'SELL'
            adjust_qty = diff
        else:
            adjust_action = 'BUY'
            adjust_qty = abs(diff)

        # Cancelaciones: si state dice 0 para este símbolo, cancelar TODOS los
        # open orders del símbolo. Si state mantiene algo, cancelar solo trails
        # cuyo size matchea el diff (el huérfano), para no tocar el trail activo.
        cancels = []
        if st_q == 0:
            cancels = list(open_orders_by_sym.get(sym, []))
        else:
            for trade in open_orders_by_sym.get(sym, []):
                qty = int(trade.order.totalQuantity)
                if qty == adjust_qty:
                    cancels.append(trade)

        actions.append({
            'symbol':         sym,
            'ibkr_qty':       ib_q,
            'state_qty':      st_q,
            'diff':           diff,
            'adjust_action':  adjust_action,
            'adjust_qty':     adjust_qty,
            'cancels':        cancels,
        })

    print("\n=== Plan de limpieza ===")
    for a in actions:
        print(f"\n{a['symbol']}: IBKR={a['ibkr_qty']:+}  state={a['state_qty']:+}  diff={a['diff']:+}")
        for trade in a['cancels']:
            o = trade.order
            print(f"  Cancelar orderId={o.orderId}  {o.action} {int(o.totalQuantity)} "
                  f"{a['symbol']}  ({o.orderType})")
        print(f"  → {a['adjust_action']} {a['adjust_qty']} {a['symbol']} @ MARKET")

    if not actions:
        print("\nNada que hacer. State y IBKR alineados.")
        ib.disconnect()
        return

    if not apply_changes:
        print("\n--dry-run: no se ejecuta nada. Usá --apply para ejecutar.")
        ib.disconnect()
        return

    print("\n=== EJECUTANDO ===")
    for a in actions:
        sym = a['symbol']
        # Cancelar trails huérfanos primero
        for trade in a['cancels']:
            try:
                ib.cancelOrder(trade.order)
                print(f"  Cancelada: orderId={trade.order.orderId} {sym}")
            except Exception as e:
                print(f"  ERROR cancelando {trade.order.orderId} {sym}: {e}")
        ib.sleep(1)

        # Adjust order: forzar SMART routing (ib.positions() devuelve el contract
        # con el exchange original, ej NASDAQ/ARCA, lo que dispara Error 10311 si
        # la config de la cuenta tiene "Preventiva" para direct routing).
        contract = Stock(sym, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        order = MarketOrder(a['adjust_action'], a['adjust_qty'])  # type: ignore
        trade = ib.placeOrder(contract, order)
        print(f"  Enviada: {a['adjust_action']} {a['adjust_qty']} {sym} (orderId={trade.order.orderId})")

        # Esperar fill hasta 30s
        filled = False
        for _ in range(60):
            ib.sleep(0.5)
            if trade.orderStatus.status == 'Filled':
                avg = float(trade.orderStatus.avgFillPrice or 0)
                print(f"     Filled @ ${avg:.4f}")
                filled = True
                break
        if not filled:
            print(f"     ⚠ No filled en 30s; status={trade.orderStatus.status}")

    ib.disconnect()
    print("\nLimpieza completada. Verificá posiciones en IB Gateway.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', default=True)
    p.add_argument('--apply', action='store_true', help='Ejecutar a market')
    args = p.parse_args()
    main(apply_changes=args.apply)
