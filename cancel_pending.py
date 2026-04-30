"""
One-shot: cancela las MarketOrder parents que quedaron queued para la apertura
del 2026-05-01 09:30 ET sin TRAIL stop adjunto (Error 328).

Uso:
    python cancel_pending.py
"""

from ib_insync import IB

HOST       = '127.0.0.1'
PORT       = 4002
CLIENT_ID  = 1   # mismo que el daemon — necesario para que reqOpenOrders() devuelva sus trades
TARGET_IDS = {799, 802, 809, 812}   # META, MSFT, NVDA, AAPL parents


def main():
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print(f"Conectado clientId={CLIENT_ID}. Pidiendo openOrders...")

    # reqAllOpenOrders trae órdenes de TODOS los clientes (incluso desconectados)
    ib.reqAllOpenOrders()
    ib.reqOpenOrders()
    ib.sleep(3)

    all_trades = ib.trades()
    print(f"  {len(all_trades)} trades visibles en la sesión")
    for t in all_trades:
        print(f"    orderId={t.order.orderId} {t.order.action} {t.order.totalQuantity} "
              f"{t.contract.symbol} type={t.order.orderType} status={t.orderStatus.status}")

    matched = [t for t in all_trades if t.order.orderId in TARGET_IDS
               and t.orderStatus.status not in ('Cancelled', 'Filled')]

    if not matched:
        print("\nNinguna de las orderIds objetivo está activa. Nada que cancelar.")
        ib.disconnect()
        return

    print(f"\nCancelando {len(matched)} órdenes...")
    for t in matched:
        ib.cancelOrder(t.order)
        print(f"  cancel solicitada para {t.order.orderId} ({t.contract.symbol})")

    ib.sleep(3)
    print("\nEstado final:")
    for t in matched:
        print(f"  {t.order.orderId} {t.contract.symbol}: {t.orderStatus.status}")

    ib.disconnect()


if __name__ == '__main__':
    main()
