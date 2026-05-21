"""
Lee NetLiquidation actual de IBKR y actualiza bot_state.json.

Usar cuando la UI muestra un equity desactualizado porque el daemon estuvo
apagado y current_equity quedó congelado en el último run.

  python3 refresh_equity.py
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from ib_insync import IB

import bot as botmod

STATE_PATH = Path(__file__).parent / 'bot_state.json'


def main():
    with open(STATE_PATH) as f:
        state = json.load(f)

    old_equity = state.get('current_equity', 0)
    old_peak = state.get('peak_equity', 0)

    ib = IB()
    print(f"Conectando a IBKR ({botmod.IB_HOST}:{botmod.IB_PORT}, "
          f"clientId={botmod.IB_CLIENT_ID})...")
    ib.connect(botmod.IB_HOST, botmod.IB_PORT, clientId=botmod.IB_CLIENT_ID, timeout=10)

    # Reintenta hasta 3 veces igual que el bot
    equity = 0.0
    for attempt in range(3):
        equity = botmod.get_equity(ib)
        if equity > 0:
            break
        print(f"Equity=$0 en intento {attempt+1}/3, esperando 3s...")
        ib.sleep(3)

    ib.disconnect()

    if equity <= 0:
        print(f"ERROR: equity sigue $0 después de 3 intentos. State no modificado.")
        return

    print(f"\nEquity en IBKR ahora: ${equity:,.2f}")
    print(f"Equity en state:      ${old_equity:,.2f}")
    print(f"Diferencia:           ${equity - old_equity:+,.2f}")

    # Backup
    backup = STATE_PATH.with_suffix('.json.bak.' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    shutil.copy(STATE_PATH, backup)
    print(f"\nBackup: {backup}")

    state['current_equity'] = equity
    # Peak solo sube, nunca baja
    if equity > old_peak:
        state['peak_equity'] = equity
        print(f"Peak actualizado: ${old_peak:,.2f} → ${equity:,.2f}")

    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)

    print("State actualizado.")


if __name__ == '__main__':
    main()
