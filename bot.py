"""
Trading Bot — Swing Strategy (Paper Trading)
Conecta a IBKR Paper Account y ejecuta la estrategia swing
sobre la canasta de tickers validada en el backtester.

Estrategia: breakout de 8 días + volumen + ADX. LONG y SHORT.
Timeframe: diario. Ejecución: cada día a las 15:55 ET (5 min antes del cierre).
Backtested 2020-2026: Sharpe 1.73 | MaxDD -11.6% | Win Rate 45.4% | PF 1.25

SETUP (una sola vez):
  1. Crear cuenta en IBKR: ibkr.com → activar Paper Trading
  2. Descargar IB Gateway: ibkr.com/en/trading/ibgateway-stable.php
  3. Iniciar IB Gateway → seleccionar "Paper Trading" → puerto 4002
  4. pip install ib_insync schedule

USO:
  python bot.py --run       # ejecutar señales ahora
  python bot.py --dry-run   # señales sin colocar órdenes
  python bot.py --status    # ver posiciones abiertas
  python bot.py --daemon    # correr automático cada día a las 15:55 ET
  python bot.py --close ALL # cerrar todas las posiciones
"""

import json
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtester import fetch, add_indicators, generate_signals_swing

try:
    from ib_insync import IB, Stock, LimitOrder, StopOrder, MarketOrder, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False


# ══════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════

TICKERS = ['NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT',
           'AMZN', 'GLD', 'SPY', 'QQQ', 'VIST']

# Parámetros ganadores del optimizador (train 2020-2023, test 2024-2026)
# Backtest portfolio: Sharpe 1.73 | MaxDD -11.6% | Win Rate 45.4% | PF 1.25
SIGNAL_PARAMS = dict(
    periods=8, vol_spike=1.1,
    adx_min=20, rsi_long_max=70, rsi_short_min=35,
)

RISK_PER_TRADE = 0.01   # 1% del equity por trade
ATR_STOP_MULT  = 1.0    # stop = entry ± ATR × 1.0
RR_RATIO       = 1.5    # target = riesgo × 1.5
MAX_POS_PCT    = 0.25   # máximo 25% del equity en una posición
ALLOW_SHORT    = True   # operar en ambas direcciones

# IBKR
IB_HOST = '127.0.0.1'
IB_PORT = 4002          # IB Gateway paper → 4002 | TWS paper → 7497
IB_CLIENT_ID = 1

STATE_FILE = 'bot_state.json'


# ══════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger('bot')


# ══════════════════════════════════════════
# ESTADO PERSISTENTE
# ══════════════════════════════════════════

def load_state():
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'positions': {}, 'last_run': None, 'total_trades': 0}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


# ══════════════════════════════════════════
# SEÑALES (reutiliza la lógica del backtester)
# ══════════════════════════════════════════

def get_signal(ticker):
    """
    Devuelve la señal del último día completo de trading.
      +1 = LONG (breakout de máximos)
      -1 = SHORT (breakout de mínimos)
       0 = sin señal
    También devuelve el ATR para calcular el stop.
    """
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    df = fetch(ticker, start, end, interval='1d')
    if df.empty or len(df) < 50:
        log.warning(f"{ticker}: datos insuficientes")
        return 0, None, None

    df = add_indicators(df)
    df = generate_signals_swing(df, **SIGNAL_PARAMS)
    df = df.dropna()

    # Usar el último día COMPLETO (no la barra del día en curso)
    last = df.iloc[-2]
    signal = int(last['signal'])
    atr_val = float(last['atr'])
    price = float(df.iloc[-1]['Close'])

    return signal, atr_val, price


def get_all_signals():
    """Devuelve dict {ticker: (signal, atr, price)} para todos los tickers."""
    signals = {}
    for t in TICKERS:
        sig, atr_val, price = get_signal(t)
        signals[t] = (sig, atr_val, price)
        direction = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(sig, '?')
        log.info(f"  {t:<6} {direction:<6}  precio=${price:.2f}  ATR={atr_val:.2f}" if price else f"  {t} sin datos")
    return signals


# ══════════════════════════════════════════
# CONEXIÓN IBKR
# ══════════════════════════════════════════

def connect_ibkr():
    if not IB_AVAILABLE:
        log.error("ib_insync no instalado. Corré: pip install ib_insync")
        return None
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=10)
        log.info(f"Conectado a IBKR (host={IB_HOST} port={IB_PORT})")
        return ib
    except Exception as e:
        log.error(f"No se pudo conectar a IBKR: {e}")
        log.error("Verificá que IB Gateway esté corriendo en modo Paper.")
        return None


def get_equity(ib):
    """Retorna el net liquidation value de la cuenta paper."""
    account = ib.accountValues()
    for av in account:
        if av.tag == 'NetLiquidation' and av.currency == 'USD':
            return float(av.value)
    return 0.0


# ══════════════════════════════════════════
# GESTIÓN DE ÓRDENES
# ══════════════════════════════════════════

def calc_position(equity, entry, atr_val, direction):
    """
    Calcula size de la posición basado en el riesgo por trade.
    Retorna: size (shares), stop, target
    """
    stop_dist = atr_val * ATR_STOP_MULT
    if stop_dist <= 0:
        return 0, 0, 0

    risk_usd  = equity * RISK_PER_TRADE
    size      = risk_usd / stop_dist
    max_size  = (equity * MAX_POS_PCT) / entry
    size      = min(size, max_size)
    size      = max(1, int(size))   # mínimo 1 share, entero

    if direction == 'LONG':
        stop   = round(entry - stop_dist, 2)
        target = round(entry + stop_dist * RR_RATIO, 2)
    else:
        stop   = round(entry + stop_dist, 2)
        target = round(entry - stop_dist * RR_RATIO, 2)

    return size, stop, target


def open_position(ib, ticker, direction, size, entry, stop, target):
    """
    Coloca una bracket order en IBKR:
      - Orden principal: limit al precio de entry
      - Stop loss: stop order
      - Take profit: limit order
    Los 3 están vinculados (OCA group).
    """
    contract = Stock(ticker, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    action      = 'BUY'  if direction == 'LONG' else 'SELL'
    close_action = 'SELL' if direction == 'LONG' else 'BUY'

    # Orden de entrada — limit a ±0.5% del precio para garantizar fill
    limit_price = round(entry * 1.005 if direction == 'LONG' else entry * 0.995, 2)

    bracket = ib.bracketOrder(
        action=action,
        quantity=size,
        limitPrice=limit_price,
        takeProfitPrice=target,
        stopLossPrice=stop,
    )

    orders = []
    for o in bracket:
        trade = ib.placeOrder(contract, o)
        orders.append(trade)
        log.info(f"  Orden colocada: {o.action} {o.totalQuantity} {ticker} @ {getattr(o,'lmtPrice',getattr(o,'auxPrice','mkt'))}")

    ib.sleep(1)
    return [o.order.orderId for o in orders]


def close_position(ib, ticker, state):
    """Cierra la posición abierta de un ticker con orden de mercado."""
    if ticker not in state['positions']:
        log.warning(f"{ticker} no tiene posición abierta.")
        return

    pos = state['positions'][ticker]
    contract = Stock(ticker, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    close_action = 'SELL' if pos['dir'] == 'LONG' else 'BUY'
    order = MarketOrder(close_action, pos['size'])  # type: ignore
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    log.info(f"Posición cerrada: {close_action} {pos['size']} {ticker}")


# ══════════════════════════════════════════
# LÓGICA PRINCIPAL DE TRADING
# ══════════════════════════════════════════

def run_signals(dry_run=False):
    """
    Ciclo principal:
    1. Genera señales de todos los tickers
    2. Compara con posiciones abiertas
    3. Cierra posiciones donde la señal cambió
    4. Abre posiciones donde hay señal nueva
    """
    state = load_state()
    log.info("═" * 50)
    log.info(f"Ejecutando señales | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Modo: {'DRY RUN (sin órdenes reales)' if dry_run else 'LIVE PAPER TRADING'}")

    log.info("Generando señales...")
    signals = get_all_signals()

    if dry_run:
        log.info("\nDRY RUN — señales generadas pero sin ejecutar.")
        for t, (sig, atr_val, price) in signals.items():
            d = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(sig, '?')
            current = state['positions'].get(t, {}).get('dir', 'FLAT')
            action = '(sin cambio)' if d == current else f'→ cambiar de {current} a {d}'
            log.info(f"  {t:<6} {d:<6} {action}")
        return

    ib = connect_ibkr()
    if not ib:
        log.error("Abortando — sin conexión a IBKR.")
        return

    equity = get_equity(ib)
    log.info(f"Equity cuenta: ${equity:,.2f}")

    for ticker, (sig, atr_val, price) in signals.items():
        if not price or not atr_val:
            continue

        current_pos = state['positions'].get(ticker)
        current_dir = current_pos['dir'] if current_pos else 'FLAT'
        new_dir     = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(sig, 'FLAT')

        if not ALLOW_SHORT and new_dir == 'SHORT':
            new_dir = 'FLAT'

        if current_dir == new_dir:
            log.info(f"{ticker}: sin cambio ({current_dir})")
            continue

        # Cerrar posición existente si la hay
        if current_pos:
            log.info(f"{ticker}: cerrando {current_dir}")
            close_position(ib, ticker, state)
            del state['positions'][ticker]
            save_state(state)

        # Abrir nueva posición si hay señal
        if new_dir in ('LONG', 'SHORT'):
            size, stop, target = calc_position(equity, price, atr_val, new_dir)
            if size < 1:
                log.warning(f"{ticker}: tamaño calculado < 1 share, skip")
                continue

            log.info(f"{ticker}: abriendo {new_dir}  "
                     f"entry={price:.2f}  stop={stop:.2f}  target={target:.2f}  size={size}")

            order_ids = open_position(ib, ticker, new_dir, size, price, stop, target)
            state['positions'][ticker] = {
                'dir':        new_dir,
                'entry':      price,
                'stop':       stop,
                'target':     target,
                'size':       size,
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
                'order_ids':  order_ids,
            }
            state['total_trades'] = state.get('total_trades', 0) + 1
            save_state(state)

    state['last_run'] = datetime.now().isoformat()
    save_state(state)
    ib.disconnect()
    log.info("Ciclo completado.")


# ══════════════════════════════════════════
# STATUS
# ══════════════════════════════════════════

def print_status():
    state = load_state()

    print("\n══ BOT STATUS ══")
    print(f"  Último run:    {state.get('last_run', 'nunca')}")
    print(f"  Total trades:  {state.get('total_trades', 0)}")

    positions = state.get('positions', {})
    if not positions:
        print("  Posiciones:    ninguna abierta\n")
        return

    print(f"\n  {'Ticker':<7} {'Dir':<6} {'Entry':>8} {'Stop':>8} {'Target':>8} {'Size':>6}  Fecha")
    print('  ' + '─' * 65)
    for t, p in positions.items():
        print(f"  {t:<7} {p['dir']:<6} {p['entry']:>8.2f} {p['stop']:>8.2f} "
              f"{p['target']:>8.2f} {p['size']:>6}  {p.get('entry_date','?')}")
    print()


# ══════════════════════════════════════════
# DAEMON — corre cada viernes al cierre
# ══════════════════════════════════════════

def start_daemon():
    if not SCHEDULE_AVAILABLE:
        print("Instalá schedule: pip install schedule")
        return

    log.info("Daemon iniciado. Ejecutará señales cada día a las 15:55 ET.")
    log.info("(Presioná Ctrl+C para detener)")

    # 15:55 ET = 5 minutos antes del cierre del mercado US
    # Lunes a viernes (schedule corre daily = todos los días pero el mercado
    # está cerrado fines de semana → sin señales, sin órdenes)
    schedule.every().day.at('15:55').do(run_signals)

    # Dry run inicial para verificar conectividad y señales
    run_signals(dry_run=True)

    while True:
        schedule.run_pending()
        time.sleep(60)


# ══════════════════════════════════════════
# CLI
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Trading Bot — Weekly Trend')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',      action='store_true', help='Ejecutar señales ahora')
    group.add_argument('--dry-run',  action='store_true', help='Generar señales sin colocar órdenes')
    group.add_argument('--status',   action='store_true', help='Ver posiciones abiertas')
    group.add_argument('--daemon',   action='store_true', help='Correr como daemon (cada día 15:55 ET)')
    group.add_argument('--close',    metavar='TICKER',    help='Cerrar posición (ticker o ALL)')
    args = parser.parse_args()

    if args.status:
        print_status()

    elif args.dry_run:
        run_signals(dry_run=True)

    elif args.run:
        run_signals(dry_run=False)

    elif args.daemon:
        start_daemon()

    elif args.close:
        state = load_state()
        ib = connect_ibkr()
        if not ib:
            return
        targets = list(state['positions'].keys()) if args.close == 'ALL' else [args.close.upper()]
        for t in targets:
            close_position(ib, t, state)
            if t in state['positions']:
                del state['positions'][t]
        save_state(state)
        ib.disconnect()


if __name__ == '__main__':
    main()
