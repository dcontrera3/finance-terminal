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
  4. pip install ib_insync

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

from backtester import (
    fetch, add_indicators,
    generate_signals_swing, generate_signals_pullback, generate_signals_weekly_trend,
)
import notifier

try:
    from ib_insync import IB, Stock, LimitOrder, StopOrder, MarketOrder, Order, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

# ══════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════

TICKERS = [
    # Tech mega-cap
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    # Financiero / defensivo / consumer staples
    'JPM', 'V', 'UNH', 'KO',
    # Fintech y tech emergente internacional
    'NU', 'MELI', 'BABA',
    # Índices
    'SPY', 'QQQ', 'IWM',
    # Commodities
    'GLD', 'SLV', 'XLE',
    # Emergente LatAm
    'VIST',
]

# Estrategias que corren en paralelo.
# Cada una opera sobre la misma canasta pero con edges distintos:
#   swing:        breakout de 8 días + vol + ADX. LONG y SHORT. Timeframe diario.
#   pullback:     continuación en tendencia cuando el precio vuelve a EMA 20 con volumen. Diario.
#   weekly_trend: cruce de EMA 10/20 en timeframe semanal con ADX > 20. Trend-follower.
#
# Fase 1 activa: trailing stop (reemplaza TP fijo) + drawdown stop global.
# Fase 3.1: weekly_trend agregada tras backtest 5y (Sharpe 1.73 aislado, correlación
# -0.09/-0.13 con swing/pullback). Stops más amplios por ser timeframe semanal:
# 2 ATR semanales = ~2 velas de ruido que cortaban los winners.
STRATEGIES = {
    'swing': {
        'signal_fn':      generate_signals_swing,
        'timeframe':      '1d',
        'params':         dict(periods=8, vol_spike=1.1, adx_min=20,
                               rsi_long_max=70, rsi_short_min=35),
        'atr_stop_mult':  1.0,      # stop inicial a 1 ATR
        'trailing_atr':   2.0,      # trailing a 2 ATR (deja correr winners)
        'risk_per_trade': 0.0075,   # 0.75% (tres estrategias → 2.25% combinado)
        'allow_short':    True,
    },
    'pullback': {
        'signal_fn':      generate_signals_pullback,
        'timeframe':      '1d',
        'params':         dict(rsi_min=40, rsi_max=65, pullback_pct=1.02, vol_min=1.2),
        'atr_stop_mult':  1.5,
        'trailing_atr':   2.0,
        'risk_per_trade': 0.0075,
        'allow_short':    False,    # pullback solo entra LONG
    },
    'weekly_trend': {
        'signal_fn':      generate_signals_weekly_trend,
        'timeframe':      '1wk',
        'params':         dict(ema_fast=10, ema_slow=20, adx_min=20,
                               rsi_long_max=75, rsi_short_min=25),
        'atr_stop_mult':  2.0,      # stop inicial más amplio (timeframe semanal)
        'trailing_atr':   3.5,      # trailing más laxo: 2 ATR semanales eran 2 velas de ruido
        'risk_per_trade': 0.0075,
        'allow_short':    True,
    },
}

MAX_POS_PCT        = 0.25   # máximo 25% del equity por posición individual
DD_PAUSE_THRESHOLD = 0.10   # -10% desde el peak → pausar nuevas aperturas

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
            s = json.load(f)
            # migración: asegurar que existe history
            if 'history' not in s:
                s['history'] = []
            return s
    return {'positions': {}, 'last_run': None, 'total_trades': 0, 'history': []}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(state, event, ticker, strategy, direction, price,
              size=None, stop=None, target=None, entry=None, pnl=None,
              indicators=None):
    """Agrega una entrada al historial de trades del bot.

    `indicators` es un dict con el snapshot al momento de la señal
    (RSI, EMAs, MACD, ADX, ATR) — solo se llena en eventos de apertura.
    """
    state['history'].append({
        'ts':         datetime.now().isoformat(timespec='seconds'),
        'event':      event,           # 'open' | 'close'
        'ticker':     ticker,
        'strategy':   strategy,
        'dir':        direction,
        'price':      round(price, 2) if price else None,
        'size':       size,
        'stop':       stop,
        'target':     target,
        'entry':      entry,
        'pnl':        round(pnl, 2) if pnl is not None else None,
        'indicators': indicators,
    })


# ══════════════════════════════════════════
# SEÑALES (reutiliza la lógica del backtester)
# ══════════════════════════════════════════

def pos_key(ticker, strategy):
    """Clave compuesta para estado persistente en JSON."""
    return f"{ticker}__{strategy}"


def _indicator_snapshot(row):
    """Extrae un snapshot serializable de indicadores desde una fila del df."""
    def _f(v, nd=2):
        try:
            x = float(v)
            if np.isnan(x):
                return None
            return round(x, nd)
        except (TypeError, ValueError):
            return None
    return {
        'rsi':       _f(row.get('rsi'), 2),
        'ema20':     _f(row.get('ema20'), 2),
        'ema50':     _f(row.get('ema50'), 2),
        'ema200':    _f(row.get('ema200'), 2),
        'macd':      _f(row.get('macd_line'), 4),
        'macd_sig':  _f(row.get('macd_sig'), 4),
        'macd_hist': _f(row.get('macd_hist'), 4),
        'adx':       _f(row.get('adx'), 2),
        'atr':       _f(row.get('atr'), 2),
        'close':     _f(row.get('Close'), 2),
        'volume':    _f(row.get('Volume'), 0),
    }


def get_signal(ticker, strategy_name):
    """
    Devuelve la señal de la última barra completa para la estrategia dada.
      +1 = LONG | -1 = SHORT | 0 = sin señal
    Retorna: (signal, atr, price, snapshot) donde snapshot es el dict de
    indicadores en la barra que generó la señal (o None si no hay datos).

    El timeframe viene por estrategia: diario o semanal. Weekly necesita más
    histórico porque add_indicators calcula EMA 200 (200 barras semanales ≈ 4 años).
    """
    cfg = STRATEGIES[strategy_name]
    timeframe = cfg.get('timeframe', '1d')

    # Window de fetch: daily 1 año es ~252 barras, weekly 5 años son ~260 barras.
    fetch_days = 365 * 5 if timeframe == '1wk' else 365
    # Umbral mínimo de barras después del dropna
    min_bars = 30 if timeframe == '1wk' else 50

    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=fetch_days)).strftime('%Y-%m-%d')

    df = fetch(ticker, start, end, interval=timeframe)
    if df.empty or len(df) < min_bars:
        log.warning(f"{ticker} [{strategy_name}]: datos insuficientes ({len(df)} barras)")
        return 0, None, None, None

    df = add_indicators(df)
    df = cfg['signal_fn'](df, **cfg['params'])
    df = df.dropna()

    if len(df) < 2:
        return 0, None, None, None

    # Última barra completa (no la del período en curso)
    last = df.iloc[-2]
    signal   = int(last['signal'])
    atr_val  = float(last['atr'])
    price    = float(df.iloc[-1]['Close'])
    snapshot = _indicator_snapshot(last)
    return signal, atr_val, price, snapshot


def get_all_signals():
    """
    Devuelve dict {(ticker, strategy): (signal, atr, price, snapshot)}
    para toda combinación ticker × estrategia.
    """
    signals = {}
    for strategy_name in STRATEGIES:
        log.info(f"── Señales {strategy_name.upper()} ──")
        for t in TICKERS:
            sig, atr_val, price, snap = get_signal(t, strategy_name)
            signals[(t, strategy_name)] = (sig, atr_val, price, snap)
            d = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(sig, '?')
            if price:
                log.info(f"  {t:<6} {d:<6}  precio=${price:.2f}  ATR={atr_val:.2f}")
            else:
                log.info(f"  {t:<6} sin datos")
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
        # Market data type 3 = delayed frozen. Si la cuenta tiene suscripción
        # live, IBKR igual devuelve datos realtime; si no, cae a delayed (~15
        # min, gratis). Sin esto, paper sin market data devuelve last=None y
        # el precio cae al cierre del día anterior.
        ib.reqMarketDataType(3)
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

def calc_position(equity, entry, atr_val, direction, strategy_name):
    """
    Calcula sizing y niveles según los parámetros de la estrategia.
    Retorna: size (shares), stop_inicial, trailing_distance (en USD)
    El trailing stop usará la distancia retornada para moverse con el precio.
    """
    cfg = STRATEGIES[strategy_name]
    stop_dist     = atr_val * cfg['atr_stop_mult']
    trail_dist    = atr_val * cfg['trailing_atr']
    if stop_dist <= 0:
        return 0, 0, 0

    risk_usd = equity * cfg['risk_per_trade']
    size     = risk_usd / stop_dist
    max_size = (equity * MAX_POS_PCT) / entry
    size     = min(size, max_size)
    size     = max(1, int(size))

    if direction == 'LONG':
        stop = round(entry - stop_dist, 2)
    else:
        stop = round(entry + stop_dist, 2)

    return size, stop, round(trail_dist, 2)


def open_position(ib, ticker, direction, size, entry, stop, trail_distance):
    """
    Coloca un bracket order con trailing stop en IBKR:
      - Parent: LimitOrder de entrada
      - Child (attached): TRAIL order que se mueve con el precio favorable
    IBKR gestiona el trailing automáticamente en su servidor.
    """
    contract = Stock(ticker, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    action       = 'BUY'  if direction == 'LONG' else 'SELL'
    close_action = 'SELL' if direction == 'LONG' else 'BUY'

    # Entrada: limit a ±0.5% del precio para garantizar fill
    limit_price = round(entry * 1.005 if direction == 'LONG' else entry * 0.995, 2)

    parent_id = ib.client.getReqId()
    parent = LimitOrder(action, size, limit_price)
    parent.orderId  = parent_id
    parent.transmit = False          # no transmitir hasta enviar el hijo

    # Child: TRAIL order — IBKR mueve el stop automáticamente
    trail = Order()
    trail.action          = close_action
    trail.orderType       = 'TRAIL'
    trail.totalQuantity   = size
    trail.auxPrice        = float(trail_distance)   # distancia en USD
    trail.trailStopPrice  = float(stop)             # stop inicial
    trail.parentId        = parent_id
    trail.transmit        = True     # dispara parent + child juntos

    parent_trade = ib.placeOrder(contract, parent)
    trail_trade  = ib.placeOrder(contract, trail)
    ib.sleep(1)

    log.info(f"  Parent: {action} {size} {ticker} @limit ${limit_price}")
    log.info(f"  Trail:  {close_action} {size} {ticker}  trail=${trail_distance:.2f}  "
             f"stop0=${stop:.2f}")

    return [parent_trade.order.orderId, trail_trade.order.orderId]


def get_market_price(ib, ticker):
    """Precio actual de mercado desde IBKR.

    Prioriza datos live/delayed reales (last, mid de bid/ask). NUNCA cae al
    campo `close` porque ése es el cierre del día anterior — usar eso como
    fallback producía entries con precios viejos.

    Retorna None si no hay datos, para que el caller pueda skipear el trade
    en vez de operar con un precio desactualizado.
    """
    import math
    def _valid(x):
        try:
            v = float(x)
            return v if not math.isnan(v) and v > 0 else None
        except (TypeError, ValueError):
            return None

    try:
        contract = Stock(ticker, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        td = ib.reqMktData(contract, '', False, False)

        # Esperar hasta 5s a que lleguen datos (delayed puede tardar más que live)
        price = None
        for _ in range(10):
            ib.sleep(0.5)
            # Campos en orden de preferencia: last real, last delayed, mid bid/ask
            for field in ('last', 'delayedLast'):
                v = _valid(getattr(td, field, None))
                if v:
                    price = v
                    break
            if price:
                break
            bid = _valid(getattr(td, 'bid', None)) or _valid(getattr(td, 'delayedBid', None))
            ask = _valid(getattr(td, 'ask', None)) or _valid(getattr(td, 'delayedAsk', None))
            if bid and ask:
                price = (bid + ask) / 2
                break

        ib.cancelMktData(contract)
        return price
    except Exception as e:
        log.warning(f"get_market_price({ticker}) falló: {e}")
        return None


def close_position(ib, key, state):
    """Cierra la posición identificada por su clave (ticker__strategy)."""
    if key not in state['positions']:
        log.warning(f"{key} no tiene posición abierta.")
        return None

    pos = state['positions'][key]
    ticker = pos.get('ticker', key.split('__')[0])
    strat  = pos.get('strategy', '?')

    contract = Stock(ticker, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    close_action = 'SELL' if pos['dir'] == 'LONG' else 'BUY'
    order = MarketOrder(close_action, pos['size'])  # type: ignore
    ib.placeOrder(contract, order)
    ib.sleep(1)

    # P&L estimado usando el último precio disponible
    exit_price = get_market_price(ib, ticker) or pos.get('entry')
    pnl = None
    if exit_price and pos.get('entry'):
        mult = 1 if pos['dir'] == 'LONG' else -1
        pnl = (exit_price - pos['entry']) * pos['size'] * mult

    log.info(f"Posición cerrada: {close_action} {pos['size']} {ticker} "
             f"({strat})  P&L≈${pnl:.2f}" if pnl is not None
             else f"Posición cerrada: {close_action} {pos['size']} {ticker} ({strat})")

    log_trade(state, 'close', ticker, strat, pos['dir'],
              price=exit_price, size=pos['size'], entry=pos.get('entry'), pnl=pnl,
              indicators=pos.get('indicators'))

    pnl_str = f"${pnl:+,.2f}" if pnl is not None else "?"
    notifier.notify(
        f"🔴 <b>Cierre</b>  {ticker} [{strat}] {pos['dir']}\n"
        f"Exit ≈ ${exit_price:.2f}  |  Entry ${pos.get('entry',0):.2f}\n"
        f"P&L ≈ <b>{pnl_str}</b>"
        if exit_price else
        f"🔴 <b>Cierre</b>  {ticker} [{strat}] {pos['dir']}  (precio indet.)"
    )
    return pnl


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
    log.info(f"Estrategias activas: {', '.join(STRATEGIES.keys())}")
    notifier.startup_msg()

    log.info("Generando señales...")
    signals = get_all_signals()

    if dry_run:
        log.info("\nDRY RUN — señales generadas pero sin ejecutar.")
        for (ticker, strat), (sig, atr_val, price, _snap) in signals.items():
            d = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(sig, '?')
            key = pos_key(ticker, strat)
            current = state['positions'].get(key, {}).get('dir', 'FLAT')
            action = '(sin cambio)' if d == current else f'→ cambiar de {current} a {d}'
            log.info(f"  [{strat:<8}] {ticker:<6} {d:<6} {action}")
        return

    ib = connect_ibkr()
    if not ib:
        log.error("Abortando — sin conexión a IBKR.")
        return

    equity = get_equity(ib)
    log.info(f"Equity cuenta: ${equity:,.2f}")

    # Precios live de IBKR (la señal se genera con yfinance pero la entrada
    # se ejecuta con el precio real del broker para evitar usar el close del
    # día anterior cuando yfinance droppea la vela parcial del día en curso).
    log.info("── Precios real-time (IBKR) ──")
    ibkr_prices = {}
    yf_price_by_ticker = {}
    for (tk, _strat), (_sig, _atr, pr, _snap) in signals.items():
        if pr and tk not in yf_price_by_ticker:
            yf_price_by_ticker[tk] = pr
    for t in TICKERS:
        p = get_market_price(ib, t)
        ibkr_prices[t] = p
        yf_p = yf_price_by_ticker.get(t)
        if p and yf_p:
            diff_pct = (p - yf_p) / yf_p * 100
            log.info(f"  {t:<6} IBKR ${p:>9.2f}   yf ${yf_p:>9.2f}   Δ {diff_pct:+.2f}%")
        elif p:
            log.info(f"  {t:<6} IBKR ${p:>9.2f}")
        else:
            log.warning(f"  {t:<6} sin precio IBKR (fallback yfinance)")

    # Tracking del peak para el drawdown stop
    peak_equity = max(state.get('peak_equity', equity), equity)
    state['peak_equity']    = peak_equity
    state['current_equity'] = equity
    current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
    dd_paused  = current_dd < -DD_PAUSE_THRESHOLD
    if dd_paused:
        log.warning(f"⚠ DRAWDOWN STOP ACTIVO  |  DD={current_dd*100:.2f}% "
                    f"(peak=${peak_equity:,.0f}, actual=${equity:,.0f}) "
                    f"→ no se abren nuevas posiciones")
        notifier.notify(
            f"⚠️ <b>Drawdown stop activo</b>\n"
            f"DD {current_dd*100:.2f}% desde peak ${peak_equity:,.0f}\n"
            f"Nuevas aperturas pausadas hasta recuperar"
        )
    else:
        log.info(f"DD actual: {current_dd*100:.2f}%  |  Peak: ${peak_equity:,.0f}")

    cycle_summary = {'opened': [], 'closed': [], 'equity': equity,
                     'dd': current_dd, 'peak': peak_equity, 'dd_paused': dd_paused}

    for (ticker, strat), (sig, atr_val, price_yf, snap) in signals.items():
        if not atr_val:
            continue

        # Precio de ejecución: EXIGIMOS precio real de IBKR. Si no hay, skipeamos
        # el trade — yfinance trae el close del día anterior y ya nos quemamos
        # con eso (MELI entrando 1855 cuando el spot era otro). Mejor perder
        # una oportunidad que entrar con data stale.
        price = ibkr_prices.get(ticker)
        if not price:
            new_dir_preview = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(sig, 'FLAT')
            if new_dir_preview != 'FLAT':
                log.warning(f"[{strat}] {ticker}: señal {new_dir_preview} descartada — "
                            f"sin precio IBKR real-time")
            continue

        cfg = STRATEGIES[strat]
        key = pos_key(ticker, strat)

        current_pos = state['positions'].get(key)
        current_dir = current_pos['dir'] if current_pos else 'FLAT'
        new_dir     = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(sig, 'FLAT')

        if not cfg['allow_short'] and new_dir == 'SHORT':
            new_dir = 'FLAT'

        if current_dir == new_dir:
            continue   # sin log para evitar ruido en canasta grande

        # Cerrar posición existente
        if current_pos:
            log.info(f"[{strat}] {ticker}: cerrando {current_dir}")
            pnl = close_position(ib, key, state)
            cycle_summary['closed'].append((ticker, strat, current_dir, pnl))
            del state['positions'][key]
            save_state(state)

        # Abrir nueva posición (pausada si estamos en drawdown stop)
        if new_dir in ('LONG', 'SHORT'):
            if dd_paused:
                log.info(f"[{strat}] {ticker}: señal {new_dir} ignorada (DD stop activo)")
                continue

            size, stop, trail_dist = calc_position(equity, price, atr_val, new_dir, strat)
            if size < 1:
                log.warning(f"[{strat}] {ticker}: tamaño calculado < 1 share, skip")
                continue

            log.info(f"[{strat}] {ticker}: abriendo {new_dir}  "
                     f"entry={price:.2f}  stop={stop:.2f}  trail=${trail_dist:.2f}  "
                     f"size={size}")

            order_ids = open_position(ib, ticker, new_dir, size, price, stop, trail_dist)
            state['positions'][key] = {
                'ticker':     ticker,
                'strategy':   strat,
                'dir':        new_dir,
                'entry':      price,
                'stop':       stop,
                'trail_dist': trail_dist,
                'size':       size,
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
                'order_ids':  order_ids,
                'indicators': snap,
            }
            state['total_trades'] = state.get('total_trades', 0) + 1

            log_trade(state, 'open', ticker, strat, new_dir,
                      price=price, size=size, stop=stop, target=None, entry=price,
                      indicators=snap)
            cycle_summary['opened'].append((ticker, strat, new_dir, price, stop, trail_dist, size))

            notifier.notify(
                f"🟢 <b>Apertura</b>  {ticker} [{strat}] {new_dir}\n"
                f"Entry ${price:.2f}  |  Size {size}\n"
                f"SL ${stop:.2f}   |  Trail ${trail_dist:.2f} ({2} ATR)"
            )

            save_state(state)

    state['last_run'] = datetime.now().isoformat()
    save_state(state)
    ib.disconnect()

    # Resumen por Telegram
    n_open   = len(cycle_summary['opened'])
    n_close  = len(cycle_summary['closed'])
    open_pos = len(state['positions'])
    if n_open or n_close:
        lines = [f"📊 <b>Ciclo completado</b>",
                 f"Equity ${cycle_summary['equity']:,.0f}  |  Posiciones activas {open_pos}"]
        if n_close:
            lines.append(f"\n<b>Cierres ({n_close}):</b>")
            for t, s, d, p in cycle_summary['closed']:
                p_str = f"${p:+,.0f}" if p is not None else "?"
                lines.append(f"  • {t} [{s}] {d} → {p_str}")
        if n_open:
            lines.append(f"\n<b>Aperturas ({n_open}):</b>")
            for t, s, d, entry, sl, trail, sz in cycle_summary['opened']:
                lines.append(f"  • {t} [{s}] {d} × {sz}  @${entry:.2f}")
        dd_tag = f"  |  DD {cycle_summary['dd']*100:.1f}%" if cycle_summary['dd'] < 0 else ""
        lines.append(f"\nPeak ${cycle_summary['peak']:,.0f}{dd_tag}")
        notifier.notify('\n'.join(lines))
    else:
        dd_tag = f"  |  DD {cycle_summary['dd']*100:.1f}%" if cycle_summary['dd'] < 0 else ""
        notifier.notify(
            f"📊 Ciclo completado — sin cambios.\n"
            f"Equity ${cycle_summary['equity']:,.0f}  |  Pos {open_pos}{dd_tag}"
        )

    log.info("Ciclo completado.")


# ══════════════════════════════════════════
# STATUS
# ══════════════════════════════════════════

def _status_lines():
    """Genera las líneas del status (reutilizable para print y Telegram)."""
    state = load_state()
    lines = []
    lines.append("══ BOT STATUS ══")
    lines.append(f"  Último run:    {state.get('last_run', 'nunca')}")
    lines.append(f"  Total trades:  {state.get('total_trades', 0)}")

    positions = state.get('positions', {})
    if not positions:
        lines.append("  Posiciones:    ninguna abierta")
        return lines, state

    lines.append("")
    lines.append(f"  {'Ticker':<7} {'Estr':<9} {'Dir':<6} {'Entry':>8} {'Stop':>8} "
                 f"{'Trail':>7} {'Size':>6}  Fecha")
    lines.append("  " + "─" * 75)
    for key, p in positions.items():
        ticker = p.get('ticker', key.split('__')[0])
        strat  = p.get('strategy', key.split('__')[-1] if '__' in key else '?')
        trail  = p.get('trail_dist', 0) or 0
        lines.append(f"  {ticker:<7} {strat:<9} {p['dir']:<6} {p['entry']:>8.2f} "
                     f"{p['stop']:>8.2f} ${trail:>6.2f} {p['size']:>6}  "
                     f"{p.get('entry_date','?')}")
    # DD info
    peak = state.get('peak_equity', 0)
    lines.append(f"\n  Peak equity:     ${peak:,.2f}" if peak else "")
    return lines, state


def print_status():
    lines, _ = _status_lines()
    print("\n" + "\n".join(lines) + "\n")


def notify_status():
    """Manda el status actual al chat de Telegram."""
    lines, state = _status_lines()
    positions = state.get('positions', {})
    msg_lines = [f"📋 <b>Status del bot</b>",
                 f"Último run: {state.get('last_run','nunca')}",
                 f"Trades totales: {state.get('total_trades',0)}",
                 f"Posiciones abiertas: {len(positions)}"]
    if positions:
        msg_lines.append("")
        for key, p in positions.items():
            ticker = p.get('ticker', key.split('__')[0])
            strat  = p.get('strategy', '?')
            trail  = p.get('trail_dist', 0) or 0
            msg_lines.append(f"• {ticker} [{strat}] {p['dir']} × {p['size']} "
                             f"@${p['entry']:.2f}  SL${p['stop']:.2f}  trail${trail:.2f}")
    sent = notifier.notify('\n'.join(msg_lines))
    if not sent:
        print("Telegram no configurado o falló el envío.")
        print_status()


def print_history(n=20):
    state = load_state()
    history = state.get('history', [])
    if not history:
        print("\nSin historial aún.\n")
        return

    print(f"\n══ HISTORIAL — últimos {min(n, len(history))} eventos ══\n")
    print(f"  {'Fecha':<20} {'Evento':<6} {'Ticker':<6} {'Estr':<9} "
          f"{'Dir':<5} {'Precio':>8} {'Size':>5} {'P&L':>10}")
    print("  " + "─" * 80)
    total_pnl = 0.0
    for e in history[-n:]:
        ts    = e.get('ts', '?')[:19].replace('T', ' ')
        ev    = e.get('event', '?')
        tk    = e.get('ticker', '?')
        st    = e.get('strategy', '?')
        d     = e.get('dir', '?')
        pr    = e.get('price', 0) or 0
        sz    = e.get('size', '') or ''
        pnl   = e.get('pnl')
        pnl_s = f"${pnl:+,.2f}" if pnl is not None else ""
        if pnl is not None:
            total_pnl += pnl
        print(f"  {ts:<20} {ev:<6} {tk:<6} {st:<9} {d:<5} "
              f"{pr:>8.2f} {str(sz):>5} {pnl_s:>10}")
    print("  " + "─" * 80)
    print(f"  P&L acumulado (eventos mostrados): ${total_pnl:+,.2f}\n")


# ══════════════════════════════════════════
# DAEMON — corre cada viernes al cierre
# ══════════════════════════════════════════

def start_daemon():
    # Timezone-aware: siempre ejecuta 15:55 hora de Nueva York (= 5 min antes
    # del cierre del mercado US), independiente del timezone del server.
    # Arregla el bug previo donde schedule.at('15:55') usaba hora local —
    # si el server estaba en Argentina corría a las 15:55 ART = 14:55 ET,
    # 65 min antes del cierre.
    from zoneinfo import ZoneInfo
    ny = ZoneInfo('America/New_York')

    log.info("Daemon iniciado. Ejecutará señales cada día hábil a las 15:55 ET.")
    log.info("(Presioná Ctrl+C para detener)")

    # Dry run inicial para verificar conectividad y señales
    run_signals(dry_run=True)

    last_run_date = None
    while True:
        now_ny = datetime.now(ny)
        # Lunes-viernes, 15:55 NY, una vez por día
        if (now_ny.weekday() < 5
                and now_ny.hour == 15 and now_ny.minute == 55
                and now_ny.date() != last_run_date):
            log.info(f"Trigger {now_ny:%Y-%m-%d %H:%M %Z} → ejecutando señales")
            try:
                run_signals()
            except Exception as e:
                log.exception(f"Error en run_signals: {e}")
            last_run_date = now_ny.date()
        time.sleep(30)


# ══════════════════════════════════════════
# CLI
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Trading Bot — Weekly Trend')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',      action='store_true', help='Ejecutar señales ahora')
    group.add_argument('--dry-run',  action='store_true', help='Generar señales sin colocar órdenes')
    group.add_argument('--status',   action='store_true', help='Ver posiciones abiertas')
    group.add_argument('--history',  nargs='?', const=20, type=int, metavar='N',
                       help='Ver últimos N eventos del historial (default 20)')
    group.add_argument('--notify-status', action='store_true',
                       help='Mandar el status actual por Telegram')
    group.add_argument('--daemon',   action='store_true', help='Correr como daemon (cada día 15:55 ET)')
    group.add_argument('--close',    metavar='TICKER',    help='Cerrar posición (ticker o ALL)')
    args = parser.parse_args()

    if args.status:
        print_status()

    elif args.history is not None:
        print_history(args.history)

    elif args.notify_status:
        notify_status()

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
        if args.close == 'ALL':
            keys = list(state['positions'].keys())
        else:
            # Cerrar todas las posiciones del ticker en todas las estrategias
            tk = args.close.upper()
            keys = [k for k in state['positions']
                    if state['positions'][k].get('ticker', k.split('__')[0]) == tk]
        for k in keys:
            close_position(ib, k, state)
            if k in state['positions']:
                del state['positions'][k]
        save_state(state)
        ib.disconnect()


if __name__ == '__main__':
    main()
