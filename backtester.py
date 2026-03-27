"""
Backtester — Financial Terminal
Valida la estrategia técnica en datos históricos antes de ir live.

Estrategia: trend-following con risk management de 1% por trade.

Uso:
    python backtester.py                        # corre tickers default
    python backtester.py AAPL MSFT GOOGL        # tickers custom
    python backtester.py --start 2019-01-01     # desde fecha específica
"""

import sys
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


# ══════════════════════════════════════════
# INDICADORES (idénticos a los del frontend)
# ══════════════════════════════════════════

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    sig  = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def bollinger(series, period=20, std_mult=2):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std_mult * sigma, mid, mid - std_mult * sigma


def adx(high, low, close, period=14):
    """Average Directional Index — mide fuerza de tendencia (no dirección)."""
    h_diff   = high.diff()
    l_diff   = -low.diff()
    plus_dm  = h_diff.where((h_diff > l_diff) & (h_diff > 0), 0.0)
    minus_dm = l_diff.where((l_diff > h_diff) & (l_diff > 0), 0.0)
    tr_raw = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_s    = tr_raw.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


# ══════════════════════════════════════════
# FETCH DE DATOS (con cache en memoria)
# ══════════════════════════════════════════

_DATA_CACHE = {}

def fetch(ticker, start, end, interval='1d'):
    key = (ticker, start, end, interval)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    try:
        raw = yf.download(ticker, start=start, end=end,
                          interval=interval, auto_adjust=True, progress=False,
                          timeout=15)
    except Exception:
        _DATA_CACHE[key] = pd.DataFrame()
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.dropna()
    _DATA_CACHE[key] = raw
    return raw


# ══════════════════════════════════════════
# GENERADOR DE SEÑALES
# ══════════════════════════════════════════

def add_indicators(df):
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']

    df['ema20']  = ema(c, 20)
    df['ema50']  = ema(c, 50)
    df['ema200'] = ema(c, 200)
    df['rsi']    = rsi(c)
    df['macd_line'], df['macd_sig'], df['macd_hist'] = macd(c)
    df['atr']    = atr(h, l, c)
    df['adx']    = adx(h, l, c)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = bollinger(c)
    df['vol_ma']    = v.rolling(20).mean()
    df['vol_ratio'] = v / df['vol_ma']
    return df


def generate_signals_pullback(df, rsi_min=40, rsi_max=65, pullback_pct=1.02, vol_min=0.8):
    """
    PULLBACK: entrada cuando el precio retrocede a la EMA 20/50 en tendencia alcista.
    Funciona bien en índices y large caps estables.
    """
    c = df['Close']
    trend    = c > df['ema200']
    momentum = (df['rsi'] > rsi_min) & (df['rsi'] < rsi_max)
    pullback = (c <= df['ema20'] * pullback_pct) & (c >= df['ema50'] * 0.97)
    vol_ok   = df['vol_ratio'] > vol_min
    df['signal'] = 0
    df.loc[trend & momentum & pullback & vol_ok, 'signal'] = 1
    return df


def generate_signals_breakout(df, vol_spike=1.3, rsi_max=75, periods=20):
    """
    BREAKOUT: entrada cuando el precio rompe máximos de N días con volumen.
    Captura momentum stocks y tendencias sostenidas.
    """
    c = df['Close']
    h = df['High']
    high_n   = h.rolling(periods).max().shift(1)
    trend    = c > df['ema200']
    breakout = c > high_n
    vol_ok   = df['vol_ratio'] > vol_spike
    rsi_ok   = df['rsi'] < rsi_max
    df['signal'] = 0
    df.loc[trend & breakout & vol_ok & rsi_ok, 'signal'] = 1
    return df


def generate_signals_swing(df, ema_fast=9, ema_slow=21, adx_min=20,
                           rsi_long_max=65, rsi_short_min=35):
    """
    SWING: señales en timeframe diario para trades de 2-7 días.
      LONG  (+1): EMA rápida cruza arriba de EMA lenta + ADX tendencia
                  + RSI no sobrecomprado + MACD hist positivo
      SHORT (-1): EMA rápida cruza abajo de EMA lenta + ADX tendencia
                  + RSI no sobrevendido + MACD hist negativo
    Más activa que weekly_trend: 10-30 señales/año por ticker.
    Funciona en acciones de alta liquidez y en índices.
    """
    c = df['Close']

    df['ema_fast'] = ema(c, ema_fast)
    df['ema_slow'] = ema(c, ema_slow)

    prev_fast = df['ema_fast'].shift(1)
    prev_slow = df['ema_slow'].shift(1)
    trend_ok  = df['adx'] > adx_min

    cross_up   = (df['ema_fast'] > df['ema_slow']) & (prev_fast <= prev_slow)
    cross_down = (df['ema_fast'] < df['ema_slow']) & (prev_fast >= prev_slow)

    macd_bull = df['macd_hist'] > 0
    macd_bear = df['macd_hist'] < 0
    vol_ok    = df['vol_ratio'] > 0.8

    df['signal'] = 0
    df.loc[cross_up   & trend_ok & (df['rsi'] < rsi_long_max)  & macd_bull & vol_ok, 'signal'] =  1
    df.loc[cross_down & trend_ok & (df['rsi'] > rsi_short_min) & macd_bear & vol_ok, 'signal'] = -1
    return df


def generate_signals_weekly_trend(df, ema_fast=10, ema_slow=20, adx_min=20, rsi_long_max=75, rsi_short_min=25):
    """
    WEEKLY TREND: sigue la tendencia semanal con cruce de EMAs.
      LONG  (+1): EMA rápida cruza arriba de EMA lenta + ADX confirma tendencia.
      SHORT (-1): EMA rápida cruza abajo de EMA lenta + ADX confirma tendencia.
    Diseñada para datos semanales. Captura movimientos de semanas a meses.
    Evita mercados laterales via ADX > adx_min.
    """
    c = df['Close']

    df['ema_fast'] = ema(c, ema_fast)
    df['ema_slow'] = ema(c, ema_slow)

    prev_fast  = df['ema_fast'].shift(1)
    prev_slow  = df['ema_slow'].shift(1)
    trend_ok   = df['adx'] > adx_min

    cross_up   = (df['ema_fast'] > df['ema_slow']) & (prev_fast <= prev_slow)
    cross_down = (df['ema_fast'] < df['ema_slow']) & (prev_fast >= prev_slow)

    df['signal'] = 0
    df.loc[cross_up   & trend_ok & (df['rsi'] < rsi_long_max),  'signal'] =  1
    df.loc[cross_down & trend_ok & (df['rsi'] > rsi_short_min), 'signal'] = -1
    return df


def generate_signals(df, strategy='breakout', **params):
    if strategy == 'pullback':
        return generate_signals_pullback(df, **params)
    if strategy == 'weekly_trend':
        return generate_signals_weekly_trend(df, **params)
    if strategy == 'swing':
        return generate_signals_swing(df, **params)
    return generate_signals_breakout(df, **params)


# ══════════════════════════════════════════
# MOTOR DE BACKTEST
# ══════════════════════════════════════════

def backtest(ticker, start='2020-01-01', end=None,
             initial_capital=10_000,
             risk_per_trade=0.01,
             atr_stop_mult=1.5,
             rr_ratio=2.0,
             strategy='breakout',
             timeframe='1d',
             signal_params=None):
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    is_weekly  = (timeframe == '1wk')
    min_bars   = 104 if is_weekly else 250  # 2 años de semanas o ~1 año de días
    ann_factor = 52  if is_weekly else 252

    raw = fetch(ticker, start, end, interval=timeframe)
    if raw.empty or len(raw) < min_bars:
        return None

    df = add_indicators(raw)
    df = generate_signals(df, strategy=strategy, **(signal_params or {}))
    df = df.dropna()

    capital  = float(initial_capital)
    position = None
    trades   = []
    equity   = []

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        date = df.index[i]

        equity.append(capital)

        # ── Gestionar posición abierta ──────────────────────────────
        if position:
            low  = float(row['Low'])
            high = float(row['High'])
            sl   = position['stop']
            tp   = position['target']
            d    = position['dir']

            sl_hit = (low <= sl) if d == 'LONG' else (high >= sl)
            tp_hit = (high >= tp) if d == 'LONG' else (low <= tp)

            if sl_hit:
                pnl = (sl - position['entry']) * position['size'] * (1 if d == 'LONG' else -1)
                capital += pnl
                trades.append({**position, 'exit': sl, 'exit_date': date,
                               'pnl': pnl, 'result': 'SL'})
                position = None

            elif tp_hit:
                pnl = (tp - position['entry']) * position['size'] * (1 if d == 'LONG' else -1)
                capital += pnl
                trades.append({**position, 'exit': tp, 'exit_date': date,
                               'pnl': pnl, 'result': 'TP'})
                position = None

        # ── Abrir nueva posición (LONG o SHORT) ─────────────────────
        sig = int(prev['signal'])
        if position is None and sig in (1, -1):
            entry     = float(row['Open'])
            stop_dist = float(prev['atr']) * atr_stop_mult
            if stop_dist <= 0 or entry <= 0:
                continue

            if sig == 1:   # LONG
                stop   = entry - stop_dist
                target = entry + stop_dist * rr_ratio
            else:          # SHORT
                stop   = entry + stop_dist
                target = entry - stop_dist * rr_ratio

            risk_usd = capital * risk_per_trade
            size     = risk_usd / stop_dist
            max_size = (capital * 0.25) / entry
            size     = min(size, max_size)

            if size > 0:
                position = {
                    'ticker':     ticker,
                    'dir':        'LONG' if sig == 1 else 'SHORT',
                    'entry':      entry,
                    'stop':       stop,
                    'target':     target,
                    'size':       size,
                    'entry_date': date,
                }

    # Cerrar posición abierta al último precio
    if position:
        exit_price = float(df.iloc[-1]['Close'])
        d   = position['dir']
        pnl = (exit_price - position['entry']) * position['size'] * (1 if d == 'LONG' else -1)
        capital += pnl
        trades.append({**position, 'exit': exit_price,
                       'exit_date': df.index[-1], 'pnl': pnl, 'result': 'OPEN'})

    equity.append(capital)

    # ── Métricas ────────────────────────────────────────────────────
    eq = pd.Series(equity)
    period_ret = eq.pct_change().dropna()

    sharpe = (period_ret.mean() / period_ret.std() * np.sqrt(ann_factor)
              if period_ret.std() > 0 else 0.0)

    rolling_max = eq.cummax()
    max_dd = float(((eq - rolling_max) / rolling_max).min() * 100)

    total_ret = (capital - initial_capital) / initial_capital * 100
    buy_hold  = (float(df.iloc[-1]['Close']) / float(df.iloc[0]['Open']) - 1) * 100

    tdf = pd.DataFrame(trades)
    n   = len(tdf)
    if n > 0:
        wins     = (tdf['pnl'] > 0).sum()
        win_rate = wins / n * 100
        avg_win  = tdf.loc[tdf['pnl'] > 0, 'pnl'].mean() if wins > 0 else 0.0
        avg_loss = tdf.loc[tdf['pnl'] <= 0, 'pnl'].mean() if wins < n else 0.0
        profit_factor = (
            tdf.loc[tdf['pnl'] > 0, 'pnl'].sum() /
            abs(tdf.loc[tdf['pnl'] <= 0, 'pnl'].sum())
            if tdf.loc[tdf['pnl'] <= 0, 'pnl'].sum() != 0 else float('inf')
        )
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0

    return {
        'ticker':         ticker,
        'strategy':       strategy,
        'timeframe':      timeframe,
        'period':         f'{start} → {end}',
        'initial':        initial_capital,
        'final':          round(capital, 2),
        'total_return':   round(total_ret, 2),
        'buy_hold':       round(buy_hold, 2),
        'sharpe':         round(sharpe, 2),
        'max_drawdown':   round(max_dd, 2),
        'n_trades':       n,
        'win_rate':       round(win_rate, 2),
        'avg_win':        round(avg_win, 2),
        'avg_loss':       round(avg_loss, 2),
        'profit_factor':  round(profit_factor, 2),
        'trades':         tdf.to_dict('records') if not tdf.empty else [],
        'equity':         eq.tolist(),
    }


# ══════════════════════════════════════════
# EVALUADOR DE CALIDAD DE LA ESTRATEGIA
# ══════════════════════════════════════════

def evaluate(result):
    """
    Semáforo de calidad: PASS / MARGINAL / FAIL
    Un backtest tiene que pasar TODOS los criterios para ir a paper trading.
    """
    strat = result.get('strategy', '')
    min_trades = {'weekly_trend': 3, 'breakout': 5, 'swing': 8}.get(strat, 10)
    if not result or result['n_trades'] < min_trades:
        return 'FAIL', f"Solo {result['n_trades'] if result else 0} trades — muestra insuficiente"

    issues = []

    if result['sharpe'] < 0.8:
        issues.append(f"Sharpe {result['sharpe']} < 0.8 (mínimo aceptable)")
    if result['max_drawdown'] < -25:
        issues.append(f"Max drawdown {result['max_drawdown']}% < -25%")
    if result['win_rate'] < 35:
        issues.append(f"Win rate {result['win_rate']}% < 35% (con R/R 2:1 necesitás al menos 34%)")
    if result['profit_factor'] < 1.2:
        issues.append(f"Profit factor {result['profit_factor']} < 1.2")
    if result['total_return'] < 0:
        issues.append(f"Return negativo: {result['total_return']}%")

    if not issues:
        return 'PASS', 'Estrategia viable para paper trading'
    if len(issues) <= 2:
        return 'MARGINAL', ' | '.join(issues)
    return 'FAIL', ' | '.join(issues)


# ══════════════════════════════════════════
# CLI / REPORTE
# ══════════════════════════════════════════

COLORS = {
    'green':  '\033[92m',
    'yellow': '\033[93m',
    'red':    '\033[91m',
    'cyan':   '\033[96m',
    'bold':   '\033[1m',
    'reset':  '\033[0m',
}

def color(text, c):
    return f"{COLORS.get(c,'')}{text}{COLORS['reset']}"

def verdict_color(v):
    return {'PASS': 'green', 'MARGINAL': 'yellow', 'FAIL': 'red'}.get(v, 'reset')


def print_summary_table(results):
    header = (f"{'Ticker':<7} {'Return%':>8} {'B&H%':>8} {'Sharpe':>7} "
              f"{'MaxDD%':>8} {'WinRate':>8} {'PF':>6} {'Trades':>7}  Veredicto")
    print(color('\n' + header, 'bold'))
    print('─' * 90)

    for r in results:
        if r is None:
            continue
        verdict, reason = evaluate(r)
        vc = verdict_color(verdict)

        line = (f"{r['ticker']:<7} {r['total_return']:>7.1f}% {r['buy_hold']:>7.1f}% "
                f"{r['sharpe']:>7.2f} {r['max_drawdown']:>7.1f}% "
                f"{r['win_rate']:>7.1f}% {r['profit_factor']:>6.2f} {r['n_trades']:>7}  "
                f"{color(verdict, vc)}")
        print(line)

    print()


def portfolio_stats(results, ann_factor=252):
    """
    Simula el comportamiento del portfolio completo sumando los P&L de todos los tickers
    en cada período. Muestra el Sharpe y drawdown consolidados.
    """
    all_trades = []
    for r in results:
        for t in r.get('trades', []):
            all_trades.append(t)

    if not all_trades:
        print("Sin trades para calcular portfolio.")
        return

    tdf = pd.DataFrame(all_trades)
    tdf['exit_date'] = pd.to_datetime(tdf['exit_date'])
    tdf = tdf.sort_values('exit_date')

    # Capital acumulado sumando P&L de todos los trades en orden cronológico
    initial = 10_000
    capital = initial
    equity  = [capital]
    for _, row in tdf.iterrows():
        capital += row['pnl']
        equity.append(capital)

    eq       = pd.Series(equity)
    rets     = eq.pct_change().dropna()
    sharpe   = rets.mean() / rets.std() * np.sqrt(ann_factor) if rets.std() > 0 else 0
    roll_max = eq.cummax()
    max_dd   = float(((eq - roll_max) / roll_max).min() * 100)

    total_ret    = (capital - initial) / initial * 100
    n            = len(tdf)
    wins         = (tdf['pnl'] > 0).sum()
    win_rate     = wins / n * 100 if n > 0 else 0
    profit_factor = (tdf.loc[tdf['pnl'] > 0, 'pnl'].sum() /
                     abs(tdf.loc[tdf['pnl'] <= 0, 'pnl'].sum())
                     if tdf.loc[tdf['pnl'] <= 0, 'pnl'].sum() != 0 else float('inf'))
    longs  = (tdf['dir'] == 'LONG').sum()  if 'dir' in tdf.columns else n
    shorts = (tdf['dir'] == 'SHORT').sum() if 'dir' in tdf.columns else 0

    verdict, reason = evaluate({
        'n_trades': n, 'sharpe': round(sharpe, 2), 'max_drawdown': round(max_dd, 2),
        'win_rate': round(win_rate, 2), 'profit_factor': round(profit_factor, 2),
        'total_return': round(total_ret, 2), 'strategy': results[0].get('strategy', '')
    })
    vc = verdict_color(verdict)

    print(color("\n══ PORTFOLIO CONSOLIDADO ══", 'cyan'))
    print(f"  Tickers:        {', '.join(r['ticker'] for r in results)}")
    print(f"  Capital:        ${initial:,.0f} → ${capital:,.2f}")
    print(f"  Retorno total:  {total_ret:+.1f}%")
    print(f"  Sharpe:         {sharpe:.2f}")
    print(f"  Max Drawdown:   {max_dd:.1f}%")
    print(f"  Trades:         {n} total  ({longs} LONG / {shorts} SHORT)")
    print(f"  Win rate:       {win_rate:.1f}%")
    print(f"  Profit Factor:  {profit_factor:.2f}")
    print(f"  Veredicto:      {color(verdict, vc)}  —  {reason}")
    print()


def print_detail(r):
    verdict, reason = evaluate(r)
    vc = verdict_color(verdict)

    print(color(f"\n══ {r['ticker']} ══", 'cyan'))
    print(f"  Período:        {r['period']}")
    print(f"  Capital:        ${r['initial']:,.0f} → ${r['final']:,.2f}")
    print(f"  Retorno:        {r['total_return']:+.1f}%  (Buy&Hold: {r['buy_hold']:+.1f}%)")
    print(f"  Sharpe:         {r['sharpe']:.2f}")
    print(f"  Max Drawdown:   {r['max_drawdown']:.1f}%")
    print(f"  Trades:         {r['n_trades']}  |  Win rate: {r['win_rate']:.1f}%")
    print(f"  Avg win/loss:   +${r['avg_win']:.2f} / -${abs(r['avg_loss']):.2f}")
    print(f"  Profit Factor:  {r['profit_factor']:.2f}")
    print(f"  Veredicto:      {color(verdict, vc)}  —  {reason}")


# ══════════════════════════════════════════
# OPTIMIZADOR — GRID SEARCH CON TRAIN/TEST
# ══════════════════════════════════════════

PULLBACK_GRID = {
    'rsi_min':      [35, 40, 45],
    'rsi_max':      [60, 65, 70],
    'pullback_pct': [1.01, 1.02, 1.04],
    'vol_min':      [0.7, 0.9, 1.1],
}

BREAKOUT_GRID = {
    'vol_spike': [1.1, 1.3, 1.5, 2.0],
    'rsi_max':   [70, 75, 80],
    'periods':   [15, 20, 30],
}

WEEKLY_TREND_GRID = {
    'ema_fast':      [8, 10, 13],
    'ema_slow':      [20, 26, 34],
    'adx_min':       [15, 20, 25],
    'rsi_long_max':  [70, 75, 80],
    'rsi_short_min': [20, 25, 30],
}

SWING_GRID = {
    'ema_fast':      [8, 9, 13],
    'ema_slow':      [21, 26, 34],
    'adx_min':       [15, 20, 25],
    'rsi_long_max':  [60, 65, 70],
    'rsi_short_min': [30, 35, 40],
}

ATR_GRID = [1.0, 1.5, 2.0]
RR_GRID  = [1.5, 2.0, 2.5, 3.0]


def grid_combinations(param_grid):
    """Genera todas las combinaciones de un dict de listas."""
    import itertools
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def score_combo(tickers, start, end, strategy, signal_params, atr_mult, rr, timeframe='1d'):
    """
    Corre backtest en todos los tickers y devuelve un score compuesto.
    Score = media de Sharpe, penalizado por tickers con MaxDD < -20%.
    """
    strat_min = {'weekly_trend': 3, 'breakout': 5, 'swing': 8}.get(strategy, 10)
    sharpes = []
    for t in tickers:
        r = backtest(t, start=start, end=end, initial_capital=10_000,
                     risk_per_trade=0.01, atr_stop_mult=atr_mult,
                     rr_ratio=rr, strategy=strategy, timeframe=timeframe,
                     signal_params=signal_params)
        if not r or r['n_trades'] < strat_min:
            sharpes.append(-1.0)
        elif r['max_drawdown'] < -20:
            sharpes.append(r['sharpe'] * 0.5)
        else:
            sharpes.append(r['sharpe'])
    return float(np.mean(sharpes))


def optimize(tickers, strategy, start_train, end_train, start_test, end_test, timeframe='1d'):
    """
    1. Grid search en el período de entrenamiento.
    2. Toma los 3 mejores combos.
    3. Valida en el período de test (datos que el optimizador no vio).
    4. El combo que mantiene mejor Sharpe en test es el ganador.
    """
    signal_grid = {'pullback': PULLBACK_GRID, 'breakout': BREAKOUT_GRID,
                   'weekly_trend': WEEKLY_TREND_GRID,
                   'swing': SWING_GRID}.get(strategy, BREAKOUT_GRID)
    total = (sum(1 for _ in grid_combinations(signal_grid))
             * len(ATR_GRID) * len(RR_GRID))

    print(color(f"\nOptimizando {strategy.upper()} en {len(tickers)} tickers "
                f"| {total} combinaciones | Train: {start_train}→{end_train}", 'cyan'))
    print(f"Test (validación fuera de muestra): {start_test}→{end_test}\n")

    # Pre-descargar todos los datos una sola vez antes del grid search
    print("  Descargando datos...", end=' ', flush=True)
    valid_tickers = []
    for t in tickers:
        df_train = fetch(t, start_train, end_train, interval=timeframe)
        fetch(t, start_test, end_test, interval=timeframe)
        min_bars = 104 if timeframe == '1wk' else 250
        if not df_train.empty and len(df_train) >= min_bars:
            valid_tickers.append(t)
            print(t, end=' ', flush=True)
        else:
            print(f"({t} sin datos)", end=' ', flush=True)
    print()
    tickers = valid_tickers

    best = []
    done = 0
    for sp in grid_combinations(signal_grid):
        for atr in ATR_GRID:
            for rr in RR_GRID:
                s = score_combo(tickers, start_train, end_train,
                                strategy, sp, atr, rr, timeframe=timeframe)
                best.append((s, sp, atr, rr))
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{total}...", end='\r', flush=True)

    best.sort(key=lambda x: -x[0])
    top3 = best[:3]

    print(color("\nTop 3 combinaciones (train):", 'bold'))
    header = f"{'#':<3} {'Score':>6}  {'ATR':>5}  {'R/R':>5}  Params señal"
    print(header)
    print('─' * 70)
    for i, (sc, sp, atr, rr) in enumerate(top3, 1):
        params_str = '  '.join(f"{k}={v}" for k, v in sp.items())
        print(f"{i:<3} {sc:>6.3f}  {atr:>5.1f}  {rr:>5.1f}  {params_str}")

    print(color("\nValidación fuera de muestra (test):", 'bold'))
    header2 = (f"{'#':<3} {'Ticker':<7} {'Return%':>8} {'Sharpe':>7} "
               f"{'MaxDD%':>8} {'WinRate':>8} {'Trades':>7}  Veredicto")
    print(header2)
    print('─' * 75)

    winner = None
    best_test_score = -999
    for i, (train_score, sp, atr, rr) in enumerate(top3, 1):
        test_scores = []
        for t in tickers:
            r = backtest(t, start=start_test, end=end_test,
                         initial_capital=10_000, risk_per_trade=0.01,
                         atr_stop_mult=atr, rr_ratio=rr,
                         strategy=strategy, timeframe=timeframe, signal_params=sp)
            if r:
                verdict, _ = evaluate(r)
                vc = verdict_color(verdict)
                print(f"#{i}  {t:<7} {r['total_return']:>7.1f}% {r['sharpe']:>7.2f} "
                      f"{r['max_drawdown']:>7.1f}% {r['win_rate']:>7.1f}% "
                      f"{r['n_trades']:>7}  {color(verdict, vc)}")
                test_scores.append(r['sharpe'] if r['n_trades'] >= 10 else -1)

        avg_test = float(np.mean(test_scores)) if test_scores else -1
        print(f"    {'→ Score test medio:':30} {avg_test:.3f}\n")
        if avg_test > best_test_score:
            best_test_score = avg_test
            winner = (sp, atr, rr)

    if winner:
        sp, atr, rr = winner
        print(color("PARÁMETROS GANADORES (train + test):", 'green'))
        print(f"  Estrategia:  {strategy}")
        print(f"  ATR stop:    {atr}×")
        print(f"  R/R ratio:   {rr}:1")
        for k, v in sp.items():
            print(f"  {k:<15} {v}")
        print(f"  Score test:  {best_test_score:.3f}")

    return winner


def main():
    parser = argparse.ArgumentParser(description='Backtester de estrategia técnica')
    parser.add_argument('tickers', nargs='*',
                        default=['NVDA','SPY','QQQ','AMD','AAPL','MSFT','META','GOOGL','AMZN','TSLA','GLD','MELI'],
                        help='Tickers a testear')
    parser.add_argument('--start',   default='2020-01-01', help='Fecha inicio (YYYY-MM-DD)')
    parser.add_argument('--end',     default=None,          help='Fecha fin (YYYY-MM-DD)')
    parser.add_argument('--capital', default=10_000, type=float, help='Capital inicial en USD')
    parser.add_argument('--risk',    default=0.01,   type=float, help='Riesgo por trade (default 0.01 = 1%%)')
    parser.add_argument('--rr',       default=2.0,    type=float, help='Risk/Reward ratio (default 2.0)')
    parser.add_argument('--strategy', default='breakout',
                        choices=['breakout', 'pullback', 'weekly_trend', 'swing'],
                        help='Estrategia: breakout, pullback, weekly_trend')
    parser.add_argument('--compare',  action='store_true', help='Comparar ambas estrategias')
    parser.add_argument('--detail',   action='store_true', help='Mostrar detalle por ticker')
    parser.add_argument('--optimize', action='store_true', help='Buscar parámetros óptimos con train/test split')
    parser.add_argument('--train-end',  default='2023-12-31', help='Fin del período de entrenamiento (default 2023-12-31)')
    parser.add_argument('--timeframe',  default='1d', choices=['1d','1wk'], help='Timeframe: 1d (diario) o 1wk (semanal)')
    parser.add_argument('--portfolio',  action='store_true', help='Mostrar estadísticas consolidadas del portfolio')
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]

    if args.optimize:
        strat = args.strategy if not args.compare else 'pullback'
        optimize(tickers, strategy=strat,
                 start_train=args.start, end_train=args.train_end,
                 start_test=args.train_end, end_test=args.end or datetime.now().strftime('%Y-%m-%d'),
                 timeframe=args.timeframe)
        return

    print(color(f"\nBacktesting {len(tickers)} ticker(s) | {args.start} → {args.end or 'hoy'} "
                f"| Capital: ${args.capital:,.0f} | Riesgo: {args.risk*100:.1f}% | R/R: {args.rr}:1", 'bold'))

    strategies = ['breakout', 'pullback'] if args.compare else [args.strategy]

    for strat in strategies:
        if args.compare:
            print(color(f"\n── Estrategia: {strat.upper()} ──", 'cyan'))

        results = []
        for t in tickers:
            print(f"  Procesando {t}...", end=' ', flush=True)
            r = backtest(t, start=args.start, end=args.end,
                         initial_capital=args.capital,
                         risk_per_trade=args.risk,
                         rr_ratio=args.rr,
                         strategy=strat,
                         timeframe=args.timeframe)
            if r:
                r['strategy'] = strat
                print('ok')
                results.append(r)
                if args.detail:
                    print_detail(r)
            else:
                print('sin datos suficientes')

        print_summary_table(results)

        if args.portfolio and results:
            ann = 52 if args.timeframe == '1wk' else 252
            portfolio_stats([r for r in results if r], ann_factor=ann)

        pass_count = sum(1 for r in results if evaluate(r)[0] == 'PASS')
        marginal   = sum(1 for r in results if evaluate(r)[0] == 'MARGINAL')
        fail_count = sum(1 for r in results if evaluate(r)[0] == 'FAIL')
        print(color(f"Resumen [{strat}]: {pass_count} PASS  |  {marginal} MARGINAL  |  {fail_count} FAIL", 'bold'))
        if pass_count > 0:
            print(color("Los tickers con PASS son candidatos para paper trading.", 'green'))
    print()



if __name__ == '__main__':
    main()
