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


# ══════════════════════════════════════════
# FETCH DE DATOS
# ══════════════════════════════════════════

def fetch(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end,
                      interval='1d', auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.dropna()
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
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = bollinger(c)
    df['vol_ma'] = v.rolling(20).mean()
    df['vol_ratio'] = v / df['vol_ma']
    return df


def generate_signals_pullback(df):
    """
    PULLBACK: entrada cuando el precio retrocede a la EMA 20/50 en tendencia alcista.
    Funciona bien en mercados que respetan las medias (SPY, QQQ, large caps estables).
    Falla en momentum stocks que no pullbackean (NVDA, AMD en bull runs).
    """
    c = df['Close']

    trend    = c > df['ema200']
    momentum = (df['rsi'] > 40) & (df['rsi'] < 65)
    # Pullback: precio entre EMA50 y EMA20, o cerca de EMA20
    pullback = (c <= df['ema20'] * 1.02) & (c >= df['ema50'] * 0.97)
    vol_ok   = df['vol_ratio'] > 0.8

    df['signal'] = 0
    df.loc[trend & momentum & pullback & vol_ok, 'signal'] = 1
    return df


def generate_signals_breakout(df):
    """
    BREAKOUT: entrada cuando el precio rompe máximos de 20 días con volumen.
    Captura momentum stocks (NVDA, AMD) y tendencias sostenidas.
    Más trades falsos en mercados laterales — el filtro EMA200 lo mitiga.
    """
    c = df['Close']
    h = df['High']

    high_20   = h.rolling(20).max().shift(1)   # máximo de los 20 días previos
    trend     = c > df['ema200']               # solo en tendencia alcista
    breakout  = c > high_20                    # rompe máximo reciente
    vol_spike = df['vol_ratio'] > 1.3          # volumen 30% sobre promedio
    rsi_ok    = df['rsi'] < 75                 # no comprar si RSI extremo

    df['signal'] = 0
    df.loc[trend & breakout & vol_spike & rsi_ok, 'signal'] = 1
    return df


def generate_signals(df, strategy='breakout'):
    if strategy == 'pullback':
        return generate_signals_pullback(df)
    return generate_signals_breakout(df)


# ══════════════════════════════════════════
# MOTOR DE BACKTEST
# ══════════════════════════════════════════

def backtest(ticker, start='2020-01-01', end=None,
             initial_capital=10_000,
             risk_per_trade=0.01,   # 1% del portfolio por trade
             atr_stop_mult=1.5,     # stop = entry - ATR × 1.5
             rr_ratio=2.0,          # target = riesgo × rr_ratio
             strategy='breakout'):  # 'breakout' o 'pullback'
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    raw = fetch(ticker, start, end)
    if raw.empty or len(raw) < 250:
        return None

    df = add_indicators(raw)
    df = generate_signals(df, strategy=strategy)
    df = df.dropna()

    capital  = float(initial_capital)
    position = None  # dict con los datos del trade abierto
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

            if low <= sl:
                # Stop loss tocado — salida al SL (worst case del día)
                pnl = (sl - position['entry']) * position['size']
                capital += pnl
                trades.append({**position, 'exit': sl, 'exit_date': date,
                               'pnl': pnl, 'result': 'SL'})
                position = None

            elif high >= tp:
                # Take profit tocado
                pnl = (tp - position['entry']) * position['size']
                capital += pnl
                trades.append({**position, 'exit': tp, 'exit_date': date,
                               'pnl': pnl, 'result': 'TP'})
                position = None

        # ── Abrir nueva posición ────────────────────────────────────
        if position is None and prev['signal'] == 1:
            entry        = float(row['Open'])   # entramos al open del día siguiente
            stop_dist    = float(prev['atr']) * atr_stop_mult
            stop         = entry - stop_dist
            target       = entry + stop_dist * rr_ratio

            risk_usd     = capital * risk_per_trade
            size         = risk_usd / stop_dist  # shares

            # No invertir más del 25% del capital en una sola posición
            max_size = (capital * 0.25) / entry
            size     = min(size, max_size)

            if size > 0 and entry > 0:
                position = {
                    'ticker':     ticker,
                    'entry':      entry,
                    'stop':       stop,
                    'target':     target,
                    'size':       size,
                    'entry_date': date,
                }

    # Cerrar posición abierta al último precio
    if position:
        exit_price = float(df.iloc[-1]['Close'])
        pnl = (exit_price - position['entry']) * position['size']
        capital += pnl
        trades.append({**position, 'exit': exit_price,
                       'exit_date': df.index[-1], 'pnl': pnl, 'result': 'OPEN'})

    equity.append(capital)

    # ── Métricas ────────────────────────────────────────────────────
    eq = pd.Series(equity)
    daily_ret = eq.pct_change().dropna()

    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

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
    if not result or result['n_trades'] < 10:
        return 'FAIL', 'Menos de 10 trades — muestra insuficiente'

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


def main():
    parser = argparse.ArgumentParser(description='Backtester de estrategia técnica')
    parser.add_argument('tickers', nargs='*',
                        default=['NVDA','SPY','QQQ','AMD','AAPL','MSFT','MELI','GLD'],
                        help='Tickers a testear')
    parser.add_argument('--start',   default='2020-01-01', help='Fecha inicio (YYYY-MM-DD)')
    parser.add_argument('--end',     default=None,          help='Fecha fin (YYYY-MM-DD)')
    parser.add_argument('--capital', default=10_000, type=float, help='Capital inicial en USD')
    parser.add_argument('--risk',    default=0.01,   type=float, help='Riesgo por trade (default 0.01 = 1%%)')
    parser.add_argument('--rr',       default=2.0,    type=float, help='Risk/Reward ratio (default 2.0)')
    parser.add_argument('--strategy', default='breakout', choices=['breakout','pullback'],
                        help='Estrategia: breakout (default) o pullback')
    parser.add_argument('--compare', action='store_true', help='Comparar ambas estrategias')
    parser.add_argument('--detail',  action='store_true', help='Mostrar detalle por ticker')
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]

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
                         strategy=strat)
            if r:
                r['strategy'] = strat
                print('ok')
                results.append(r)
                if args.detail:
                    print_detail(r)
            else:
                print('sin datos suficientes')

        print_summary_table(results)

        pass_count = sum(1 for r in results if evaluate(r)[0] == 'PASS')
        marginal   = sum(1 for r in results if evaluate(r)[0] == 'MARGINAL')
        fail_count = sum(1 for r in results if evaluate(r)[0] == 'FAIL')
        print(color(f"Resumen [{strat}]: {pass_count} PASS  |  {marginal} MARGINAL  |  {fail_count} FAIL", 'bold'))
        if pass_count > 0:
            print(color("Los tickers con PASS son candidatos para paper trading.", 'green'))
    print()



if __name__ == '__main__':
    main()
