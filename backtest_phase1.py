"""
Backtest comparativo: Fase 1 (trailing stop + drawdown stop + VIX filter)
contra el baseline actual. Muestra impacto real de cada mejora.
"""

from datetime import datetime, timedelta
import pandas as pd
from backtester import backtest, fetch

TICKERS = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO',
    'NU', 'MELI', 'BABA',
    'SPY', 'QQQ', 'IWM',
    'GLD', 'SLV', 'XLE',
    'VIST',
]

STRATEGIES = {
    'swing': dict(
        strategy='swing',
        timeframe='1d',
        signal_params=dict(periods=8, vol_spike=1.1, adx_min=20,
                           rsi_long_max=70, rsi_short_min=35),
        atr_stop_mult=1.0,
        rr_ratio=1.5,
        risk_per_trade=0.0075,
    ),
    'pullback': dict(
        strategy='pullback',
        timeframe='1d',
        signal_params=dict(rsi_min=40, rsi_max=65, pullback_pct=1.02, vol_min=1.2),
        atr_stop_mult=1.5,
        rr_ratio=2.0,
        risk_per_trade=0.0075,
    ),
}


def run_portfolio(tickers, start, end, capital, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    all_trades = []
    for strat_name, cfg in STRATEGIES.items():
        merged = {**cfg, **extra_kwargs}
        for t in tickers:
            r = backtest(t, start=start, end=end,
                         initial_capital=capital, **merged)
            if r and r.get('trades'):
                for trade in r['trades']:
                    trade['strategy'] = strat_name
                    trade['ticker']   = t
                    all_trades.append(trade)
    return all_trades


def metrics(all_trades, initial_capital):
    if not all_trades:
        return {'capital': initial_capital, 'return': 0, 'sharpe': 0,
                'maxdd': 0, 'pf': 0, 'wr': 0, 'n': 0, 'avg_win': 0, 'avg_loss': 0}

    df = pd.DataFrame(all_trades)
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df = df.sort_values('exit_date').reset_index(drop=True)

    capital = initial_capital
    equity  = [capital]
    for _, row in df.iterrows():
        capital += row['pnl']
        equity.append(capital)

    eq = pd.Series(equity)
    ret = eq.pct_change().dropna()
    sharpe = (ret.mean() / ret.std() * (252 ** 0.5)) if ret.std() > 0 else 0.0
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)

    wins = (df['pnl'] > 0).sum()
    n = len(df)
    wr = wins / n * 100 if n else 0
    pf = (df.loc[df['pnl'] > 0, 'pnl'].sum() /
          abs(df.loc[df['pnl'] <= 0, 'pnl'].sum())
          if df.loc[df['pnl'] <= 0, 'pnl'].sum() else float('inf'))

    avg_win  = df.loc[df['pnl'] > 0, 'pnl'].mean()  if wins > 0 else 0
    avg_loss = df.loc[df['pnl'] <= 0, 'pnl'].mean() if wins < n else 0

    return {
        'capital':  capital,
        'return':   (capital - initial_capital) / initial_capital * 100,
        'sharpe':   sharpe,
        'maxdd':    max_dd,
        'pf':       pf,
        'wr':       wr,
        'n':        n,
        'avg_win':  avg_win,
        'avg_loss': avg_loss,
    }


def print_comparison(label, initial, variants):
    print(f"\n{'═' * 90}")
    print(f"  {label}")
    print(f"{'═' * 90}")
    print(f"  {'Variante':<30} {'Capital':>10} {'Ret%':>8} {'Sharpe':>7} "
          f"{'MaxDD%':>8} {'PF':>6} {'WR%':>6} {'Trades':>7}  {'AvgW/L':>12}")
    print("  " + "─" * 88)
    baseline = variants[0][1]
    for name, m in variants:
        marker = "►" if name != variants[0][0] else " "
        avg = f"{m['avg_win']:.1f}/{m['avg_loss']:.1f}"
        print(f" {marker}{name:<30} ${m['capital']:>9,.0f} {m['return']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['maxdd']:>7.2f}% {m['pf']:>6.2f} "
              f"{m['wr']:>5.1f}% {m['n']:>7}  {avg:>12}")

    # Delta vs baseline
    print("  " + "─" * 88)
    print(f"  Cambios vs baseline:")
    for name, m in variants[1:]:
        d_ret    = m['return'] - baseline['return']
        d_sharpe = m['sharpe'] - baseline['sharpe']
        d_dd     = m['maxdd']  - baseline['maxdd']
        d_pf     = m['pf']     - baseline['pf']
        print(f"    {name:<30}  Δret {d_ret:+6.2f}%  ΔSharpe {d_sharpe:+5.2f}  "
              f"ΔDD {d_dd:+5.2f}%  ΔPF {d_pf:+5.2f}")


def main():
    end = datetime.now().strftime('%Y-%m-%d')
    start_5y = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    start_ytd = '2026-01-01'
    warmup = (datetime.strptime(start_ytd, '%Y-%m-%d') - timedelta(days=730)).strftime('%Y-%m-%d')

    # Pre-fetch del VIX (una sola vez, reutilizable)
    print("Fetching VIX data...")
    vix_df = fetch('^VIX', start_5y, end, interval='1d')
    print(f"VIX barras: {len(vix_df)}")

    # Configuraciones a comparar
    variantes = [
        ('baseline (TP fijo)',              {}),
        ('+ trailing stop 2 ATR',           {'trailing_atr': 2.0}),
        ('+ trailing + DD stop -10%',       {'trailing_atr': 2.0, 'dd_pause_threshold': 0.10}),
        ('+ trailing + DD + VIX>35 filter', {'trailing_atr': 2.0, 'dd_pause_threshold': 0.10,
                                             'vix_df': vix_df, 'vix_pause_above': 35}),
    ]

    # ── 5 años, $10,000 ──
    print("\nCorriendo backtest 5 años...")
    results_5y = []
    for name, kwargs in variantes:
        print(f"  {name}...")
        trades = run_portfolio(TICKERS, start_5y, end, 10_000, extra_kwargs=kwargs)
        results_5y.append((name, metrics(trades, 10_000)))

    print_comparison(f"5 AÑOS  ({start_5y} → {end})  |  Capital $10,000",
                     10_000, results_5y)

    # ── YTD 2026, $5,000 ──
    print("\nCorriendo simulación YTD 2026...")
    results_ytd = []
    for name, kwargs in variantes:
        print(f"  {name}...")
        kwargs_with_sim = {**kwargs, 'sim_start': start_ytd}
        trades = run_portfolio(TICKERS, warmup, end, 5_000, extra_kwargs=kwargs_with_sim)
        results_ytd.append((name, metrics(trades, 5_000)))

    print_comparison(f"YTD 2026  ({start_ytd} → {end})  |  Capital $5,000",
                     5_000, results_ytd)


if __name__ == '__main__':
    main()
