"""
Backtest completo del portfolio del bot.
Corre swing + pullback sobre los 24 tickers y combina P&L.
También simula desde 1-abril-2026 con capital inicial configurable.
"""

from datetime import datetime, timedelta
import pandas as pd
from backtester import backtest  # noqa: F401

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
    'swing': {
        'fn_kwargs': dict(
            strategy='swing',
            timeframe='1d',
            signal_params=dict(periods=8, vol_spike=1.1, adx_min=20,
                               rsi_long_max=70, rsi_short_min=35),
            atr_stop_mult=1.0,
            rr_ratio=1.5,
            risk_per_trade=0.0075,
        ),
    },
    'pullback': {
        'fn_kwargs': dict(
            strategy='pullback',
            timeframe='1d',
            signal_params=dict(rsi_min=40, rsi_max=65, pullback_pct=1.02, vol_min=1.2),
            atr_stop_mult=1.5,
            rr_ratio=2.0,
            risk_per_trade=0.0075,
        ),
    },
}


def run_strategy(strategy_name, tickers, start, end, capital):
    """Corre una estrategia sobre toda la canasta, capital individual para cada ticker."""
    results = []
    cfg = STRATEGIES[strategy_name]
    per_ticker_capital = capital   # cada ticker trackea su propio equity simulado
    for t in tickers:
        r = backtest(t, start=start, end=end,
                     initial_capital=per_ticker_capital,
                     **cfg['fn_kwargs'])
        if r:
            r['strategy'] = strategy_name
            results.append(r)
    return results


def combined_stats(all_results, initial_capital, label):
    """
    Agrega P&L de todos los trades (swing + pullback) cronológicamente y
    calcula métricas del portfolio combinado.
    """
    all_trades = []
    for r in all_results:
        for t in r.get('trades', []):
            t['strategy'] = r['strategy']
            t['ticker']   = r['ticker']
            all_trades.append(t)

    if not all_trades:
        print(f"\n{label}: sin trades.\n")
        return

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
    n    = len(df)
    wr   = wins / n * 100 if n else 0
    pf   = (df.loc[df['pnl'] > 0, 'pnl'].sum() /
            abs(df.loc[df['pnl'] <= 0, 'pnl'].sum()) if df.loc[df['pnl'] <= 0, 'pnl'].sum() else float('inf'))

    n_swing    = (df['strategy'] == 'swing').sum()
    n_pullback = (df['strategy'] == 'pullback').sum()
    pnl_swing  = df.loc[df['strategy'] == 'swing',    'pnl'].sum()
    pnl_pull   = df.loc[df['strategy'] == 'pullback', 'pnl'].sum()

    total_ret = (capital - initial_capital) / initial_capital * 100
    n_long  = (df['dir'] == 'LONG').sum()
    n_short = (df['dir'] == 'SHORT').sum()

    print(f"\n══ {label} ══")
    print(f"  Capital:         ${initial_capital:,.0f} → ${capital:,.2f}")
    print(f"  Retorno total:   {total_ret:+.2f}%")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd:.2f}%")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"  Win Rate:        {wr:.1f}%  ({wins}/{n})")
    print(f"  Trades totales:  {n}  ({n_long} LONG / {n_short} SHORT)")
    print(f"    swing:         {n_swing}  |  P&L ${pnl_swing:+,.2f}")
    print(f"    pullback:      {n_pullback}  |  P&L ${pnl_pull:+,.2f}")

    # Top winners y losers
    top_win = df.nlargest(5, 'pnl')[['ticker', 'strategy', 'dir', 'exit_date', 'pnl']]
    top_los = df.nsmallest(5, 'pnl')[['ticker', 'strategy', 'dir', 'exit_date', 'pnl']]
    print(f"\n  Top 5 ganadores:")
    for _, row in top_win.iterrows():
        print(f"    {row['ticker']:<6} [{row['strategy']:<8}] {row['dir']:<5} "
              f"{row['exit_date'].date()}  ${row['pnl']:+,.2f}")
    print(f"  Top 5 perdedores:")
    for _, row in top_los.iterrows():
        print(f"    {row['ticker']:<6} [{row['strategy']:<8}] {row['dir']:<5} "
              f"{row['exit_date'].date()}  ${row['pnl']:+,.2f}")

    return df


def main():
    end   = datetime.now().strftime('%Y-%m-%d')
    start_5y = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    start_sim = '2026-01-01'   # simulación YTD 2026

    # ── Backtest de 5 años con $10,000 ───────────────────────────────
    print("═" * 80)
    print("BACKTEST 5 AÑOS  —  Portfolio completo (24 tickers, swing + pullback)")
    print(f"Período: {start_5y} → {end}")
    print("Capital inicial: $10,000  |  Riesgo por trade: 0.75% por estrategia")
    print("═" * 80)

    all_results_5y = []
    for strat in STRATEGIES:
        r = run_strategy(strat, TICKERS, start_5y, end, capital=10_000)
        all_results_5y.extend(r)

    combined_stats(all_results_5y, initial_capital=10_000,
                   label='PORTFOLIO COMBINADO 5 AÑOS')

    # ── Simulación desde 1 abril 2026 con $5,000 ─────────────────────
    # Traemos 2 años de data histórica para warmup de indicadores
    # pero los trades solo se ejecutan desde 1-abril
    warmup_start = (datetime.strptime(start_sim, '%Y-%m-%d') - timedelta(days=730)).strftime('%Y-%m-%d')

    print("\n\n" + "═" * 80)
    print(f"SIMULACIÓN  —  $5,000 desde {start_sim} hasta hoy")
    print(f"Período trading: {start_sim} → {end}")
    print(f"Warmup data:     {warmup_start} → {start_sim}  (para cálculo de indicadores)")
    print("═" * 80)

    all_results_apr = []
    for strat in STRATEGIES:
        cfg = STRATEGIES[strat]
        for t in TICKERS:
            r = backtest(t, start=warmup_start, end=end,
                         initial_capital=5_000,
                         sim_start=start_sim,
                         **cfg['fn_kwargs'])
            if r:
                r['strategy'] = strat
                all_results_apr.append(r)

    combined_stats(all_results_apr, initial_capital=5_000,
                   label=f'SIMULACIÓN YTD 2026  ($5,000 desde {start_sim})')


if __name__ == '__main__':
    main()
