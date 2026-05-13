"""
Backtest consolidado post-Fase 3.1:
Compara el portfolio Fase 1 (swing + pullback) contra Fase 1 + weekly_trend,
en 5 años y YTD 2026. Es el reporte de baseline tras agregar weekly al bot.

Uso: python3.10 backtest_consolidated.py
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backtester import backtest

TICKERS = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO', 'COST',
    'CAT',
    'XLU',
    'NU', 'MELI', 'BABA',
    'SPY', 'QQQ', 'IWM',
    'GLD', 'SLV', 'XLE',
    'VIST',
]

COMMON = dict(
    dd_pause_threshold=0.10,
    risk_per_trade=0.0075,
)

STRATEGIES = {
    'swing': dict(
        strategy='swing', timeframe='1d',
        signal_params=dict(periods=8, vol_spike=1.1, adx_min=20,
                           rsi_long_max=70, rsi_short_min=35),
        atr_stop_mult=1.0, rr_ratio=1.5, trailing_atr=2.0,
    ),
    'pullback': dict(
        strategy='pullback', timeframe='1d',
        signal_params=dict(rsi_min=40, rsi_max=65, pullback_pct=1.02, vol_min=1.2),
        atr_stop_mult=1.5, rr_ratio=2.0, trailing_atr=2.0,
    ),
    'weekly_trend': dict(
        strategy='weekly_trend', timeframe='1wk',
        signal_params=dict(ema_fast=10, ema_slow=20, adx_min=20,
                           rsi_long_max=75, rsi_short_min=25),
        atr_stop_mult=2.0, rr_ratio=2.0, trailing_atr=3.5,
    ),
}


def run_portfolio(strat_names, tickers, start, end, capital, sim_start=None):
    """Corre las estrategias dadas sobre la canasta. Devuelve trades combinados."""
    all_trades = []
    for name in strat_names:
        cfg = {**STRATEGIES[name], **COMMON}
        if sim_start:
            cfg['sim_start'] = sim_start
        for t in tickers:
            r = backtest(t, start=start, end=end, initial_capital=capital, **cfg)
            if r and r.get('trades'):
                for trade in r['trades']:
                    trade['strategy'] = name
                    trade['ticker']   = t
                    all_trades.append(trade)
    return all_trades


def metrics(trades, initial):
    """Métricas agregadas del portfolio."""
    if not trades:
        return {'capital': initial, 'return': 0, 'sharpe': 0, 'maxdd': 0,
                'pf': 0, 'wr': 0, 'n': 0, 'avg_win': 0, 'avg_loss': 0,
                'closed': 0, 'open': 0}

    df = pd.DataFrame(trades)
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df = df.sort_values('exit_date').reset_index(drop=True)

    capital = initial
    equity  = [capital]
    for _, row in df.iterrows():
        capital += row['pnl']
        equity.append(capital)

    eq  = pd.Series(equity)
    ret = eq.pct_change().dropna()
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)

    wins   = (df['pnl'] > 0).sum()
    n      = len(df)
    wr     = wins / n * 100 if n else 0
    pf_loss = abs(df.loc[df['pnl'] <= 0, 'pnl'].sum())
    pf     = (df.loc[df['pnl'] > 0, 'pnl'].sum() / pf_loss) if pf_loss > 0 else float('inf')

    avg_win  = df.loc[df['pnl'] > 0, 'pnl'].mean()  if wins > 0 else 0
    avg_loss = df.loc[df['pnl'] <= 0, 'pnl'].mean() if wins < n else 0

    closed = int((df['result'] != 'OPEN').sum()) if 'result' in df.columns else n
    opens  = int((df['result'] == 'OPEN').sum()) if 'result' in df.columns else 0

    return {
        'capital': capital, 'return': (capital - initial) / initial * 100,
        'sharpe': sharpe, 'maxdd': max_dd, 'pf': pf, 'wr': wr,
        'n': n, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'closed': closed, 'open': opens,
    }


def metrics_by_strategy(trades, initial):
    """Breakdown métricas por estrategia."""
    by = {}
    for name in sorted(set(t['strategy'] for t in trades)):
        sub = [t for t in trades if t['strategy'] == name]
        by[name] = metrics(sub, initial)
    return by


def print_header(title):
    print(f"\n{'═' * 100}")
    print(f"  {title}")
    print(f"{'═' * 100}")


def print_variant(label, m, delta=None):
    pf_s = f"{m['pf']:.2f}" if m['pf'] != float('inf') else 'inf'
    line = (f"  {label:<22} Ret {m['return']:>7.2f}%  Sharpe {m['sharpe']:>5.2f}  "
            f"MaxDD {m['maxdd']:>6.2f}%  PF {pf_s:>5}  WR {m['wr']:>5.1f}%  "
            f"Trades {m['n']:>5}")
    print(line)
    if delta:
        d_ret, d_sh, d_dd, d_pf = delta
        print(f"  {'':22} Δret {d_ret:+6.2f}%  ΔSharpe {d_sh:+5.2f}  "
              f"ΔDD {d_dd:+5.2f}%  ΔPF {d_pf:+5.2f}")


def compare_section(title, strat_sets, tickers, start, end, capital, sim_start=None):
    """Corre cada variante y printea tabla comparativa."""
    print_header(title)

    results = []
    for label, strats in strat_sets:
        print(f"  Corriendo {label}...")
        trades = run_portfolio(strats, tickers, start, end, capital, sim_start=sim_start)
        results.append((label, strats, metrics(trades, capital), trades))

    baseline = results[0][2]
    print(f"\n  Capital inicial: ${capital:,}")
    print(f"  {'-' * 96}")
    for i, (label, _, m, _) in enumerate(results):
        delta = None
        if i > 0:
            delta = (m['return'] - baseline['return'],
                     m['sharpe'] - baseline['sharpe'],
                     m['maxdd']  - baseline['maxdd'],
                     m['pf']     - baseline['pf'])
        print_variant(label, m, delta)

    # Breakdown por estrategia de la variante última (la más completa)
    last_label, _, _, last_trades = results[-1]
    by_strat = metrics_by_strategy(last_trades, capital)
    if by_strat:
        print(f"\n  Breakdown por estrategia  ({last_label}):")
        for name, m in by_strat.items():
            print_variant(f"  └ {name}", m)


def main():
    end          = datetime.now().strftime('%Y-%m-%d')
    start_5y     = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    start_ytd    = '2026-01-01'
    warmup_start = (datetime.strptime(start_ytd, '%Y-%m-%d') - timedelta(days=730)) \
                    .strftime('%Y-%m-%d')

    variants = [
        ('Fase 1 (sw+pb)',        ['swing', 'pullback']),
        ('Fase 1 + weekly',       ['swing', 'pullback', 'weekly_trend']),
    ]

    print(f"\n{'═' * 100}")
    print(f"  BACKTEST CONSOLIDADO POST-FASE 3.1")
    print(f"  Fecha: {end}  |  Canasta: {len(TICKERS)} tickers")
    print(f"{'═' * 100}")

    compare_section(f"5 AÑOS  ({start_5y} → {end})  |  Capital $10,000",
                    variants, TICKERS, start_5y, end, 10_000)

    compare_section(f"YTD 2026  ({start_ytd} → {end})  |  Capital $5,000",
                    variants, TICKERS, warmup_start, end, 5_000, sim_start=start_ytd)

    print()


if __name__ == '__main__':
    main()
