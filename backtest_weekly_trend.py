"""
Backtest Fase 3.1 — Weekly Trend
Valida si la estrategia weekly_trend tiene edge real sobre los 24 tickers
y si diversifica (baja correlación) con swing y pullback.

Criterios de aprobación (todos deben cumplirse para pasar a paper):
  - Sharpe > 0.8 aislado
  - Profit Factor > 1.2
  - Correlación de retornos mensuales con swing y pullback < 0.5
  - Al menos 20 trades en el período de 5 años

Uso: python3.10 backtest_weekly_trend.py
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backtester import backtest

TICKERS = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO',
    'NU', 'MELI', 'BABA',
    'SPY', 'QQQ', 'IWM',
    'GLD', 'SLV', 'XLE',
    'VIST',
]

# Risk management común: DD stop global + riesgo por trade.
# El trailing se define por estrategia porque cada timeframe necesita lo suyo.
COMMON_RISK = dict(
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
    # Weekly trend: timeframe semanal requiere stops más amplios que diario.
    # 2 ATR semanales eran 2 velas de ruido normal — los winners se cortaban antes.
    'weekly_trend': dict(
        strategy='weekly_trend', timeframe='1wk',
        signal_params=dict(ema_fast=10, ema_slow=20, adx_min=20,
                           rsi_long_max=75, rsi_short_min=25),
        atr_stop_mult=2.0, rr_ratio=2.0, trailing_atr=3.5,
    ),
}


def run_strategy(strat_name, tickers, start, end, capital):
    """Corre una estrategia sobre toda la canasta. Devuelve lista de trades."""
    cfg = {**STRATEGIES[strat_name], **COMMON_RISK}
    all_trades = []
    for t in tickers:
        r = backtest(t, start=start, end=end, initial_capital=capital, **cfg)
        if r and r.get('trades'):
            for trade in r['trades']:
                trade['strategy'] = strat_name
                trade['ticker']   = t
                all_trades.append(trade)
    return all_trades


def metrics(trades, initial_capital, ann_factor=252):
    """Métricas agregadas para una lista de trades."""
    if not trades:
        return {'capital': initial_capital, 'return': 0, 'sharpe': 0,
                'maxdd': 0, 'pf': 0, 'wr': 0, 'n': 0, 'avg_win': 0, 'avg_loss': 0}

    df = pd.DataFrame(trades)
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df = df.sort_values('exit_date').reset_index(drop=True)

    capital = initial_capital
    equity  = [capital]
    for _, row in df.iterrows():
        capital += row['pnl']
        equity.append(capital)

    eq  = pd.Series(equity)
    ret = eq.pct_change().dropna()
    sharpe = (ret.mean() / ret.std() * np.sqrt(ann_factor)) if ret.std() > 0 else 0.0
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)

    wins = (df['pnl'] > 0).sum()
    n    = len(df)
    wr   = wins / n * 100 if n else 0
    pf_loss = abs(df.loc[df['pnl'] <= 0, 'pnl'].sum())
    pf   = (df.loc[df['pnl'] > 0, 'pnl'].sum() / pf_loss) if pf_loss > 0 else float('inf')

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


def monthly_returns(trades, index):
    """Serie de retornos mensuales (USD) a partir de trades. Reindexa para correlacionar."""
    if not trades:
        return pd.Series(0.0, index=index)
    df = pd.DataFrame(trades)
    df['exit_date'] = pd.to_datetime(df['exit_date']).dt.tz_localize(None)
    df['month'] = df['exit_date'].dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    monthly.index = monthly.index.to_timestamp()
    return monthly.reindex(index, fill_value=0.0)


def print_strategy_summary(name, m):
    avg = f"{m['avg_win']:.1f}/{m['avg_loss']:.1f}"
    print(f"  {name:<15} Ret {m['return']:>7.2f}%  Sharpe {m['sharpe']:>5.2f}  "
          f"MaxDD {m['maxdd']:>6.2f}%  PF {m['pf']:>5.2f}  "
          f"WR {m['wr']:>5.1f}%  Trades {m['n']:>4}  AvgW/L {avg:>12}")


def evaluate_weekly(m, corrs):
    """Semáforo de aprobación para weekly trend."""
    issues = []
    if m['n'] < 20:
        issues.append(f"Trades {m['n']} < 20 (muestra insuficiente)")
    if m['sharpe'] < 0.8:
        issues.append(f"Sharpe {m['sharpe']:.2f} < 0.8")
    if m['pf'] < 1.2:
        issues.append(f"PF {m['pf']:.2f} < 1.2")
    for other, c in corrs.items():
        if abs(c) >= 0.5:
            issues.append(f"Correlación con {other} {c:+.2f} >= 0.5 (no diversifica)")

    if not issues:
        return 'PASS', 'Weekly trend aprobado para agregar al bot paper'
    if len(issues) <= 1 and m['sharpe'] >= 0.6:
        return 'MARGINAL', ' | '.join(issues)
    return 'FAIL', ' | '.join(issues)


def main():
    end       = datetime.now().strftime('%Y-%m-%d')
    start_5y  = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    capital   = 10_000

    print(f"\n{'═' * 92}")
    print(f"  BACKTEST FASE 3.1 — Weekly Trend aislado + correlación con swing/pullback")
    print(f"  Período: {start_5y} → {end}  |  Canasta: {len(TICKERS)} tickers  |  Capital: ${capital:,}")
    print(f"{'═' * 92}\n")

    results = {}
    trades_by_strat = {}
    for name in ('swing', 'pullback', 'weekly_trend'):
        print(f"Corriendo {name}...")
        trades = run_strategy(name, TICKERS, start_5y, end, capital)
        trades_by_strat[name] = trades
        results[name] = metrics(trades, capital)

    # ── Métricas aisladas ──
    print(f"\n{'─' * 92}")
    print(f"  MÉTRICAS AISLADAS (cada estrategia sobre capital $10,000 completo, no combinado)")
    print(f"{'─' * 92}")
    for name in ('swing', 'pullback', 'weekly_trend'):
        print_strategy_summary(name, results[name])

    # ── Correlación de retornos mensuales ──
    month_index = pd.date_range(start_5y, end, freq='MS')
    monthly = {
        name: monthly_returns(trades_by_strat[name], month_index)
        for name in ('swing', 'pullback', 'weekly_trend')
    }
    corr_matrix = pd.DataFrame(monthly).corr()

    print(f"\n{'─' * 92}")
    print(f"  MATRIZ DE CORRELACIÓN (retornos mensuales USD)")
    print(f"{'─' * 92}")
    print(corr_matrix.round(2).to_string())

    # ── Combinado (portfolio Fase 1 + weekly) ──
    fase1_trades = trades_by_strat['swing'] + trades_by_strat['pullback']
    combo_trades = fase1_trades + trades_by_strat['weekly_trend']
    fase1 = metrics(fase1_trades, capital)
    combo = metrics(combo_trades, capital)

    print(f"\n{'─' * 92}")
    print(f"  PORTFOLIO COMBINADO (capital compartido)")
    print(f"{'─' * 92}")
    print_strategy_summary('Fase 1 (sw+pb)', fase1)
    print_strategy_summary('Fase 1 + weekly', combo)
    d_ret = combo['return'] - fase1['return']
    d_sh  = combo['sharpe'] - fase1['sharpe']
    d_dd  = combo['maxdd']  - fase1['maxdd']
    d_pf  = combo['pf']     - fase1['pf']
    print(f"\n  Δ al agregar weekly:  Δret {d_ret:+6.2f}%  ΔSharpe {d_sh:+5.2f}  "
          f"ΔDD {d_dd:+5.2f}%  ΔPF {d_pf:+5.2f}")

    # ── Veredicto ──
    wt_corrs = {
        'swing':    corr_matrix.loc['weekly_trend', 'swing'],
        'pullback': corr_matrix.loc['weekly_trend', 'pullback'],
    }
    verdict, reason = evaluate_weekly(results['weekly_trend'], wt_corrs)

    print(f"\n{'═' * 92}")
    print(f"  VEREDICTO: {verdict}")
    print(f"  {reason}")
    print(f"{'═' * 92}\n")


if __name__ == '__main__':
    main()
