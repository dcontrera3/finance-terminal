"""
Backtest comparativo Fase 2.4 — Filtro RSI condicional al ADX en swing.

Compara dos configuraciones del swing sobre 5 años y 24 tickers:
  A: filtro actual (rsi_long_max=70, rsi_short_min=35)
  B: filtro condicional — si adx > 45, relaja a (rsi_relax_max=100, rsi_relax_min=0)

Umbral ADX=45 derivado de TRAIN ONLY (2021-2023) en analyze_rsi_filter_train.py.

Pullback y weekly_trend NO se tocan (no usan filtro RSI igual al swing).

Criterio de aceptación (out-of-sample):
  Sharpe ≥ +0.1 sin que MaxDD empeore más de 1pt.
  Si pasa: deploy. Si no: archivar.

Out-of-sample split:
  Train: 2021-05 → 2023-12 (donde se derivó el umbral 45)
  Test:  2024-01 → 2026-05 (decisorio, no visto al elegir umbral)

Uso: python3.10 backtest_rsi_conditional.py
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

COMMON = dict(
    risk_per_trade=0.0075,
    dd_pause_threshold=0.10,
    initial_capital=1_000_000,
)

ADX_THRESHOLD = 45  # derivado de TRAIN ONLY

STRATEGIES_A = {
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

# Variante B: swing con filtro RSI condicional al ADX
STRATEGIES_B = {**STRATEGIES_A}
STRATEGIES_B['swing'] = dict(
    strategy='swing', timeframe='1d',
    signal_params=dict(periods=8, vol_spike=1.1, adx_min=20,
                       rsi_long_max=70, rsi_short_min=35,
                       adx_relax_threshold=ADX_THRESHOLD,
                       rsi_relax_max=100, rsi_relax_min=0),
    atr_stop_mult=1.0, rr_ratio=1.5, trailing_atr=2.0,
)


def run_portfolio(strategies, tickers, start, end):
    all_trades = []
    for name, cfg in strategies.items():
        full_cfg = {**cfg, **COMMON}
        for t in tickers:
            r = backtest(t, start=start, end=end, **full_cfg)
            if r and r.get('trades'):
                for trade in r['trades']:
                    trade['strategy'] = name
                    trade['ticker']   = t
                    all_trades.append(trade)
    return all_trades


def metrics(trades, initial_capital, period_years):
    if not trades:
        return None

    df = pd.DataFrame(trades)
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df = df.sort_values('exit_date').reset_index(drop=True)

    capital = initial_capital
    equity_curve = [capital]
    for _, row in df.iterrows():
        capital += row['pnl']
        equity_curve.append(capital)

    ec = pd.Series(equity_curve)
    returns = ec.pct_change().dropna()
    n_trades = len(df)
    trades_per_year = n_trades / period_years if period_years > 0 else n_trades
    sharpe = (returns.mean() / returns.std() * np.sqrt(trades_per_year)
              if returns.std() > 0 else 0)
    running_max = ec.cummax()
    dd = (ec - running_max) / running_max
    maxdd = float(dd.min()) * 100

    win  = df[df['pnl'] > 0]
    loss = df[df['pnl'] <= 0]
    gw   = win['pnl'].sum() if len(win) else 0
    gl   = abs(loss['pnl'].sum()) if len(loss) else 0
    pf   = gw / gl if gl > 0 else float('inf')

    return {
        'final_capital':  capital,
        'return_pct':     (capital - initial_capital) / initial_capital * 100,
        'n_trades':       n_trades,
        'win_rate':       len(win) / n_trades * 100 if n_trades else 0,
        'avg_win':        win['pnl'].mean() if len(win) else 0,
        'avg_loss':       loss['pnl'].mean() if len(loss) else 0,
        'profit_factor':  pf if pf != float('inf') else None,
        'sharpe':         sharpe,
        'maxdd_pct':      maxdd,
    }


def run_period(label, start, end):
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    print(f"\n{'='*80}")
    print(f"PERÍODO: {label}   {start} → {end}   ({years:.2f} años)")
    print('='*80)

    trades_a = run_portfolio(STRATEGIES_A, TICKERS, start, end)
    trades_b = run_portfolio(STRATEGIES_B, TICKERS, start, end)
    m_a = metrics(trades_a, 1_000_000, years)
    m_b = metrics(trades_b, 1_000_000, years)

    print(f"\n{'Métrica':<22} {'A (actual)':>20} {'B (cond ADX>'+str(ADX_THRESHOLD)+')':>22} {'Δ':>12}")
    print('-' * 80)
    rows = [
        ('Capital final',  f"${m_a['final_capital']:>14,.0f}",  f"${m_b['final_capital']:>14,.0f}",  f"${m_b['final_capital']-m_a['final_capital']:>+10,.0f}"),
        ('Return %',       f"{m_a['return_pct']:>14.2f}%",       f"{m_b['return_pct']:>14.2f}%",       f"{m_b['return_pct']-m_a['return_pct']:>+10.2f}pt"),
        ('Sharpe',         f"{m_a['sharpe']:>15.3f}",            f"{m_b['sharpe']:>15.3f}",            f"{m_b['sharpe']-m_a['sharpe']:>+10.3f}"),
        ('MaxDD %',        f"{m_a['maxdd_pct']:>14.2f}%",        f"{m_b['maxdd_pct']:>14.2f}%",        f"{m_b['maxdd_pct']-m_a['maxdd_pct']:>+10.2f}pt"),
        ('N trades',       f"{m_a['n_trades']:>15d}",            f"{m_b['n_trades']:>15d}",            f"{m_b['n_trades']-m_a['n_trades']:>+10d}"),
        ('Win rate %',     f"{m_a['win_rate']:>14.1f}%",         f"{m_b['win_rate']:>14.1f}%",         f"{m_b['win_rate']-m_a['win_rate']:>+10.1f}pt"),
        ('Avg win $',      f"${m_a['avg_win']:>14,.0f}",         f"${m_b['avg_win']:>14,.0f}",         f"${m_b['avg_win']-m_a['avg_win']:>+10,.0f}"),
        ('Avg loss $',     f"${m_a['avg_loss']:>14,.0f}",        f"${m_b['avg_loss']:>14,.0f}",        f"${m_b['avg_loss']-m_a['avg_loss']:>+10,.0f}"),
        ('Profit factor',  f"{m_a['profit_factor']:>15.2f}" if m_a['profit_factor'] else "          inf", f"{m_b['profit_factor']:>15.2f}" if m_b['profit_factor'] else "          inf", ""),
    ]
    for label_, va, vb, delta in rows:
        print(f"{label_:<22} {va:>20} {vb:>22} {delta:>12}")

    return m_a, m_b


end_full   = datetime.now().strftime('%Y-%m-%d')
start_full = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
end_train  = '2023-12-31'
start_test = '2024-01-01'

print("BACKTEST COMPARATIVO Fase 2.4 — Filtro RSI condicional ADX en swing")
print(f"Universo: {len(TICKERS)} tickers")
print(f"Estrategias: swing modificado, pullback y weekly_trend sin tocar")
print(f"Umbral ADX = {ADX_THRESHOLD} (derivado de TRAIN ONLY)")

m_full_a, m_full_b = run_period('FULL 5Y (sanity)', start_full, end_full)
m_tr_a,   m_tr_b   = run_period('TRAIN (in-sample, donde se derivó umbral)', start_full, end_train)
m_te_a,   m_te_b   = run_period('TEST (out-of-sample) ← DECISORIO', start_test, end_full)

print("\n" + "="*80)
print("VEREDICTO (criterio out-of-sample)")
print("="*80)
sharpe_delta = m_te_b['sharpe'] - m_te_a['sharpe']
maxdd_delta  = m_te_b['maxdd_pct'] - m_te_a['maxdd_pct']
print(f"  Δ Sharpe (test):  {sharpe_delta:+.3f}   (criterio: ≥ +0.1)")
print(f"  Δ MaxDD  (test):  {maxdd_delta:+.2f}pt  (criterio: ≥ -1pt)")
print()
sharpe_ok = sharpe_delta >= 0.1
maxdd_ok  = maxdd_delta >= -1.0
if sharpe_ok and maxdd_ok:
    print("  ✓ PASA ambos criterios → DEPLOY")
else:
    reasons = []
    if not sharpe_ok: reasons.append(f"Sharpe sube solo {sharpe_delta:+.3f} (< +0.1)")
    if not maxdd_ok:  reasons.append(f"MaxDD empeora {maxdd_delta:+.2f}pt (> -1pt)")
    print(f"  ✗ NO PASA → archivar")
    for r in reasons:
        print(f"    · {r}")
