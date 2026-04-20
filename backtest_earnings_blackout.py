"""
Backtest del Earnings Blackout Filter.
Compara portfolio consolidado (Fase 1 + weekly_trend) con y sin el filtro
de earnings blackout. El filtro descarta trades cuya entrada cae dentro
de los N días previos a un earnings histórico del mismo ticker.

Metodología: filtrado post-hoc (descarta entries en blackout window).
No simula cierres anticipados todavía. Para una primera aproximación alcanza:
el impacto principal está en evitar abrir justo antes del evento.

Uso: python3.10 backtest_earnings_blackout.py
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

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

COMMON = dict(dd_pause_threshold=0.10, risk_per_trade=0.0075)

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


def fetch_earnings_map(tickers):
    """Construye dict {ticker: [dates naive]} de earnings históricos.
    ETFs sin earnings devuelven lista vacía (no hay filtro que aplicar)."""
    out = {}
    for t in tickers:
        try:
            ed = yf.Ticker(t).earnings_dates
            if ed is None or len(ed) == 0:
                out[t] = []
                continue
            dates = [pd.Timestamp(d).tz_localize(None).normalize() for d in ed.index]
            out[t] = sorted(dates)
        except Exception:
            out[t] = []
    return out


def in_blackout(entry_date, ticker, earnings_map, blackout_days):
    """True si entry_date cae dentro de los `blackout_days` días previos
    a algún earnings del ticker. Inclusivo en ambos extremos: 0 a blackout_days."""
    entry = pd.Timestamp(entry_date)
    if entry.tz is not None:
        entry = entry.tz_localize(None)
    entry = entry.normalize()
    for edate in earnings_map.get(ticker, []):
        delta = (edate - entry).days
        if 0 <= delta <= blackout_days:
            return True
    return False


def run_portfolio(strat_names, tickers, start, end, capital, sim_start=None):
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


def apply_blackout_filter(trades, earnings_map, blackout_days):
    """Descarta trades que abren dentro del blackout window."""
    return [t for t in trades
            if not in_blackout(t['entry_date'], t['ticker'], earnings_map, blackout_days)]


def metrics(trades, initial):
    if not trades:
        return {'capital': initial, 'return': 0, 'sharpe': 0, 'maxdd': 0,
                'pf': 0, 'wr': 0, 'n': 0}
    df = pd.DataFrame(trades)
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df = df.sort_values('exit_date').reset_index(drop=True)

    capital = initial
    equity = [capital]
    for _, row in df.iterrows():
        capital += row['pnl']
        equity.append(capital)

    eq = pd.Series(equity)
    ret = eq.pct_change().dropna()
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)

    wins = (df['pnl'] > 0).sum()
    n = len(df)
    wr = wins / n * 100 if n else 0
    pf_loss = abs(df.loc[df['pnl'] <= 0, 'pnl'].sum())
    pf = (df.loc[df['pnl'] > 0, 'pnl'].sum() / pf_loss) if pf_loss > 0 else float('inf')

    return {
        'capital': capital, 'return': (capital - initial) / initial * 100,
        'sharpe': sharpe, 'maxdd': max_dd, 'pf': pf, 'wr': wr, 'n': n,
    }


def print_section(title, variants, capital):
    print(f"\n{'═' * 102}")
    print(f"  {title}  |  Capital inicial ${capital:,}")
    print(f"{'═' * 102}")
    print(f"  {'Variante':<30} {'Ret':>9} {'Sharpe':>8} {'MaxDD':>9} {'PF':>6} "
          f"{'WR':>6} {'Trades':>7}  {'Trades filt.':>12}")
    print(f"  {'-' * 100}")
    baseline = variants[0][1]
    for name, m, n_filtered in variants:
        pf_s = f"{m['pf']:.2f}" if m['pf'] != float('inf') else 'inf'
        filt = f"(-{n_filtered})" if n_filtered > 0 else ""
        print(f"  {name:<30} {m['return']:>8.2f}% {m['sharpe']:>8.2f} "
              f"{m['maxdd']:>8.2f}% {pf_s:>6} {m['wr']:>5.1f}% {m['n']:>7}  {filt:>12}")
    print(f"  {'-' * 100}")
    print(f"  Cambios vs baseline:")
    for name, m, _ in variants[1:]:
        d_ret = m['return'] - baseline['return']
        d_sh  = m['sharpe'] - baseline['sharpe']
        d_dd  = m['maxdd']  - baseline['maxdd']
        d_pf  = m['pf']     - baseline['pf']
        print(f"    {name:<30} Δret {d_ret:+6.2f}%  ΔSharpe {d_sh:+5.2f}  "
              f"ΔDD {d_dd:+5.2f}%  ΔPF {d_pf:+5.2f}")


def by_strategy_breakdown(trades, initial, label):
    print(f"\n  Breakdown por estrategia — {label}")
    print(f"  {'-' * 100}")
    for name in sorted(set(t['strategy'] for t in trades)):
        sub = [t for t in trades if t['strategy'] == name]
        m = metrics(sub, initial)
        pf_s = f"{m['pf']:.2f}" if m['pf'] != float('inf') else 'inf'
        print(f"    {name:<20} Ret {m['return']:>7.2f}%  Sharpe {m['sharpe']:>5.2f}  "
              f"MaxDD {m['maxdd']:>7.2f}%  PF {pf_s:>5}  WR {m['wr']:>5.1f}%  "
              f"Trades {m['n']:>5}")


def main():
    end          = datetime.now().strftime('%Y-%m-%d')
    start_5y     = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    start_ytd    = '2026-01-01'
    warmup_start = (datetime.strptime(start_ytd, '%Y-%m-%d') - timedelta(days=730)) \
                    .strftime('%Y-%m-%d')
    strat_names = ['swing', 'pullback', 'weekly_trend']

    print(f"\n{'═' * 102}")
    print(f"  EARNINGS BLACKOUT BACKTEST  |  Fecha {end}")
    print(f"  Filtro: descarta trades que abren 0-3 días antes de earnings del mismo ticker")
    print(f"{'═' * 102}")

    print("\nCargando earnings históricos...")
    earnings_map = fetch_earnings_map(TICKERS)
    has_earnings = sum(1 for t in TICKERS if earnings_map[t])
    total = sum(len(v) for v in earnings_map.values())
    print(f"  {has_earnings}/{len(TICKERS)} tickers con earnings | {total} fechas totales")
    etfs = [t for t in TICKERS if not earnings_map[t]]
    print(f"  Sin earnings (ETFs/no data): {', '.join(etfs) or 'ninguno'}")

    # ── 5 años ──
    capital_5y = 10_000
    print("\nCorriendo portfolio 5 años...")
    trades_5y_all = run_portfolio(strat_names, TICKERS, start_5y, end, capital_5y)

    variants_5y = []
    m_none = metrics(trades_5y_all, capital_5y)
    variants_5y.append(('Sin blackout (baseline)', m_none, 0))

    for days in (2, 3, 5):
        filtered = apply_blackout_filter(trades_5y_all, earnings_map, days)
        n_removed = len(trades_5y_all) - len(filtered)
        variants_5y.append((f'Blackout {days} días antes',
                            metrics(filtered, capital_5y), n_removed))

    print_section(f"5 AÑOS  ({start_5y} → {end})", variants_5y, capital_5y)

    # Breakdown 3 días (nuestro candidato principal)
    filtered_3 = apply_blackout_filter(trades_5y_all, earnings_map, 3)
    by_strategy_breakdown(filtered_3, capital_5y, 'con blackout 3 días (5y)')

    # ── YTD 2026 ──
    capital_ytd = 5_000
    print("\n\nCorriendo portfolio YTD 2026...")
    trades_ytd_all = run_portfolio(strat_names, TICKERS, warmup_start, end,
                                   capital_ytd, sim_start=start_ytd)

    variants_ytd = []
    m_none_ytd = metrics(trades_ytd_all, capital_ytd)
    variants_ytd.append(('Sin blackout (baseline)', m_none_ytd, 0))

    for days in (2, 3, 5):
        filtered = apply_blackout_filter(trades_ytd_all, earnings_map, days)
        n_removed = len(trades_ytd_all) - len(filtered)
        variants_ytd.append((f'Blackout {days} días antes',
                             metrics(filtered, capital_ytd), n_removed))

    print_section(f"YTD 2026  ({start_ytd} → {end})", variants_ytd, capital_ytd)

    filtered_3_ytd = apply_blackout_filter(trades_ytd_all, earnings_map, 3)
    by_strategy_breakdown(filtered_3_ytd, capital_ytd, 'con blackout 3 días (YTD)')

    print()


if __name__ == '__main__':
    main()
