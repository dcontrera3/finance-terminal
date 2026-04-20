"""
Simulación YTD 2026 de weekly_trend con parámetros validados
(trailing 3.5 ATR, stop inicial 2.0 ATR). Muestra:
  - Trades cerrados YTD con tabla detallada
  - Posiciones que estarían abiertas al cierre del período
  - Métricas YTD
  - Últimos 20 trades históricos (5 años) como muestra del tipo de movimiento

Uso: python3.10 backtest_weekly_ytd.py
"""

from datetime import datetime, timedelta
import pandas as pd

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

WEEKLY_CFG = dict(
    strategy='weekly_trend',
    timeframe='1wk',
    signal_params=dict(ema_fast=10, ema_slow=20, adx_min=20,
                       rsi_long_max=75, rsi_short_min=25),
    atr_stop_mult=2.0,
    trailing_atr=3.5,
    rr_ratio=2.0,
    risk_per_trade=0.0075,
    dd_pause_threshold=0.10,
)


def run_weekly(tickers, start, end, capital, sim_start=None):
    """Corre weekly_trend sobre la canasta. Devuelve trades con ticker taggeado."""
    cfg = dict(WEEKLY_CFG)
    if sim_start:
        cfg['sim_start'] = sim_start
    all_trades = []
    for t in tickers:
        r = backtest(t, start=start, end=end, initial_capital=capital, **cfg)
        if r and r.get('trades'):
            for trade in r['trades']:
                trade['ticker'] = t
                all_trades.append(trade)
    return all_trades


def print_trades_table(title, trades, show_all=False, limit=None):
    print(f"\n{'─' * 112}")
    print(f"  {title}  ({len(trades)} trades)")
    print(f"{'─' * 112}")
    if not trades:
        print("  Sin trades en el período.")
        return
    print(f"  {'#':<4} {'Ticker':<7} {'Dir':<6} {'Entry date':<12} {'Entry':>10} "
          f"{'Exit date':<12} {'Exit':>10} {'Size':>6} {'PnL $':>10} {'Result':<8}")
    print(f"  {'-'*110}")

    shown = trades if show_all else (trades[-limit:] if limit else trades)
    for i, t in enumerate(shown, 1):
        ed = pd.Timestamp(t['entry_date']).strftime('%Y-%m-%d')
        xd = pd.Timestamp(t['exit_date']).strftime('%Y-%m-%d')
        pnl = t['pnl']
        pnl_str = f"{'+' if pnl >= 0 else ''}{pnl:.2f}"
        print(f"  {i:<4} {t['ticker']:<7} {t['dir']:<6} {ed:<12} "
              f"${t['entry']:>9.2f} {xd:<12} ${t['exit']:>9.2f} "
              f"{t['size']:>6.1f} {pnl_str:>10} {t['result']:<8}")


def print_open_positions(title, trades):
    """Posiciones que quedaron abiertas (result='OPEN') al final del backtest."""
    opens = [t for t in trades if t.get('result') == 'OPEN']
    print(f"\n{'─' * 112}")
    print(f"  {title}  ({len(opens)} posiciones)")
    print(f"{'─' * 112}")
    if not opens:
        print("  No hay posiciones abiertas al día de hoy.")
        return
    print(f"  {'#':<4} {'Ticker':<7} {'Dir':<6} {'Entry date':<12} {'Entry':>10} "
          f"{'Último':>10} {'Stop':>10} {'Size':>6} {'PnL $':>10} {'Días':>5}")
    print(f"  {'-'*110}")
    today = pd.Timestamp(datetime.now().date())
    for i, t in enumerate(opens, 1):
        ed = pd.Timestamp(t['entry_date']).tz_localize(None) \
             if pd.Timestamp(t['entry_date']).tz is not None else pd.Timestamp(t['entry_date'])
        days_open = (today - ed.normalize()).days
        pnl = t['pnl']
        pnl_str = f"{'+' if pnl >= 0 else ''}{pnl:.2f}"
        print(f"  {i:<4} {t['ticker']:<7} {t['dir']:<6} {ed.strftime('%Y-%m-%d'):<12} "
              f"${t['entry']:>9.2f} ${t['exit']:>9.2f} ${t['stop']:>9.2f} "
              f"{t['size']:>6.1f} {pnl_str:>10} {days_open:>5}")


def summary(trades, initial, label):
    closed = [t for t in trades if t.get('result') != 'OPEN']
    opens  = [t for t in trades if t.get('result') == 'OPEN']

    closed_pnl = sum(t['pnl'] for t in closed)
    open_pnl   = sum(t['pnl'] for t in opens)   # PnL no realizado

    capital_after = initial + closed_pnl
    equity_now    = capital_after + open_pnl

    wins   = sum(1 for t in closed if t['pnl'] > 0)
    n      = len(closed)
    wr     = (wins / n * 100) if n else 0
    gross_win  = sum(t['pnl'] for t in closed if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in closed if t['pnl'] <= 0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    print(f"\n{'═' * 112}")
    print(f"  {label}")
    print(f"{'═' * 112}")
    print(f"  Capital inicial:            ${initial:>10,.2f}")
    print(f"  P&L cerrado:                ${closed_pnl:>+10,.2f}  ({n} trades)")
    print(f"  P&L no realizado (abiertas):${open_pnl:>+10,.2f}  ({len(opens)} posiciones)")
    print(f"  Capital tras cierres:       ${capital_after:>10,.2f}")
    print(f"  Equity estimado hoy:        ${equity_now:>10,.2f}   "
          f"({(equity_now - initial) / initial * 100:+.2f}% vs inicial)")
    print(f"  Win rate:                    {wr:.1f}%   |   PF: {pf:.2f}" if pf != float('inf')
          else f"  Win rate:                    {wr:.1f}%   |   PF: inf")


def main():
    end          = datetime.now().strftime('%Y-%m-%d')
    start_ytd    = '2026-01-01'
    warmup_start = (datetime.strptime(start_ytd, '%Y-%m-%d') - timedelta(days=730)) \
                    .strftime('%Y-%m-%d')
    start_5y     = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

    ytd_capital = 5_000
    hist_capital = 10_000

    # ── YTD 2026 ──
    print(f"\n{'═' * 112}")
    print(f"  SIMULACIÓN YTD 2026  ({start_ytd} → {end})  weekly_trend")
    print(f"  Canasta: {len(TICKERS)} tickers  |  Capital: ${ytd_capital:,}")
    print(f"  Warmup: {warmup_start}  (para que las EMAs tengan historia)")
    print(f"{'═' * 112}")
    print("Corriendo YTD...")
    trades_ytd = run_weekly(TICKERS, warmup_start, end, ytd_capital, sim_start=start_ytd)

    # Orden cronológico para la tabla
    trades_ytd_sorted = sorted(trades_ytd, key=lambda t: pd.Timestamp(t['entry_date']))

    print_trades_table("TRADES YTD 2026 (cronológico)",
                       [t for t in trades_ytd_sorted if t.get('result') != 'OPEN'],
                       show_all=True)
    print_open_positions("POSICIONES ABIERTAS AL DÍA DE HOY", trades_ytd_sorted)
    summary(trades_ytd, ytd_capital, "RESUMEN YTD 2026")

    # ── Últimos 20 trades históricos (5 años) ──
    print(f"\n\n{'═' * 112}")
    print(f"  MUESTRA: últimos 20 trades cerrados del histórico 5 años")
    print(f"  (Para que veas qué tipo de movimientos captura el weekly)")
    print(f"{'═' * 112}")
    print("Corriendo histórico 5 años...")
    trades_5y = run_weekly(TICKERS, start_5y, end, hist_capital)
    closed_5y = sorted([t for t in trades_5y if t.get('result') != 'OPEN'],
                       key=lambda t: pd.Timestamp(t['exit_date']))
    print_trades_table("ÚLTIMOS 20 CIERRES HISTÓRICOS", closed_5y, limit=20)


if __name__ == '__main__':
    main()
