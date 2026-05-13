"""
Evalúa 10 candidatos para expandir el universo del bot.

Metodología (corrección sobre v1):
  - El criterio per-ticker (Sharpe individual > 1) es demasiado estricto y
    descarta candidatos que SÍ contribuyen al portfolio. El verdadero test es:
    agregar el candidato al portfolio actual, ¿mejora Sharpe sin destruir DD?

Pasos:
  1. Backtest 5y de cada ticker (24 actuales + 10 candidatos) con las 3 estrategias.
  2. Métricas baseline: portfolio con 24 tickers.
  3. Por cada candidato: portfolio con 24 + ese candidato. Delta vs baseline.
  4. Bonus: correlación de retornos mensuales de precio vs canasta actual,
     y Sharpe individual del candidato (informativo).

Criterios de aceptación del candidato (corregidos):
  - ΔRet > 0  (debe agregar retorno absoluto)
  - ΔSharpe >= -0.05  (degradación marginal aceptable, el bot ya está en Sharpe 1.36)
  - ΔMaxDD >= -2.0pt  (DD no puede empeorar más de 2pt; positivo = mejora)

Nota sobre signos: MaxDD se reporta negativo (ej -18%). Si el portfolio extendido
queda en -17%, ΔDD = +1pt (mejora). Si queda en -20%, ΔDD = -2pt (empeora).

Uso: python3.10 backtest_universe_expansion.py
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backtester import backtest, fetch

CURRENT_UNIVERSE = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO',
    'NU', 'MELI', 'BABA',
    'SPY', 'QQQ', 'IWM',
    'GLD', 'SLV', 'XLE',
    'VIST',
]

CANDIDATES = [
    ('LLY',  'Eli Lilly',         'Healthcare / Pharma'),
    ('JNJ',  'Johnson & Johnson', 'Healthcare / Diversified'),
    ('HD',   'Home Depot',        'Consumer Discretionary'),
    ('MCD',  'McDonald\'s',       'Consumer Discretionary'),
    ('COST', 'Costco',            'Consumer Staples'),
    ('BAC',  'Bank of America',   'Financials / Banking'),
    ('GS',   'Goldman Sachs',     'Financials / IB'),
    ('CAT',  'Caterpillar',       'Industrials'),
    ('XOM',  'ExxonMobil',        'Energy'),
    ('XLU',  'Utilities ETF',     'Utilities'),
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


def ticker_trades(ticker, start, end, capital):
    """Backtest del ticker con las 3 estrategias. Devuelve trades combinados."""
    all_trades = []
    for name, cfg in STRATEGIES.items():
        merged = {**cfg, **COMMON}
        r = backtest(ticker, start=start, end=end, initial_capital=capital, **merged)
        if not r or not r.get('trades'):
            continue
        for t in r['trades']:
            t['strategy'] = name
            t['ticker']   = ticker
            all_trades.append(t)
    return all_trades


def portfolio_metrics(trades, initial):
    """Métricas del portfolio dados los trades combinados."""
    if not trades:
        return {'ret': 0, 'sharpe': 0, 'maxdd': 0, 'n': 0, 'wr': 0, 'pf': 0}

    df = pd.DataFrame(trades)
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df = df.sort_values('exit_date').reset_index(drop=True)

    cap = initial
    eq  = [cap]
    for _, row in df.iterrows():
        cap += row['pnl']
        eq.append(cap)

    eq_s   = pd.Series(eq)
    ret_s  = eq_s.pct_change().dropna()
    sharpe = (ret_s.mean() / ret_s.std() * np.sqrt(252)) if ret_s.std() > 0 else 0.0
    max_dd = float(((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100)

    n      = len(df)
    wins   = (df['pnl'] > 0).sum()
    wr     = wins / n * 100 if n else 0
    pf_loss = abs(df.loc[df['pnl'] <= 0, 'pnl'].sum())
    pf     = (df.loc[df['pnl'] > 0, 'pnl'].sum() / pf_loss) if pf_loss > 0 else float('inf')

    return {
        'ret':    (cap - initial) / initial * 100,
        'sharpe': sharpe,
        'maxdd':  max_dd,
        'n':      n,
        'wr':     wr,
        'pf':     pf,
    }


def monthly_returns(ticker, start, end):
    df = fetch(ticker, start, end, interval='1d')
    if df.empty:
        return None
    monthly = df['Close'].resample('ME').last()
    return monthly.pct_change().dropna()


def basket_monthly_returns(tickers, start, end):
    series = []
    for t in tickers:
        r = monthly_returns(t, start, end)
        if r is not None and len(r) > 12:
            series.append(r.rename(t))
    df = pd.concat(series, axis=1)
    return df.mean(axis=1)


def main():
    end      = datetime.now().strftime('%Y-%m-%d')
    start_5y = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    capital  = 10_000

    print(f"\n{'═' * 110}")
    print(f"  EVALUACIÓN DE EXPANSIÓN DE UNIVERSO  (portfolio-level test)")
    print(f"  Período: {start_5y} → {end}  |  Capital: ${capital:,}")
    print(f"  Criterios: ΔRet > 0  Y  ΔSharpe >= -0.05  Y  ΔMaxDD >= -2.0pt")
    print(f"{'═' * 110}\n")

    # 1) Backtest de cada ticker del universo actual + candidatos
    print(f"  [1/4] Backtest individual de los 24 tickers actuales...")
    universe_trades = {}
    for tk in CURRENT_UNIVERSE:
        universe_trades[tk] = ticker_trades(tk, start_5y, end, capital)
        print(f"      {tk:<5} → {len(universe_trades[tk]):>4} trades")

    print(f"\n  [2/4] Backtest individual de los 10 candidatos...")
    candidate_trades = {}
    candidate_meta   = {}
    for tk, name, sector in CANDIDATES:
        candidate_trades[tk] = ticker_trades(tk, start_5y, end, capital)
        candidate_meta[tk]   = (name, sector)
        print(f"      {tk:<5} {name:<22} ({sector:<26}) → {len(candidate_trades[tk]):>4} trades")

    # 2) Baseline portfolio: 24 actuales
    print(f"\n  [3/4] Computando baseline portfolio (24 actuales)...")
    baseline_trades = [t for trades in universe_trades.values() for t in trades]
    baseline = portfolio_metrics(baseline_trades, capital)
    print(f"      Baseline: Sharpe {baseline['sharpe']:.2f}  Ret {baseline['ret']:+.2f}%  "
          f"MaxDD {baseline['maxdd']:.2f}%  PF {baseline['pf']:.2f}  Trades {baseline['n']}")

    # 3) Para cada candidato: portfolio 24 + candidato
    print(f"\n  [4/4] Computando portfolio con cada candidato agregado...\n")
    print(f"  {'Tk':<5} {'Nombre':<22} {'Sector':<26} "
          f"{'Sharpe':>7} {'ΔSh':>6} {'Ret':>9} {'ΔRet':>7} "
          f"{'MaxDD':>7} {'ΔDD':>6} {'Tr':>4} {'Veredicto':<14}")
    print(f"  {'-' * 130}")

    # Correlación: precomputar
    basket_ret = basket_monthly_returns(CURRENT_UNIVERSE, start_5y, end)

    results = []
    for tk, _, _ in CANDIDATES:
        name, sector = candidate_meta[tk]
        c_trades = candidate_trades.get(tk, [])
        if not c_trades:
            print(f"  {tk:<5} {name:<22} {sector:<26}  sin trades, skip")
            continue

        ext_trades = baseline_trades + c_trades
        ext = portfolio_metrics(ext_trades, capital)

        d_sharpe = ext['sharpe'] - baseline['sharpe']
        d_ret    = ext['ret']    - baseline['ret']
        d_dd     = ext['maxdd']  - baseline['maxdd']

        r_ticker = monthly_returns(tk, start_5y, end)
        corr = None
        if r_ticker is not None:
            aligned = pd.concat([r_ticker, basket_ret], axis=1, join='inner').dropna()
            if len(aligned) > 12:
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

        # Standalone sharpe del candidato (informativo)
        solo = portfolio_metrics(c_trades, capital)

        passes = d_ret > 0 and d_sharpe >= -0.05 and d_dd >= -2.0
        verdict = 'PASA' if passes else 'NO'

        results.append({
            'ticker': tk, 'name': name, 'sector': sector,
            'sharpe_solo': solo['sharpe'], 'ret_solo': solo['ret'],
            'sharpe_port': ext['sharpe'], 'd_sharpe': d_sharpe,
            'ret_port':    ext['ret'],    'd_ret':    d_ret,
            'maxdd_port':  ext['maxdd'],  'd_dd':     d_dd,
            'n_solo':      solo['n'],
            'corr':        corr,
            'passes':      passes,
        })

        print(f"  {tk:<5} {name:<22} {sector:<26} "
              f"{ext['sharpe']:>7.2f} {d_sharpe:>+6.2f} {ext['ret']:>+8.2f}% {d_ret:>+6.2f}% "
              f"{ext['maxdd']:>+6.2f}% {d_dd:>+5.2f} {solo['n']:>4} {verdict:<14}")

    # 4) Survivors y resumen
    survivors = [r for r in results if r['passes']]
    print(f"\n{'═' * 110}")
    print(f"  RESUMEN")
    print(f"{'═' * 110}")
    print(f"  Baseline portfolio (24 tickers):")
    print(f"    Sharpe {baseline['sharpe']:.2f}  Ret {baseline['ret']:+.2f}%  "
          f"MaxDD {baseline['maxdd']:.2f}%  Trades {baseline['n']}")

    print(f"\n  Supervivientes ({len(survivors)}):")
    for r in sorted(survivors, key=lambda x: -x['d_sharpe']):
        corr_str = f"corr {r['corr']:.2f}" if r['corr'] is not None else "corr n/a"
        print(f"    {r['ticker']:<5} {r['name']:<22} "
              f"ΔSharpe {r['d_sharpe']:+.2f}  ΔRet {r['d_ret']:+.2f}%  "
              f"ΔDD {r['d_dd']:+.2f}%  ({corr_str}, solo-Sh {r['sharpe_solo']:.2f})")

    if survivors:
        # Portfolio final con TODOS los supervivientes juntos
        print(f"\n  Portfolio con TODOS los supervivientes agregados:")
        all_extra = [t for r in survivors for t in candidate_trades[r['ticker']]]
        final_trades = baseline_trades + all_extra
        final = portfolio_metrics(final_trades, capital)
        print(f"    Sharpe {final['sharpe']:.2f}  Ret {final['ret']:+.2f}%  "
              f"MaxDD {final['maxdd']:.2f}%  Trades {final['n']}")
        print(f"    Δ vs baseline:  ΔSharpe {final['sharpe']-baseline['sharpe']:+.2f}  "
              f"ΔRet {final['ret']-baseline['ret']:+.2f}%  "
              f"ΔDD {final['maxdd']-baseline['maxdd']:+.2f}%")

        # Combinaciones específicas (excluir los marginales)
        # Solo los que aportan >= +3% retorno
        impactful = [r for r in survivors if r['d_ret'] >= 3.0]
        if impactful and len(impactful) < len(survivors):
            print(f"\n  Portfolio solo con candidatos de ΔRet >= +3% "
                  f"({', '.join(r['ticker'] for r in impactful)}):")
            impactful_extra = [t for r in impactful for t in candidate_trades[r['ticker']]]
            ft = baseline_trades + impactful_extra
            fm = portfolio_metrics(ft, capital)
            print(f"    Sharpe {fm['sharpe']:.2f}  Ret {fm['ret']:+.2f}%  "
                  f"MaxDD {fm['maxdd']:.2f}%  Trades {fm['n']}")
            print(f"    Δ vs baseline:  ΔSharpe {fm['sharpe']-baseline['sharpe']:+.2f}  "
                  f"ΔRet {fm['ret']-baseline['ret']:+.2f}%  "
                  f"ΔDD {fm['maxdd']-baseline['maxdd']:+.2f}%")

        # Solo CAT (el más potente standalone)
        if any(r['ticker'] == 'CAT' for r in survivors):
            print(f"\n  Portfolio solo con CAT (mejor standalone):")
            ft = baseline_trades + candidate_trades['CAT']
            fm = portfolio_metrics(ft, capital)
            print(f"    Sharpe {fm['sharpe']:.2f}  Ret {fm['ret']:+.2f}%  "
                  f"MaxDD {fm['maxdd']:.2f}%  Trades {fm['n']}")
            print(f"    Δ vs baseline:  ΔSharpe {fm['sharpe']-baseline['sharpe']:+.2f}  "
                  f"ΔRet {fm['ret']-baseline['ret']:+.2f}%  "
                  f"ΔDD {fm['maxdd']-baseline['maxdd']:+.2f}%")

    # Rechazados con explicación
    rejected = [r for r in results if not r['passes']]
    if rejected:
        print(f"\n  Rechazados ({len(rejected)}):")
        for r in sorted(rejected, key=lambda x: x['d_sharpe']):
            reasons = []
            if r['d_ret'] <= 0:
                reasons.append(f"ΔRet {r['d_ret']:+.2f}%")
            if r['d_sharpe'] < -0.05:
                reasons.append(f"ΔSh {r['d_sharpe']:+.2f}")
            if r['d_dd'] < -2.0:
                reasons.append(f"ΔDD {r['d_dd']:+.2f}pt")
            print(f"    {r['ticker']:<5} {r['name']:<22} "
                  f"motivos: {', '.join(reasons)}")
    print()


if __name__ == '__main__':
    main()
