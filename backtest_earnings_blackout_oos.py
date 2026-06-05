"""
backtest_earnings_blackout_oos.py — Validación OUT-OF-SAMPLE del earnings blackout.

Motivación: incidente 2026-06-04. AVGO LONG gapeó -15% al open tras earnings de
Broadcom (2026-06-03 AMC). El TRAIL GTC ejecutó bien pero filleó $49 abajo del
stop: -$20,623 en un trade que estaba +6%. Ningún stop protege contra gaps; la
única defensa es no estar expuesto al evento.

El script viejo (backtest_earnings_blackout.py, abril) solo filtraba ENTRADAS
post-hoc y miraba todo el período junto (in-sample). Esto extiende con:

  1. Regla A — entry blackout: descarta trades que abren 0-N días antes de un
     earnings del ticker (N = 2, 3, 5).
  2. Regla B — early exit: si un trade quedaba abierto a través de un earnings,
     simula el cierre al close de la última sesión segura (AMC: el mismo día del
     reporte; BMO: la sesión anterior). El P&L se recalcula con el close real.
  3. Split train/test idéntico a Fase 2.1 / 2.4 / ranking ADX:
        Train: 2021 → 2023-12-31   (in-sample)
        Test:  2024-01-01 → hoy    (out-of-sample, DECISORIO)
     Criterio: ΔSharpe (test) ≥ +0.10 y ΔMaxDD (test) ≥ -1pt vs baseline.

Sesgo conocido (conservador contra el filtro): la Regla B no simula re-entrada
post-earnings, así que los winners que seguían corriendo después del reporte
pierden esa cola. Si aun así valida OOS, el caso es sólido.

Uso: python3.10 backtest_earnings_blackout_oos.py [end_train] [start_test]
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from backtester import backtest, fetch

# Universo actual del bot (27 tickers, post-expansión 2026-05-13)
TICKERS = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO', 'COST',
    'CAT', 'XLU',
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

WARMUP_START = '2019-01-01'   # data extra para indicadores (EMA200 diaria, semanales)
SIM_START    = '2021-01-01'
END_TRAIN    = sys.argv[1] if len(sys.argv) > 1 else '2023-12-31'
START_TEST   = sys.argv[2] if len(sys.argv) > 2 else '2024-01-01'
CAPITAL      = 10_000


# ══════════════════════════════════════════
# EARNINGS
# ══════════════════════════════════════════

def fetch_earnings_map(tickers, since):
    """{ticker: [(fecha naive normalizada, is_amc), ...]} ordenado.
    is_amc=True si el reporte fue after-market-close (hora >= 12 ET):
    el gap pega en la sesión SIGUIENTE; BMO pega en la apertura del mismo día.
    ETFs sin earnings → lista vacía."""
    out = {}
    floor = pd.Timestamp(since)
    for t in tickers:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=60)
            if ed is None or len(ed) == 0:
                out[t] = []
                continue
            entries = []
            for d in ed.index:
                ts = pd.Timestamp(d)
                if ts.tz is not None:
                    ts = ts.tz_localize(None)
                if pd.isna(ts) or ts < floor:
                    continue
                entries.append((ts.normalize(), ts.hour >= 12))
            out[t] = sorted(set(entries))
        except Exception:
            out[t] = []
    return out


def gap_sessions(earnings, cal):
    """Convierte cada earnings en la sesión donde pega el gap.
    AMC en día D → primera sesión > D. BMO en día D → primera sesión >= D."""
    out = []
    for edate, is_amc in earnings:
        idx = cal.searchsorted(edate, side='right' if is_amc else 'left')
        if idx < len(cal):
            out.append(cal[idx])
    return sorted(set(out))


# ══════════════════════════════════════════
# REGLAS
# ══════════════════════════════════════════

def norm(d):
    ts = pd.Timestamp(d)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def in_blackout(entry_date, earnings, blackout_days):
    """Regla A: True si la entrada cae 0-N días antes de un earnings."""
    entry = norm(entry_date)
    for edate, _ in earnings:
        delta = (edate - entry).days
        if 0 <= delta <= blackout_days:
            return True
    return False


def apply_early_exit(trades, gaps_map, closes_map, cal):
    """Regla B: si un trade cruza un gap de earnings, lo cierra al close de la
    última sesión previa al gap y recalcula el P&L. Retorna (trades, stats)."""
    out, n_cut, pnl_delta = [], 0, 0.0
    for t in trades:
        gaps = gaps_map.get(t['ticker'], [])
        closes = closes_map.get(t['ticker'])
        if not gaps or closes is None:
            out.append(t)
            continue
        entry_d, exit_d = norm(t['entry_date']), norm(t['exit_date'])
        hit = next((g for g in gaps if entry_d < g <= exit_d), None)
        if hit is None:
            out.append(t)
            continue
        # Última sesión estrictamente antes del gap (>= día de entrada)
        idx = cal.searchsorted(hit, side='left') - 1
        if idx < 0:
            out.append(t)
            continue
        safe_day = max(cal[idx], entry_d)
        new_exit = closes.get(safe_day)
        if new_exit is None or pd.isna(new_exit):
            out.append(t)
            continue
        sign = 1 if t['dir'] == 'LONG' else -1
        new_pnl = (float(new_exit) - t['entry']) * t['size'] * sign
        pnl_delta += new_pnl - t['pnl']
        n_cut += 1
        out.append({**t, 'exit': float(new_exit), 'exit_date': safe_day,
                    'pnl': new_pnl, 'result': 'EARN_EXIT'})
    return out, n_cut, pnl_delta


# ══════════════════════════════════════════
# PORTFOLIO + MÉTRICAS (idéntico al script de abril)
# ══════════════════════════════════════════

def run_portfolio(strat_names, tickers, start, end, capital, sim_start):
    all_trades = []
    for name in strat_names:
        cfg = {**STRATEGIES[name], **COMMON, 'sim_start': sim_start}
        for t in tickers:
            r = backtest(t, start=start, end=end, initial_capital=capital, **cfg)
            if r and r.get('trades'):
                for trade in r['trades']:
                    trade['strategy'] = name
                    trade['ticker']   = t
                    all_trades.append(trade)
    return all_trades


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

    return {'capital': capital, 'return': (capital - initial) / initial * 100,
            'sharpe': sharpe, 'maxdd': max_dd, 'pf': pf, 'wr': wr, 'n': n}


def split_window(trades, end_train, start_test):
    train = [t for t in trades if norm(t['entry_date']) <= pd.Timestamp(end_train)]
    test  = [t for t in trades if norm(t['entry_date']) >= pd.Timestamp(start_test)]
    return train, test


def print_table(title, variants, capital):
    print(f"\n{title}")
    print(f"  {'Variante':<34} {'Ret':>9} {'Sharpe':>8} {'MaxDD':>9} {'PF':>6} "
          f"{'WR':>6} {'Trades':>7}")
    print(f"  {'-' * 84}")
    baseline = variants[0][1]
    for name, m in variants:
        pf_s = f"{m['pf']:.2f}" if m['pf'] != float('inf') else 'inf'
        print(f"  {name:<34} {m['return']:>8.2f}% {m['sharpe']:>8.2f} "
              f"{m['maxdd']:>8.2f}% {pf_s:>6} {m['wr']:>5.1f}% {m['n']:>7}")
    print(f"  {'-' * 84}")
    for name, m in variants[1:]:
        d_sh = m['sharpe'] - baseline['sharpe']
        d_dd = m['maxdd']  - baseline['maxdd']
        d_rt = m['return'] - baseline['return']
        print(f"    {name:<34} Δret {d_rt:+7.2f}%  ΔSharpe {d_sh:+5.2f}  ΔDD {d_dd:+5.2f}pt")


def main():
    end = datetime.now().strftime('%Y-%m-%d')
    strat_names = list(STRATEGIES)

    print("=" * 88)
    print(f"EARNINGS BLACKOUT — VALIDACIÓN OUT-OF-SAMPLE  |  {end}")
    print(f"Train: {SIM_START} → {END_TRAIN}   |   Test: {START_TEST} → {end}  ← DECISORIO")
    print(f"Criterio: ΔSharpe (test) ≥ +0.10  y  ΔMaxDD (test) ≥ -1pt  vs baseline")
    print("=" * 88)

    print("\nCargando earnings históricos (limit=60 por ticker)...")
    earnings_map = fetch_earnings_map(TICKERS, since='2020-10-01')
    n_has = sum(1 for t in TICKERS if earnings_map[t])
    print(f"  {n_has}/{len(TICKERS)} tickers con earnings | "
          f"{sum(len(v) for v in earnings_map.values())} fechas desde 2020-10")
    no_earn = [t for t in TICKERS if not earnings_map[t]]
    print(f"  Sin earnings (ETFs/no data): {', '.join(no_earn) or 'ninguno'}")
    for t in TICKERS:
        if earnings_map[t] and len(earnings_map[t]) < 18:
            print(f"  ⚠ {t}: solo {len(earnings_map[t])} earnings — historial incompleto, "
                  f"el filtro queda subestimado en ese ticker")

    print("\nCorriendo portfolio 2021 → hoy (27 tickers × 3 estrategias)...")
    trades_all = run_portfolio(strat_names, TICKERS, WARMUP_START, end,
                               CAPITAL, sim_start=SIM_START)
    print(f"  {len(trades_all)} trades generados")

    # Calendario y closes diarios para la Regla B
    spy = fetch('SPY', WARMUP_START, end, interval='1d')
    cal = pd.DatetimeIndex([norm(d) for d in spy.index])
    closes_map, gaps_map = {}, {}
    for t in TICKERS:
        px = fetch(t, WARMUP_START, end, interval='1d')
        closes_map[t] = ({norm(d): float(c) for d, c in px['Close'].items()}
                         if not px.empty else None)
        gaps_map[t] = gap_sessions(earnings_map[t], cal)

    # ── Variantes ──
    def build_variants(trades):
        variants = [('Sin blackout (baseline)', metrics(trades, CAPITAL))]
        for days in (2, 3, 5):
            filt = [t for t in trades
                    if not in_blackout(t['entry_date'], earnings_map[t['ticker']], days)]
            variants.append((f'A: entry blackout {days}d (-{len(trades)-len(filt)} tr)',
                             metrics(filt, CAPITAL)))
        ee, n_cut, dpnl = apply_early_exit(trades, gaps_map, closes_map, cal)
        variants.append((f'B: early exit ({n_cut} cortados, Δ${dpnl:+,.0f})',
                         metrics(ee, CAPITAL)))
        filt3 = [t for t in trades
                 if not in_blackout(t['entry_date'], earnings_map[t['ticker']], 3)]
        combo, n_cut2, dpnl2 = apply_early_exit(filt3, gaps_map, closes_map, cal)
        variants.append((f'A3+B combo ({n_cut2} cortados, Δ${dpnl2:+,.0f})',
                         metrics(combo, CAPITAL)))
        return variants

    train_tr, test_tr = split_window(trades_all, END_TRAIN, START_TEST)
    print(f"  Split por entry_date: {len(train_tr)} train | {len(test_tr)} test")

    variants_train = build_variants(train_tr)
    variants_test  = build_variants(test_tr)

    print_table(f"TRAIN (in-sample)  {SIM_START} → {END_TRAIN}", variants_train, CAPITAL)
    print_table(f"TEST (out-of-sample) ← DECISORIO  {START_TEST} → {end}",
                variants_test, CAPITAL)

    # ── Veredicto ──
    print("\n" + "=" * 88)
    print("VEREDICTO EN TEST (ΔSharpe ≥ +0.10 y ΔMaxDD ≥ -1pt)")
    print("=" * 88)
    base = variants_test[0][1]
    for name, m in variants_test[1:]:
        d_sh = m['sharpe'] - base['sharpe']
        d_dd = m['maxdd'] - base['maxdd']
        ok = d_sh >= 0.10 and d_dd >= -1.0
        print(f"  {name:<44} ΔSharpe {d_sh:+5.2f}  ΔDD {d_dd:+5.2f}pt  "
              f"{'✓ PASA' if ok else '✗ no pasa'}")
    print("\nNota: la Regla B no simula re-entrada post-earnings (sesgo conservador")
    print("contra el filtro). Los ETFs no tienen earnings y no se ven afectados.")


if __name__ == '__main__':
    main()
