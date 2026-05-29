"""
Backtest de POLÍTICAS DE RESOLUCIÓN DE CONTRADICCIONES entre estrategias.

Pregunta que responde (disparada por COST 2026-05-29):
  Cuando dos estrategias dan señales OPUESTAS sobre el mismo ticker, ¿conviene
  "elegir la mejor" en vez del first-come-first-served que corre hoy el bot?

Políticas comparadas (todas sobre el MISMO set de trades naturales):
  ALL       Toma todas las señales (lo que asume hoy backtest_consolidated;
            las contradicciones se auto-hedgean). Baseline histórico.
  FCFS      First-come-first-served: si ya hay una posición opuesta abierta de
            otra estrategia, descarta la nueva. Lo que el bot hace HOY.
  TREND     En una contradicción gana el lado alineado con la tendencia macro
            (signo de Close - EMA200 diaria del ticker en la fecha del conflicto).
            Si el challenger gana, se trunca el incumbente en esa fecha y se abre
            el challenger. Parameter-free, theory-driven.
  PRIORITY  Gana la estrategia con mayor Sharpe histórico (ranking derivado SOLO
            del train, sin lookahead). Mismo mecanismo de truncado.

Metodología honesta (igual que vol-targeting y rsi-conditional ya archivados):
  TRAIN 2021-2023  →  TEST out-of-sample 2024-2026.
  Criterio para tocar bot.py: la política candidata debe batir a FCFS en TEST
  con ΔSharpe >= +0.1 y MaxDD no peor.

LIMITACIÓN del modelo: el truncado del incumbente sale a Close de la barra del
conflicto (modelo de "flip al cierre de la barra de decisión"). No simula el
camino intradía. Es un modelo de primer orden del DECISION RULE, suficiente para
un screen out-of-sample; si pasa, se valida con un simulador día a día.

Uso: python3.10 backtest_conflict_policy.py
"""

import copy
from datetime import datetime

import numpy as np
import pandas as pd

from backtester import fetch, add_indicators, backtest
from backtest_consolidated import metrics, STRATEGIES, COMMON, TICKERS

TF_BY_STRAT = {'swing': '1d', 'pullback': '1d', 'weekly_trend': '1wk'}
CAPITAL = 10_000


def norm_ts(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


# ──────────────────────────────────────────────────────────────────────
# 1. Trades naturales por (estrategia, ticker)  +  dataframes de apoyo
# ──────────────────────────────────────────────────────────────────────
def natural_trades(tickers, start, end):
    """Corre backtest() por estrategia/ticker (igual que backtest_consolidated)
    y devuelve la lista combinada de trades, cada uno taggeado."""
    out = []
    for name, base in STRATEGIES.items():
        cfg = {**base, **COMMON}
        for t in tickers:
            r = backtest(t, start=start, end=end, initial_capital=CAPITAL, **cfg)
            if not r or not r.get('trades'):
                continue
            for tr in r['trades']:
                out.append({
                    'ticker':     t,
                    'strategy':   name,
                    'dir':        tr['dir'],
                    'entry':      float(tr['entry']),
                    'size':       float(tr['size']),
                    'entry_date': norm_ts(tr['entry_date']),
                    'exit_date':  norm_ts(tr['exit_date']),
                    'exit':       float(tr['exit']),
                    'pnl':        float(tr['pnl']),
                    'result':     tr['result'],
                })
    return out


def support_dfs(tickers, start, end):
    """Por ticker: df diario y semanal con indicadores, para lookup de
    EMA200/Close (tendencia macro y precio de truncado)."""
    dfs = {}
    for t in tickers:
        d = {}
        raw_d = fetch(t, start, end, '1d')
        if not raw_d.empty:
            di = add_indicators(raw_d.copy())
            di.index = [norm_ts(x) for x in di.index]
            d['1d'] = di
        raw_w = fetch(t, start, end, '1wk')
        if not raw_w.empty:
            wi = add_indicators(raw_w.copy())
            wi.index = [norm_ts(x) for x in wi.index]
            d['1wk'] = wi
        dfs[t] = d
    return dfs


def _row_at_or_before(df, date):
    if df is None or df.empty:
        return None
    idx = df.index
    pos = idx.searchsorted(date, side='right') - 1
    if pos < 0:
        return None
    return df.iloc[pos]


def trend_sign(dfs, ticker, date):
    """+1 si Close > EMA200 diaria en la fecha, -1 si no. 0 si no hay dato."""
    row = _row_at_or_before(dfs.get(ticker, {}).get('1d'), date)
    if row is None or pd.isna(row.get('ema200')):
        return 0
    return 1 if float(row['Close']) > float(row['ema200']) else -1


def trunc_price(dfs, ticker, tf, date):
    """Close de la barra <= date en el timeframe del incumbente."""
    row = _row_at_or_before(dfs.get(ticker, {}).get(tf), date)
    if row is None:
        return None
    return float(row['Close'])


# ──────────────────────────────────────────────────────────────────────
# 2. Resolución de conflictos
# ──────────────────────────────────────────────────────────────────────
def _truncate(inc, dfs, at_date):
    """Cierra el incumbente a Close de la barra del conflicto. Muta el dict."""
    tf = TF_BY_STRAT[inc['strategy']]
    px = trunc_price(dfs, inc['ticker'], tf, at_date)
    if px is None:
        return False
    sign = 1 if inc['dir'] == 'LONG' else -1
    inc['exit'] = px
    inc['exit_date'] = at_date
    inc['pnl'] = (px - inc['entry']) * inc['size'] * sign
    inc['result'] = 'FLIP'
    return True


def resolve(trades, policy, dfs, rank=None):
    """Devuelve (accepted_trades, n_conflicts, n_truncated)."""
    trades = sorted((copy.deepcopy(t) for t in trades),
                    key=lambda x: x['entry_date'])

    if policy == 'ALL':
        return trades, 0, 0

    accepted = []
    n_conf = 0
    n_trunc = 0

    for c in trades:
        opp = [a for a in accepted
               if a['ticker'] == c['ticker']
               and a['dir'] != c['dir']
               and a['entry_date'] <= c['entry_date'] <= a['exit_date']]
        if not opp:
            accepted.append(c)
            continue

        n_conf += 1

        if policy == 'FCFS':
            # el/los incumbente(s) llegaron primero → se descarta el challenger
            continue

        if policy == 'TREND':
            ts = trend_sign(dfs, c['ticker'], c['entry_date'])
            c_dir = 1 if c['dir'] == 'LONG' else -1
            challenger_wins = (ts != 0 and ts == c_dir)

        elif policy == 'PRIORITY':
            c_rank = rank[c['strategy']]
            best_inc_rank = max(rank[a['strategy']] for a in opp)
            # gana el challenger solo si es ESTRICTAMENTE de mayor prioridad
            challenger_wins = c_rank > best_inc_rank
        else:
            raise ValueError(policy)

        if not challenger_wins:
            continue

        # challenger gana: truncar incumbentes opuestos y abrir el challenger
        for a in opp:
            if _truncate(a, dfs, c['entry_date']):
                n_trunc += 1
        accepted.append(c)

    return accepted, n_conf, n_trunc


# ──────────────────────────────────────────────────────────────────────
# 3. Reporte
# ──────────────────────────────────────────────────────────────────────
def strategy_sharpe_rank(trades):
    """Ranking de estrategias por Sharpe (1 = peor). Derivado del set dado."""
    sh = {}
    for name in set(t['strategy'] for t in trades):
        sub = [t for t in trades if t['strategy'] == name]
        sh[name] = metrics(sub, CAPITAL)['sharpe']
    order = sorted(sh, key=lambda k: sh[k])  # ascendente
    return {name: i + 1 for i, name in enumerate(order)}, sh


def fmt(label, m, base=None, extra=''):
    pf = f"{m['pf']:.2f}" if m['pf'] != float('inf') else 'inf'
    line = (f"  {label:<10} Ret {m['return']:>8.2f}%  Sharpe {m['sharpe']:>5.2f}  "
            f"MaxDD {m['maxdd']:>7.2f}%  PF {pf:>5}  WR {m['wr']:>5.1f}%  "
            f"N {m['n']:>4}{extra}")
    print(line)
    if base is not None:
        print(f"  {'':10} vs FCFS:  Δret {m['return']-base['return']:+7.2f}%  "
              f"ΔSharpe {m['sharpe']-base['sharpe']:+5.2f}  "
              f"ΔMaxDD {m['maxdd']-base['maxdd']:+6.2f}%")


def run_window(title, tickers, start, end, rank):
    print(f"\n{'='*92}\n  {title}\n{'='*92}")
    trades = natural_trades(tickers, start, end)
    dfs = support_dfs(tickers, start, end)

    results = {}
    info = {}
    for pol in ['ALL', 'FCFS', 'TREND', 'PRIORITY']:
        acc, nc, nt = resolve(trades, pol, dfs, rank)
        results[pol] = metrics(acc, CAPITAL)
        info[pol] = (nc, nt, len(acc))

    base = results['FCFS']
    for pol in ['ALL', 'FCFS', 'TREND', 'PRIORITY']:
        nc, nt, nacc = info[pol]
        extra = f"  [conf={nc} trunc={nt}]" if pol != 'ALL' else ''
        fmt(pol, results[pol], base=None if pol == 'FCFS' else base, extra=extra)
    return results


def main():
    end = datetime.now().strftime('%Y-%m-%d')
    train_start, train_end = '2021-01-01', '2023-12-31'
    test_start,  test_end  = '2024-01-01', end

    print(f"\n{'#'*92}")
    print(f"  POLÍTICAS DE CONTRADICCIÓN  |  {len(TICKERS)} tickers  |  capital ${CAPITAL:,}")
    print(f"  TRAIN {train_start}→{train_end}   TEST {test_start}→{test_end}")
    print(f"{'#'*92}")

    # Ranking de estrategias derivado SOLO del train (sin lookahead)
    train_trades = natural_trades(TICKERS, train_start, train_end)
    rank, sh = strategy_sharpe_rank(train_trades)
    print("\n  Ranking PRIORITY (Sharpe train, 1=peor):")
    for name in sorted(rank, key=lambda k: rank[k]):
        print(f"    {rank[name]}. {name:14s} Sharpe_train={sh[name]:.2f}")

    run_window(f"TRAIN  {train_start} → {train_end}", TICKERS, train_start, train_end, rank)
    test_res = run_window(f"TEST (OUT-OF-SAMPLE)  {test_start} → {test_end}",
                          TICKERS, test_start, test_end, rank)

    # ── Veredicto ────────────────────────────────────────────────────
    print(f"\n{'='*92}\n  VEREDICTO (criterio: batir FCFS en TEST con ΔSharpe>=+0.1 y MaxDD no peor)\n{'='*92}")
    base = test_res['FCFS']
    for pol in ['ALL', 'TREND', 'PRIORITY']:
        m = test_res[pol]
        d_sh = m['sharpe'] - base['sharpe']
        d_dd = m['maxdd'] - base['maxdd']
        passes = (d_sh >= 0.1) and (d_dd >= -1e-9)
        verdict = 'PASA' if passes else 'NO PASA'
        print(f"  {pol:10s} ΔSharpe {d_sh:+.2f}  ΔMaxDD {d_dd:+.2f}%  →  {verdict}")


if __name__ == '__main__':
    main()
