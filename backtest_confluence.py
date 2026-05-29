"""
Backtest de CONFLUENCIA entre estrategias (Fase 2.2, variante constructiva).

Hipótesis (base académica): cuando dos estrategias independientes coinciden en
el MISMO lado del mismo ticker, la señal tiene mayor expectancy. Vale la pena
sobre-ponderar esos setups.

Se evalúa SOBRE el portfolio post-FCFS (lo que el bot realmente sostiene hoy):
primero se resuelven contradicciones con first-come-first-served, después se
analiza la confluencia sobre los trades aceptados.

"Confirmado" = al abrir el trade C ya había una posición de OTRA estrategia,
MISMO lado, abierta en ese ticker (C.entry dentro de [A.entry, A.exit]).
"Solitario" = no había confirmación al entrar (incluye al líder de un cluster).

PARTE 1 — Diagnóstico: ¿los confirmados ganan más que los solitarios?
  Gate. Si no hay diferencia de expectancy, la idea muere.

PARTE 2 — Políticas vs FCFS (solo si el diagnóstico da edge):
  FCFS         baseline (bot actual)
  CONF_ONLY    quedarse solo con los confirmados (concentrar)
  SIZEUP_1.5x  FCFS + pierna confirmada a 1.5x size
  SIZEUP_2x    FCFS + pierna confirmada a 2x size

Criterio para tocar bot.py: batir FCFS en TEST con ΔSharpe>=+0.1 y MaxDD no peor.

LIMITACIÓN: el size-up escala el PnL del trade linealmente (modelo de primer
orden). Ignora el cap de 25% por posición y el recálculo de risk_usd. Suficiente
para un screen out-of-sample; si pasa, validar con simulador día a día.

Uso: python3.10 backtest_confluence.py
"""

import copy
from datetime import datetime

import pandas as pd

from backtest_conflict_policy import natural_trades, resolve, CAPITAL
from backtest_consolidated import metrics, TICKERS


def tag_confluence(accepted):
    """Marca cada trade con 'confirmed'=True/False sobre el set post-FCFS."""
    trades = sorted(accepted, key=lambda x: x['entry_date'])
    for c in trades:
        confirmed = any(
            a is not c
            and a['ticker'] == c['ticker']
            and a['strategy'] != c['strategy']
            and a['dir'] == c['dir']
            and a['entry_date'] <= c['entry_date'] <= a['exit_date']
            for a in trades
        )
        c['confirmed'] = confirmed
    return trades


def bucket_metrics(trades, flag):
    sub = [t for t in trades if t['confirmed'] == flag]
    m = metrics(sub, CAPITAL)
    exp = (sum(t['pnl'] for t in sub) / len(sub)) if sub else 0.0
    return m, exp, len(sub)


def diagnostic(label, trades):
    print(f"\n  {label}")
    print(f"  {'-'*78}")
    print(f"  {'bucket':<12}{'N':>5}{'WR':>8}{'avgPnL':>10}{'totalPnL':>12}{'Sharpe':>8}")
    for flag, name in [(False, 'solitario'), (True, 'confirmado')]:
        m, exp, n = bucket_metrics(trades, flag)
        tot = sum(t['pnl'] for t in trades if t['confirmed'] == flag)
        print(f"  {name:<12}{n:>5}{m['wr']:>7.1f}%{exp:>10.1f}{tot:>12,.0f}{m['sharpe']:>8.2f}")


def apply_policy(accepted, policy):
    """Devuelve lista de trades (copias) con la política aplicada."""
    out = []
    for t in accepted:
        t = copy.deepcopy(t)
        if policy == 'FCFS':
            out.append(t)
        elif policy == 'CONF_ONLY':
            if t['confirmed']:
                out.append(t)
        elif policy == 'SIZEUP_1.5x':
            if t['confirmed']:
                t['pnl'] *= 1.5
            out.append(t)
        elif policy == 'SIZEUP_2x':
            if t['confirmed']:
                t['pnl'] *= 2.0
            out.append(t)
        else:
            raise ValueError(policy)
    return out


def fmt(label, m, base=None):
    pf = f"{m['pf']:.2f}" if m['pf'] != float('inf') else 'inf'
    print(f"  {label:<13} Ret {m['return']:>8.2f}%  Sharpe {m['sharpe']:>5.2f}  "
          f"MaxDD {m['maxdd']:>7.2f}%  PF {pf:>5}  WR {m['wr']:>5.1f}%  N {m['n']:>4}")
    if base is not None:
        print(f"  {'':13} vs FCFS:  Δret {m['return']-base['return']:+7.2f}%  "
              f"ΔSharpe {m['sharpe']-base['sharpe']:+5.2f}  "
              f"ΔMaxDD {m['maxdd']-base['maxdd']:+6.2f}%")


def run_window(title, start, end):
    print(f"\n{'='*88}\n  {title}\n{'='*88}")
    raw = natural_trades(TICKERS, start, end)
    accepted, _, _ = resolve(raw, 'FCFS', dfs=None)
    tagged = tag_confluence(accepted)

    n_conf = sum(1 for t in tagged if t['confirmed'])
    print(f"  Trades post-FCFS: {len(tagged)}   confirmados: {n_conf} "
          f"({100*n_conf/len(tagged):.1f}%)")

    diagnostic("DIAGNÓSTICO buckets", tagged)

    print(f"\n  POLÍTICAS")
    print(f"  {'-'*78}")
    base = metrics(apply_policy(tagged, 'FCFS'), CAPITAL)
    res = {}
    for pol in ['FCFS', 'CONF_ONLY', 'SIZEUP_1.5x', 'SIZEUP_2x']:
        m = metrics(apply_policy(tagged, pol), CAPITAL)
        res[pol] = m
        fmt(pol, m, base=None if pol == 'FCFS' else base)
    return res


def main():
    end = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'#'*88}")
    print(f"  CONFLUENCIA ENTRE ESTRATEGIAS  |  {len(TICKERS)} tickers  |  ${CAPITAL:,}")
    print(f"{'#'*88}")

    run_window("TRAIN  2021-01-01 → 2023-12-31", '2021-01-01', '2023-12-31')
    test = run_window(f"TEST (OUT-OF-SAMPLE)  2024-01-01 → {end}", '2024-01-01', end)

    print(f"\n{'='*88}\n  VEREDICTO (batir FCFS en TEST con ΔSharpe>=+0.1 y MaxDD no peor)\n{'='*88}")
    base = test['FCFS']
    for pol in ['CONF_ONLY', 'SIZEUP_1.5x', 'SIZEUP_2x']:
        m = test[pol]
        d_sh = m['sharpe'] - base['sharpe']
        d_dd = m['maxdd'] - base['maxdd']
        passes = (d_sh >= 0.1) and (d_dd >= -1e-9)
        print(f"  {pol:13s} ΔSharpe {d_sh:+.2f}  ΔMaxDD {d_dd:+.2f}%  →  "
              f"{'PASA' if passes else 'NO PASA'}")


if __name__ == '__main__':
    main()
