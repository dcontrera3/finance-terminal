"""
backtest_portfolio_oos.py — Validación OUT-OF-SAMPLE del ranking de señales por ADX.

El commit 04bf401 eligió ranking por ADX y tope de exposición 100% mirando TODO
el período 2021-2026 (in-sample). Las variantes apalancadas (150%/200%) quedaron
marcadas como "pendiente de validación out-of-sample". Esto es exactamente lo que
mató a vol-targeting (Sharpe OOS -0.27) y al filtro RSI condicional (Sharpe -0.31):
decisiones que brillaban in-sample y se derrumbaban en datos no vistos.

Mismo split que esos análisis (consistencia metodológica):
  Train: 2021 → 2023-12-31   (in-sample, donde el ranking ADX "se descubre" ganador)
  Test:  2024-01-01 → hoy    (out-of-sample, DECISORIO — nunca se vio al decidir)

Criterio de validación (idéntico a Fase 2.1 / 2.4):
  - El candidato (adx) debe batir al baseline (fcfs) en el MISMO tope, en el TEST:
        Δ Sharpe (test) ≥ +0.10  y  Δ MaxDD (test) ≥ -1pt (no empeora más de 1pt).
  - Las variantes apalancadas (>100%) sólo pasan si mantienen Sharpe en test sin
    inflar MaxDD: el apalancamiento que sólo funciona in-sample es overfitting.

Reusa load_data() y run() de backtest_portfolio.py SIN tocar la lógica del sistema:
sólo parte el calendario. Capital fresco por ventana (evaluación independiente).

Uso:  python3.10 backtest_portfolio_oos.py [end_train] [start_test]
"""
import sys
import pandas as pd

import backtest_portfolio as bp

# Rango FULL para descargar una sola vez; el split se hace sobre el calendario.
FULL_START = '2021-01-01'
END_TRAIN  = sys.argv[1] if len(sys.argv) > 1 else '2023-12-31'
START_TEST = sys.argv[2] if len(sys.argv) > 2 else '2024-01-01'

SCORERS = bp.SCORERS
CAPS    = bp.CAPS
BASELINE = 'fcfs'      # comportamiento del bot ANTES del ranking ADX
CANDIDATE = 'adx'      # lo que el commit puso en producción


def spy_bh(daily, cal):
    if 'SPY' not in daily or len(cal) == 0:
        return None
    return (daily['SPY']['C'][cal[-1]] / daily['SPY']['O'][cal[0]] - 1) * 100


def print_table(title, daily, feats, cal):
    bh = spy_bh(daily, cal)
    yrs = max((cal[-1] - cal[0]).days / 365.25, 1e-9)
    print(f"\n{title}  | {cal[0].date()} → {cal[-1].date()} | {len(cal)} días | {yrs:.2f} años")
    print(f"{'Scorer':>8} {'Tope':>8} {'Ret%':>7} {'CAGR%':>7} {'Sharpe':>7} "
          f"{'MaxDD%':>8} {'GrPk':>6} {'Trades':>7} {'Win%':>5}")
    print("─" * 78)
    res = {}
    for scorer in SCORERS:
        for c in CAPS:
            r = bp.run(c, scorer, daily, feats, cal)
            res[(scorer, c)] = r
            print(f"{r['scorer']:>8} {r['cap']:>8} {r['total_ret']:>7.1f} "
                  f"{r['cagr']:>7.1f} {r['sharpe']:>7.2f} {r['maxdd']:>8.1f} "
                  f"{r['gross_peak']:>5.0f}% {r['n_trades']:>7} {r['win_rate']:>4.0f}%")
        print("─" * 78)
    if bh is not None:
        print(f"Referencia: SPY buy & hold = {bh:+.1f}%  "
              f"(CAGR {((1+bh/100)**(1/yrs)-1)*100:.1f}%)")
    return res, bh


def main():
    # Descarga FULL una sola vez, luego corta el calendario.
    bp.START, bp.END = FULL_START, bp.END
    daily, feats, cal = bp.load_data()
    if not daily:
        print("Sin datos. Abortando.")
        return

    end_train = pd.Timestamp(END_TRAIN)
    start_test = pd.Timestamp(START_TEST)
    cal_train = cal[cal <= end_train]
    cal_test  = cal[cal >= start_test]
    if len(cal_train) == 0 or len(cal_test) == 0:
        print("Split inválido: una de las ventanas quedó vacía.")
        return

    print("=" * 78)
    print("VALIDACIÓN OUT-OF-SAMPLE — Ranking de señales por ADX + tope de exposición")
    print(f"Baseline = '{BASELINE}'  |  Candidato = '{CANDIDATE}'  |  Caps = {CAPS}")
    print("=" * 78)

    train_res, _ = print_table("TRAIN (in-sample)", daily, feats, cal_train)
    test_res, bh_test = print_table("TEST (out-of-sample) ← DECISORIO", daily, feats, cal_test)

    # ── Veredicto: candidato vs baseline en el MISMO tope, en TEST ──
    print("\n" + "=" * 78)
    print("VEREDICTO — adx vs fcfs en TEST (criterio: ΔSharpe ≥ +0.10 y ΔMaxDD ≥ -1pt)")
    print("=" * 78)
    print(f"{'Tope':>8} {'ΔRet%':>8} {'ΔSharpe':>9} {'ΔMaxDD':>9} {'Veredicto':>14}")
    print("─" * 78)
    passes = []
    for c in CAPS:
        b = test_res[(BASELINE, c)]
        a = test_res[(CANDIDATE, c)]
        d_ret = a['total_ret'] - b['total_ret']
        d_sh  = a['sharpe'] - b['sharpe']
        d_dd  = a['maxdd'] - b['maxdd']   # maxdd es negativo; más alto (menos negativo) = mejor
        ok = d_sh >= 0.10 and d_dd >= -1.0
        passes.append((c, ok))
        verdict = "✓ PASA" if ok else "✗ no pasa"
        print(f"{b['cap']:>8} {d_ret:>+8.1f} {d_sh:>+9.2f} {d_dd:>+8.1f}pt {verdict:>14}")

    # ── Chequeo de apalancamiento: ¿sobrevive >100% en OOS? ──
    print("\n" + "=" * 78)
    print("APALANCAMIENTO EN TEST — ¿el candidato adx aguanta gross > 100% OOS?")
    print("=" * 78)
    base100 = test_res[(CANDIDATE, 1.0)]
    print(f"adx @ 100% (ancla)  → Sharpe {base100['sharpe']:.2f} | "
          f"Ret {base100['total_ret']:+.1f}% | MaxDD {base100['maxdd']:.1f}%")
    for c in [x for x in CAPS if x > 1.0]:
        r = test_res[(CANDIDATE, c)]
        d_sh = r['sharpe'] - base100['sharpe']
        d_dd = r['maxdd'] - base100['maxdd']
        flag = "vale el riesgo" if (d_sh >= 0.0 and d_dd >= -3.0) else "NO compensa (overfit de leverage)"
        print(f"adx @ {r['cap']:<5}        → Sharpe {r['sharpe']:.2f} ({d_sh:+.2f}) | "
              f"Ret {r['total_ret']:+.1f}% | MaxDD {r['maxdd']:.1f}% ({d_dd:+.1f}pt) → {flag}")

    if bh_test is not None:
        print(f"\nSPY buy&hold en TEST = {bh_test:+.1f}%")
    chosen_ok = dict(passes).get(1.0, False)
    print("\n" + "=" * 78)
    print(f"CONFIG EN PRODUCCIÓN (adx @ 100%): "
          f"{'VALIDADA out-of-sample ✓' if chosen_ok else 'NO validada OOS — revisar ✗'}")
    print("=" * 78)


if __name__ == '__main__':
    main()
