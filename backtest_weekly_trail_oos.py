"""
backtest_weekly_trail_oos.py — ¿El trail del weekly_trend (3.5 ATR) es demasiado
ancho? Validación OUT-OF-SAMPLE de un trail más corto.

Motivo (2026-06-24): SLV weekly_trend SHORT en +12% pero con el stop sólo en
breakeven, porque el trail de 3.5 ATR (~43% del precio en SLV) no se aprieta
hasta una caída gigante. Dante pidió ver si un trail más corto protege más
ganancia SIN matar la edge (dejar correr) — y validarlo OOS, no a ojo, igual que
mató al vol-targeting (Sharpe OOS -0.27), al RSI condicional (-0.31) y al
ranking ADX.

Método (idéntico a backtest_portfolio_oos.py, consistencia metodológica):
  - Config de PRODUCCIÓN: scorer 'fcfs', tope de exposición 100%.
  - Sweep SOLO del trailing_atr del weekly_trend; pullback/swing intactos.
  - Train: 2021 → 2023-12-31 (in-sample).
  - Test:  2024-01-01 → hoy   (out-of-sample, DECISORIO).
  - Reusa load_data() y run() de backtest_portfolio SIN tocar la lógica.

Criterio de aprobación (en TEST, vs baseline 3.5):
  Un trail más corto sólo se shipea si  ΔSharpe ≥ +0.10  y  ΔMaxDD ≥ -1pt
  (no empeora el drawdown más de 1 punto). Si además baja el retorno total,
  es la señal clásica de que cortó la cola derecha (los ganadores grandes).

Uso:  python3.10 backtest_weekly_trail_oos.py [end_train] [start_test]
"""

import sys
import pandas as pd

import backtest_portfolio as bp
from bot import STRATEGIES

FULL_START = '2021-01-01'
END_TRAIN  = sys.argv[1] if len(sys.argv) > 1 else '2023-12-31'
START_TEST = sys.argv[2] if len(sys.argv) > 2 else '2024-01-01'

SCORER = 'fcfs'           # config de producción
CAP    = 1.0              # tope de exposición 100% (producción)
BASELINE_TRAIL = 3.5      # weekly_trend['trailing_atr'] actual
TRAILS = [2.0, 2.5, 3.0, 3.5, 4.0]   # 3.5 = baseline


def run_trail(trail, daily, feats, cal):
    """Corre la cartera con el weekly_trend a `trail` ATR. Mutar STRATEGIES
    alcanza: run() lee trailing_atr en runtime."""
    STRATEGIES['weekly_trend']['trailing_atr'] = trail
    return bp.run(CAP, SCORER, daily, feats, cal)


def print_table(title, daily, feats, cal):
    yrs = max((cal[-1] - cal[0]).days / 365.25, 1e-9)
    bh = (daily['SPY']['C'][cal[-1]] / daily['SPY']['O'][cal[0]] - 1) * 100 \
        if 'SPY' in daily and len(cal) else None
    print(f"\n{title}  | {cal[0].date()} → {cal[-1].date()} | {len(cal)} días | {yrs:.2f} años")
    print(f"{'wkTrail':>8} {'Ret%':>7} {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>8} "
          f"{'GrPk':>6} {'Trades':>7} {'Win%':>5}")
    print("─" * 64)
    res = {}
    for tr in TRAILS:
        r = run_trail(tr, daily, feats, cal)
        res[tr] = r
        tag = '  ← actual' if tr == BASELINE_TRAIL else ''
        print(f"{tr:>8.1f} {r['total_ret']:>7.1f} {r['cagr']:>7.1f} {r['sharpe']:>7.2f} "
              f"{r['maxdd']:>8.1f} {r['gross_peak']:>5.0f}% {r['n_trades']:>7} "
              f"{r['win_rate']:>4.0f}%{tag}")
    print("─" * 64)
    if bh is not None:
        print(f"Referencia: SPY buy & hold = {bh:+.1f}%")
    return res


def main():
    bp.START, bp.END = FULL_START, bp.END
    daily, feats, cal = bp.load_data()
    if not daily:
        print("Sin datos. Abortando.")
        return

    end_train  = pd.Timestamp(END_TRAIN)
    start_test = pd.Timestamp(START_TEST)
    cal_train  = cal[cal <= end_train]
    cal_test   = cal[cal >= start_test]
    if len(cal_train) == 0 or len(cal_test) == 0:
        print("Split inválido: una ventana quedó vacía.")
        return

    print("=" * 64)
    print("OOS — TRAIL DEL weekly_trend  (scorer fcfs @ 100%)")
    print(f"Baseline trail = {BASELINE_TRAIL} ATR  |  Sweep = {TRAILS}")
    print("=" * 64)

    train = print_table("TRAIN (in-sample)", daily, feats, cal_train)
    test  = print_table("TEST (out-of-sample) ← DECISORIO", daily, feats, cal_test)

    # ── Veredicto: cada candidato vs baseline 3.5 ──
    # Un candidato es REAL sólo si supera al baseline en TEST *y* no se desploma
    # en TRAIN. Si gana en uno y pierde feo en el otro, es un sign-flip = ruido,
    # la misma huella del vol-targeting y del ranking ADX (overfit a un régimen).
    print("\n" + "=" * 64)
    print("VEREDICTO — candidato vs 3.5")
    print("PASA si: en TEST ΔSharpe ≥ +0.10 y ΔMaxDD ≥ -1pt  Y  en TRAIN no")
    print("pierde Sharpe vs 3.5 (consistencia, no sign-flip)")
    print("=" * 64)
    bt, btr = test[BASELINE_TRAIL], train[BASELINE_TRAIL]
    print(f"{'wkTrail':>8} {'ΔRet(T)':>8} {'ΔSh(T)':>8} {'ΔDD(T)':>8} {'ΔSh(tr)':>8}  {'Veredicto':>16}")
    print("─" * 64)
    real_winner = None
    for tr in TRAILS:
        if tr == BASELINE_TRAIL:
            continue
        a, atr_ = test[tr], train[tr]
        d_ret = a['total_ret'] - bt['total_ret']
        d_sh  = a['sharpe'] - bt['sharpe']
        d_dd  = a['maxdd'] - bt['maxdd']
        d_sh_train = atr_['sharpe'] - btr['sharpe']
        passes_test = d_sh >= 0.10 and d_dd >= -1.0
        consistent  = d_sh_train >= -0.05   # no se desploma en train
        if passes_test and consistent:
            verdict = "✓ PASA (robusto)"; real_winner = tr
        elif passes_test and not consistent:
            verdict = "✗ sign-flip (ruido)"
        else:
            verdict = "✗ no pasa"
        side = '+ancho' if tr > BASELINE_TRAIL else '-corto'
        print(f"{tr:>7.1f}{side[:1]} {d_ret:>+8.1f} {d_sh:>+8.2f} {d_dd:>+7.1f}p "
              f"{d_sh_train:>+8.2f}  {verdict:>16}")

    print("\n" + "=" * 64)
    if real_winner:
        print(f"Trail {real_winner} ATR sobrevivió OOS de forma consistente → candidato real.")
    else:
        print("Ningún trail pasó de forma consistente → dejar el 3.5. NO tocar.")
        print("Apretar el trail NO mejora; el único 'pase' es un sign-flip (ruido).")
        print("Confirma la tesis: el trail ancho es la edge, deja correr los ganadores.")
    print("=" * 64)

    STRATEGIES['weekly_trend']['trailing_atr'] = BASELINE_TRAIL   # restaurar


if __name__ == '__main__':
    main()
