"""
Analiza el gross exposure histórico del bot reconstruyendo el snapshot de
posiciones abiertas en cada momento del history.

Para cada evento (open/close), calcula:
  - Cantidad de posiciones abiertas en ese momento
  - Gross exposure = sum(entry × size) / current_equity
  - Net exposure  = sum(entry × size × signo) / current_equity

Reporta el pico de cada métrica, distribución y línea de tiempo.
"""
import json
from datetime import datetime
from collections import defaultdict

PATH = '/home/futit/workspace/financial-terminal/bot_state.json'

with open(PATH) as f:
    s = json.load(f)

history = sorted(s.get('history', []), key=lambda h: h.get('ts', ''))
current_equity = s.get('current_equity') or 1_000_000

# Reconstruir secuencia de eventos con snapshot de posiciones abiertas
open_positions = {}  # key (ticker, strategy) → {entry, size, dir}
snapshots = []  # lista de {ts, num_pos, gross_notional, net_notional, positions}

for h in history:
    key = (h['ticker'], h.get('strategy', '?'))
    if h['event'] == 'open':
        open_positions[key] = {
            'entry': h.get('entry') or h.get('price'),
            'size':  h.get('size'),
            'dir':   h.get('dir'),
        }
    elif h['event'] == 'close':
        open_positions.pop(key, None)

    # Snapshot después del evento
    gross = sum(p['entry'] * p['size'] for p in open_positions.values()
                if p['entry'] and p['size'])
    net   = sum(p['entry'] * p['size'] * (1 if p['dir'] == 'LONG' else -1)
                for p in open_positions.values()
                if p['entry'] and p['size'])
    snapshots.append({
        'ts':             h['ts'],
        'event':          h['event'],
        'ticker':         h['ticker'],
        'num_pos':        len(open_positions),
        'gross_notional': gross,
        'net_notional':   net,
        'positions':      list(open_positions.keys()),
    })

# Métricas agregadas
print("=" * 75)
print(f"GROSS EXPOSURE ANALYSIS — equity actual: ${current_equity:,.0f}")
print(f"Período: {history[0]['ts'][:10]} → {history[-1]['ts'][:10]}")
print(f"Total eventos: {len(history)}")
print("=" * 75)

# Pico de posiciones simultáneas
max_pos = max(snapshots, key=lambda x: x['num_pos'])
print(f"\nPico de posiciones simultáneas: {max_pos['num_pos']}")
print(f"  Fecha: {max_pos['ts'][:19]}")
print(f"  Posiciones: {[f'{t}/{st}' for t,st in max_pos['positions']]}")

# Pico de gross exposure (notional)
max_gross = max(snapshots, key=lambda x: x['gross_notional'])
gross_pct = max_gross['gross_notional'] / current_equity * 100
print(f"\nPico de gross exposure: ${max_gross['gross_notional']:,.0f}  ({gross_pct:.1f}% del equity)")
print(f"  Fecha: {max_gross['ts'][:19]}")
print(f"  N posiciones: {max_gross['num_pos']}")
print(f"  Posiciones: {[f'{t}/{st}' for t,st in max_gross['positions']]}")

# Net exposure (long - short) pico absoluto
max_net = max(snapshots, key=lambda x: abs(x['net_notional']))
net_pct = max_net['net_notional'] / current_equity * 100
print(f"\nPico de net exposure: ${max_net['net_notional']:+,.0f}  ({net_pct:+.1f}% del equity)")
print(f"  Fecha: {max_net['ts'][:19]}")

# Distribución de gross exposure
print(f"\n{'─' * 75}")
print("Distribución de gross exposure (% del equity actual):")
print(f"{'─' * 75}")
buckets = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 150), (150, 999)]
for lo, hi in buckets:
    count = sum(1 for s_ in snapshots
                if lo <= s_['gross_notional']/current_equity*100 < hi)
    bar = '█' * int(count / max(1, len(snapshots)) * 50)
    label = f"{lo:>3}–{hi:<3}%" if hi < 999 else f">{lo}%   "
    print(f"  {label}: {count:>3} momentos  {bar}")

# Días con >100% (leverage natural)
over_100 = [s_ for s_ in snapshots if s_['gross_notional']/current_equity*100 > 100]
print(f"\nMomentos con gross > 100% (leverage natural): {len(over_100)}")
if over_100:
    print(f"  Ejemplos:")
    for s_ in over_100[:5]:
        pct = s_['gross_notional']/current_equity*100
        print(f"    {s_['ts'][:19]}  {pct:.1f}%  {s_['num_pos']} posiciones")

# Línea de tiempo: ventanas con gross > 50%
print(f"\n{'─' * 75}")
print("Top 10 momentos por gross exposure:")
print(f"{'─' * 75}")
top = sorted(snapshots, key=lambda x: -x['gross_notional'])[:10]
print(f"{'Fecha':<20} {'Evento':<7} {'N':>3} {'Gross $':>14} {'%Eq':>7} {'Net $':>14}")
for s_ in top:
    g_pct = s_['gross_notional'] / current_equity * 100
    print(f"{s_['ts'][:19]:<20} {s_['event']:<7} {s_['num_pos']:>3} "
          f"${s_['gross_notional']:>12,.0f} {g_pct:>6.1f}% ${s_['net_notional']:>+12,.0f}")

# Estado actual
print(f"\n{'─' * 75}")
print("Estado AHORA:")
print(f"{'─' * 75}")
current_open = s.get('positions', {})
gross_now = sum(p['entry'] * p['size'] for p in current_open.values())
net_now = sum(p['entry'] * p['size'] * (1 if p['dir'] == 'LONG' else -1)
              for p in current_open.values())
print(f"  Posiciones abiertas: {len(current_open)}")
print(f"  Gross exposure: ${gross_now:,.0f}  ({gross_now/current_equity*100:.1f}% del equity)")
print(f"  Net exposure:   ${net_now:+,.0f}  ({net_now/current_equity*100:+.1f}% del equity)")
for k, p in current_open.items():
    notion = p['entry'] * p['size']
    pct = notion / current_equity * 100
    print(f"    {k}: {p['dir']:5s}  ${notion:>10,.0f}  ({pct:>4.1f}% equity)")
