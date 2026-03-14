#!/usr/bin/env python3
"""
CEDEAR Terminal - Servidor Local
Resuelve CORS para Yahoo Finance. Requiere: pip install yfinance flask flask-cors
Uso: python server.py
"""

import json
import sys

# Auto-instalar dependencias si faltan
def install(pkg):
    import subprocess
    print(f"Instalando {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import yfinance as yf
except ImportError:
    install("yfinance")
    import yfinance as yf

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
except ImportError:
    install("flask")
    install("flask-cors")
    from flask import Flask, jsonify, request
    from flask_cors import CORS

try:
    import anthropic
except ImportError:
    install("anthropic")
    import anthropic

import math
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Permite requests desde el HTML abierto localmente

INTERVAL_MAP = {
    "1D":  ("1d",  "6mo"),
    "1W":  ("1wk", "2y"),
    "1M":  ("1mo", "5y"),
    "3M":  ("3mo", "10y"),
}

def clean(val):
    """Convierte NaN/Inf a None para JSON válido"""
    if val is None: return None
    try:
        if math.isnan(val) or math.isinf(val): return None
        return round(float(val), 4)
    except: return None

@app.route("/quote/<ticker>")
def quote(ticker):
    tf = request.args.get("tf", "1M")
    interval, period = INTERVAL_MAP.get(tf, ("1mo", "5y"))

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, auto_adjust=True)

        if hist.empty:
            return jsonify({"error": f"No data for {ticker}"}), 404

        rows = []
        for ts, row in hist.iterrows():
            rows.append({
                "date":   ts.strftime("%Y-%m-%d"),
                "open":   clean(row["Open"]),
                "high":   clean(row["High"]),
                "low":    clean(row["Low"]),
                "close":  clean(row["Close"]),
                "volume": int(row["Volume"]) if row["Volume"] == row["Volume"] else 0,
            })

        # Filtrar filas con datos nulos
        rows = [r for r in rows if all(r[k] is not None for k in ["open","high","low","close"])]

        cur  = rows[-1]["close"] if rows else None
        prev = rows[-2]["close"] if len(rows) > 1 else cur
        chg  = round((cur - prev) / prev * 100, 2) if cur and prev else 0

        info = {}
        try:
            info = tk.fast_info
        except: pass

        return jsonify({
            "ticker":        ticker.upper(),
            "current_price": cur,
            "change_pct":    chg,
            "currency":      "USD",
            "timeframe":     tf,
            "rows":          len(rows),
            "data":          rows,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/news/<ticker>")
def news(ticker):
    try:
        from datetime import date as date_cls
        tk = yf.Ticker(ticker)

        # Noticias
        news_items = []
        try:
            for item in (tk.news or [])[:8]:
                # yfinance >= 0.2.x uses nested 'content' structure
                content = item.get('content', None)
                if content:
                    pub_date = content.get('pubDate', '')
                    try:
                        from datetime import datetime as dt
                        date_str = dt.strptime(pub_date[:10], '%Y-%m-%d').strftime('%d/%m/%Y') if pub_date else ''
                    except:
                        date_str = pub_date[:10] if pub_date else ''
                    url = (content.get('canonicalUrl') or {}).get('url', '') or \
                          (content.get('clickThroughUrl') or {}).get('url', '')
                    publisher = (content.get('provider') or {}).get('displayName', '')
                    news_items.append({
                        'title':     content.get('title', ''),
                        'publisher': publisher,
                        'link':      url,
                        'date':      date_str,
                    })
                else:
                    # legacy format
                    ts = item.get('providerPublishTime', 0)
                    news_items.append({
                        'title':     item.get('title', ''),
                        'publisher': item.get('publisher', ''),
                        'link':      item.get('link', ''),
                        'date':      datetime.fromtimestamp(ts).strftime('%d/%m/%Y') if ts else '',
                    })
        except: pass

        # Fecha de balance (earnings)
        earnings_date = None
        days_to_earnings = None
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                ed_list = cal.get('Earnings Date', [])
                if ed_list:
                    earnings_date = str(ed_list[0])[:10]
            elif cal is not None and hasattr(cal, 'T'):
                cal = cal.T
                if 'Earnings Date' in cal.columns:
                    earnings_date = str(cal['Earnings Date'].iloc[0])[:10]

            if earnings_date:
                ed_dt = datetime.strptime(earnings_date[:10], '%Y-%m-%d').date()
                days_to_earnings = (ed_dt - date_cls.today()).days
        except: pass

        return jsonify({
            'ticker':           ticker.upper(),
            'news':             news_items,
            'earnings_date':    earnings_date,
            'days_to_earnings': days_to_earnings,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/advice", methods=["POST"])
def advice():
    try:
        d = request.json

        def fmt(v, decimals=2):
            return f"${v:.{decimals}f}" if v is not None else "N/D"

        res_str  = ", ".join(fmt(r) for r in d.get("resistance", [])) or "No detectadas"
        sup_str  = ", ".join(fmt(s) for s in d.get("support", []))    or "No detectadas"
        bb_pos   = d.get("bb_position", "N/D")
        rsi      = d.get("rsi", 0)
        macd_h   = d.get("macd_hist", 0)
        price    = d.get("price", 0)
        ema20    = d.get("ema20", 0)
        ema50    = d.get("ema50", 0)
        ema200   = d.get("ema200", 0)
        bull     = d.get("bull_score", 0)
        total    = d.get("total_signals", 8)

        prompt = f"""Sos un analista técnico experto de mercados financieros. Analizá los indicadores técnicos del activo {d.get('ticker')} y emití una recomendación clara y honesta. Si las señales son mixtas o el momento no es favorable, recomendá no operar.

INDICADORES TÉCNICOS — {d.get('ticker')} ({d.get('timeframe')}):

Precio: {fmt(price)} | Variación 24h: {d.get('chg_24h', 0):+.2f}%

EMAs:
  EMA 20:  {fmt(ema20)}  → precio {'SOBRE' if price > ema20 else 'BAJO'} EMA20
  EMA 50:  {fmt(ema50)}  → precio {'SOBRE' if price > ema50 else 'BAJO'} EMA50
  EMA 200: {fmt(ema200)}  → precio {'SOBRE' if price > ema200 else 'BAJO'} EMA200
  Alineación: EMA20 {'>' if ema20 > ema50 else '<'} EMA50 {'>' if ema50 > ema200 else '<'} EMA200

RSI(14): {rsi:.1f} → {'SOBRECOMPRA' if rsi > 70 else 'SOBREVENTA' if rsi < 30 else 'ZONA NEUTRAL'}

MACD(12,26,9):
  Línea: {fmt(d.get('macd_line'), 3)} | Señal: {fmt(d.get('macd_signal'), 3)}
  Histograma: {d.get('macd_hist', 0):+.3f} ({'ALCISTA' if macd_h > 0 else 'BAJISTA'})
  MACD {'sobre' if d.get('macd_line',0) > d.get('macd_signal',0) else 'bajo'} señal

Bollinger Bands(20,2):
  Superior: {fmt(d.get('bb_upper'))} | Media: {fmt(d.get('bb_mid'))} | Inferior: {fmt(d.get('bb_lower'))}
  Posición del precio: {bb_pos}

ATR(14): {fmt(d.get('atr'))} | Volumen vs media 20p: x{d.get('vol_ratio', 1):.1f}

Niveles clave:
  Resistencias: {res_str}
  Soportes: {sup_str}

Score técnico: {bull}/{total} señales alcistas ({bull/total*100:.0f}%)

Respondé en español con EXACTAMENTE este formato:

---
SEÑAL: [COMPRA LONG | VENTA SHORT | NEUTRO — NO OPERAR]

ANÁLISIS:
[2-3 párrafos concisos. Destacá los factores determinantes: qué confluye, qué contradice, y por qué el momento es o no es favorable. Sé directo.]

PLAN: [Solo si hay señal clara. Si no, escribí "Sin operación recomendada en este momento."]
• Entrada: $X
• Stop Loss: $X (−X%)
• Take Profit: $X (+X%)
• Ratio R/R: 1:X

RIESGO: [BAJO | MEDIO | ALTO]
CONFIANZA: [X/10]

⚠️ Análisis técnico automatizado. No es asesoramiento financiero.
---"""

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"advice": message.content[0].text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "server": "CEDEAR Terminal", "time": datetime.now().isoformat()})

@app.route("/manifest.json")
def manifest():
    import os
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "application/manifest+json; charset=utf-8"}
    return "manifest.json not found", 404

@app.route("/")
def index():
    import os
    html_path = os.path.join(os.path.dirname(__file__), "cedear_terminal.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    return "Finance Terminal — HTML file not found", 404

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  CEDEAR TERMINAL — Servidor Local")
    print("="*50)
    print(f"  ✅ Servidor corriendo en http://localhost:5000")
    print(f"  📂 Abrí cedear_terminal.html en tu navegador")
    print(f"  🛑 Para detener: Ctrl+C")
    print("="*50 + "\n")
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
