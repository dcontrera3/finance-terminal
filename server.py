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
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except ImportError:
    install("flask")
    install("flask-cors")
    install("flask-limiter")
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

try:
    import anthropic
except ImportError:
    install("anthropic")
    import anthropic


import math
import os
import requests as http
from datetime import datetime, timedelta
from functools import wraps

# IBKR price stream deshabilitado por default: una segunda conexión a IBKR desde
# el server roba market data lines al bot y genera error 10197 "sesiones
# competidoras". Para reactivar (cuando tengamos arquitectura unificada):
#   export IBKR_PRICES_ENABLED=true
ibkr_stream = None
if os.environ.get('IBKR_PRICES_ENABLED', 'false').lower() == 'true':
    try:
        from ibkr_prices import stream as ibkr_stream
        ibkr_stream.start()
    except Exception as _e:
        print(f"[server] ibkr_prices no disponible: {_e}", flush=True)
        ibkr_stream = None

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300 per minute"],
    storage_uri="memory://",
    on_breach=lambda limit: (jsonify({"error": "Demasiadas requests. Esperá un momento."}), 429),
)

# ── Supabase config ──
SUPABASE_URL        = os.environ.get('SUPABASE_URL', '').rstrip('/')
SUPABASE_KEY        = os.environ.get('SUPABASE_KEY', '')
SUPABASE_ANON_KEY   = os.environ.get('SUPABASE_ANON_KEY', SUPABASE_KEY)
SUPABASE_JWT_SECRET = os.environ.get('SUPABASE_JWT_SECRET', '')

def sb_headers():
    return {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
    }

# ── Data cache — evita requests redundantes a Yahoo Finance ──
_data_cache = {}  # key -> (payload, expires_at)

def _cache_get(key):
    entry = _data_cache.get(key)
    if not entry:
        return None
    payload, expires_at = entry
    if datetime.utcnow() < expires_at:
        return payload
    del _data_cache[key]
    return None

def _cache_set(key, payload, ttl_seconds):
    # Evict entradas antiguas si el cache crece demasiado
    if len(_data_cache) > 500:
        now = datetime.utcnow()
        expired = [k for k, (_, exp) in _data_cache.items() if now >= exp]
        for k in expired:
            del _data_cache[k]
    _data_cache[key] = (payload, datetime.utcnow() + timedelta(seconds=ttl_seconds))

# ── Auth — validación via Supabase API con caché de 5 min ──
_auth_cache = {}  # token_prefix -> (user_id, expires_at)

def _validate_token(token):
    """Valida el token contra la API de Supabase. Cachea el resultado 5 min."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return 'anon'
    cache_key = token[:32]  # primeros 32 chars como clave
    cached = _auth_cache.get(cache_key)
    if cached:
        user_id, expires_at = cached
        if datetime.utcnow() < expires_at:
            return user_id
    try:
        r = http.get(
            f'{SUPABASE_URL}/auth/v1/user',
            headers={'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {token}'},
            timeout=5
        )
        if r.status_code == 200:
            user_id = r.json().get('id', 'unknown')
            _auth_cache[cache_key] = (user_id, datetime.utcnow() + timedelta(minutes=5))
            return user_id
    except Exception:
        pass
    return None

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not SUPABASE_URL or not SUPABASE_KEY:
            request.user_id = 'anon'
            return f(*args, **kwargs)
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No autorizado'}), 401
        user_id = _validate_token(auth_header[7:])
        if not user_id:
            return jsonify({'error': 'No autorizado'}), 401
        request.user_id = user_id
        return f(*args, **kwargs)
    return decorated

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

def _get_sparkline(ticker):
    """Últimos 7 cierres diarios. Cache 1 hora."""
    cache_key = f"spark:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        hist = yf.Ticker(ticker).history(period='10d', interval='1d', auto_adjust=True)
        closes = [clean(float(c)) for c in hist['Close'].tolist()
                  if c == c and c is not None][-7:]
        _cache_set(cache_key, closes, ttl_seconds=3600)
        return closes
    except:
        _cache_set(cache_key, [], ttl_seconds=300)
        return []

@app.route("/config")
def config():
    return jsonify({
        "supabase_url":      SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
    })

@app.route("/quote/<ticker>")
@limiter.limit("30 per minute")
@require_auth
def quote(ticker):
    tf = request.args.get("tf", "1M")
    interval, period = INTERVAL_MAP.get(tf, ("1mo", "5y"))
    # 1D más volátil → 90s. Timeframes largos cambian poco → 5 min.
    ttl = 90 if tf == "1D" else 300
    cache_key = f"quote:{ticker.upper()}:{tf}"
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)

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

        # Precio real-time desde fast_info (evita caché de history)
        cur, chg = None, 0
        try:
            fi         = tk.fast_info
            cur        = clean(fi.get('lastPrice') or fi.get('regularMarketPrice'))
            prev_close = clean(fi.get('previousClose') or fi.get('regularMarketPreviousClose'))
            if cur and prev_close:
                chg = round((cur - prev_close) / prev_close * 100, 2)
        except:
            pass

        # Fallback: último cierre del histórico
        if not cur and rows:
            cur  = rows[-1]["close"]
            prev = rows[-2]["close"] if len(rows) > 1 else cur
            chg  = round((cur - prev) / prev * 100, 2) if cur and prev else 0

        result = {
            "ticker":        ticker.upper(),
            "current_price": cur,
            "change_pct":    chg,
            "currency":      "USD",
            "timeframe":     tf,
            "rows":          len(rows),
            "data":          rows,
        }
        _cache_set(cache_key, result, ttl_seconds=ttl)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/price/<ticker>")
@limiter.limit("120 per minute")
@require_auth
def price(ticker):
    """Precio real-time. Prioriza IBKR si está conectado; fallback a yfinance. Cache 15s."""
    ticker_u = ticker.upper()

    # IBKR real-time primero. El stream devuelve solo si hay dato fresco (<60s).
    if ibkr_stream:
        ib_data = ibkr_stream.get_prices([ticker_u]).get(ticker_u)
        if ib_data:
            result = {
                "ticker": ticker_u,
                "current_price": ib_data['price'],
                "change_pct": ib_data['change_pct'],
                "spark": _get_sparkline(ticker),
                "source": "ibkr",
            }
            return jsonify(result)

    cache_key = f"price:{ticker_u}"
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)
    try:
        fi         = yf.Ticker(ticker).fast_info
        cur        = clean(fi.get('lastPrice') or fi.get('regularMarketPrice'))
        prev_close = clean(fi.get('previousClose') or fi.get('regularMarketPreviousClose'))
        chg        = round((cur - prev_close) / prev_close * 100, 2) if cur and prev_close else 0
        result     = {"ticker": ticker_u, "current_price": cur, "change_pct": chg,
                      "spark": _get_sparkline(ticker), "source": "yfinance"}
        _cache_set(cache_key, result, ttl_seconds=15)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/prices_ibkr")
@limiter.limit("120 per minute")
@require_auth
def prices_ibkr():
    """Batch de precios desde IBKR. Devuelve solo los que tienen dato fresco.
    Query: ?tickers=NVDA,SPY,QQQ"""
    tickers_raw = request.args.get('tickers', '').strip()
    if not tickers_raw:
        return jsonify({"prices": {}, "connected": False})
    tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
    if not ibkr_stream or not ibkr_stream.connected:
        return jsonify({"prices": {}, "connected": False})
    prices = ibkr_stream.get_prices(tickers)
    return jsonify({"prices": prices, "connected": True, "source": "ibkr"})


@app.route("/ibkr/status")
@require_auth
def ibkr_status():
    if not ibkr_stream:
        return jsonify({"available": False, "connected": False})
    return jsonify(ibkr_stream.status())

@app.route("/news/<ticker>")
@limiter.limit("20 per minute")
@require_auth
def news(ticker):
    cache_key = f"news:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)
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

        result = {
            'ticker':           ticker.upper(),
            'news':             news_items,
            'earnings_date':    earnings_date,
            'days_to_earnings': days_to_earnings,
        }
        _cache_set(cache_key, result, ttl_seconds=600)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/advice", methods=["POST"])
@limiter.limit("10 per minute")
@require_auth
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


@app.route("/sync", methods=["GET"])
@require_auth
def sync_get():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return jsonify({}), 200
    try:
        uid    = request.user_id
        prefix = f'{uid}:'
        r      = http.get(
            f'{SUPABASE_URL}/rest/v1/app_data',
            headers=sb_headers(),
            params={'key': f'like.{prefix}%'},
            timeout=5
        )
        rows = r.json()
        return jsonify({
            row['key'][len(prefix):]: row['value']
            for row in rows
            if isinstance(row, dict) and row.get('key', '').startswith(prefix)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/sync", methods=["POST"])
@require_auth
def sync_post():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return jsonify({'ok': False}), 200
    try:
        data       = request.json
        uid        = request.user_id
        actual_key = f'{uid}:{data["key"]}'
        headers    = {**sb_headers(), 'Prefer': 'resolution=merge-duplicates,return=minimal'}
        http.post(
            f'{SUPABASE_URL}/rest/v1/app_data',
            headers=headers,
            json={'key': actual_key, 'value': data['value'], 'updated_at': datetime.utcnow().isoformat()},
            timeout=5
        )
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Bot monitor ─────────────────────────────────────────────
BOT_STATE_PATH = os.path.join(os.path.dirname(__file__), 'bot_state.json')

def _read_bot_state():
    """Lee bot_state.json. Devuelve estructura vacía si no existe o falla."""
    try:
        with open(BOT_STATE_PATH, 'r') as f:
            s = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'positions': {}, 'history': [], 'last_run': None,
            'total_trades': 0, 'peak_equity': 0, 'current_equity': None,
        }
    s.setdefault('positions',           {})
    s.setdefault('history',             [])
    s.setdefault('last_run',            None)
    s.setdefault('total_trades',        0)
    s.setdefault('peak_equity',         0)
    s.setdefault('current_equity',      None)
    s.setdefault('capital_contributed', 1_000_000)
    s.setdefault('cash_flows',          [])
    s.setdefault('equity_history',      [])
    return s


def _write_bot_state(s):
    """Escribe bot_state.json atómicamente para evitar corrupción si el bot
    está corriendo en paralelo."""
    tmp = BOT_STATE_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(s, f, indent=2, default=str)
    os.replace(tmp, BOT_STATE_PATH)


def _compute_metrics(state):
    """Calcula métricas agregadas del history del bot."""
    history = state.get('history', [])
    closes  = [e for e in history if e.get('event') == 'close' and e.get('pnl') is not None]

    wins   = [e for e in closes if e['pnl'] > 0]
    losses = [e for e in closes if e['pnl'] < 0]

    total_pnl  = sum(e['pnl'] for e in closes)
    gross_win  = sum(e['pnl'] for e in wins)
    gross_loss = abs(sum(e['pnl'] for e in losses))
    pf         = (gross_win / gross_loss) if gross_loss > 0 else (float('inf') if gross_win > 0 else 0)

    def agg(events):
        cl = [e for e in events if e.get('event') == 'close' and e.get('pnl') is not None]
        w  = [e for e in cl if e['pnl'] > 0]
        l_ = [e for e in cl if e['pnl'] < 0]
        gw = sum(e['pnl'] for e in w)
        gl = abs(sum(e['pnl'] for e in l_))
        return {
            'trades':        len(cl),
            'pnl':           round(sum(e['pnl'] for e in cl), 2),
            'wins':          len(w),
            'losses':        len(l_),
            'win_rate':      round(len(w) / len(cl) * 100, 1) if cl else 0,
            'avg_win':       round(gw / len(w), 2) if w else 0,
            'avg_loss':      round(gl / len(l_), 2) if l_ else 0,
            'profit_factor': round(gw / gl, 2) if gl > 0 else (None if not w else 999),
        }

    by_strategy = {}
    for strat in {e.get('strategy') for e in closes if e.get('strategy')}:
        by_strategy[strat] = agg([e for e in history if e.get('strategy') == strat])

    by_ticker = {}
    for tk in {e.get('ticker') for e in closes if e.get('ticker')}:
        by_ticker[tk] = agg([e for e in history if e.get('ticker') == tk])

    # Equity curve: cumulative pnl ordenado cronológicamente
    sorted_closes = sorted(closes, key=lambda e: e.get('ts', ''))
    cum = 0
    equity_curve = []
    for e in sorted_closes:
        cum += e['pnl']
        equity_curve.append({'ts': e.get('ts'), 'cum_pnl': round(cum, 2)})

    return {
        'overall': {
            'trades':        len(closes),
            'pnl':           round(total_pnl, 2),
            'wins':          len(wins),
            'losses':        len(losses),
            'win_rate':      round(len(wins) / len(closes) * 100, 1) if closes else 0,
            'avg_win':       round(gross_win / len(wins), 2) if wins else 0,
            'avg_loss':      round(gross_loss / len(losses), 2) if losses else 0,
            'profit_factor': round(pf, 2) if pf != float('inf') else None,
        },
        'by_strategy':  by_strategy,
        'by_ticker':    by_ticker,
        'equity_curve': equity_curve,
    }


@app.route("/bot/state")
@require_auth
def bot_state():
    """Estado crudo del bot: posiciones abiertas, history, equity, último run."""
    s = _read_bot_state()
    # Drawdown actual desde peak
    peak = s.get('peak_equity') or 0
    cur  = s.get('current_equity')
    dd   = round((cur - peak) / peak * 100, 2) if (peak and cur) else None
    return jsonify({
        'positions':           s.get('positions', {}),
        'history':             s.get('history', [])[-200:],   # últimos 200 eventos
        'last_run':            s.get('last_run'),
        'total_trades':        s.get('total_trades', 0),
        'peak_equity':         peak,
        'current_equity':      cur,
        'drawdown_pct':        dd,
        'positions_count':     len(s.get('positions', {})),
        'capital_contributed': s.get('capital_contributed', 1_000_000),
        'cash_flows':          s.get('cash_flows', []),
    })


@app.route("/bot/cashflow", methods=['POST'])
@limiter.limit("30 per minute")
@require_auth
def bot_cashflow():
    """Registra un aporte (deposit) o retiro (withdrawal) de capital.

    Body JSON:
      type:   "deposit" | "withdrawal"
      amount: número positivo en USD
      ts:     ISO date opcional (default: ahora)
      note:   string opcional
    """
    data = request.get_json(silent=True) or {}
    cf_type = data.get('type')
    if cf_type not in ('deposit', 'withdrawal'):
        return jsonify({'error': 'type must be deposit or withdrawal'}), 400
    try:
        amount = float(data.get('amount', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'amount must be a number'}), 400
    if amount <= 0:
        return jsonify({'error': 'amount must be > 0'}), 400

    ts   = (data.get('ts') or datetime.utcnow().isoformat(timespec='seconds'))
    note = (data.get('note') or '').strip()

    s = _read_bot_state()
    delta = amount if cf_type == 'deposit' else -amount
    s['capital_contributed'] = round(float(s.get('capital_contributed', 0)) + delta, 2)
    s['cash_flows'] = list(s.get('cash_flows', [])) + [{
        'ts':     ts,
        'type':   cf_type,
        'amount': round(amount, 2),
        'note':   note,
    }]
    _write_bot_state(s)

    return jsonify({
        'ok':                  True,
        'capital_contributed': s['capital_contributed'],
        'cash_flows':          s['cash_flows'],
    })


@app.route("/bot/metrics")
@require_auth
def bot_metrics():
    """Agregados: P&L total, win rate, por estrategia, por ticker, equity curve."""
    s = _read_bot_state()
    return jsonify(_compute_metrics(s))


# Canasta de tickers que opera el bot. Mantener en sync con TICKERS en bot.py.
BOT_TICKERS = [
    'NVDA', 'AMD', 'AAPL', 'NFLX', 'META', 'MSFT', 'AMZN',
    'GOOGL', 'TSLA', 'AVGO',
    'JPM', 'V', 'UNH', 'KO',
    'NU', 'MELI', 'BABA',
    'SPY', 'QQQ', 'IWM',
    'GLD', 'SLV', 'XLE',
    'VIST',
]

@app.route("/bot/tickers")
@require_auth
def bot_tickers():
    """Canasta de tickers que opera el bot (24). Usada para popular el seguimiento."""
    return jsonify({'tickers': BOT_TICKERS})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "server": "CEDEAR Terminal", "time": datetime.now().isoformat()})

@app.route("/sw.js")
def service_worker():
    import os
    sw_path = os.path.join(os.path.dirname(__file__), "sw.js")
    if os.path.exists(sw_path):
        with open(sw_path, "r", encoding="utf-8") as f:
            return f.read(), 200, {
                "Content-Type": "application/javascript; charset=utf-8",
                "Service-Worker-Allowed": "/",
            }
    return "sw.js not found", 404

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
