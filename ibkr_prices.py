"""
Fetch de precios IBKR para la UI.

Mantiene una conexión persistente a IB Gateway/TWS en un thread background con
su propio event loop asyncio. Los precios se obtienen via `reqHistoricalData`
con bars diarias (3 días), que NO requiere suscripción de market data live y
no entra en conflicto con las sesiones del bot.

De cada ticker se extrae:
  - current_price = último daily bar close (actualiza durante el día)
  - prev_close    = daily bar anterior al actual
  - change_pct    = (current - prev_close) / prev_close * 100

Cache corto (15s) para amortiguar ráfagas del frontend. Si IB Gateway no está
disponible, `get_prices()` devuelve {} y el caller cae al fallback de yfinance.
"""

import asyncio
import math
import os
import threading
import time
from typing import Dict, List, Optional

try:
    from ib_insync import IB, Stock, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

IB_HOST = os.environ.get('IB_HOST', '127.0.0.1')
IB_PORT = int(os.environ.get('IB_PORT', '4002'))
# ClientId distinto al del bot (que usa 1) para no chocar.
IB_CLIENT_ID = int(os.environ.get('IB_CLIENT_ID_PRICES', '7'))

RECONNECT_DELAY_S = 15
CACHE_TTL_S = 15
FETCH_TIMEOUT_S = 12


def _valid(x) -> Optional[float]:
    try:
        v = float(x)
        return v if not math.isnan(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


class IBKRPriceStream:
    def __init__(self):
        self.ib: Optional['IB'] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.connected = False
        self._qualified: Dict[str, object] = {}   # ticker -> Contract
        self._cache: Dict[str, dict] = {}         # ticker -> {price, change_pct, ts}
        self._cache_lock = threading.Lock()
        self._inflight: Dict[str, asyncio.Future] = {}  # ticker -> future (coalesces)
        self._started = False

    def start(self):
        if self._started or not IB_AVAILABLE:
            return
        self._started = True
        self.thread = threading.Thread(target=self._run, daemon=True, name='ibkr-prices')
        self.thread.start()

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        util.patchAsyncio()
        while True:
            try:
                self.loop.run_until_complete(self._connect_and_idle())
            except Exception as e:
                print(f"[ibkr_prices] loop error: {e}", flush=True)
            self.connected = False
            self._qualified.clear()
            time.sleep(RECONNECT_DELAY_S)

    async def _connect_and_idle(self):
        self.ib = IB()
        try:
            await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=10)
            self.connected = True
            print(f"[ibkr_prices] conectado a IBKR (clientId={IB_CLIENT_ID})", flush=True)
        except Exception as e:
            print(f"[ibkr_prices] no se pudo conectar: {e}", flush=True)
            self.connected = False
            return
        while self.ib.isConnected():
            await asyncio.sleep(1)
        print("[ibkr_prices] conexión perdida", flush=True)

    async def _qualify(self, ticker: str):
        if ticker in self._qualified:
            return self._qualified[ticker]
        contract = Stock(ticker, 'SMART', 'USD')
        try:
            await self.ib.qualifyContractsAsync(contract)
            self._qualified[ticker] = contract
            return contract
        except Exception as e:
            print(f"[ibkr_prices] qualify {ticker} falló: {e}", flush=True)
            return None

    async def _fetch_one(self, ticker: str) -> Optional[dict]:
        """Pide 3 daily bars de un ticker. Devuelve {price, change_pct, ts} o None."""
        contract = await self._qualify(ticker)
        if not contract:
            return None
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='3 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
            )
        except Exception as e:
            print(f"[ibkr_prices] reqHistoricalData {ticker} falló: {e}", flush=True)
            return None
        if not bars:
            return None
        last = bars[-1]
        price = _valid(last.close)
        if not price:
            return None
        prev_close = _valid(bars[-2].close) if len(bars) >= 2 else None
        chg = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0
        return {'price': round(price, 4), 'change_pct': chg, 'ts': time.time()}

    async def _fetch_batch(self, tickers: List[str]) -> Dict[str, dict]:
        """Fetch en paralelo con coalescing: si ya hay un fetch para ese ticker, reusa."""
        tasks = {}
        for t in tickers:
            if t in self._inflight:
                tasks[t] = self._inflight[t]
            else:
                fut = asyncio.ensure_future(self._fetch_one(t))
                self._inflight[t] = fut
                tasks[t] = fut
        out = {}
        try:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        finally:
            for t in tickers:
                self._inflight.pop(t, None)
        for t, res in zip(tasks.keys(), results):
            if isinstance(res, dict):
                out[t] = res
        return out

    def get_prices(self, tickers) -> Dict[str, dict]:
        """Retorna {ticker: {price, change_pct}}. Usa cache de 15s."""
        if not self.connected or not self.loop:
            return {}
        tickers = [t.upper() for t in tickers]
        now = time.time()
        out = {}
        missing = []
        with self._cache_lock:
            for t in tickers:
                c = self._cache.get(t)
                if c and now - c['ts'] < CACHE_TTL_S:
                    out[t] = {'price': c['price'], 'change_pct': c['change_pct']}
                else:
                    missing.append(t)
        if not missing:
            return out
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._fetch_batch(missing), self.loop
            )
            fresh = fut.result(timeout=FETCH_TIMEOUT_S + 2)
        except Exception as e:
            print(f"[ibkr_prices] fetch_batch error: {e}", flush=True)
            fresh = {}
        with self._cache_lock:
            for t, data in fresh.items():
                self._cache[t] = data
                out[t] = {'price': data['price'], 'change_pct': data['change_pct']}
        return out

    def status(self) -> dict:
        return {
            'available': IB_AVAILABLE,
            'connected': self.connected,
            'host': IB_HOST,
            'port': IB_PORT,
            'client_id': IB_CLIENT_ID,
            'qualified_contracts': len(self._qualified),
            'cached_prices': len(self._cache),
        }


# Singleton global. Importar y arrancar desde server.py.
stream = IBKRPriceStream()
