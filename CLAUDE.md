# Financial Terminal — CEDEAR Terminal

## Qué es este proyecto

Terminal financiera para traders de CEDEARs (ADRs en pesos argentinos). Permite hacer análisis técnico completo, gestionar una cartera, simular operaciones con paper trading, y recibir recomendaciones generadas por IA.

---

## Arquitectura

**Frontend:** HTML5 single-file (`cedear_terminal.html` - 2391 líneas). CSS + JavaScript inline. Sin framework, sin build step. Desplegable directamente en el browser.

**Backend:** Python + Flask (`server.py` - 352 líneas). Bridge entre el frontend y Yahoo Finance + Claude API.

**Despliegue:** Gunicorn en producción. Configurado para Render.com (`render.yaml`). Puerto 5000 por defecto o via `PORT` env var.

**Persistencia:** localStorage (offline-first). Supabase opcional para sync entre dispositivos.

---

## Stack

- **Frontend:** Vanilla JS, Chart.js 3.9.1 (con plugins: financial, zoom, date adapter)
- **Backend:** Python 3, Flask 3.1.3, Gunicorn 23.0.0
- **Data:** yfinance 1.2.0 (Yahoo Finance)
- **IA:** Anthropic SDK 0.84.0, modelo `claude-sonnet-4-6`
- **Sync opcional:** Supabase
- **Fonts:** Space Mono, Bebas Neue, DM Sans

---

## Variables de entorno

```
ANTHROPIC_API_KEY   (requerida para /advice)
SUPABASE_URL        (opcional, para sync)
SUPABASE_KEY        (opcional, para sync)
PORT                (opcional, default 5000)
```

---

## Estructura de archivos

```
cedear_terminal.html   Frontend completo (HTML + CSS + JS)
server.py              API Flask + bridge yfinance
requirements.txt       Dependencias Python
manifest.json          Config PWA
render.yaml            Config despliegue Render.com
```

---

## API Routes (server.py)

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/quote/<ticker>?tf=1W` | OHLCV histórico + precio real-time |
| GET | `/price/<ticker>` | Precio actual via fast_info |
| GET | `/news/<ticker>` | Noticias + fecha de earnings |
| POST | `/advice` | Recomendación de Claude con todos los indicadores |
| GET/POST | `/sync` | Sync con Supabase |
| GET | `/health` | Status del servidor |
| GET | `/` | Sirve el HTML |

**Timeframes y mapping a yfinance:**
- 1D → intervalo 1d, período 6mo
- 1W → intervalo 1wk, período 2y
- 1M → intervalo 1mo, período 5y
- 3M → intervalo 3mo, período 10y

---

## Módulos del frontend (cedear_terminal.html)

El archivo está organizado con comentarios de sección. Orden aproximado:

1. **CSS** (líneas 19-449) — Variables de diseño, componentes, responsive
2. **HTML** (líneas 450-683) — Estructura, modales, nav
3. **JavaScript** (líneas 685-2349):
   - CONFIG — endpoint API, cartera default, colores
   - DATA PERSISTENCE — localStorage, sync Supabase
   - MATH — EMA, RSI, MACD, Bollinger, ATR, pivots
   - FETCH LAYER — llamadas a la API
   - SIDEBAR — cartera con precios en tiempo real
   - RIGHT PANEL — indicadores técnicos
   - CHARTS — inicialización y config Chart.js
   - LOAD CHART — lógica principal de carga de datos
   - WATCHLIST — acciones sin posición
   - NEWS & EARNINGS — modal de noticias
   - PORTFOLIO MGMT — agregar/editar posiciones
   - TECHNICAL ANALYSIS ENGINE — señales y sizing
   - SIMULATOR — paper trading
   - EVENT LISTENERS

---

## Funciones clave (JS)

| Función | Propósito |
|---------|-----------|
| `loadChart()` | Fetcha datos, renderiza los 3 gráficos |
| `renderRightPanel()` | Muestra indicadores + señal actual |
| `openAdvice()` | Flujo completo de recomendación IA |
| `analyzeIndicators()` | Scoring técnico (bull/bear points) |
| `addSimPosition()` | Abrir trade en el simulador |
| `cashSimPosition()` | Cerrar trade al precio actual |
| `renderSimulator()` | Tabla de posiciones abiertas/cerradas |
| `loadAllPrices()` | Actualización de precios cada 15 segundos |

---

## Funcionalidades principales

### 1. Análisis Técnico
Indicadores calculados en el browser con datos de yfinance:
- EMA 20, 50, 200
- RSI 14 períodos
- MACD (12, 26, 9)
- Bollinger Bands (20 períodos, 2 sigma)
- ATR 14 períodos
- Soportes y resistencias (detección de pivots)
- Ratio de volumen

3 gráficos sincronizados: candlestick (con EMAs), RSI, MACD.

### 2. Cartera (Portfolio)
- Seguimiento de posiciones en USD y cantidad de acciones
- Precio real-time via fast_info de yfinance
- Porcentaje de allocación
- Capital invertido vs. liquid cash

CEDEARs default: NVDA, SPY, QQQ, GLD, AMD, MELI, SLV.

### 3. Watchlist
- Seguimiento de acciones sin necesidad de tener posición
- Actualización de precios en tiempo real
- Acceso rápido a noticias y análisis IA
- Integración con simulador con un click

### 4. Simulador de Trading (Paper Trading)
- LONG y SHORT
- Stop Loss y Take Profit manuales
- Tracking de posiciones abiertas y cerradas
- P&L en USD y porcentaje
- Balance acumulado de posiciones cerradas
- Detección automática de hits de SL/TP

### 5. Recomendaciones IA
- POST a `/advice` con todos los indicadores + contexto de cartera
- Claude devuelve: señal (LONG/SHORT/NEUTRO), confianza, narrativa, entry, SL, TP, R/R, sizing recomendado
- Si hay earnings en 3 días o menos, fuerza señal NEUTRO
- Un click para importar la recomendación al simulador

---

## Estructuras de datos clave

```javascript
// Posición en cartera
{ ticker, name, usd, qty, color }

// Cache de precios
{ [ticker]: { price, chg } }  // actualiza cada 15s

// Posición en simulador
{ id, ticker, dir, qty, entry, sl, tp, date, usd }

// Resultado de IA
{ signal, confidence, bullPts, bearPts, narrative, entry, sl, tp, rr, risk, recommendedUSD, recommendedShares, earningsWarning, daysToEarnings }
```

---

## Decisiones de arquitectura importantes

**Precio real-time vs histórico:** `history()` de yfinance devuelve el close cacheado, no el precio live. Se usa `fast_info` para el precio actual y se sobreescribe el último cierre. El priceCache en localStorage persiste entre sesiones y se actualiza cada 15 segundos.

**Single-file frontend:** Sin build step ni framework. Permite despliegue trivial y edición directa. Trade-off: archivo largo pero con secciones bien marcadas.

**USD vs Quantity:** Las posiciones guardan tanto `usd` (capital invertido, fijo) como `qty` (calculado). El USD es el anclaje real; la cantidad flota con el precio.

**Offline-first:** localStorage como fuente primaria. Supabase solo para sync entre dispositivos si está configurado.

**Scoring de indicadores:** Sistema de puntos bull/bear (bullPts, bearPts). Umbral 62%+ = LONG, 38%- = SHORT, resto NEUTRO. Confianza escala de 4 a 10 basado en el diferencial.

**Position sizing:** `riskFactor = (confidence - 3) / 7 * 0.40`. Máximo 50% del capital líquido.

---

## Diseño visual

Tema cyberpunk oscuro (dark blue/cyan). Variables CSS en `:root`. Fuentes: Space Mono para monospace, Bebas Neue para display, DM Sans para cuerpo.

Layout: Sidebar izquierdo (cartera) | Área central (gráficos) | Panel derecho (indicadores). Mobile: nav colapsable < 650px.

---

## Cómo correr en local

```bash
pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-... python server.py
# Abre http://localhost:5000
```

El servidor auto-instala dependencias si faltan (lógica en el bloque `__main__`).
