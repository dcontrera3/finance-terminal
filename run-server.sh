#!/bin/bash
# Wrapper que carga .env y levanta el servidor Flask (frontend + API).
# Uso: ./run-server.sh
#   Luego abrí http://localhost:5000 en el browser.

set -e

cd "$(dirname "$0")"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

exec python3.10 server.py
