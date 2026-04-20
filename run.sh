#!/bin/bash
# Wrapper que carga .env y ejecuta el bot.
# Uso: ./run.sh <flag>
#   ./run.sh --daemon
#   ./run.sh --status
#   ./run.sh --history
#   ./run.sh --notify-status
#   ./run.sh --close ALL
#   ./run.sh --dry-run

set -e

# Movernos al directorio del proyecto sin importar desde dónde se llamó
cd "$(dirname "$0")"

# Cargar .env si existe (export automático de las variables)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

exec python3.10 bot.py "$@"
