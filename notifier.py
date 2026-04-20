"""
Notificador por Telegram. Activo solo si las env vars están seteadas.
  TELEGRAM_BOT_TOKEN   — token del bot (obtenido de @BotFather)
  TELEGRAM_CHAT_ID     — chat_id al que enviar los mensajes

Setup rápido:
  1. Abrir Telegram, buscar @BotFather, enviar /newbot
  2. Seguir los pasos → te devuelve el TOKEN
  3. Buscar tu propio bot (el que creaste), mandale /start
  4. Obtener tu chat_id desde:
       https://api.telegram.org/bot<TOKEN>/getUpdates
     (buscar "chat":{"id":NNNNNN,...})
  5. Exportar:
       export TELEGRAM_BOT_TOKEN="..."
       export TELEGRAM_CHAT_ID="..."
"""

import os
import logging

import requests

log = logging.getLogger('bot')

TOKEN   = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
ENABLED = bool(TOKEN and CHAT_ID)


def notify(text):
    """
    Envía un mensaje al chat de Telegram configurado.
    Si las env vars no están seteadas, no hace nada (silencioso).
    Nunca levanta excepción — fallos de red no deben romper el trading.
    """
    if not ENABLED:
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            'chat_id': CHAT_ID,
            'text': text,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True,
        }, timeout=10)
        if r.status_code != 200:
            log.warning(f"Telegram respondió {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        log.warning(f"Telegram notify falló: {e}")
        return False


def startup_msg():
    if ENABLED:
        log.info("Telegram: notificaciones activas")
    else:
        log.info("Telegram: desactivado (setea TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID)")
