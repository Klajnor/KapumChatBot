#!/usr/bin/env bash
#
# Исправляет владельца файлов бота (brain:brain), перезапускает сервис
# и показывает лог.
#
# Использование (на сервере):
#   sudo ./restart.sh

set -euo pipefail

APP_DIR="/opt/KapumChatBot"
SERVICE="chat_bot.service"
OWNER="brain:brain"

if [ "$(id -u)" -ne 0 ]; then
    echo "Ошибка: запустите от root (sudo ./restart.sh)" >&2
    exit 1
fi

echo "Владелец $OWNER для $APP_DIR..."
chown -R "$OWNER" "$APP_DIR"

echo "Перезапуск $SERVICE..."
systemctl restart "$SERVICE"
systemctl --no-pager --lines=0 status "$SERVICE" || true

echo "Лог (Ctrl+C для выхода):"
exec journalctl -u "$SERVICE" -f
