#!/usr/bin/env bash
#
# Исправляет владельца файлов бота (brain:brain), ставит права 600 на
# критичные файлы (секреты, БД), перезапускает сервис и показывает лог.
#
# Использование (на сервере):
#   sudo ./restart.sh

set -euo pipefail

APP_DIR="/opt/KapumChatBot"
SERVICE="chat_bot.service"
OWNER="brain:brain"
STATE_DIR="/var/lib/kapum-chatbot"   # StateDirectory сервиса: БД и токен Telegraph

# Критичные файлы: секреты и данные (шаблоны, отсутствующие файлы пропускаются)
SECRET_GLOBS=(
    "$APP_DIR/.env"
    "$APP_DIR/.mcp.json"
    "$APP_DIR"/*.db "$APP_DIR"/*.db-wal "$APP_DIR"/*.db-shm
    "$APP_DIR/telegraph_account.json"
    "$STATE_DIR"/*.db "$STATE_DIR"/*.db-wal "$STATE_DIR"/*.db-shm
    "$STATE_DIR/telegraph_account.json"
)

if [ "$(id -u)" -ne 0 ]; then
    echo "Ошибка: запустите от root (sudo ./restart.sh)" >&2
    exit 1
fi

echo "Владелец $OWNER для $APP_DIR..."
chown -R "$OWNER" "$APP_DIR"
if [ -d "$STATE_DIR" ]; then
    chown -R "$OWNER" "$STATE_DIR"
fi

echo "Права 600 на критичные файлы..."
for f in "${SECRET_GLOBS[@]}"; do
    if [ -f "$f" ]; then
        chmod 600 -- "$f"
        echo "  600 $f"
    fi
done

echo "Перезапуск $SERVICE..."
systemctl restart "$SERVICE"
systemctl --no-pager --lines=0 status "$SERVICE" || true

echo "Лог (Ctrl+C для выхода):"
# Без exec: иначе при запуске через `source`/`. restart.sh` он заменяет
# оболочку терминала, и Ctrl+C закрывает сессию. Ctrl+C гасит только
# journalctl, скрипт завершается штатно с кодом 0.
trap 'echo; echo "Выход из лога."; exit 0' INT TERM
journalctl -u "$SERVICE" -f || true
