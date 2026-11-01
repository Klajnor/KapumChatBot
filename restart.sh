#!/usr/bin/env bash
#
# Возвращает hardening к состоянию из README/юнита: код и venv — root:root
# (сервис только читает), конфиги — root:brain 640, изменяемые данные —
# brain:brain. Перезапускает сервис и показывает лог.
#
# Использование (на сервере):
#   sudo ./restart.sh
#
# Не запускайте через `source`/`.`: скрипт меняет set -e и не должен
# переопределять trap/exit интерактивной оболочки (Ctrl+C закрывал SSH-сессию).

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    echo "Ошибка: запускайте ./restart.sh как отдельный процесс, не через source/." >&2
    return 1
fi

set -euo pipefail

APP_DIR="/opt/KapumChatBot"
SERVICE="chat_bot.service"
OWNER="brain:brain"
CONFIG_OWNER="root:brain"
STATE_DIR="/var/lib/kapum-chatbot"   # StateDirectory сервиса: БД и токен Telegraph

# Конфиги читает сервис, но менять их вправе только root.
CONFIG_FILES=(
    "$APP_DIR/.env.linux"
    "$APP_DIR/.env"
    "$APP_DIR/.mcp.json"
)

# Изменяемые данные: БД и токен Telegraph (отсутствующие файлы пропускаются).
DATA_GLOBS=(
    "$APP_DIR"/*.db "$APP_DIR"/*.db-wal "$APP_DIR"/*.db-shm
    "$APP_DIR/telegraph_account.json"
    "$STATE_DIR"/*.db "$STATE_DIR"/*.db-wal "$STATE_DIR"/*.db-shm
    "$STATE_DIR/telegraph_account.json"
)

if [ "$(id -u)" -ne 0 ]; then
    echo "Ошибка: запустите от root (sudo ./restart.sh)" >&2
    exit 1
fi

echo "Код $APP_DIR: root:root, без права записи для группы и остальных..."
chown -R root:root "$APP_DIR"
chmod -R go-w "$APP_DIR"
# Старые __pycache__ могли остаться от предыдущих версий и принадлежат root.
# Сервис работает с PYTHONDONTWRITEBYTECODE=1 и не должен пытаться их удалять.
find "$APP_DIR" -type d -name '__pycache__' -prune -exec rm -rf -- {} +

echo "Конфиги: $CONFIG_OWNER 640..."
for f in "${CONFIG_FILES[@]}"; do
    if [ -f "$f" ]; then
        chown "$CONFIG_OWNER" -- "$f"
        chmod 640 -- "$f"
        echo "  640 $f"
    fi
done

if [ -d "$STATE_DIR" ]; then
    echo "Данные $STATE_DIR: $OWNER 700..."
    chown -R "$OWNER" "$STATE_DIR"
    chmod 700 "$STATE_DIR"
fi

echo "Данные в $APP_DIR: $OWNER 600..."
for f in "${DATA_GLOBS[@]}"; do
    if [ -f "$f" ]; then
        chown "$OWNER" -- "$f"
        chmod 600 -- "$f"
        echo "  600 $f"
    fi
done

echo "Перезапуск $SERVICE..."
systemctl daemon-reload
systemctl restart "$SERVICE"
systemctl --no-pager --lines=0 status "$SERVICE" || true

echo "Лог (Ctrl+C останавливает только журнал и возвращает в консоль):"
journalctl -u "$SERVICE" -f || true
