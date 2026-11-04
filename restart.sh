#!/usr/bin/env bash
#
# Возвращает hardening к состоянию из README/юнита: код и venv — root:root
# (сервис только читает), конфиги — root:brain 640, изменяемые данные —
# brain:brain. Перезапускает сервис и показывает лог.
#
# Использование (на сервере):
#   sudo ./restart.sh               # перезапуск + слежение за журналом
#   sudo ./restart.sh --no-follow   # то же без journalctl -f (для скриптов)
#
# Не запускайте через `source`/`.`: скрипт меняет set -e и не должен
# переопределять trap/exit интерактивной оболочки (Ctrl+C закрывал SSH-сессию).

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    echo "Ошибка: запускайте ./restart.sh как отдельный процесс, не через source/." >&2
    return 1
fi

set -euo pipefail

NO_FOLLOW=0
case "${1:-}" in
    --no-follow) NO_FOLLOW=1 ;;
    "") ;;
    *)
        echo "Ошибка: неизвестный аргумент: $1" >&2
        echo "Использование: sudo ./restart.sh [--no-follow]" >&2
        exit 1
        ;;
esac

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

echo "Данные: $OWNER, каталоги 700, файлы 600 (без перехода по ссылкам)..."
# Каталог данных доступен сервису на запись: проверки -f/-L перед chown/chmod
# не защищают от подмены между проверкой и операцией. Используем дескрипторы.
/usr/bin/python3 "$APP_DIR/scripts/secure_data_permissions.py" \
    --user brain "$STATE_DIR" "${DATA_GLOBS[@]}"

echo "Перезапуск $SERVICE..."
systemctl daemon-reload
systemctl restart "$SERVICE"
systemctl --no-pager --lines=0 status "$SERVICE" || true

if [ "$NO_FOLLOW" -eq 1 ]; then
    echo "Перезапуск выполнен. Журнал: journalctl -u $SERVICE -f"
    exit 0
fi

echo "Лог (Ctrl+C останавливает только журнал и возвращает в консоль):"
journalctl -u "$SERVICE" -f || true
