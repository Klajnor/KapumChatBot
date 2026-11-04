#!/usr/bin/env bash
#
# Автообновление бота на сервере: сверяет HEAD /opt/KapumChatBot с
# origin/master и при расхождении обновляет код, зависимости (только если
# менялся requirements.txt) и перезапускает сервис через restart.sh.
# Запускается от root таймером kapum-update.timer (или вручную); бот и
# сервисный пользователь brain код по-прежнему не трогают.
#
# Использование (на сервере):
#   sudo ./scripts/update.sh
#
# Поведение:
#   - изменений нет          — тихий выход (код 0);
#   - сбой pre-flight/fetch  — отказ без изменений; уведомление владельцу
#     или грязные tracked-     одно, до восстановления (state-файл в /run);
#     файлы
#   - обновление             — reset --hard origin/master (до стопа сервиса),
#                              pip install при изменившемся requirements.txt,
#                              права+рестарт (restart.sh --no-follow),
#                              health-check;
#   - провал health-check    — откат на прежний коммит со старыми
#                              зависимостями, повторный рестарт и проверка.
#
# Не запускайте через `source`/`.`: скрипт ставит set -e и берёт flock.

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    echo "Ошибка: запускайте scripts/update.sh как отдельный процесс, не через source/." >&2
    return 1
fi

set -euo pipefail

APP_DIR="/opt/KapumChatBot"
SERVICE="chat_bot.service"
BRANCH="master"
REMOTE="origin"
# Read-only deploy-ключ без парольной фразы (README, «Автообновление»).
# Не путать с ключом load_git_key.sh — тот остаётся для ручных сессий.
DEPLOY_KEY="/root/id_ed25519_kapum_deploy"
VENV_PIP="$APP_DIR/.venv/bin/pip"
STATE_FILE="/run/kapum-update.state"   # ok | block | failed:<sha>
LOCK_FILE="/run/kapum-update.lock"
HEALTH_DEADLINE=60       # секунд на старт сервиса после рестарта
HEALTH_FALLBACK_WAIT=20  # секунд стабильности юнита без HEALTH_PORT

# ---------------- уведомления и состояние ----------------

_env_value() {
    # Значение переменной из .env.linux/.env (первой найденной), без кавычек
    # и пробелов. Код 1 — переменной нет.
    local key="$1" f raw
    for f in "$APP_DIR/.env.linux" "$APP_DIR/.env"; do
        [ -r "$f" ] || continue
        raw=$(grep -m1 -E "^[[:space:]]*${key}=" "$f" || true)
        [ -n "$raw" ] || continue
        printf '%s' "${raw#*=}" | tr -d "\"'" | tr -d '[:space:]'
        return 0
    done
    return 1
}

_notify() {
    # Сообщение владельцу через Bot API; сбой Telegram не ломает обновление.
    local token ids id
    if ! token=$(_env_value BOT_TOKEN); then
        echo "Предупреждение: BOT_TOKEN не найден — уведомление не отправлено." >&2
        return 0
    fi
    ids=$(_env_value OWNER_IDS || true)
    [ -n "$ids" ] || ids=$(_env_value ADMIN_IDS || true)
    if [ -z "$ids" ]; then
        echo "Предупреждение: OWNER_IDS пуст — уведомление не отправлено." >&2
        return 0
    fi
    for id in ${ids//,/ }; do
        curl -fsS --max-time 10 \
            "https://api.telegram.org/bot${token}/sendMessage" \
            --data-urlencode "chat_id=${id}" \
            --data-urlencode "text=$1" >/dev/null 2>&1 \
            || echo "Предупреждение: Telegram не принял сообщение для ${id}." >&2
    done
    return 0
}

_state_read() {
    if [ -r "$STATE_FILE" ]; then
        cat "$STATE_FILE" 2>/dev/null || true
    fi
}

_state_write() {
    printf '%s\n' "$1" >"$STATE_FILE" 2>/dev/null \
        || echo "Предупреждение: не удалось записать $STATE_FILE." >&2
    return 0
}

_blocked() {
    # $1 — причина по-русски (в журнал), $2 — по-английски (в Telegram).
    # Одно уведомление на серию блокировок, а не каждые 10 минут.
    echo "Ошибка: $1" >&2
    if [ "$(_state_read)" != "block" ]; then
        _notify "Kapum bot: auto-update is blocked — $2 No changes were applied."
        _state_write block
    else
        echo "Блокировка повторяется — уведомление уже отправлялось." >&2
    fi
    exit 1
}

# ---------------- health-check ----------------

_systemctl_active() {
    [ "$(systemctl is-active "$SERVICE" 2>/dev/null || true)" = "active" ]
}

_n_restarts() {
    systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null || echo 0
}

_wait_healthy() {
    # Готовность после рестарта: GET /health при включённом HEALTH_PORT
    # (эндпоинт поднимается в конце инициализации бота), иначе — активность
    # юнита без роста NRestarts (ловит crash-loop с Restart=always).
    local host port deadline base
    host=$(_env_value HEALTH_HOST || true)
    host=${host:-127.0.0.1}
    port=$(_env_value HEALTH_PORT || true)
    deadline=$((SECONDS + HEALTH_DEADLINE))
    if [ -n "$port" ] && [ "$port" != "0" ]; then
        while [ "$SECONDS" -lt "$deadline" ]; do
            if curl -fsS --max-time 3 "http://${host}:${port}/health" >/dev/null 2>&1; then
                sleep 5
                _systemctl_active && return 0
                return 1
            fi
            sleep 2
        done
        return 1
    fi
    base=$(_n_restarts)
    while [ "$SECONDS" -lt "$deadline" ]; do
        _systemctl_active || return 1
        if [ "$SECONDS" -ge "$HEALTH_FALLBACK_WAIT" ]; then
            [ "$(_n_restarts)" = "$base" ] && return 0
            return 1
        fi
        sleep 2
    done
    return 1
}

# ---------------- проверка предусловий ----------------

if [ "$(id -u)" -ne 0 ]; then
    echo "Ошибка: запустите от root (sudo ./scripts/update.sh)." >&2
    exit 1
fi

# Один запуск одновременно: таймер не должен накладываться на долгий pip.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Предыдущий запуск ещё работает — выходим." >&2
    exit 0
fi

[ -d "$APP_DIR" ] \
    || _blocked "каталог $APP_DIR не найден." \
        "app directory $APP_DIR is missing."
[ -r "$DEPLOY_KEY" ] \
    || _blocked "deploy-ключ не найден: $DEPLOY_KEY (README, «Автообновление»)." \
        "deploy key $DEPLOY_KEY is missing (see README, Auto-update)."
[ -x "$VENV_PIP" ] \
    || _blocked "не найден pip виртуального окружения: $VENV_PIP." \
        "venv pip not found: $VENV_PIP."

cd "$APP_DIR"

case "$(git remote get-url "$REMOTE" 2>/dev/null || true)" in
    git@github.com:*|ssh://git@github.com/*) ;;
    *)
        # Сам URL не печатаем: в нём может быть токен.
        _blocked "origin не использует SSH — deploy-ключ не применяется." \
            "origin remote is not SSH. Run: git remote set-url origin git@github.com:Klajnor/KapumChatBot.git"
        ;;
esac

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$CURRENT_BRANCH" = "$BRANCH" ] \
    || _blocked "рабочая копия не на ветке $BRANCH (сейчас $CURRENT_BRANCH)." \
        "server checkout is not on branch $BRANCH."

run_git() {
    GIT_SSH_COMMAND="ssh -i $DEPLOY_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
        git "$@"
}

echo "Проверка $REMOTE/$BRANCH..."
if ! run_git fetch "$REMOTE" "$BRANCH"; then
    _blocked "git fetch $REMOTE $BRANCH не удался (сеть, ключ или доступ к репозиторию)." \
        "git fetch failed (network, deploy key or repository access)."
fi

OLD_HEAD=$(git rev-parse HEAD)
NEW_HEAD=$(git rev-parse "$REMOTE/$BRANCH")

if [ "$OLD_HEAD" = "$NEW_HEAD" ]; then
    echo "Изменений нет (${OLD_HEAD:0:7} $(git log -1 --format=%s "$OLD_HEAD"))."
    _state_write ok
    exit 0
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    git status --porcelain --untracked-files=no >&2 || true
    _blocked "tracked-файлы в $APP_DIR локально изменены — обновление отменено." \
        "/opt/KapumChatBot has local changes to tracked files; update refused (git status)."
fi

# Этот коммит уже пробовал обновиться и упал: ждём новый, не спамим.
if [ "$(_state_read)" = "failed:$NEW_HEAD" ]; then
    echo "Обновление до ${NEW_HEAD:0:7} уже проваливалось — жду нового коммита в $REMOTE/$BRANCH." >&2
    exit 0
fi

NEW_SUBJECT=$(git log -1 --format=%s "$NEW_HEAD")
echo "Обновление: ${OLD_HEAD:0:7} → ${NEW_HEAD:0:7} ($NEW_SUBJECT)"

# ---------------- применение и проверка ----------------

_deps_install() {
    # $1/$2 — старый и новый коммиты. Сервис останавливаем только здесь —
    # минимум даунтайма (код на диске уже обновлён, процесс работает старый).
    if git diff --quiet "$1" "$2" -- requirements.txt; then
        echo "requirements.txt не менялся — зависимости не переустанавливаю."
        return 0
    fi
    echo "requirements.txt изменился — остановка сервиса и pip install..."
    systemctl stop "$SERVICE" || return 1
    "$VENV_PIP" install -r "$APP_DIR/requirements.txt" || return 1
    return 0
}

_apply_update() {
    git reset --hard "$NEW_HEAD" || return 1
    _deps_install "$OLD_HEAD" "$NEW_HEAD" || return 1
    "$APP_DIR/restart.sh" --no-follow || return 1
    echo "Health-check (до ${HEALTH_DEADLINE} с)..."
    _wait_healthy
}

_rollback() {
    # Best effort: вернуть код и зависимости, поднять сервис, вернуть статус.
    if git reset --hard "$OLD_HEAD"; then
        echo "Код откачен на ${OLD_HEAD:0:7}."
    else
        echo "Ошибка: не удалось откатить код." >&2
    fi
    if git diff --quiet "$OLD_HEAD" "$NEW_HEAD" -- requirements.txt; then
        :
    else
        "$VENV_PIP" install -r "$APP_DIR/requirements.txt" \
            || echo "Ошибка: не удалось восстановить старые зависимости." >&2
    fi
    "$APP_DIR/restart.sh" --no-follow || echo "Ошибка: рестарт после отката." >&2
    _wait_healthy
}

if _apply_update; then
    _state_write ok
    echo "Обновление успешно."
    _notify "Kapum bot updated: ${OLD_HEAD:0:7} → ${NEW_HEAD:0:7} (${NEW_SUBJECT}). Health check OK."
else
    echo "Ошибка: сервис не готов после обновления — откат." >&2
    if _rollback; then
        echo "Откат удался, сервис здоров."
        _notify "Kapum bot: update to ${NEW_HEAD:0:7} failed; rolled back to ${OLD_HEAD:0:7}. Service is healthy. Details: journalctl -u kapum-update"
    else
        echo "Ошибка: откат не восстановил сервис — нужно вмешательство." >&2
        _notify "Kapum bot: update to ${NEW_HEAD:0:7} failed and rollback to ${OLD_HEAD:0:7} did not restore the service. Manual fix required (journalctl -u kapum-update -u $SERVICE)."
    fi
    # Не пробовать этот коммит снова до нового коммита в origin/master.
    _state_write "failed:$NEW_HEAD"
    exit 1
fi
