#!/usr/bin/env bash
# Автообновление кода бота и ObsidianBrain; запуск от root через systemd.
# Конфиги, SQLite и хранилища заметок не обновляются и не откатываются.
# Использование: sudo ./scripts/update.sh

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    echo "Ошибка: запускайте scripts/update.sh отдельным процессом." >&2
    return 1
fi

set -euo pipefail
umask 022

APP_DIR="/opt/KapumChatBot"
BRAIN_DIR="/opt/brain"
BOT_PY="$APP_DIR/.venv/bin/python"
BRAIN_PY="$BRAIN_DIR/.venv/bin/python3"
BOT_KEY="/root/id_ed25519_kapum_deploy"
BRAIN_KEY="/root/id_ed25519_brain_deploy"
SERVICE="chat_bot.service"
SYNC_UNIT="brain-sync.service"
PULL_UNIT="brain-pull.service"
PULL_TIMER="brain-pull.timer"
STATE_DIR="/var/lib/kapum-update"
LOCK_FILE="/run/kapum-update.lock"
SUPPORT="$APP_DIR/scripts/update_support.py"
HEALTH_DEADLINE=60
STABLE_SECONDS=20

UPD_BOT=0 UPD_BRAIN=0 BLOCKED=0
BOT_OLD="" BOT_NEW="" BRAIN_OLD="" BRAIN_NEW=""
BOT_REQ=0 BRAIN_REQ=0 BOT_TOUCHED=0 BRAIN_TOUCHED=0
SYNC_WAS_ACTIVE=0 TIMER_WAS_ACTIVE=0 TRANSACTION=0 KEEP_BACKUP=0
RUN_DIR="" BACKUP_DIR=""

_git() {
    # Хуки из репозитория не должны выполняться от root.
    local dir="$1"
    shift
    git -c core.hooksPath=/dev/null -C "$dir" "$@"
}

_support() { "$BOT_PY" -I "$SUPPORT" "$@"; }
_system_support() { /usr/bin/python3 -I "$SUPPORT" "$@"; }

_notify() {
    # Токен читается внутри Python и не попадает в argv, stderr или журнал.
    _support notify "$APP_DIR" "$1" || echo "Предупреждение: уведомление не доставлено." >&2
}

_state() { cat "$STATE_DIR/$1.state" 2>/dev/null || true; }
_save_state() { printf '%s\n' "$2" >"$STATE_DIR/$1.state"; }

_blocked() {
    BLOCKED=1
    echo "Обновление $1 заблокировано: $2" >&2
    if [ "$(_state "$1")" != "block" ]; then
        _notify "Kapum auto-update: $1 blocked — $2 This repository was not changed."
        _save_state "$1" block
    fi
}

_fetch() {
    # Только один ключ, без агента, паролей и интерактивного подтверждения.
    # Host key GitHub должен быть проверен и установлен вручную заранее.
    local dir="$1" key="$2"
    GIT_TERMINAL_PROMPT=0 \
    GIT_SSH_COMMAND="ssh -F /dev/null -i $key -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=yes -o ConnectTimeout=15" \
        timeout --kill-after=10s 60s git -c core.hooksPath=/dev/null -C "$dir" \
        fetch --no-tags --recurse-submodules=no origin \
        +refs/heads/master:refs/remotes/origin/master >/dev/null 2>&1
}

_check_repo() {
    # Результат в REPO_OLD/REPO_NEW/REPO_UPD; блокировка одного клона не
    # мешает проверить другой. /opt/brain может отсутствовать.
    local name="$1" dir="$2" key="$3" py="$4" expected="$5" old new dirty
    REPO_OLD="" REPO_NEW="" REPO_UPD=0
    if [ "$name" = brain ] && [ ! -e "$dir" ]; then return 0; fi
    if ! _system_support trusted "$dir" "$name"; then
        _blocked "$name" "code must be root-owned and not writable by the service (see README)."
        return 0
    fi
    if [ ! -d "$dir/.git" ] || [ "$(_git "$dir" symbolic-ref --short -q HEAD)" != master ]; then
        _blocked "$name" "expected a Git clone on branch master in $dir."
        return 0
    fi
    case "$(_git "$dir" remote get-url origin)" in
        "git@github.com:$expected"|"ssh://git@github.com/$expected") ;;
        *) _blocked "$name" "origin must be SSH for $expected (see README)."; return 0 ;;
    esac
    if [ ! -r "$key" ] || [ ! -x "$py" ] || [ -L "$dir/.venv" ]; then
        _blocked "$name" "deploy key or .venv Python is missing (see README)."
        return 0
    fi
    echo "$name: проверка origin/master..."
    if ! _fetch "$dir" "$key"; then
        _blocked "$name" "git fetch failed; check network, deploy key and known_hosts."
        return 0
    fi
    old=$(_git "$dir" rev-parse --verify HEAD)
    new=$(_git "$dir" rev-parse --verify 'origin/master^{commit}')
    if [ "$old" = "$new" ]; then
        _save_state "$name" ok
        return 0
    fi
    if ! dirty=$(_git "$dir" status --porcelain --untracked-files=no) || [ -n "$dirty" ]; then
        _blocked "$name" "local tracked changes; commit or resolve them manually."
        return 0
    fi
    if ! _git "$dir" merge-base --is-ancestor "$old" "$new"; then
        _blocked "$name" "local commits or diverged history; only fast-forward is allowed."
        return 0
    fi
    if ! _system_support target "$dir" "$old" "$new"; then
        _blocked "$name" "target conflicts with local files or protected runtime paths."
        return 0
    fi
    if [ "$(_state "$name")" = "failed:$new" ]; then
        echo "$name: коммит ${new:0:7} уже откатывался, ожидаю следующий."
        return 0
    fi
    REPO_OLD="$old" REPO_NEW="$new" REPO_UPD=1
    echo "$name: ${old:0:7} → ${new:0:7}"
}

_active() { systemctl is-active --quiet "$1"; }

_pause_indexer() {
    # Не прерываем длительный oneshot с платными LLM-запросами. Сначала
    # выключаем триггер, затем проверяем, не успел ли начаться прогон.
    if _active "$PULL_TIMER"; then
        TIMER_WAS_ACTIVE=1
        systemctl stop "$PULL_TIMER" || return 1
    fi
    case "$(systemctl show "$PULL_UNIT" -p ActiveState --value 2>/dev/null)" in
        active|activating|deactivating)
            echo "brain-pull ещё работает — обновление мозга отложено."
            UPD_BRAIN=0
            return 0
            ;;
    esac
    case "$(systemctl show "$SYNC_UNIT" -p ActiveState --value 2>/dev/null)" in
        active|activating) SYNC_WAS_ACTIVE=1 ;;
    esac
}

_resume_timer() {
    if [ "$TIMER_WAS_ACTIVE" -eq 1 ]; then
        systemctl start "$PULL_TIMER" || return 1
        TIMER_WAS_ACTIVE=0
    fi
}

_unit_fingerprint() {
    # PID тоже нужен: число NRestarts обнуляется при ручном рестарте.
    systemctl show "$1" -p MainPID -p NRestarts
}

_wait_healthy() {
    local start=$SECONDS ready_since=-1 bot_base sync_base="" url
    bot_base=$(_unit_fingerprint "$SERVICE") || return 1
    if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then
        sync_base=$(_unit_fingerprint "$SYNC_UNIT") || return 1
    fi
    url=$(_support health-url "$APP_DIR") || return 1
    while [ "$((SECONDS - start))" -lt "$HEALTH_DEADLINE" ]; do
        _active "$SERVICE" || return 1
        [ "$(_unit_fingerprint "$SERVICE")" = "$bot_base" ] || return 1
        if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then
            _active "$SYNC_UNIT" || return 1
            [ "$(_unit_fingerprint "$SYNC_UNIT")" = "$sync_base" ] || return 1
        fi
        if [ -z "$url" ] || _support http-check "$url"; then
            if [ "$ready_since" -lt 0 ]; then ready_since=$SECONDS; fi
            if [ "$((SECONDS - ready_since))" -ge "$STABLE_SECONDS" ]; then return 0; fi
        else
            ready_since=-1
        fi
        sleep 2
    done
    return 1
}

_backup_venv() {
    # Снимок, а не pip freeze: так возвращаются удалённые/новые пакеты и
    # точные старые версии. Reflink при поддержке ФС, иначе обычная копия.
    local destination="${BACKUP_DIR:-$RUN_DIR}"
    cp -a --reflink=auto -- "$1/.venv" "$destination/$2-venv" || return 1
    chmod 700 "$destination/$2-venv"
}

_restore_venv() {
    local dir="$1" name="$2" source="${BACKUP_DIR:-$RUN_DIR}"
    [ -d "$source/$name-venv" ] || return 0
    mv -- "$dir/.venv" "$source/$name-broken-venv" || return 1
    mv -- "$source/$name-venv" "$dir/.venv" || return 1
    chmod 755 "$dir/.venv"
}

_install_deps() {
    timeout --kill-after=10s 300s "$1" -m pip --disable-pip-version-check \
        install -r "$2/requirements.txt"
}

_ready_to_advance() {
    # Повторная проверка после копирования окружений: администратор мог
    # изменить checkout во время fetch/подготовки. Чужую работу не откатываем.
    local dir="$1" old="$2" new="$3" dirty
    [ "$(_git "$dir" rev-parse HEAD)" = "$old" ] || return 1
    dirty=$(_git "$dir" status --porcelain --untracked-files=no) || return 1
    [ -z "$dirty" ] || return 1
    _system_support target "$dir" "$old" "$new"
}

_restart_services() {
    /bin/bash "$APP_DIR/restart.sh" --no-follow || return 1
    if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then systemctl start "$SYNC_UNIT" || return 1; fi
    if [ "$UPD_BRAIN" -eq 1 ]; then
        # Запуск только от brain, без инструментов записи и без вызовов LLM.
        timeout --kill-after=5s 40s runuser -u brain -- "$BOT_PY" -I "$SUPPORT" \
            mcp-check "$BRAIN_PY" "$BRAIN_DIR" || return 1
    fi
    _wait_healthy
}

_apply_update() {
    # Копии окружений создаются до остановки сервисов. При нехватке места
    # рабочий код/окружение ещё не затронуты.
    if [ "$BOT_REQ" -eq 1 ]; then _backup_venv "$APP_DIR" bot || return 1; fi
    if [ "$BRAIN_REQ" -eq 1 ]; then _backup_venv "$BRAIN_DIR" brain || return 1; fi
    TRANSACTION=1
    systemctl stop "$SERVICE" || return 1
    if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then systemctl stop "$SYNC_UNIT" || return 1; fi
    if [ "$UPD_BOT" -eq 1 ]; then
        _ready_to_advance "$APP_DIR" "$BOT_OLD" "$BOT_NEW" || return 1
        BOT_TOUCHED=1
        _git "$APP_DIR" merge --ff-only --no-edit "$BOT_NEW" || return 1
        "$BOT_PY" -I "$SUPPORT" syntax "$APP_DIR/bot" "$APP_DIR/scripts" || return 1
    fi
    if [ "$UPD_BRAIN" -eq 1 ]; then
        _ready_to_advance "$BRAIN_DIR" "$BRAIN_OLD" "$BRAIN_NEW" || return 1
        BRAIN_TOUCHED=1
        _git "$BRAIN_DIR" merge --ff-only --no-edit "$BRAIN_NEW" || return 1
        "$BRAIN_PY" -I "$SUPPORT" syntax "$BRAIN_DIR" || return 1
    fi
    if [ "$BOT_REQ" -eq 1 ]; then _install_deps "$BOT_PY" "$APP_DIR" || return 1; fi
    if [ "$BRAIN_REQ" -eq 1 ]; then _install_deps "$BRAIN_PY" "$BRAIN_DIR" || return 1; fi
    _restart_services
}

_rollback() {
    # При ошибке любого шага не объявляем откат успешным только потому,
    # что процесс бота активен. Не откатываем не затронутые репозитории.
    local result=0
    systemctl stop "$SERVICE" || return 1
    if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then systemctl stop "$SYNC_UNIT" || return 1; fi
    if [ "$BOT_TOUCHED" -eq 1 ]; then _git "$APP_DIR" reset --hard "$BOT_OLD" || result=1; fi
    if [ "$BRAIN_TOUCHED" -eq 1 ]; then _git "$BRAIN_DIR" reset --hard "$BRAIN_OLD" || result=1; fi
    _restore_venv "$APP_DIR" bot || result=1
    _restore_venv "$BRAIN_DIR" brain || result=1
    _restart_services || result=1
    return "$result"
}

_summary() {
    if [ "$UPD_BOT" -eq 1 ]; then printf 'bot %s → %s; ' "${BOT_OLD:0:7}" "${BOT_NEW:0:7}"; fi
    if [ "$UPD_BRAIN" -eq 1 ]; then printf 'brain %s → %s; ' "${BRAIN_OLD:0:7}" "${BRAIN_NEW:0:7}"; fi
}

_failed_update() {
    if _rollback; then
        _notify "Kapum auto-update failed [$(_summary)]. Previous code and environments restored; service checks OK."
    else
        KEEP_BACKUP=1
        _notify "Kapum auto-update failed [$(_summary)]; rollback is incomplete. Manual intervention required: journalctl -u kapum-update -u chat_bot. Backups: ${BACKUP_DIR:-$RUN_DIR}"
    fi
    TRANSACTION=0
    if [ "$UPD_BOT" -eq 1 ]; then _save_state bot "failed:$BOT_NEW"; fi
    if [ "$UPD_BRAIN" -eq 1 ]; then _save_state brain "failed:$BRAIN_NEW"; fi
}

_on_exit() {
    local status=$?
    trap - EXIT INT TERM
    if [ "$TRANSACTION" -eq 1 ]; then _failed_update || true; fi
    if [ "$KEEP_BACKUP" -eq 1 ] && [ "$TIMER_WAS_ACTIVE" -eq 1 ]; then
        echo "brain-pull.timer оставлен выключенным до ручного восстановления." >&2
    elif ! _resume_timer; then
        _notify "Kapum auto-update: could not resume brain-pull.timer; check systemd manually."
        status=1
    fi
    if [ -n "$BACKUP_DIR" ] && [ "$KEEP_BACKUP" -eq 0 ]; then rm -rf -- "$BACKUP_DIR"; fi
    if [ -n "$RUN_DIR" ]; then rm -rf -- "$RUN_DIR"; fi
    exit "$status"
}

_main() {
    if [ "$(id -u)" -ne 0 ]; then echo "Ошибка: нужен root." >&2; return 1; fi
    # Минимальная проверка доверия до первого выполнения helper от root.
    local path mode
    for path in / /opt "$APP_DIR" "$APP_DIR/scripts" "$SUPPORT"; do
        [ ! -L "$path" ] && [ "$(stat -c %u "$path")" = 0 ] || return 1
        mode=$(stat -c %a "$path")
        [ "$((8#$mode & 022))" -eq 0 ] || return 1
    done
    _system_support trusted "$APP_DIR" bot || return 1
    for path in git ssh timeout flock systemctl runuser; do command -v "$path" >/dev/null || return 1; done
    exec 9>"$LOCK_FILE"
    flock -n 9 || return 0
    mkdir -p -- "$STATE_DIR"
    chmod 700 "$STATE_DIR"
    RUN_DIR=$(mktemp -d /run/kapum-update.XXXXXXXX)
    # Helper остаётся неизменным, даже если обновление заменит сам скрипт.
    # В каталоге нет секретов; большие снимки venv храним на диске в /var/lib,
    # а не в обычно маленьком tmpfs /run.
    cp -- "$SUPPORT" "$RUN_DIR/update_support.py"
    chmod 755 "$RUN_DIR"
    chmod 644 "$RUN_DIR/update_support.py"
    SUPPORT="$RUN_DIR/update_support.py"
    trap _on_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    _check_repo bot "$APP_DIR" "$BOT_KEY" "$BOT_PY" Klajnor/KapumChatBot.git
    BOT_OLD=$REPO_OLD BOT_NEW=$REPO_NEW UPD_BOT=$REPO_UPD
    _check_repo brain "$BRAIN_DIR" "$BRAIN_KEY" "$BRAIN_PY" Klajnor/ObsidianBrain.git
    BRAIN_OLD=$REPO_OLD BRAIN_NEW=$REPO_NEW UPD_BRAIN=$REPO_UPD
    if [ "$UPD_BRAIN" -eq 1 ]; then _pause_indexer || return 1; fi
    if [ "$UPD_BOT" -eq 0 ] && [ "$UPD_BRAIN" -eq 0 ]; then return "$BLOCKED"; fi
    BACKUP_DIR=$(mktemp -d "$STATE_DIR/backup.XXXXXXXX")

    if [ "$UPD_BOT" -eq 1 ] && ! _git "$APP_DIR" diff --quiet "$BOT_OLD" "$BOT_NEW" -- requirements.txt; then BOT_REQ=1; fi
    if [ "$UPD_BRAIN" -eq 1 ] && ! _git "$BRAIN_DIR" diff --quiet "$BRAIN_OLD" "$BRAIN_NEW" -- requirements.txt; then BRAIN_REQ=1; fi
    if _apply_update; then
        TRANSACTION=0
        if [ "$UPD_BOT" -eq 1 ]; then _save_state bot ok; fi
        if [ "$UPD_BRAIN" -eq 1 ]; then _save_state brain ok; fi
        _notify "Kapum updated: $(_summary) Service checks OK."
    elif [ "$TRANSACTION" -eq 1 ]; then
        _failed_update
        return 1
    else
        _notify "Kapum auto-update: could not back up .venv. No code was changed; check free disk space."
        return 1
    fi
    return "$BLOCKED"
}

_main "$@"
