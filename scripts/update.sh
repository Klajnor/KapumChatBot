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
TIMER_REPORT="unchanged (disabled or not installed)"
CURRENT_STAGE="preflight" FAILURE_STAGE="" FAILURE_CODE=0 FAILURE_DETAIL="" STEP_DETAIL=""
ROLLING_BACK=0 ROLLBACK_ERRORS="" ROLLBACK_REPORT="not needed (no code changed)" OUTCOME=""
RUN_DIR="" BACKUP_DIR=""

_git() {
    # Хуки из репозитория не должны выполняться от root.
    local dir="$1"
    shift
    git -c core.hooksPath=/dev/null -C "$dir" "$@"
}

_support() { "$BOT_PY" -I "$SUPPORT" "$@"; }
_system_support() { /usr/bin/python3 -I "$SUPPORT" "$@"; }

_stage_hint() {
    case "$1" in
        *.backup) printf 'Cannot copy .venv; check free disk space and directory permissions.' ;;
        *.dependencies) printf 'pip install failed or timed out (limit 300s); rollback uses the saved .venv.' ;;
        *.checkout) printf 'Git checkout failed; check repository permissions and local changes.' ;;
        brain.mcp) printf 'MCP probe failed or timed out (limit 40s); check brain startup and dependencies.' ;;
        timer.*) printf 'systemd timer operation failed; check LoadState and ActiveState below.' ;;
        *) printf 'Command failed; see the journal for details.' ;;
    esac
}

_record_failure() {
    # Только наши метки/безопасные сообщения helper, не сырой stderr команд.
    local stage="$1" code="$2" detail="${3:-}"
    if [ -z "$detail" ]; then detail=$(_stage_hint "$stage"); fi
    echo "Ошибка этапа $stage (код $code): $detail" >&2
    if [ "$ROLLING_BACK" -eq 1 ]; then
        ROLLBACK_ERRORS+="$stage (exit $code): ${detail:0:200}; "
    elif [ -z "$FAILURE_STAGE" ]; then
        FAILURE_STAGE="$stage" FAILURE_CODE="$code" FAILURE_DETAIL="${detail:0:700}"
    fi
}

_step() {
    local stage="$1" code
    shift
    CURRENT_STAGE="$stage" STEP_DETAIL=""
    echo "Этап: $stage"
    if "$@"; then return 0; else
        code=$?
        _record_failure "$stage" "$code" "$STEP_DETAIL"
        return "$code"
    fi
}

_checked_command() {
    # Helper выводит контролируемую диагностику, без исходного кода/секретов.
    local output code
    STEP_DETAIL=""
    if output=$("$@" 2>&1); then
        printf '%s' "$output"
    else
        code=$?
        case "$output" in
            "Update check failed:"*) STEP_DETAIL="${output:0:700}" ;;
            *) STEP_DETAIL="$(_stage_hint "$CURRENT_STAGE") Exit code: $code." ;;
        esac
        printf '%s\n' "$STEP_DETAIL" >&2
        return "$code"
    fi
}

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
        _notify "Kapum auto-update: $1 blocked.
Stage: preflight.$1
Reason: $2
${STEP_DETAIL:0:700}
This repository was not changed.
Logs: journalctl -u kapum-update -n 80 --no-pager"
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
    if ! _checked_command _system_support trusted "$dir" "$name"; then
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
    if _fetch "$dir" "$key"; then :; else
        local code=$?
        _blocked "$name" "git fetch failed (exit $code); check network, deploy key and known_hosts."
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
    if ! _checked_command _system_support target "$dir" "$old" "$new"; then
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

_remember_timer() {
    # Включённый, но остановленный предыдущим обновлением таймер тоже
    # возвращаем. Намеренно disabled/inactive таймер не включаем.
    if _active "$PULL_TIMER"; then
        TIMER_WAS_ACTIVE=1
    else
        case "$(systemctl is-enabled "$PULL_TIMER" 2>/dev/null)" in
            enabled|enabled-runtime) TIMER_WAS_ACTIVE=1 ;;
        esac
    fi
}

_pull_running() {
    case "$(systemctl show "$PULL_UNIT" -p ActiveState --value 2>/dev/null)" in
        active|activating|deactivating) return 0 ;;
        *) return 1 ;;
    esac
}

_defer_brain() {
    echo "brain-pull ещё работает — обновление мозга отложено до следующей проверки."
    UPD_BRAIN=0
}

_pause_indexer() {
    # Работающий oneshot не ждём и не прерываем; даже таймер не трогаем.
    # Повторная проверка после остановки триггера закрывает гонку запуска.
    _remember_timer
    if _pull_running; then _defer_brain; return 0; fi
    if _active "$PULL_TIMER"; then
        _step timer.pause systemctl stop "$PULL_TIMER" || return 1
    fi
    if _pull_running; then _defer_brain; return 0; fi
    case "$(systemctl show "$SYNC_UNIT" -p ActiveState --value 2>/dev/null)" in
        active|activating) SYNC_WAS_ACTIVE=1 ;;
    esac
}

_resume_timer() {
    if [ "$TIMER_WAS_ACTIVE" -eq 1 ]; then
        if ! _active "$PULL_TIMER"; then
            if ! _step timer.resume systemctl start "$PULL_TIMER"; then
                TIMER_REPORT="FAILED to resume; inspect brain-pull.timer"
                return 1
            fi
        fi
        if ! _active "$PULL_TIMER"; then
            TIMER_REPORT="FAILED: timer is inactive after start"
            _record_failure timer.resume 1 "$TIMER_REPORT"
            return 1
        fi
        TIMER_REPORT="active (schedule restored)"
        TIMER_WAS_ACTIVE=0
    fi
}

_unit_fingerprint() {
    # PID тоже нужен: число NRestarts обнуляется при ручном рестарте.
    systemctl show "$1" -p MainPID -p NRestarts
}

_wait_healthy() {
    CURRENT_STAGE=health.wait
    local start=$SECONDS ready_since=-1 bot_base sync_base="" url health_detail=""
    bot_base=$(_unit_fingerprint "$SERVICE") || {
        _record_failure health.bot.state 1 "Cannot read systemd process state."; return 1;
    }
    if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then
        sync_base=$(_unit_fingerprint "$SYNC_UNIT") || {
            _record_failure health.indexer.state 1 "Cannot read watch indexer state."; return 1;
        }
    fi
    url=$(_support health-url "$APP_DIR") || {
        _record_failure health.config 1 "Check HEALTH_HOST and HEALTH_PORT."; return 1;
    }
    while [ "$((SECONDS - start))" -lt "$HEALTH_DEADLINE" ]; do
        _active "$SERVICE" || {
            _record_failure health.bot.active 1 "$SERVICE is not active."; return 1;
        }
        [ "$(_unit_fingerprint "$SERVICE")" = "$bot_base" ] || {
            _record_failure health.bot.stability 1 "Bot PID or NRestarts changed (crash loop)."; return 1;
        }
        if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then
            _active "$SYNC_UNIT" || {
                _record_failure health.indexer.active 1 "$SYNC_UNIT is not active."; return 1;
            }
            [ "$(_unit_fingerprint "$SYNC_UNIT")" = "$sync_base" ] || {
                _record_failure health.indexer.stability 1 "Indexer PID or NRestarts changed."; return 1;
            }
        fi
        if [ -z "$url" ] || _checked_command _support http-check "$url"; then
            if [ "$ready_since" -lt 0 ]; then ready_since=$SECONDS; fi
            if [ "$((SECONDS - ready_since))" -ge "$STABLE_SECONDS" ]; then return 0; fi
        else
            ready_since=-1
            health_detail="$STEP_DETAIL"
        fi
        sleep 2
    done
    _record_failure health.deadline 1 "Readiness deadline ${HEALTH_DEADLINE}s exceeded. $health_detail"
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
    [ "$(_git "$dir" rev-parse HEAD)" = "$old" ] || { STEP_DETAIL="HEAD changed since preflight."; return 1; }
    dirty=$(_git "$dir" status --porcelain --untracked-files=no) || return 1
    [ -z "$dirty" ] || { STEP_DETAIL="Tracked files changed since preflight."; return 1; }
    _checked_command _system_support target "$dir" "$old" "$new"
}

_restart_services() {
    _step bot.restart /bin/bash "$APP_DIR/restart.sh" --no-follow || return 1
    if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then _step indexer.start systemctl start "$SYNC_UNIT" || return 1; fi
    if [ "$UPD_BRAIN" -eq 1 ]; then
        # Запуск только от brain, без инструментов записи и без вызовов LLM.
        _step brain.mcp _checked_command timeout --kill-after=5s 40s runuser -u brain -- \
            "$BOT_PY" -I "$SUPPORT" mcp-check "$BRAIN_PY" "$BRAIN_DIR" || return 1
    fi
    _wait_healthy
}

_apply_update() {
    # Копии окружений создаются до остановки сервисов. При нехватке места
    # рабочий код/окружение ещё не затронуты.
    if [ "$BOT_REQ" -eq 1 ]; then _step bot.backup _backup_venv "$APP_DIR" bot || return 1; fi
    if [ "$BRAIN_REQ" -eq 1 ]; then _step brain.backup _backup_venv "$BRAIN_DIR" brain || return 1; fi
    TRANSACTION=1
    _step bot.stop systemctl stop "$SERVICE" || return 1
    if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then _step indexer.stop systemctl stop "$SYNC_UNIT" || return 1; fi
    if [ "$UPD_BOT" -eq 1 ]; then
        _step bot.recheck _ready_to_advance "$APP_DIR" "$BOT_OLD" "$BOT_NEW" || return 1
        BOT_TOUCHED=1
        _step bot.checkout _git "$APP_DIR" merge --ff-only --no-edit "$BOT_NEW" || return 1
        _step bot.syntax _checked_command "$BOT_PY" -I "$SUPPORT" syntax "$APP_DIR/bot" "$APP_DIR/scripts" || return 1
    fi
    if [ "$UPD_BRAIN" -eq 1 ]; then
        _step brain.recheck _ready_to_advance "$BRAIN_DIR" "$BRAIN_OLD" "$BRAIN_NEW" || return 1
        BRAIN_TOUCHED=1
        _step brain.checkout _git "$BRAIN_DIR" merge --ff-only --no-edit "$BRAIN_NEW" || return 1
        _step brain.syntax _checked_command "$BRAIN_PY" -I "$SUPPORT" syntax "$BRAIN_DIR" || return 1
    fi
    if [ "$BOT_REQ" -eq 1 ]; then _step bot.dependencies _install_deps "$BOT_PY" "$APP_DIR" || return 1; fi
    if [ "$BRAIN_REQ" -eq 1 ]; then _step brain.dependencies _install_deps "$BRAIN_PY" "$BRAIN_DIR" || return 1; fi
    _restart_services
}

_rollback() {
    # При ошибке любого шага не объявляем откат успешным только потому,
    # что процесс бота активен. Не откатываем не затронутые репозитории.
    local result=0
    _step rollback.bot.stop systemctl stop "$SERVICE" || return 1
    if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then _step rollback.indexer.stop systemctl stop "$SYNC_UNIT" || return 1; fi
    if [ "$BOT_TOUCHED" -eq 1 ]; then _step rollback.bot.checkout _git "$APP_DIR" reset --hard "$BOT_OLD" || result=1; fi
    if [ "$BRAIN_TOUCHED" -eq 1 ]; then _step rollback.brain.checkout _git "$BRAIN_DIR" reset --hard "$BRAIN_OLD" || result=1; fi
    _step rollback.bot.environment _restore_venv "$APP_DIR" bot || result=1
    _step rollback.brain.environment _restore_venv "$BRAIN_DIR" brain || result=1
    _restart_services || result=1
    return "$result"
}

_summary() {
    if [ "$UPD_BOT" -eq 1 ]; then printf 'bot %s → %s; ' "${BOT_OLD:0:7}" "${BOT_NEW:0:7}"; fi
    if [ "$UPD_BRAIN" -eq 1 ]; then printf 'brain %s → %s; ' "${BRAIN_OLD:0:7}" "${BRAIN_NEW:0:7}"; fi
}

_failed_update() {
    OUTCOME=failed ROLLING_BACK=1
    if _rollback; then
        ROLLBACK_REPORT="complete. Previous code and environments restored; service checks OK."
    else
        KEEP_BACKUP=1
        ROLLBACK_REPORT="rollback is incomplete; manual intervention required. ${ROLLBACK_ERRORS:0:700}"
    fi
    TRANSACTION=0 ROLLING_BACK=0
    if [ "$UPD_BOT" -eq 1 ]; then _save_state bot "failed:$BOT_NEW"; fi
    if [ "$UPD_BRAIN" -eq 1 ]; then _save_state brain "failed:$BRAIN_NEW"; fi
}

_unit_summary() {
    local unit="$1" info
    if info=$(systemctl show "$unit" -p LoadState -p ActiveState -p SubState -p Result \
        -p ExecMainStatus -p NRestarts 2>/dev/null | tr '\n' ' '); then
        printf '%s: %s\n' "$unit" "${info:0:350}"
    else
        printf '%s: state unavailable\n' "$unit"
    fi
}

_notify_result() {
    local backup="" units
    if [ "$OUTCOME" = success ] && [ -z "$FAILURE_STAGE" ]; then
        _notify "Kapum updated: $(_summary) Service checks OK.
$PULL_TIMER: $TIMER_REPORT"
    elif [ -n "$OUTCOME" ] || [ -n "$FAILURE_STAGE" ]; then
        if [ "$KEEP_BACKUP" -eq 1 ]; then backup="Backups: ${BACKUP_DIR:-$RUN_DIR}"; fi
        units="$(_unit_summary "$SERVICE")
$(_unit_summary "$PULL_UNIT")
$(_unit_summary "$PULL_TIMER")"
        if [ "$SYNC_WAS_ACTIVE" -eq 1 ]; then units+=$'\n'"$(_unit_summary "$SYNC_UNIT")"; fi
        _notify "Kapum auto-update failed [$(_summary)].
Stage: ${FAILURE_STAGE:-$CURRENT_STAGE} (exit $FAILURE_CODE)
Reason: ${FAILURE_DETAIL:-Command failed; see the journal for details.}
Rollback: $ROLLBACK_REPORT
$PULL_TIMER: $TIMER_REPORT
$units
$backup
Logs: journalctl -u kapum-update -u chat_bot -u brain-pull -n 100 --no-pager"
    fi
}

_on_exit() {
    local status=$?
    trap - EXIT INT TERM
    if [ "$status" -ne 0 ] && [ -z "$FAILURE_STAGE" ] && [ "$BLOCKED" -eq 0 ]; then
        OUTCOME=failed
        _record_failure "$CURRENT_STAGE" "$status" "Update exited before completion or was interrupted."
    fi
    if [ "$TRANSACTION" -eq 1 ]; then
        if [ "$status" -eq 0 ]; then status=1; fi
        if [ -z "$FAILURE_STAGE" ]; then _record_failure "$CURRENT_STAGE" "$status" "Update was interrupted."; fi
        _failed_update || true
    fi
    # Возвращаем расписание при любом исходе, даже при неполном откате.
    if ! _resume_timer; then
        if [ "$status" -eq 0 ]; then status=1; fi
    fi
    _notify_result || true
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

    _remember_timer
    _check_repo bot "$APP_DIR" "$BOT_KEY" "$BOT_PY" Klajnor/KapumChatBot.git
    BOT_OLD=$REPO_OLD BOT_NEW=$REPO_NEW UPD_BOT=$REPO_UPD
    _check_repo brain "$BRAIN_DIR" "$BRAIN_KEY" "$BRAIN_PY" Klajnor/ObsidianBrain.git
    BRAIN_OLD=$REPO_OLD BRAIN_NEW=$REPO_NEW UPD_BRAIN=$REPO_UPD
    if [ "$UPD_BRAIN" -eq 1 ]; then
        if ! _pause_indexer; then OUTCOME=failed; return 1; fi
    fi
    if [ "$UPD_BOT" -eq 0 ] && [ "$UPD_BRAIN" -eq 0 ]; then return "$BLOCKED"; fi
    BACKUP_DIR=$(mktemp -d "$STATE_DIR/backup.XXXXXXXX")

    if [ "$UPD_BOT" -eq 1 ] && ! _git "$APP_DIR" diff --quiet "$BOT_OLD" "$BOT_NEW" -- requirements.txt; then BOT_REQ=1; fi
    if [ "$UPD_BRAIN" -eq 1 ] && ! _git "$BRAIN_DIR" diff --quiet "$BRAIN_OLD" "$BRAIN_NEW" -- requirements.txt; then BRAIN_REQ=1; fi
    if _apply_update; then
        TRANSACTION=0
        OUTCOME=success ROLLBACK_REPORT="not needed (new code and service checks OK)"
        if [ "$UPD_BOT" -eq 1 ]; then _save_state bot ok; fi
        if [ "$UPD_BRAIN" -eq 1 ]; then _save_state brain ok; fi
    elif [ "$TRANSACTION" -eq 1 ]; then
        _failed_update
        return 1
    else
        OUTCOME=failed
        return 1
    fi
    return "$BLOCKED"
}

_main "$@"
