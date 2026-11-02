#!/usr/bin/env bash
#
# Загрузка Git-ключа в агент текущей Bash-сессии:
#   source ./load_git_key.sh
# Путь к ключу намеренно фиксирован; сам ключ хранится вне проекта.
# Не меняем set/trap оболочки и не останавливаем чужие агенты.
# После загрузки скрипт проверяет ssh-add -l: окружение агента живёт только
# в той оболочке, где выполнен source.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Ошибка: используйте source ./load_git_key.sh, иначе окружение агента не сохранится в текущей оболочке.\n' >&2
    exit 1
fi

_load_git_key_main() {
    local key_file="/root/id_ed25519_github"
    local tool agent_env="" agent_status=0

    for tool in ssh-agent ssh-add; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            printf 'Ошибка: команда %s не найдена; установите OpenSSH client.\n' "$tool" >&2
            return 1
        fi
    done

    if [[ ! -f "$key_file" ]]; then
        printf 'Ошибка: файл ключа не найден: %s\n' "$key_file" >&2
        return 1
    fi
    if [[ ! -r "$key_file" ]]; then
        printf 'Ошибка: нет прав на чтение ключа: %s\n' "$key_file" >&2
        return 1
    fi
    if [[ ! -s "$key_file" ]]; then
        printf 'Ошибка: файл ключа пуст: %s\n' "$key_file" >&2
        return 1
    fi

    # ssh-add -l: 0 — есть ключи, 1 — агент пуст, 2 — агент недоступен.
    ssh-add -l >/dev/null 2>&1 || agent_status=$?
    case "$agent_status" in
        0|1) ;;
        2)
            if ! agent_env=$(ssh-agent -s); then
                printf 'Ошибка: не удалось запустить ssh-agent.\n' >&2
                return 1
            fi
            if [[ -z "$agent_env" ]]; then
                printf 'Ошибка: ssh-agent не вернул окружение.\n' >&2
                return 1
            fi
            ;;
        *)
            printf 'Ошибка: проверка ssh-agent завершилась с кодом %s.\n' "$agent_status" >&2
            return 1
            ;;
    esac

    # Новое окружение сначала проверяем в подоболочке: при отказе ssh-add
    # завершаем только созданный нами агент, окружение вызывающей сессии сохраняем.
    if ! (
        if [[ -n "$agent_env" ]] && ! eval "$agent_env" >/dev/null; then
            printf 'Ошибка: не удалось применить окружение ssh-agent.\n' >&2
            exit 1
        fi
        if ! ssh-add "$key_file"; then
            printf 'Ошибка: не удалось загрузить ключ %s; проверьте формат, права (600) и пароль ключа.\n' "$key_file" >&2
            if [[ -n "$agent_env" ]]; then
                ssh-agent -k >/dev/null 2>&1 || printf 'Предупреждение: не удалось остановить созданный ssh-agent.\n' >&2
            fi
            exit 1
        fi
        # Проверяем итог до изменения окружения вызывающей оболочки: иначе при
        # сбое агента в сессии остался бы нерабочий SSH_AUTH_SOCK.
        if ! ssh-add -l >/dev/null 2>&1; then
            printf 'Ошибка: ssh-add -l не видит агента после загрузки (SSH_AUTH_SOCK=%s).\n' "${SSH_AUTH_SOCK:-не задан}" >&2
            printf 'Подсказка: source и ssh-add -L должны выполняться в одной оболочке.\n' >&2
            if [[ -n "$agent_env" ]]; then
                ssh-agent -k >/dev/null 2>&1 || printf 'Предупреждение: не удалось остановить созданный ssh-agent.\n' >&2
            fi
            exit 1
        fi
    ); then
        return 1
    fi

    if [[ -n "$agent_env" ]]; then
        eval "$agent_env" >/dev/null || return 1
    fi
    printf 'Ключ %s загружен; SSH_AUTH_SOCK=%s\n' "$key_file" "${SSH_AUTH_SOCK:-не задан}"
}

if _load_git_key_main; then
    unset -f _load_git_key_main
    return 0
else
    unset -f _load_git_key_main
    return 1
fi
