#!/usr/bin/env bash
#
# Рекурсивно удаляет папки __pycache__ (кэш скомпилированного Python).
# Полезно перед загрузкой на хостинг или после смены версии Python.
#
# Использование:
#   scripts/clean_pycache.sh              # удалить во всём проекте
#   scripts/clean_pycache.sh -n           # только показать (-DryRun)
#   scripts/clean_pycache.sh -p PATH      # очистить другую папку
#
# Совместимо с bash 3.2 (macOS) и bash 4+ (Linux, Git Bash на Windows).

set -euo pipefail

usage() {
    sed -n '3,10p' "$0" | sed 's/^# \{0,1\}//'
}

DRY_RUN=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"   # корень проекта: скрипт лежит в <root>/scripts

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        -p|--path)
            if [ $# -lt 2 ]; then
                echo "Ошибка: опции $1 нужен путь" >&2
                exit 2
            fi
            ROOT="$2"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Неизвестная опция: $1" >&2; usage >&2; exit 2 ;;
        *) ROOT="$1" ;;
    esac
    shift
done

if [ ! -d "$ROOT" ]; then
    echo "Ошибка: папка не найдена: $ROOT" >&2
    exit 1
fi
ROOT="$(cd "$ROOT" && pwd)"

echo "Поиск __pycache__ под: $ROOT"

# Собираем список заранее: -prune не даёт заходить внутрь найденных папок.
dirs=()
while IFS= read -r -d '' dir; do
    dirs+=("$dir")
done < <(find "$ROOT" -type d -name '__pycache__' -prune -print0)

if [ "${#dirs[@]}" -eq 0 ]; then
    echo "Нечего чистить: папок __pycache__ нет."
    exit 0
fi

removed=0
freed_kb=0

for dir in "${dirs[@]}"; do
    files="$(find "$dir" -type f | wc -l | tr -d '[:space:]')"
    kb="$(du -sk "$dir" 2>/dev/null | cut -f1 || true)"
    kb="${kb:-0}"
    mb="$(awk -v k="$kb" 'BEGIN { printf "%.2f", k / 1024 }')"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] $dir  ($files файлов, $mb МБ)"
        continue
    fi

    if rm -rf -- "$dir"; then
        removed=$((removed + 1))
        freed_kb=$((freed_kb + kb))
        echo "Удалена $dir  ($files файлов, $mb МБ)"
    else
        echo "Предупреждение: не удалось удалить $dir" >&2
    fi
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry-run: будет удалено папок: ${#dirs[@]}"
else
    awk -v n="$removed" -v t="${#dirs[@]}" -v k="$freed_kb" \
        'BEGIN { printf "Готово: удалено %d из %d папок; освобождено %.2f МБ\n", n, t, k / 1024 }'
fi
