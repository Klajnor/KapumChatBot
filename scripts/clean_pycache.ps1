<#
.SYNOPSIS
    Рекурсивно удаляет папки __pycache__ (кэш скомпилированного Python).
    Полезно перед загрузкой на хостинг или после смены версии Python.

.EXAMPLE
    .\scripts\clean_pycache.ps1              # удалить во всём проекте
    .\scripts\clean_pycache.ps1 -DryRun      # только показать, ничего не удаляя
    .\scripts\clean_pycache.ps1 -Path C:\other\project
    powershell -ExecutionPolicy Bypass -File .\scripts\clean_pycache.ps1
#>
[CmdletBinding()]
param(
    # Корень очистки: все вложенные папки __pycache__. По умолчанию — корень
    # проекта (скрипт лежит в <root>\scripts\).
    [string]$Path,

    # Только показать, что будет удалено.
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = if ($Path) { $Path } else { Split-Path -Parent $PSScriptRoot }
if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    Write-Error "Папка не найдена: $root"
    exit 1
}
$root = (Resolve-Path -LiteralPath $root).ProviderPath

Write-Host "Поиск __pycache__ под: $root"

$dirs = @(
    Get-ChildItem -LiteralPath $root -Recurse -Directory -Filter "__pycache__" `
        -Force -ErrorAction SilentlyContinue
)

if ($dirs.Count -eq 0) {
    Write-Host "Нечего чистить: папок __pycache__ нет."
    exit 0
}

$removed = 0
$freed = 0

foreach ($dir in $dirs) {
    $files = @(Get-ChildItem -LiteralPath $dir.FullName -File -Force `
        -ErrorAction SilentlyContinue)
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $bytes) { $bytes = 0 }
    $mb = [math]::Round($bytes / 1MB, 2)

    if ($DryRun) {
        Write-Host ("[dry-run] {0}  ({1} файлов, {2} МБ)" -f $dir.FullName, $files.Count, $mb)
        continue
    }

    try {
        Remove-Item -LiteralPath $dir.FullName -Recurse -Force
        $removed++
        $freed += $bytes
        Write-Host ("Удалена {0}  ({1} файлов, {2} МБ)" -f $dir.FullName, $files.Count, $mb)
    } catch {
        Write-Warning ("Не удалось удалить {0}: {1}" -f $dir.FullName, $_.Exception.Message)
    }
}

if ($DryRun) {
    Write-Host ("Dry-run: будет удалено папок: {0}" -f $dirs.Count)
} else {
    Write-Host ("Готово: удалено {0} из {1} папок; освобождено {2} МБ." -f `
        $removed, $dirs.Count, [math]::Round($freed / 1MB, 2))
}
