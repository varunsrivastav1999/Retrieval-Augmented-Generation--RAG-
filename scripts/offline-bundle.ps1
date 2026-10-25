# Windows (PowerShell) version of scripts/offline-bundle.sh
# Run with internet:  powershell -ExecutionPolicy Bypass -File scripts\offline-bundle.ps1
# On the site:        docker load -i industrial-rag-images.tar ; docker compose up -d
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
docker compose build
docker compose pull postgres qdrant redis
$images = docker compose config --images | Sort-Object -Unique
Write-Host "Saving images:`n$($images -join "`n")"
docker save -o industrial-rag-images.tar $images
Get-Item industrial-rag-images.tar | Select-Object Name, Length
