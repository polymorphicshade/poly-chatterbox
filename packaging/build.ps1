# Builds dist\PolyVoiceChanger\PolyVoiceChanger.exe: the live voice changer GUI with torch (CUDA 12.8) bundled.
# Usage:  powershell -ExecutionPolicy Bypass -File packaging\build.ps1 [-Zip]
# -Zip also writes dist\PolyVoiceChanger.zip to hand to friends.
param([switch]$Zip)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

uv run --extra chatterbox --extra stream --with "pyinstaller>=6.11" pyinstaller --noconfirm `
    --distpath dist --workpath build packaging\PolyVoiceChanger.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# build\ only holds PyInstaller's intermediate files, including a PolyVoiceChanger.exe that can't run on its own.
Remove-Item build -Recurse -Force

Copy-Item packaging\README-friends.txt dist\PolyVoiceChanger\README.txt

# Smoke test: load the model, convert a test clip, then report (the EXE has no console; it writes to its log).
$log = Join-Path $env:LOCALAPPDATA "PolyVoiceChanger\voice-changer.log"
$proc = Start-Process dist\PolyVoiceChanger\PolyVoiceChanger.exe -ArgumentList "--check" -Wait -PassThru
Get-Content $log -Tail 5
if ($proc.ExitCode -ne 0) { Write-Error "The --check run failed; see $log" }

if ($Zip) {
    Remove-Item dist\PolyVoiceChanger.zip -ErrorAction SilentlyContinue
    tar.exe -a -c -f dist\PolyVoiceChanger.zip -C dist PolyVoiceChanger
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Wrote dist\PolyVoiceChanger.zip"
}
