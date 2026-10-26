# Builds dist\launcher\PolyVoiceChanger.exe: a ~45 MB launcher that installs the voice changer on first run
# (Python, PyTorch with CUDA and poly-chatterbox from GitHub) into %LOCALAPPDATA%\PolyVoiceChanger, then starts it.
# Usage:  powershell -ExecutionPolicy Bypass -File packaging\build-launcher.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# The tools the launcher carries, pinned and checked against their published SHA-256.
$UvVersion = "0.13.5"
$UvSha256 = "a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a"
$MinGitTag = "v2.56.0.windows.4"
$MinGitFile = "MinGit-2.56.0.4-64-bit.zip"
$MinGitSha256 = "e3bf78351fe777fba47ddee5950082f8a98bb0f02604fdc7ba3ee2cb68ffa877"

$downloads = "packaging\downloads"
New-Item -ItemType Directory -Force $downloads | Out-Null

function Get-Pinned($url, $path, $sha256) {
    if (-not ((Test-Path $path) -and (Get-FileHash $path -Algorithm SHA256).Hash -eq $sha256)) {
        Write-Host "Downloading $url"
        Invoke-WebRequest $url -OutFile $path -UseBasicParsing
        if ((Get-FileHash $path -Algorithm SHA256).Hash -ne $sha256) { throw "Checksum mismatch for $url" }
    }
}

Get-Pinned "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" `
    "$downloads\uv.zip" $UvSha256
Expand-Archive "$downloads\uv.zip" "$downloads\uv" -Force
Copy-Item "$downloads\uv\uv.exe" "$downloads\uv.exe" -Force
Get-Pinned "https://github.com/git-for-windows/git/releases/download/$MinGitTag/$MinGitFile" `
    "$downloads\MinGit.zip" $MinGitSha256
# The launcher re-unpacks its tools when this changes.
Set-Content "$downloads\tools-version.txt" "uv $UvVersion, $MinGitFile" -Encoding ascii

uv run --no-project --python 3.12 --with "pyinstaller>=6.11" pyinstaller --noconfirm `
    --distpath dist\launcher --workpath build\launcher packaging\PolyVoiceChangerSetup.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item build\launcher -Recurse -Force

Copy-Item packaging\README-friends.txt dist\launcher\README.txt
Get-Item dist\launcher\PolyVoiceChanger.exe | Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
