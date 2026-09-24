param(
    [string]$OutputPath = ".secrets/llm.key"
)

$ErrorActionPreference = "Stop"

$secureKey = Read-Host "请输入新的 LLM API Key" -AsSecureString
$encryptedKey = ConvertFrom-SecureString $secureKey

$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$output = [IO.Path]::GetFullPath((Join-Path $root $OutputPath))
if (-not $output.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
    throw "输出路径必须位于项目目录内"
}

$directory = Split-Path -Parent $output
New-Item -ItemType Directory -Force -Path $directory | Out-Null
[IO.File]::WriteAllText($output, $encryptedKey)
$secureKey.Dispose()

Write-Output "API Key 已使用 Windows DPAPI 加密保存：$output"
