param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8020,
    [switch]$UseFileSearch
)

$ErrorActionPreference = "Stop"

Import-Module Microsoft.PowerShell.Security -ErrorAction Stop

$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$keyPath = Join-Path $projectRoot ".secrets/llm.key"
if (-not (Test-Path -LiteralPath $keyPath)) {
    throw "未找到 DPAPI 密钥文件，请先运行 scripts/set_llm_key.ps1"
}

$encryptedKey = Get-Content -LiteralPath $keyPath -Raw
$secureKey = ConvertTo-SecureString $encryptedKey
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)

try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $env:LLM_API_KEY = $plainKey
    $env:LLM_BASE_URL = "https://api.deepseek.com"
    $env:LLM_MODEL = "deepseek-v4-flash"
    $env:USE_FILE_SEARCH = if ($UseFileSearch) { "true" } else { "false" }
    $env:USE_NEO4J = if ($UseFileSearch) { "false" } else { "true" }
    $env:USE_HYBRID = if ($UseFileSearch) { "false" } else { "true" }
    $env:USE_LLM_PLANNER = "false"
    $env:USE_LLM_QUERY_ANALYZER = "true"
    $env:USE_LLM_RESPONDER = "true"
    $env:PYTHONIOENCODING = "utf-8"

    Push-Location $projectRoot
    & .\.venv\Scripts\python.exe -m uvicorn app.api.main:app `
        --app-dir backend `
        --host $HostAddress `
        --port $Port
}
finally {
    Pop-Location
    $plainKey = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secureKey.Dispose()
}
