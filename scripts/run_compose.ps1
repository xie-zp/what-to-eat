param(
    [switch]$Rebuild,
    [switch]$UseLLMPlanner
)

$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$keyPath = Join-Path $projectRoot ".secrets/llm.key"
if (-not (Test-Path -LiteralPath $keyPath)) {
    throw "未找到 DPAPI 密钥文件，请先运行 scripts/set_llm_key.cmd"
}

$encryptedKey = Get-Content -LiteralPath $keyPath -Raw
$secureKey = ConvertTo-SecureString $encryptedKey
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)

try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $env:LLM_API_KEY = $plainKey
    $env:LLM_BASE_URL = "https://api.deepseek.com"
    $env:LLM_MODEL = "deepseek-v4-flash"
    $env:USE_LLM_QUERY_ANALYZER = "true"
    $env:USE_LLM_RESPONDER = "true"
    if ($UseLLMPlanner) {
        $env:USE_LLM_PLANNER = "true"
    }
    else {
        $env:USE_LLM_PLANNER = "false"
    }
    $env:USE_HYBRID = "true"

    Push-Location $projectRoot
    try {
        if ($Rebuild) {
            docker compose build backend
        }
        docker compose up -d
        docker compose ps
    }
    finally {
        Pop-Location
    }
}
finally {
    $plainKey = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secureKey.Dispose()
}
