$ErrorActionPreference = "Stop"

$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$output = [IO.Path]::GetFullPath(
    (Join-Path $root ".secrets/llm.key")
)
if (-not $output.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
    throw "输出路径必须位于项目目录内"
}

$plainKey = (Get-Clipboard -Raw).Trim()
if (-not $plainKey) {
    throw "剪贴板为空，请先复制新的 API Key"
}
if (-not $plainKey.StartsWith("sk-") -or $plainKey.Length -lt 20) {
    throw "剪贴板内容不像有效 API Key"
}

$secureKey = ConvertTo-SecureString $plainKey -AsPlainText -Force
$plainKey = $null
$encryptedKey = ConvertFrom-SecureString $secureKey

$directory = Split-Path -Parent $output
New-Item -ItemType Directory -Force -Path $directory | Out-Null
[IO.File]::WriteAllText($output, $encryptedKey)
$secureKey.Dispose()
Add-Type -AssemblyName PresentationCore
[System.Windows.Clipboard]::Clear()

Write-Output "API Key 已加密保存到 $output"
