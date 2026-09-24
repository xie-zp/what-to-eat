$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$output = [IO.Path]::GetFullPath((Join-Path $root ".secrets/llm.key"))
if (-not $output.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
    throw "输出路径必须位于项目目录内"
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "设置 LLM API Key"
$form.Width = 520
$form.Height = 190
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.MinimizeBox = $false

$label = New-Object System.Windows.Forms.Label
$label.Text = "请输入新生成并轮换后的 API Key："
$label.Left = 20
$label.Top = 22
$label.Width = 460
$form.Controls.Add($label)

$textbox = New-Object System.Windows.Forms.TextBox
$textbox.Left = 20
$textbox.Top = 52
$textbox.Width = 460
$textbox.UseSystemPasswordChar = $true
$form.Controls.Add($textbox)

$saveButton = New-Object System.Windows.Forms.Button
$saveButton.Text = "保存"
$saveButton.Left = 300
$saveButton.Top = 96
$saveButton.Width = 85
$saveButton.DialogResult = [System.Windows.Forms.DialogResult]::OK
$form.Controls.Add($saveButton)

$cancelButton = New-Object System.Windows.Forms.Button
$cancelButton.Text = "取消"
$cancelButton.Left = 395
$cancelButton.Top = 96
$cancelButton.Width = 85
$cancelButton.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$form.Controls.Add($cancelButton)

$form.AcceptButton = $saveButton
$form.CancelButton = $cancelButton

$result = $form.ShowDialog()
if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
    $form.Dispose()
    exit 0
}

$plainKey = $textbox.Text.Trim()
$textbox.Clear()
$form.Dispose()

if (-not $plainKey) {
    [System.Windows.Forms.MessageBox]::Show(
        "API Key 不能为空。",
        "保存失败",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Warning
    ) | Out-Null
    exit 1
}

$secureKey = ConvertTo-SecureString $plainKey -AsPlainText -Force
$plainKey = $null
$encryptedKey = ConvertFrom-SecureString $secureKey

$directory = Split-Path -Parent $output
New-Item -ItemType Directory -Force -Path $directory | Out-Null
[IO.File]::WriteAllText($output, $encryptedKey)
$secureKey.Dispose()

[System.Windows.Forms.MessageBox]::Show(
    "API Key 已使用 Windows DPAPI 加密保存。`n`n$output",
    "保存成功",
    [System.Windows.Forms.MessageBoxButtons]::OK,
    [System.Windows.Forms.MessageBoxIcon]::Information
) | Out-Null
