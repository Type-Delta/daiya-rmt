param(
    [Parameter(Mandatory = $true)][string]$Worktree,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$AdapterPath,
    [Parameter(Mandatory = $true)][string]$MergedOutputDir,
    [Parameter(Mandatory = $true)][string]$Ct2OutputDir,
    [Parameter(Mandatory = $true)][string]$StatusDir
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $StatusDir, (Join-Path $StatusDir "temp") | Out-Null
$env:PYTHONPATH = Join-Path $Worktree "training\whisper\src"
$env:PYTHONUNBUFFERED = "1"
$env:TEMP = Join-Path $StatusDir "temp"
$env:TMP = $env:TEMP

$arguments = @(
    "-m", "daiya_whisper_lora.cli", "merge",
    "--adapter-path", $AdapterPath,
    "--base-model", "openai/whisper-large-v3",
    "--merged-output-dir", $MergedOutputDir,
    "--ct2-output-dir", $Ct2OutputDir,
    "--quantization", "int8_float16",
    "--skip-wer",
    "--device", "cuda"
)

[ordered]@{
    launcher_pid = $PID
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    job = "m31_merge_and_ct2_conversion"
    adapter_path = $AdapterPath
    arguments = $arguments
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $StatusDir "launch.json") -Encoding utf8

& $Python @arguments 1>> (Join-Path $StatusDir "stdout.log") 2>> (Join-Path $StatusDir "stderr.log")
$exitCode = $LASTEXITCODE
[ordered]@{
    launcher_pid = $PID
    completed_at = (Get-Date).ToUniversalTime().ToString("o")
    exit_code = $exitCode
    merged_output_dir = $MergedOutputDir
    ct2_output_dir = $Ct2OutputDir
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $StatusDir "exit.json") -Encoding utf8
exit $exitCode
