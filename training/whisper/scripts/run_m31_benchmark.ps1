param(
    [Parameter(Mandatory = $true)][string]$Worktree,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$DatasetDir,
    [Parameter(Mandatory = $true)][string]$Models,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$StatusDir
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir, $StatusDir, (Join-Path $StatusDir "temp") | Out-Null
$env:PYTHONUNBUFFERED = "1"
$env:TEMP = Join-Path $StatusDir "temp"
$env:TMP = $env:TEMP

$arguments = @(
    (Join-Path $Worktree "lab\asr_eval.py"),
    "--models", $Models,
    "--dataset-dir", $DatasetDir,
    "--manifest", (Join-Path $Worktree "docs\experiments\m31\m31-benchmark-v1.jsonl"),
    "--sample-id-field", "sample_id",
    "--split-manifest", (Join-Path $Worktree "docs\experiments\m31\m31-split-v1.jsonl"),
    "--required-split", "benchmark",
    "--output-dir", $OutputDir,
    "--benchmark-strategies", "isolated,rolling_initial_prompt",
    "--rolling-prompt-turns", "3",
    "--rolling-prompt-chars", "512",
    "--device", "cuda",
    "--compute-type", "int8_float16",
    "--language", "th",
    "--beam-size", "5",
    "--no-condition-on-previous-text",
    "--short-utterance-seconds", "3.0",
    "--bootstrap-samples", "2000",
    "--bootstrap-seed", "1337"
)

[ordered]@{
    launcher_pid = $PID
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    job = "m31_primary_benchmark"
    models = $Models
    arguments = $arguments
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $StatusDir "launch.json") -Encoding utf8

& $Python @arguments 1>> (Join-Path $StatusDir "stdout.log") 2>> (Join-Path $StatusDir "stderr.log")
$exitCode = $LASTEXITCODE
[ordered]@{
    launcher_pid = $PID
    completed_at = (Get-Date).ToUniversalTime().ToString("o")
    exit_code = $exitCode
    output_dir = $OutputDir
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $StatusDir "exit.json") -Encoding utf8
exit $exitCode
