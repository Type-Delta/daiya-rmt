param(
    [Parameter(Mandatory = $true)][string]$Worktree,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$DatasetDir,
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$StatusDir,
    [ValidateSet("isolated", "rolling-initial-prompt")][string]$PromptStrategy = "rolling-initial-prompt"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir, $StatusDir, (Join-Path $StatusDir "temp") | Out-Null
$env:PYTHONPATH = Join-Path $Worktree "training\whisper\src"
$env:PYTHONUNBUFFERED = "1"
$env:TEMP = Join-Path $StatusDir "temp"
$env:TMP = $env:TEMP

$arguments = @(
    "-m", "daiya_whisper_lora.cli", "probe-checkpoints",
    "--run-dir", $RunDir,
    "--base-model", "openai/whisper-large-v3",
    "--dataset-dir", $DatasetDir,
    "--output-dir", $OutputDir,
    "--split-manifest", (Join-Path $Worktree "docs\experiments\m31\m31-split-v1.jsonl"),
    "--selector-manifest", (Join-Path $Worktree "docs\experiments\m31\m31-generation-gate-v1.jsonl"),
    "--split", "validation",
    "--primary-metric", "micro_cer",
    "--generation-failure-policy", "raise",
    "--prompt-strategy", $PromptStrategy,
    "--prompt-max-tokens", "64",
    "--prompt-fields", "context_before",
    "--rolling-prompt-turns", "3",
    "--rolling-prompt-chars", "512",
    "--device", "cuda",
    "--load-in-4bit",
    "--fp16"
)

[ordered]@{
    launcher_pid = $PID
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    job = "m31_checkpoint_probe"
    prompt_strategy = $PromptStrategy
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
