param(
    [Parameter(Mandatory = $true)][string]$Worktree,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$DatasetDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$StatusDir
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir, $StatusDir, (Join-Path $StatusDir "temp") | Out-Null
$env:PYTHONPATH = Join-Path $Worktree "training\whisper\src"
$env:PYTHONUNBUFFERED = "1"
$env:TEMP = Join-Path $StatusDir "temp"
$env:TMP = $env:TEMP

$splitManifest = Join-Path $Worktree "docs\experiments\m31\m31-split-v1.jsonl"
$arguments = @(
    "-m", "daiya_whisper_lora.cli", "train",
    "--dataset-dir", $DatasetDir,
    "--model-name-or-path", "openai/whisper-large-v3",
    "--output-dir", $OutputDir,
    "--split-manifest", $splitManifest,
    "--seed", "42",
    "--num-train-epochs", "2",
    "--per-device-train-batch-size", "2",
    "--per-device-eval-batch-size", "2",
    "--gradient-accumulation-steps", "8",
    "--learning-rate", "2e-5",
    "--warmup-steps", "50",
    "--eval-steps", "100",
    "--save-steps", "100",
    "--logging-steps", "25",
    "--lora-r", "16",
    "--lora-alpha", "32",
    "--lora-dropout", "0.05",
    "--lora-target-modules", "q_proj,k_proj,v_proj,out_proj,fc1,fc2",
    "--load-in-4bit",
    "--fp16",
    "--gradient-checkpointing",
    "--prompt-conditioning",
    "--prompt-max-tokens", "64",
    "--prompt-fields", "context_before",
    "--predict-with-generate",
    "--generation-max-length", "225",
    "--load-best-model-at-end"
)

$launch = [ordered]@{
    launcher_pid = $PID
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    worktree = $Worktree
    python = $Python
    dataset_dir = $DatasetDir
    output_dir = $OutputDir
    split_manifest = $splitManifest
    arguments = $arguments
}
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $StatusDir "launch.json") -Encoding utf8

$stdout = Join-Path $StatusDir "stdout.log"
$stderr = Join-Path $StatusDir "stderr.log"
& $Python @arguments 1>> $stdout 2>> $stderr
$exitCode = $LASTEXITCODE

$completion = [ordered]@{
    launcher_pid = $PID
    completed_at = (Get-Date).ToUniversalTime().ToString("o")
    exit_code = $exitCode
    output_dir = $OutputDir
}
$completion | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $StatusDir "exit.json") -Encoding utf8
exit $exitCode
