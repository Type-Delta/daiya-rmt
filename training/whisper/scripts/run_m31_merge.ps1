param(
    [Parameter(Mandatory = $true)][string]$Worktree,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$AdapterPath,
    [Parameter(Mandatory = $true)][string]$MergedOutputDir,
    [Parameter(Mandatory = $true)][string]$Ct2OutputDir,
    [Parameter(Mandatory = $true)][string]$StatusDir
)

$ErrorActionPreference = "Stop"
foreach ($path in @($MergedOutputDir, $Ct2OutputDir)) {
    if ((Test-Path -LiteralPath $path) -and @(Get-ChildItem -LiteralPath $path -Force).Count -gt 0) {
        throw "Refusing to reuse non-empty model output directory: $path"
    }
}
if ((Test-Path -LiteralPath $StatusDir) -and @(Get-ChildItem -LiteralPath $StatusDir -Force).Count -gt 0) {
    throw "Refusing to reuse non-empty status directory: $StatusDir"
}
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

$adapterModel = Join-Path $AdapterPath "adapter_model.safetensors"
$adapterFingerprint = if (Test-Path -LiteralPath $adapterModel) {
    (Get-FileHash -LiteralPath $adapterModel -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$gitCommit = (& git -C $Worktree rev-parse HEAD).Trim()
$runtime = (& $Python -c "import json,platform; import ctranslate2,peft,torch,transformers; print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'ctranslate2':ctranslate2.__version__}))") | ConvertFrom-Json

[ordered]@{
    launcher_pid = $PID
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    job = "m31_merge_and_ct2_conversion"
    worktree = $Worktree
    git_commit = $gitCommit
    python = $Python
    runtime = $runtime
    adapter_path = $AdapterPath
    adapter_model_sha256 = $adapterFingerprint
    arguments = $arguments
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $StatusDir "launch.json") -Encoding utf8

& $Python @arguments 1>> (Join-Path $StatusDir "stdout.log") 2>> (Join-Path $StatusDir "stderr.log")
$exitCode = $LASTEXITCODE
$ct2Model = Join-Path $Ct2OutputDir "model.bin"
[ordered]@{
    launcher_pid = $PID
    completed_at = (Get-Date).ToUniversalTime().ToString("o")
    exit_code = $exitCode
    merged_output_dir = $MergedOutputDir
    ct2_output_dir = $Ct2OutputDir
    ct2_model_sha256 = if (Test-Path -LiteralPath $ct2Model) {
        (Get-FileHash -LiteralPath $ct2Model -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $StatusDir "exit.json") -Encoding utf8
exit $exitCode
