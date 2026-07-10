param(
    [Parameter(Mandatory = $true)][string]$StatusDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$MilestoneStep = 100,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$checkpoint = Join-Path $OutputDir ("checkpoint-{0}" -f $MilestoneStep)
$exitPath = Join-Path $StatusDir "exit.json"
$stderrPath = Join-Path $StatusDir "stderr.log"
$stdoutPath = Join-Path $StatusDir "stdout.log"

while ($true) {
    if (Test-Path -LiteralPath $exitPath) {
        $exitState = Get-Content -Raw -LiteralPath $exitPath | ConvertFrom-Json
        [ordered]@{
            status = if ($exitState.exit_code -eq 0) { "completed" } else { "failed" }
            exit = $exitState
            latest_stdout = if (Test-Path $stdoutPath) { @(Get-Content $stdoutPath -Tail 20) } else { @() }
            latest_stderr = if (Test-Path $stderrPath) { @(Get-Content $stderrPath -Tail 20) } else { @() }
        } | ConvertTo-Json -Depth 6
        exit [int]$exitState.exit_code
    }
    if (Test-Path -LiteralPath $checkpoint) {
        [ordered]@{
            status = "milestone"
            checkpoint = $checkpoint
            observed_at = (Get-Date).ToUniversalTime().ToString("o")
            latest_stdout = if (Test-Path $stdoutPath) { @(Get-Content $stdoutPath -Tail 20) } else { @() }
            latest_stderr = if (Test-Path $stderrPath) { @(Get-Content $stderrPath -Tail 20) } else { @() }
        } | ConvertTo-Json -Depth 5
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}
