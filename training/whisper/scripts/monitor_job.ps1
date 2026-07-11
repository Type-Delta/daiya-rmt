param(
    [Parameter(Mandatory = $true)][string]$StatusDir,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$launchPath = Join-Path $StatusDir "launch.json"
$exitPath = Join-Path $StatusDir "exit.json"
$stdoutPath = Join-Path $StatusDir "stdout.log"
$stderrPath = Join-Path $StatusDir "stderr.log"

while ($true) {
    if (Test-Path -LiteralPath $exitPath) {
        $exitState = Get-Content -Raw -LiteralPath $exitPath | ConvertFrom-Json
        [ordered]@{
            status = if ($exitState.exit_code -eq 0) { "completed" } else { "failed" }
            exit = $exitState
            latest_stdout = if (Test-Path $stdoutPath) { @(Get-Content $stdoutPath -Tail 30) } else { @() }
            latest_stderr = if (Test-Path $stderrPath) { @(Get-Content $stderrPath -Tail 30) } else { @() }
        } | ConvertTo-Json -Depth 6
        exit [int]$exitState.exit_code
    }
    if (Test-Path -LiteralPath $launchPath) {
        $launchState = Get-Content -Raw -LiteralPath $launchPath | ConvertFrom-Json
        if ($null -eq (Get-Process -Id ([int]$launchState.launcher_pid) -ErrorAction SilentlyContinue)) {
            [ordered]@{
                status = "launcher_missing_without_exit_record"
                launcher_pid = $launchState.launcher_pid
                latest_stdout = if (Test-Path $stdoutPath) { @(Get-Content $stdoutPath -Tail 30) } else { @() }
                latest_stderr = if (Test-Path $stderrPath) { @(Get-Content $stderrPath -Tail 30) } else { @() }
            } | ConvertTo-Json -Depth 5
            exit 2
        }
    }
    Start-Sleep -Seconds $PollSeconds
}
