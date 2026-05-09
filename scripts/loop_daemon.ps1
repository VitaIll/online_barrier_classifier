# Self-exciting agentic loop daemon for online_barrier_classifier.
# Compatible with Windows PowerShell 5.1+ and PowerShell 7+.
#
# Each iteration spawns a fresh `npx claude-code -p` subprocess that executes
# one round per RESEARCH/ROUND_TEMPLATE.md. The daemon is the round-driver;
# Claude Code's in-app scheduler is NOT used (so the loop survives Claude Code
# app close / restart).
#
# Wall-clock timeout per round, exponential back-off on failure, max-
# consecutive-failure circuit-breaker. Heartbeat written to
# RESEARCH/.loop_heartbeat.json so `make loop-status` can report liveness.
#
# Usage:
#   pwsh scripts/loop_daemon.ps1                      # run forever
#   pwsh scripts/loop_daemon.ps1 -MaxRounds 8         # run 8 rounds and exit
#   pwsh scripts/loop_daemon.ps1 -SleepSeconds 30     # 30s gap between rounds
#   pwsh scripts/loop_daemon.ps1 -DryRun              # don't actually spawn claude
#
# Stop with Ctrl+C. The current iteration finishes its round before exit.

[CmdletBinding()]
param(
    [int]$MaxRounds = 0,                       # 0 = unbounded
    [int]$RoundTimeoutSeconds = 1800,          # 30 min hard cap
    [int]$SleepSeconds = 60,                   # delay between rounds
    [int]$BackoffOnFailureSeconds = 300,       # delay after a failed round
    [int]$MaxConsecutiveFailures = 3,          # halt loop after N back-to-back fails
    [string]$RepoRoot = "",
    [string]$PromptFile = "",
    [string]$Effort = "xhigh",
    [string]$PermissionMode = "bypassPermissions",
    [string]$Model = "opus",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"

# Resolve paths AFTER param block (PS 5.1 evaluates default expressions
# inconsistently when invoked via -File; defer until $PSScriptRoot is reliable).
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path $MyInvocation.MyCommand.Path }
if (-not $RepoRoot) { $RepoRoot = (Join-Path $ScriptDir "..") }
if (-not $PromptFile) { $PromptFile = (Join-Path $ScriptDir "round_prompt.md") }
$RepoRoot = (Resolve-Path $RepoRoot).Path
$Heartbeat = Join-Path $RepoRoot "RESEARCH\.loop_heartbeat.json"
$LogDir = Join-Path $RepoRoot ".tmp"
$LogFile = Join-Path $LogDir "loop_daemon.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Heartbeat) | Out-Null

function Write-Hb {
    param($obj)
    $obj | ConvertTo-Json -Depth 4 | Set-Content -Path $Heartbeat -Encoding UTF8
}

function Log {
    param([string]$msg)
    $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    $line = "[$ts] $msg"
    $line | Tee-Object -Append -FilePath $LogFile
}

if (-not (Test-Path $PromptFile)) {
    Log "ERROR: prompt file not found: $PromptFile"
    exit 1
}
$Prompt = Get-Content $PromptFile -Raw

Log "loop_daemon starting"
Log "  repo=$RepoRoot"
Log "  prompt=$PromptFile ($($Prompt.Length) chars)"
Log "  budget=${RoundTimeoutSeconds}s  sleep=${SleepSeconds}s  effort=$Effort  model=$Model  perm=$PermissionMode  dry=$DryRun"

$round = 0
$consecutiveFailures = 0

while ($true) {
    $round += 1
    if ($MaxRounds -gt 0 -and $round -gt $MaxRounds) {
        Log "MaxRounds=$MaxRounds reached; halting"
        Write-Hb @{ status = "halted"; round_index = $round - 1; reason = "max_rounds_reached" }
        break
    }

    $startUtc = [DateTime]::UtcNow
    Write-Hb @{
        status = "running"
        round_index = $round
        started_at = $startUtc.ToString("o")
        timeout_seconds = $RoundTimeoutSeconds
        consecutive_failures = $consecutiveFailures
    }
    Log "round_index=$round starting"

    if ($DryRun) {
        Start-Sleep -Seconds 5
        $exit = 0
    } else {
        # `npx -y @anthropic-ai/claude-code -p ...`
        # -p (print) = headless, exits after the agent stops
        # --add-dir = grant tool access to the repo
        # --permission-mode = bypassPermissions (matches settings.local.json)
        # --effort = xhigh (max reasoning)
        # --setting-sources user,project,local = honor everything in settings*.json
        # The prompt is piped via stdin so we don't shell-escape a large string.
        $cliArgs = @(
            "-y", "@anthropic-ai/claude-code",
            "-p",
            "--add-dir", $RepoRoot,
            "--permission-mode", $PermissionMode,
            "--effort", $Effort,
            "--model", $Model,
            "--setting-sources", "user,project,local",
            "--allow-dangerously-skip-permissions",
            "--verbose"
        )

        $job = Start-Job -ScriptBlock {
            param($promptText, $argsForClaude, $cwd)
            Set-Location $cwd
            $promptText | & npx @argsForClaude 2>&1
            exit $LASTEXITCODE
        } -ArgumentList $Prompt, $cliArgs, $RepoRoot

        $finished = Wait-Job -Job $job -Timeout $RoundTimeoutSeconds
        if (-not $finished) {
            Log "round_index=$round TIMEOUT after ${RoundTimeoutSeconds}s; stopping job"
            Stop-Job -Job $job
            Receive-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
            Remove-Job -Job $job -Force
            $exit = 124
        } else {
            $output = Receive-Job -Job $job -Keep
            $exit = if ($job.State -eq "Completed") { 0 } else { 1 }
            $tail = ($output | Select-Object -Last 30) -join "`n"
            Log "round_index=$round output tail:`n$tail"
            Remove-Job -Job $job -Force
        }
    }

    $endUtc = [DateTime]::UtcNow
    $dt = ($endUtc - $startUtc).TotalSeconds

    if ($exit -eq 0) {
        Log "round_index=$round completed in ${dt}s"
        $consecutiveFailures = 0
        Write-Hb @{
            status = "ok"
            round_index = $round
            started_at = $startUtc.ToString("o")
            finished_at = $endUtc.ToString("o")
            duration_seconds = [math]::Round($dt, 1)
            consecutive_failures = 0
        }
        Log "sleeping ${SleepSeconds}s before next round"
        Start-Sleep -Seconds $SleepSeconds
    } else {
        $consecutiveFailures += 1
        Log "round_index=$round FAILED in ${dt}s (exit=$exit, consecutive=$consecutiveFailures)"
        Write-Hb @{
            status = "fail"
            round_index = $round
            exit_code = $exit
            started_at = $startUtc.ToString("o")
            finished_at = $endUtc.ToString("o")
            duration_seconds = [math]::Round($dt, 1)
            consecutive_failures = $consecutiveFailures
        }
        if ($consecutiveFailures -ge $MaxConsecutiveFailures) {
            Log "$MaxConsecutiveFailures consecutive failures; halting daemon"
            Write-Hb @{
                status = "halted"
                round_index = $round
                reason = "max_consecutive_failures"
                consecutive_failures = $consecutiveFailures
            }
            break
        }
        Log "backing off ${BackoffOnFailureSeconds}s before next round"
        Start-Sleep -Seconds $BackoffOnFailureSeconds
    }
}

Log "loop_daemon exiting"
