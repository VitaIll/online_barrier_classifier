# Self-exciting agentic loop daemon for barrier_classifier.
#
# Runs continuously, invoking one round per iteration via the Claude Code CLI.
# Independent of Claude Code app state (each iteration spawns a fresh subprocess).
# Round budget enforced via wall-clock timeout. Round failures don't kill the loop.
#
# Usage:
#   pwsh scripts/loop_daemon.ps1
#   pwsh scripts/loop_daemon.ps1 -MaxRounds 24    # cap total rounds
#   pwsh scripts/loop_daemon.ps1 -SleepSeconds 30
#   pwsh scripts/loop_daemon.ps1 -DryRun          # don't actually invoke claude
#
# Health: each iteration writes RESEARCH/.loop_heartbeat.json with timestamp,
# round number, status, wall-clock time. `make loop-status` consumes it.

[CmdletBinding()]
param(
    [int]$MaxRounds = 0,                       # 0 = unbounded
    [int]$RoundTimeoutSeconds = 1800,          # 30 min hard cap
    [int]$SleepSeconds = 60,                   # delay between rounds
    [int]$BackoffOnFailureSeconds = 300,       # delay after a failed round
    [int]$MaxConsecutiveFailures = 3,          # halt loop after N back-to-back fails
    [string]$RepoRoot = (Join-Path $PSScriptRoot ".."),
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path $RepoRoot).Path
$Heartbeat = Join-Path $RepoRoot "RESEARCH\.loop_heartbeat.json"
$LogFile = Join-Path $RepoRoot ".tmp\loop_daemon.log"
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

function Write-Hb {
    param($obj)
    $obj | ConvertTo-Json -Depth 4 | Set-Content -Path $Heartbeat -Encoding UTF8
}

function Log {
    param([string]$msg)
    $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    "[$ts] $msg" | Tee-Object -Append -FilePath $LogFile
}

Log "loop_daemon starting; repo=$RepoRoot dry=$DryRun max=$MaxRounds budget=${RoundTimeoutSeconds}s"

$round = 0
$consecutiveFailures = 0
while ($true) {
    $round += 1
    if ($MaxRounds -gt 0 -and $round -gt $MaxRounds) { Log "MaxRounds=$MaxRounds reached; halting"; break }

    $start = Get-Date
    Write-Hb @{ status = "running"; round_index = $round; started_at = $start.ToString("o");
                 timeout_seconds = $RoundTimeoutSeconds; consecutive_failures = $consecutiveFailures }

    Log "round_index=$round starting"
    if ($DryRun) {
        Start-Sleep -Seconds 5
        $exit = 0
        $stdout = "(dry run)"
        $stderr = ""
    } else {
        # Invoke Claude Code CLI in headless mode against the scheduled task prompt.
        # `claude --task <id>` triggers one fire of that task, then exits.
        $claudeExe = Get-Command claude -ErrorAction SilentlyContinue
        if (-not $claudeExe) {
            Log "ERROR: 'claude' CLI not found in PATH"
            Write-Hb @{ status = "error"; round_index = $round; error = "claude CLI not in PATH"; started_at = $start.ToString("o") }
            $consecutiveFailures += 1
        } else {
            $job = Start-Job -ScriptBlock {
                param($exe)
                & $exe.Source --task barrier-classifier-research-round 2>&1
                exit $LASTEXITCODE
            } -ArgumentList $claudeExe
            $finished = Wait-Job -Job $job -Timeout $RoundTimeoutSeconds
            if (-not $finished) {
                Log "round_index=$round TIMEOUT after ${RoundTimeoutSeconds}s; stopping job"
                Stop-Job -Job $job
                Receive-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
                Remove-Job -Job $job -Force
                $exit = 124
                $stdout = ""
                $stderr = "timeout"
            } else {
                $stdout = (Receive-Job -Job $job -Keep) -join "`n"
                $exit = $job.State -eq "Completed" ? 0 : 1
                Remove-Job -Job $job -Force
                $stderr = ""
            }
        }
    }

    $end = Get-Date
    $dt = ($end - $start).TotalSeconds
    if ($exit -eq 0) {
        Log "round_index=$round completed in ${dt}s"
        $consecutiveFailures = 0
        Write-Hb @{ status = "ok"; round_index = $round; started_at = $start.ToString("o");
                     finished_at = $end.ToString("o"); duration_seconds = $dt;
                     consecutive_failures = 0 }
    } else {
        Log "round_index=$round FAILED in ${dt}s (exit=$exit)"
        $consecutiveFailures += 1
        Write-Hb @{ status = "fail"; round_index = $round; exit_code = $exit;
                     started_at = $start.ToString("o"); finished_at = $end.ToString("o");
                     duration_seconds = $dt; consecutive_failures = $consecutiveFailures;
                     stderr_tail = ($stderr | Select-Object -Last 20) }
        if ($consecutiveFailures -ge $MaxConsecutiveFailures) {
            Log "$MaxConsecutiveFailures consecutive failures; halting daemon"
            Write-Hb @{ status = "halted"; round_index = $round; reason = "max_consecutive_failures";
                         consecutive_failures = $consecutiveFailures }
            break
        }
        Log "backing off ${BackoffOnFailureSeconds}s before next round"
        Start-Sleep -Seconds $BackoffOnFailureSeconds
        continue
    }

    Log "sleeping ${SleepSeconds}s before next round"
    Start-Sleep -Seconds $SleepSeconds
}

Log "loop_daemon exiting"
