#!/usr/bin/env bash
# Drive the remaining long-running stages to completion unattended.
#
# Both the lyrics fetch and the NLI stance scorer are resumable and idempotent: they
# skip anything already present in their on-disk caches. This script therefore just
# re-runs them until there is nothing left to do, then produces the analysis.
#
# Usage:  scripts/finish_pipeline.sh [LYRICS_PID] [NLI_PID]
# Passing the PIDs of already-running jobs makes the script wait for them rather than
# starting a competing second copy.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LYRICS_PID="${1:-}"
NLI_PID="${2:-}"
LOG=data/pipeline_supervisor.log

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_for_pid() {
  local pid="$1" name="$2"
  [ -z "$pid" ] && return 0
  while ps -p "$pid" > /dev/null 2>&1; do
    sleep 120
  done
  log "$name finished"
}

log "supervisor started"

# 1. Let any in-flight jobs finish before starting our own copies, so we never run
#    two model processes competing for the same GPU.
wait_for_pid "$LYRICS_PID" "in-flight lyrics fetch"
log "running lyrics fetch to completion"
$PY -m music_tastes.fetch_lyrics --workers 4 >> data/lyrics_fetch.log 2>&1
log "lyrics fetch complete"

wait_for_pid "$NLI_PID" "in-flight NLI"

# 2. Score stances. Re-run until a pass adds nothing, since each pass picks up lyrics
#    that landed after the previous pass started.
prev=-1
for attempt in $(seq 1 12); do
  count=$(find data/cache/nli -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
  log "NLI pass $attempt starting (scored so far: $count)"
  if [ "$count" = "$prev" ]; then
    log "no progress since last pass; stopping"
    break
  fi
  prev="$count"
  $PY -m music_tastes.stance_nli >> data/nli.log 2>&1
done
log "NLI scoring complete"

# 3. Analysis. Each stage is cheap and deterministic.
for stage in features-a coverage trends report; do
  log "running $stage"
  $PY -m music_tastes.cli "$stage" >> data/analysis.log 2>&1 || log "$stage FAILED"
done

$PY -m music_tastes.gold_set >> data/analysis.log 2>&1 || log "gold_set FAILED"

log "supervisor done -- see reports/findings.md"
