#!/usr/bin/env bash
# Drive the remaining long-running stages to completion, then rebuild the analysis.
#
# Every stage is resumable and idempotent: each skips whatever is already in its
# on-disk cache. This script therefore just re-runs them until a pass adds nothing,
# then regenerates the report.
#
# Usage: scripts/finish_pipeline.sh [LYRICS_PID] [NLI_PID] [ACOUSTIC_PID]
# Passing PIDs of already-running jobs makes the script wait for them rather than
# starting competing copies.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LYRICS_PID="${1:-}"
NLI_PID="${2:-}"
ACOUSTIC_PID="${3:-}"
LOG=data/pipeline_supervisor.log

log() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

wait_for_pid() {
  local pid="$1" name="$2"
  [ -z "$pid" ] && return 0
  while ps -p "$pid" > /dev/null 2>&1; do sleep 60; done
  log "$name finished"
}

count() { find "$1" -name '*.json' 2>/dev/null | wc -l | tr -d ' '; }

log "=== supervisor started ==="

# --- lyrics -----------------------------------------------------------------
wait_for_pid "$LYRICS_PID" "in-flight lyrics fetch"
for attempt in 1 2 3; do
  before=$(ls data/cache/lyrics_cache 2>/dev/null | wc -l | tr -d ' ')
  log "lyrics pass $attempt (cached: $before)"
  # --no-api: the search quota is exhausted; verified slug URLs need no API.
  $PY -m music_tastes.fetch_lyrics --no-api --workers 3 >> data/lyrics_fetch2.log 2>&1
  after=$(ls data/cache/lyrics_cache 2>/dev/null | wc -l | tr -d ' ')
  [ "$after" = "$before" ] && break
done
log "lyrics complete"

# --- stance -----------------------------------------------------------------
wait_for_pid "$NLI_PID" "in-flight NLI"
prev=-1
for attempt in $(seq 1 10); do
  now=$(count data/cache/nli)
  log "NLI pass $attempt (scored: $now)"
  [ "$now" = "$prev" ] && { log "no progress; stopping"; break; }
  prev="$now"
  $PY -m music_tastes.stance_nli >> data/nli2.log 2>&1
done
log "NLI complete"

# --- acoustic ---------------------------------------------------------------
wait_for_pid "$ACOUSTIC_PID" "in-flight acoustic"
prev=-1
for attempt in 1 2 3; do
  now=$(count data/cache/mbid)
  log "acoustic pass $attempt (mbids: $now)"
  [ "$now" = "$prev" ] && break
  prev="$now"
  $PY -m music_tastes.enrich_acoustic >> data/acoustic.log 2>&1
done
log "acoustic complete"

# --- analysis ---------------------------------------------------------------
for stage in features-a coverage trends report; do
  log "running $stage"
  $PY -m music_tastes.cli "$stage" >> data/analysis.log 2>&1 || log "$stage FAILED"
done
$PY -m music_tastes.gold_set >> data/analysis.log 2>&1 || log "gold_set FAILED"

log "=== supervisor done -- see reports/findings.md ==="
