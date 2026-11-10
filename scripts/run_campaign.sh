#!/usr/bin/env bash
# scripts/run_campaign.sh — from a set-up node to a finished full sweep, in one command.
#
#   bash scripts/run_campaign.sh start  <DEVICE> [--trials N] [--workloads B1,C3] [--recalibrate]
#                                                [--skip-tests] [--dry-run]
#   bash scripts/run_campaign.sh status <DEVICE>     progress, counts by status, rough ETA
#   bash scripts/run_campaign.sh resume <DEVICE>     continue the saved campaign (finished jobs are skipped)
#   bash scripts/run_campaign.sh stop   <DEVICE>     stop a running campaign (resume later)
#
# `start` detaches itself (setsid + nohup, stdin from /dev/null): closing ssh or letting the laptop
# sleep does not stop it. Steps, each stopping the run if it fails:
#   1. machine + git state      2. unit tests      3. smoke test (any error / validation failure stops)
#   4. calibration (only if results/calibration/<DEVICE>.json is missing, or --recalibrate)
#   5. full sweep (fixed campaign id, so it can be resumed)      6. summary of row statuses
# Files: results/logs/<DEVICE>.log (everything), results/logs/<DEVICE>.campaign (saved settings).
# <DEVICE> is a file name in configs/devices/ (without .json); `python3 -m bench describe` helps.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

die()  { echo "run_campaign: $*" >&2; exit 2; }
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }
CMD="${1:-}"; DEV="${2:-}"
[ -n "$CMD" ] && [ -n "$DEV" ] || die "usage: $0 start|status|resume|stop <DEVICE> [options]"
shift 2
CFG="configs/devices/$DEV.json"
[ -f "$CFG" ] || die "$CFG not found (existing: $(ls configs/devices | sed 's/\.json$//' | tr '\n' ' '))"
LOGDIR=results/logs; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$DEV.log"; STATE="$LOGDIR/$DEV.campaign"; PIDF="$LOGDIR/$DEV.pid"
CLASS="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['device_class'])" "$CFG")"
SIDE="$CLASS"; SMOKE_DEV=$([ "$CLASS" = gpu ] && echo cuda || echo cpu)

running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

# counts of this campaign's rows by status (results/data/<side>/<DEVICE>/*.csv)
count_rows() {
    python3 - "$1" "$SIDE" "$DEV" "${BENCH_OUT:-results/data}" <<'PY'
import csv, glob, sys, collections
camp, side, dev, root = sys.argv[1:5]
c = collections.Counter()
for f in glob.glob(f"{root}/{side}/{dev}/*.csv"):
    for r in csv.DictReader(open(f)):
        if r["run_id"].startswith(camp + "-"):
            c[r["status"]] += 1
            if "unstable" in (r.get("flag") or ""):
                c["(unstable)"] += 1
print("  ".join(f"{k}={v}" for k, v in sorted(c.items())) or "no rows yet")
PY
}

case "$CMD" in
start)
    TRIALS=3; WORKLOADS=""; RECAL=0; SKIP_TESTS=0; DRY=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --trials) TRIALS="$2"; shift 2 ;;
            --workloads) WORKLOADS="$2"; shift 2 ;;
            --recalibrate) RECAL=1; shift ;;
            --skip-tests) SKIP_TESTS=1; shift ;;
            --dry-run) DRY=1; shift ;;
            *) die "unknown option $1" ;;
        esac
    done
    if [ "$DRY" = 1 ]; then
        exec python3 -m bench sweep --device-config "$CFG" --trials "$TRIALS" ${WORKLOADS:+--workloads "$WORKLOADS"} --dry-run
    fi
    running && die "a campaign is already running for $DEV (pid $(cat "$PIDF")): use status / stop"
    CAMPAIGN="c_$(echo "$DEV" | tr 'A-Z' 'a-z')_$(date -u +%Y%m%dT%H%MZ)"
    printf 'CAMPAIGN=%s\nTRIALS=%s\nWORKLOADS=%s\nRECAL=%s\nSKIP_TESTS=%s\n' \
        "$CAMPAIGN" "$TRIALS" "$WORKLOADS" "$RECAL" "$SKIP_TESTS" > "$STATE"
    setsid -f nohup bash "$0" _run "$DEV" full >> "$LOG" 2>&1 < /dev/null
    echo "started campaign $CAMPAIGN for $DEV in the background (safe to close ssh / sleep the laptop)"
    echo "  progress : bash scripts/run_campaign.sh status $DEV"
    echo "  live log : tail -f $LOG"
    ;;
resume)
    [ -f "$STATE" ] || die "no saved campaign for $DEV ($STATE): use start"
    running && die "already running (pid $(cat "$PIDF"))"
    setsid -f nohup bash "$0" _run "$DEV" resume >> "$LOG" 2>&1 < /dev/null
    echo "resuming $(sed -n 's/^CAMPAIGN=//p' "$STATE") for $DEV in the background; status: bash scripts/run_campaign.sh status $DEV"
    ;;
stop)
    running || die "nothing running for $DEV"
    PID="$(cat "$PIDF")"
    kill -- -"$PID" 2>/dev/null || kill "$PID"
    echo "=== stopped by user $(stamp)" >> "$LOG"
    echo "stopped $DEV (pid $PID); continue later with: bash scripts/run_campaign.sh resume $DEV"
    ;;
status)
    [ -f "$STATE" ] || die "no campaign started for $DEV"
    . "$STATE"
    echo "campaign : $CAMPAIGN   (trials=$TRIALS${WORKLOADS:+, workloads=$WORKLOADS})"
    if running; then echo "state    : RUNNING (pid $(cat "$PIDF"))"
    elif grep -q "=== campaign end" "$LOG" 2>/dev/null; then echo "state    : FINISHED ($(grep "=== campaign end" "$LOG" | tail -1 | sed 's/=== //'))"
    else echo "state    : NOT RUNNING (stopped or failed: see the log; resume with: bash scripts/run_campaign.sh resume $DEV)"; fi
    echo "step     : $(grep "^=== step" "$LOG" | tail -1 | sed 's/=== //')"
    python3 - "$LOG" <<'PY'
import re, sys, time, os
txt = open(sys.argv[1], errors="replace").read()
m = list(re.finditer(r"\[sweep\] campaign=\S+ device=\S+ jobs/trial=(\d+) trials=(\d+)", txt))
if m:
    total = int(m[-1].group(1)) * int(m[-1].group(2))
    done = len(re.findall(r"^\[t\d+ \d+/\d+\]", txt[m[-1].start():], re.M))
    starts = re.findall(r"=== step 5/6 sweep \(started (\d+)\)", txt)
    line = f"progress : {done}/{total} jobs this session"
    if starts and done:
        el = time.time() - int(starts[-1])
        line += f", elapsed {el/3600:.1f} h, ETA ~{el/done*(total-done)/3600:.1f} h (resumed jobs are skipped)"
    print(line)
PY
    echo "rows     : $(count_rows "$CAMPAIGN")"
    echo "--- last log lines ---"; tail -n 4 "$LOG" | cut -c1-200
    ;;
_run)
    MODE="${1:-full}"
    . "$STATE"
    echo $$ > "$PIDF"
    trap 'rm -f "$PIDF"' EXIT
    echo "=== campaign $CAMPAIGN ($MODE) $(stamp) on $(hostname) ==="
    echo "=== step 1/6 machine: $(python3 -c 'from bench import devices; print(devices.cpu_model())') | class=$CLASS"
    [ "$CLASS" = gpu ] && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
    if [ -n "$(git status --porcelain -- bench analysis configs scripts Makefile 2>/dev/null)" ]; then
        echo "WARNING: uncommitted changes in the code: rows will be flagged dirty_tree"
    fi
    if [ "$MODE" = full ]; then
        if [ "$SKIP_TESTS" = 1 ]; then echo "=== step 2/6 tests: skipped (--skip-tests)"
        else
            echo "=== step 2/6 tests"
            python3 -m pytest -q tests || { echo "=== STOP: unit tests failed $(stamp)"; exit 1; }
        fi
        echo "=== step 3/6 smoke"
        SMOKE_OUT="results/data/smoke/$CAMPAIGN"
        python3 -m bench smoke --device "$SMOKE_DEV" --out "$SMOKE_OUT" || true
        BAD="$(python3 - "$SMOKE_OUT" <<'PY'
import csv, glob, sys
bad = [f"{r['workload_id']} {r['status']} {r.get('validation','')}" for f in glob.glob(f"{sys.argv[1]}/**/*.csv", recursive=True)
       for r in csv.DictReader(open(f)) if r["status"] in ("error", "validation_failed")]
print("\n".join(bad))
PY
)"
        if [ -n "$BAD" ]; then echo "$BAD"; echo "=== STOP: smoke test has errors or validation failures $(stamp)"; exit 1; fi
        echo "smoke ok: $(python3 - "$SMOKE_OUT" <<'PY'
import csv, glob, sys, collections
c = collections.Counter(r["status"] for f in glob.glob(f"{sys.argv[1]}/**/*.csv", recursive=True) for r in csv.DictReader(open(f)))
print("  ".join(f"{k}={v}" for k, v in sorted(c.items())))
PY
)"
        if [ "$RECAL" = 1 ] || [ ! -f "results/calibration/$DEV.json" ]; then
            echo "=== step 4/6 calibration"
            python3 scripts/calibrate.py --device-config "$CFG" || { echo "=== STOP: calibration failed $(stamp)"; exit 1; }
        else
            echo "=== step 4/6 calibration: using existing results/calibration/$DEV.json (--recalibrate to redo)"
        fi
    else
        echo "=== resume: skipping tests, smoke and calibration"
    fi
    echo "=== step 5/6 sweep (started $(date +%s))"
    rc=0
    python3 -m bench sweep --device-config "$CFG" --trials "$TRIALS" --campaign "$CAMPAIGN" \
        ${WORKLOADS:+--workloads "$WORKLOADS"} || rc=$?
    echo "=== step 6/6 summary: $(count_rows "$CAMPAIGN")"
    echo "=== campaign end $(stamp) rc=$rc (rc=1 means at least one job crashed: check 'rc=' lines above; resume re-runs only those)"
    ;;
*) die "unknown command $CMD (start|status|resume|stop)" ;;
esac
