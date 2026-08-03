#!/bin/bash
# Submit the whole noise sweep as one dependency graph and return immediately.
#
#   ./scc/launch_sweep.sh
#
# The graph:
#
#   noise_generate ──┬── svm_noise ──┐
#                    ├── mert_noise ─┤
#                    ├── cnn_noise ──┼── sweep_report
#                    ├── crnn_noise ─┤
#                    ├── ast_noise ──┤
#                    └── panns_noise ┘
#
# Nothing here needs a human, a browser session, or an interactive shell. Once submitted it runs
# to completion whether or not anyone is logged in -- that is the entire point of using the
# scheduler rather than holding a session open.
#
# WHY EVERY EVAL IS GATED. `-hold_jid` releases a held job when its predecessor COMPLETES, not
# when it SUCCEEDS. During the 2026-08-02 retrain a pipeline that died in one second released
# three training jobs against the previous build. Each adapter therefore re-checks the corpus
# manifest itself; a failed generate cannot be silently scored.
set -uo pipefail

REPO="${RISE_REPO:-/projectnb/rise-grid/maxliu2k/instrument-robustness}"
DATA="${RISE_DATA_ROOT:-/project/rise-grid/maxliu2k/all-samples}"
NOISE="${RISE_NOISE_ROOT:-/projectnb/rise-grid/noise-sources}"
VP="${VENV_PRETRAINED:-/projectnb/rise-grid/maxliu2k/venv_pretrained}"
VM="${VENV_MERT:-/projectnb/rise-grid/maxliu2k/venv_mert}"
HF="${HF_HOME:-/projectnb/rise-grid/maxliu2k/hf_cache}"
MAIL="${RISE_MAIL:-}"                       # set to get SGE begin/abort/end mail
STATUS="$REPO/sweep_status.log"

mailopt=()
[ -n "$MAIL" ] && mailopt=(-m abe -M "$MAIL")

say() { printf '%s\n' "$*" | tee -a "$STATUS"; }

# ---- preflight -------------------------------------------------------------------------------
# Each of these has already cost a run at least once. Fail here, loudly, rather than three hours
# into generation.
fail=0
[ -d "$DATA/pipeline" ]                       || { echo "no pipeline/ under $DATA" >&2; fail=1; }
[ -f "$DATA/pipeline/dataset_freeze.json" ]   || { echo "dataset is not sealed: run pipeline_rebuild.qsub" >&2; fail=1; }
[ -d "$NOISE/ESC-50-master/audio" ]           || { echo "no ESC-50 audio under $NOISE" >&2; fail=1; }
[ -f "$NOISE/ESC-50-master/meta/esc50.csv" ]  || { echo "no ESC-50 meta under $NOISE" >&2; fail=1; }
avail=$(df -Pk "$DATA" | awk 'NR==2{print int($4/1048576)}')
[ "${avail:-0}" -ge 18 ] || { echo "only ${avail}G free at $DATA; the corpus needs ~15G" >&2; fail=1; }
[ "$fail" -eq 0 ] || { echo "PREFLIGHT FAILED - nothing submitted" >&2; exit 1; }

say "=== sweep launched $(date) ==="
say "    repo $REPO"
say "    data $DATA   (${avail}G free)"

cd "$REPO" || exit 1
COMMON="RISE_REPO=$REPO,RISE_DATA_ROOT=$DATA"

# ---- 1. generate the corpus ONCE -------------------------------------------------------------
# Every model must read the same realized files or predictions are not paired, and the paired
# bootstrap and cluster sign test in noise_stats.py require pairing. The audit's NOISE-001 found
# PANNs scored against a different corpus than the others, silently invalidating every
# comparison involving it.
gen=$(qsub -terse "${mailopt[@]}" -N noise_generate \
      -v "$COMMON,RISE_NOISE_ROOT=$NOISE" \
      -o "$REPO/noise_generate.log" scc/noise_generate.qsub)
say "  noise_generate  $gen"

# ---- 2. one evaluation per model, all held on the corpus -------------------------------------
evals=""
for spec in "svm:scc/svm_noise.qsub:" \
            "mert:scc/mert_noise.qsub:NOISE_VENV=$VM,HF_HOME=$HF" \
            "cnn:scc/rise_noise_eval.qsub:RISE_MODEL=cnn" \
            "crnn:scc/rise_noise_eval.qsub:RISE_MODEL=crnn" \
            "ast:scc/rise_noise_eval.qsub:RISE_MODEL=ast,RISE_VENV=$VP,HF_HOME=$HF" \
            "panns:scc/rise_noise_eval.qsub:RISE_MODEL=panns,RISE_VENV=$VP"; do
    m="${spec%%:*}"; rest="${spec#*:}"; script="${rest%%:*}"; extra="${rest#*:}"
    vars="$COMMON"; [ -n "$extra" ] && vars="$vars,$extra"
    jid=$(qsub -terse "${mailopt[@]}" -N "${m}_noise" -hold_jid "$gen" \
          -v "$vars" -o "$REPO/${m}_noise.log" "$script")
    evals="${evals:+$evals,}$jid"
    say "  ${m}_noise      $jid"
done

# ---- 3. one report, held on all six ----------------------------------------------------------
rep=$(qsub -terse "${mailopt[@]}" -N sweep_report -hold_jid "$evals" \
      -v "$COMMON" -o "$REPO/sweep_report.log" scc/sweep_report.qsub)
say "  sweep_report    $rep"
say ""
say "Watch:   qstat -u \$USER"
say "Status:  cat $STATUS"
say "Result:  docs/NOISE_RESULTS.md once sweep_report finishes"
