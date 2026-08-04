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
DEMAND="${RISE_DEMAND_ROOT:-/project/rise-grid/maxliu2k/noise-sources/DEMAND}"
VC="${VENV_CORE:-/projectnb/rise-grid/maxliu2k/venv}"
VP="${VENV_PRETRAINED:-/projectnb/rise-grid/maxliu2k/venv_pretrained}"
VM="${VENV_MERT:-/projectnb/rise-grid/maxliu2k/venv_mert}"
HF="${HF_HOME:-/projectnb/rise-grid/maxliu2k/hf_cache}"
MAIL="${RISE_MAIL:-}"                       # set to get SGE mail; see the policy below
STATUS="$REPO/sweep_status.log"

# MAIL POLICY: silence unless something breaks, plus exactly one "it is done".
#
# This used to be `-m abe` on all eight jobs -- abort, BEGIN and end. Eight begin-mails and
# eight end-mails per launch, times two launches on 2026-08-03 (the first aborted six times),
# is roughly twenty messages, none of which carried a decision. Mail you learn to delete
# unread is worse than no mail: the one that matters arrives in the same pile.
#
# So: every worker mails ONLY on abort. sweep_report, the single terminal job, also mails on
# end -- and because it is held on all six evaluations and runs even when they fail, that one
# message is a true "the sweep is over, read docs/NOISE_RESULTS.md" for the whole graph.
mailfail=()                                 # workers: abort only
maildone=()                                 # terminal job: abort + end
if [ -n "$MAIL" ]; then
    mailfail=(-m a  -M "$MAIL")
    maildone=(-m ae -M "$MAIL")
fi

say() { printf '%s\n' "$*" | tee -a "$STATUS"; }

# ---- preflight -------------------------------------------------------------------------------
# Each of these has already cost a run at least once. Fail here, loudly, rather than three hours
# into generation.
fail=0
[ -d "$DATA/pipeline" ]                       || { echo "no pipeline/ under $DATA" >&2; fail=1; }
[ -f "$DATA/pipeline/dataset_freeze.json" ]   || { echo "dataset is not sealed: run pipeline_rebuild.qsub" >&2; fail=1; }

# WHICH CORPORA ARE ACTUALLY NEEDED comes from config, not from this file's assumptions.
#
# On 2026-08-04 this preflight passed and generation died anyway: RISE_DEMAND_ROOT was set in
# the launching shell but never passed to the jobs, and the preflight only knew about ESC-50, so
# nothing checked the path the jobs would actually resolve. Asking config which corpora the grid
# requires means the check cannot drift out of step with the taxonomy again.
needs=$(cd "$REPO" && PYTHONPATH="$REPO/src" python - <<'PY' 2>/dev/null
from instrument_robustness.config import NOISE_TYPES
from instrument_robustness.noise_sweep import ESC50_TARGETS, DEMAND_TARGETS
print("esc50" if any(t in ESC50_TARGETS for t in NOISE_TYPES) else "")
print("demand" if any(t in DEMAND_TARGETS for t in NOISE_TYPES) else "")
PY
)
[ -n "$needs" ] || { echo "could not ask config which noise corpora are required" >&2; fail=1; }

if grep -qx esc50 <<<"$needs"; then
    [ -d "$NOISE/ESC-50-master/audio" ]           || { echo "no ESC-50 audio under $NOISE" >&2; fail=1; }
    [ -f "$NOISE/ESC-50-master/meta/esc50.csv" ]  || { echo "no ESC-50 meta under $NOISE" >&2; fail=1; }
fi
if grep -qx demand <<<"$needs"; then
    # Check the exact path the JOBS will resolve, which is the one this script exports below.
    [ -d "$DEMAND" ] || { echo "no DEMAND corpus at $DEMAND (set RISE_DEMAND_ROOT)" >&2; fail=1; }
    [ -f "$DEMAND/DKITCHEN/ch01.wav" ] || { echo "DEMAND at $DEMAND has no DKITCHEN/ch01.wav" >&2; fail=1; }
fi
avail=$(df -Pk "$DATA" | awk 'NR==2{print int($4/1048576)}')
[ "${avail:-0}" -ge 18 ] || { echo "only ${avail}G free at $DATA; the corpus needs ~15G" >&2; fail=1; }

# Every venv this launcher is about to name, checked BEFORE anything is submitted. On
# 2026-08-03 the scripts defaulted to `$REPO/.venv`, which has never existed: nine jobs were
# accepted, and six of them died -- two at the source line and two only later, at `import
# torch`, after running a whole job under the system python. The scheduler will happily queue
# a graph that cannot possibly run; this is the check that says so in one second instead.
for v in "$VC" "$VP" "$VM"; do
    [ -f "$v/bin/activate" ] || { echo "no venv at $v (expected bin/activate)" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "PREFLIGHT FAILED - nothing submitted" >&2; exit 1; }

say "=== sweep launched $(date) ==="
say "    repo $REPO"
say "    data $DATA   (${avail}G free)"

cd "$REPO" || exit 1
COMMON="RISE_REPO=$REPO,RISE_DATA_ROOT=$DATA,RISE_NOISE_ROOT=$NOISE,RISE_DEMAND_ROOT=$DEMAND"

# ---- 1. generate the corpus ONCE -------------------------------------------------------------
# Every model must read the same realized files or predictions are not paired, and the paired
# bootstrap and cluster sign test in noise_stats.py require pairing. The audit's NOISE-001 found
# PANNs scored against a different corpus than the others, silently invalidating every
# comparison involving it.
# RISE_SKIP_GENERATE=1 reuses the corpus already on disk and submits only the evaluations.
#
# For when generation SUCCEEDED and the evaluations did not -- which is exactly what happened on
# 2026-08-04, when all six aborted on FileExistsError against stale results while the 60,240-file
# corpus sat there complete and validated. Regenerating in that situation burns 86 minutes to
# reproduce bytes that already exist, which CLAUDE.md calls out by name: do not spend compute to
# paper over a tool that cannot express "just the evals".
#
# The evals still verify the manifest themselves, so a corpus that is NOT complete cannot be
# silently scored just because this flag was set.
if [ "${RISE_SKIP_GENERATE:-0}" = "1" ]; then
    [ -f "$DATA/work/windows_noisy/noise_manifest.json" ] || {
        echo "RISE_SKIP_GENERATE=1 but there is no noise manifest at $DATA" >&2; exit 1; }
    gen=""
    say "  noise_generate  SKIPPED (reusing corpus at $DATA/work/windows_noisy)"
else
    gen=$(qsub -terse "${mailfail[@]}" -N noise_generate \
          -v "$COMMON,RISE_VENV=$VC" \
          -o "$REPO/noise_generate.log" scc/noise_generate.qsub)
    say "  noise_generate  $gen"
fi

# ---- 2. one evaluation per model, all held on the corpus -------------------------------------
evals=""
# Every entry names its venv EXPLICITLY. Relying on each script's internal default is what
# broke the 2026-08-03 run: the default was wrong in four scripts and right in two, so the
# two that happened to be passed a venv by this launcher were the only two that activated.
for spec in "svm:scc/svm_noise.qsub:RISE_VENV=$VC" \
            "mert:scc/mert_noise.qsub:RISE_VENV=$VM,HF_HOME=$HF" \
            "cnn:scc/rise_noise_eval.qsub:RISE_MODEL=cnn,RISE_VENV=$VC" \
            "crnn:scc/rise_noise_eval.qsub:RISE_MODEL=crnn,RISE_VENV=$VC" \
            "ast:scc/rise_noise_eval.qsub:RISE_MODEL=ast,RISE_VENV=$VP,HF_HOME=$HF" \
            "panns:scc/rise_noise_eval.qsub:RISE_MODEL=panns,RISE_VENV=$VP"; do
    m="${spec%%:*}"; rest="${spec#*:}"; script="${rest%%:*}"; extra="${rest#*:}"
    vars="$COMMON"; [ -n "$extra" ] && vars="$vars,$extra"
    hold=(); [ -n "$gen" ] && hold=(-hold_jid "$gen")
    jid=$(qsub -terse "${mailfail[@]}" -N "${m}_noise" "${hold[@]}" \
          -v "$vars" -o "$REPO/${m}_noise.log" "$script")
    evals="${evals:+$evals,}$jid"
    say "  ${m}_noise      $jid"
done

# ---- 3. one report, held on all six ----------------------------------------------------------
rep=$(qsub -terse "${maildone[@]}" -N sweep_report -hold_jid "$evals" \
      -v "$COMMON,RISE_VENV=$VC" -o "$REPO/sweep_report.log" scc/sweep_report.qsub)
say "  sweep_report    $rep"
say ""
say "Watch:   qstat -u \$USER"
say "Status:  cat $STATUS"
say "Result:  docs/NOISE_RESULTS.md once sweep_report finishes"
