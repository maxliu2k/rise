# Activate a venv, or kill the job. Sourced by the scc/*.qsub scripts; not executable on its own.
#
# WHY THIS EXISTS, precisely.
#
# `source /nonexistent/bin/activate` prints one line to stderr and returns non-zero. A qsub
# script running under `set -uo pipefail` -- no `-e` -- then CONTINUES, using whatever `python`
# the module system left on PATH. The job does not stop; it changes interpreter.
#
# On 2026-08-03 that turned a single wrong path into six dead jobs. The sweep's core venv is
# `<parent>/venv`, but four scripts defaulted to `$repo/.venv`, which has never existed:
#
#   noise_generate, svm_noise   had `set -e`, so they died AT the source line -- one clear error
#   cnn_noise, crnn_noise       had no `-e`, so they ran a full job under the SYSTEM python and
#                               only fell over later at `import torch`, reporting
#                               `ModuleNotFoundError` -- an error that says nothing about venvs
#
# The second failure mode is the dangerous one: the misleading symptom cost more time than the
# actual bug, and a job that gets far enough to import a *different* build of a library would
# not have crashed at all. So: a wrong or missing interpreter is fatal here, at activation,
# before any of it can matter.
#
# Usage:
#   . "$(dirname "$0")/activate_venv.sh"     # but qsub copies the script, so prefer:
#   . "$noise_repo/scc/activate_venv.sh"
#   rise_activate "$noise_venv"

rise_activate() {
    venv_path="$1"

    if [ ! -f "$venv_path/bin/activate" ]; then
        echo "FATAL: no venv at $venv_path (expected $venv_path/bin/activate)" >&2
        echo "       build one with scc/build_venvs.qsub, or pass -v RISE_VENV=/path/to/venv" >&2
        exit 1
    fi

    # shellcheck disable=SC1091
    . "$venv_path/bin/activate"

    # Activation can "succeed" and still leave another python first on PATH -- a stale module
    # load, or a venv whose activate script was copied without its bin/. Check the thing we
    # actually care about rather than trusting the exit status.
    venv_python="$(command -v python || true)"
    case "$venv_python" in
        "$venv_path"/bin/python*) ;;
        *)  echo "FATAL: sourced $venv_path/bin/activate but python resolves to '${venv_python:-<none>}'" >&2
            exit 1 ;;
    esac

    echo "venv: $venv_path ($(python -V 2>&1))"
}
