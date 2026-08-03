"""Marks this directory as a regular package so `python -m unittest tests.test_noise` resolves HERE.

WHY THIS FILE EXISTS. Without an `__init__.py` this directory is only a PEP 420 namespace
portion, and the import machinery treats that as a last resort: it records the portion, keeps
scanning sys.path, and any *regular* package named `tests` found later wins outright.

Several widely-installed packages ship exactly that. On this project's Windows checkout,
`import tests.test_noise` resolved to ultralytics' bundled suite in site-packages and failed
with `ModuleNotFoundError: No module named 'tests.test_noise'`, while the identical command
worked on the SCC venv purely because nothing there had claimed the name.

The loud failure is the lucky case. `scc/noise_generate.qsub`, `scc/svm_noise.qsub` and
`scc/mert_noise.qsub` run `python -m unittest tests.test_noise -q` as a GATE before generating
or scoring anything. Had the shadowing package happened to contain a module of a matching name,
that gate would have run someone else's passing tests and waved the job through.

One empty-ish file makes the repo's suite a regular package, which wins the path scan outright.
"""
