"""Instrument classification and noise-robustness study.

Modules:
    config      — every tunable constant; change CLASSES to rescope the study
    prep_data   — download, inventory, codec check, cache, grouped split
    train       — medium CNN, multi-seed training, evaluation
    noise_eval  — SNR sweep (currently 2-class-era; see FINDINGS.md)

Run as modules from the repo root, e.g.:
    python -m instrument_robustness.prep_data --inventory-only
"""

__version__ = "0.1.0"
