# Model weights

The previous copied checkpoints were removed because they belonged to the superseded dataset
build and, for AST/PANNs, were not the checkpoints behind the reported results.

After all six models are retrained and finalized on the frozen 8,374-source build, run:

```bash
python -m instrument_robustness.bundle_weights
python -m instrument_robustness.bundle_weights --check
```

`EXTERNAL_WEIGHTS.json` preserves the exact identities of the historical AST and PANNs fine-tunes,
including the PANNs release URL and the AST Git-LFS commit/path. They are provenance pointers, not
current models: both checkpoints were trained on the old 8,378-source build and must not be used
for new results.
