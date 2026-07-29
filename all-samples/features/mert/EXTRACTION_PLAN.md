# MERT (m-a-p/MERT-v1-95M) — on-the-fly extraction plan (pretrained)

**Status:** frozen-probe baseline and guarded final evaluation implemented. Raw processor inputs are
not materialized; time-pooled hidden states are cached once under
`features/mert/{train,val}.npz` so the frozen 95M-parameter backbone is not recomputed during every
probe epoch. The Hugging Face model and remote code are pinned to commit
`12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.

**Input contract:** raw waveform @ **24 kHz** via MERT's own processor
(`Wav2Vec2FeatureExtractor`, `trust_remote_code=True`). Not the 22050 set raw; not the Step-6 stats.

**Flow (in the training DataLoader):**
1. Load the Step-5 normalized 22050 window.
2. `proc = pretrained_extractors.build_mert_processor()` (once).
3. `x = pretrained_extractors.mert_batch_input(windows, proc)` → `input_values` @ 24 kHz.
4. `model(x, output_hidden_states=True).hidden_states` → 13 layers × (T, 768).

**Model / probing decision (documented):** start with a **frozen-feature probe** — freeze MERT,
mean-pool over time, learn a weighted sum over the 13 hidden layers + a linear 12-way head. Use
inverse-frequency class weights because the current window counts exceed the configured 1.5
max/min threshold. Switch to **fine-tuning** only if the probe plateaus. `extract_mert` extracts
train/validation embeddings only; `train_mert` tunes the probe learning rate using validation
macro-F1 and never loads test.

After validation selection is frozen, `finalize_mert` refits a fresh probe on train+validation for
the selected best epoch. It then extracts test embeddings with the exact saved MERT model revision
and evaluates test once. A status record and the existence of `test.npz` prevent a second access.

**Deps:** `pip install -e ".[mert]"` installs PyTorch and the MERT-compatible Transformers 4.38
release (then downloads `m-a-p/MERT-v1-95M` on first extraction).

**Noise experiments:** add noise to the 22050 window first, THEN call `mert_input` — identical code path to clean.
