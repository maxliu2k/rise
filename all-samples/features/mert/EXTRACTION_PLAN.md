# MERT (m-a-p/MERT-v1-95M) — on-the-fly extraction plan (pretrained)

**Status:** deferred (build when training this branch). Inputs are NOT materialized to disk.

**Input contract:** raw waveform @ **24 kHz** via MERT's own processor
(`Wav2Vec2FeatureExtractor`, `trust_remote_code=True`). Not the 22050 set raw; not the Step-6 stats.

**Flow (in the training DataLoader):**
1. Load the Step-5 normalized 22050 window.
2. `proc = pretrained_extractors.build_mert_processor()` (once).
3. `x = pretrained_extractors.mert_input(y, proc)` → `input_values` @ 24 kHz.
4. `model(x, output_hidden_states=True).hidden_states` → 13 layers × (T, 768).

**Model / probing decision (documented):** start with a **frozen-feature probe** — freeze MERT,
mean-pool over time, learn a weighted sum over the 13 hidden layers + a linear 9-way head. Switch to
**fine-tuning** only if the probe plateaus. (`build_mert_model()` returns the frozen backbone.)

**Deps:** `pip install torch torchaudio transformers` (downloads `m-a-p/MERT-v1-95M`).

**Noise experiments:** add noise to the 22050 window first, THEN call `mert_input` — identical code path to clean.
