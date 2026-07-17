# PANNs CNN14 — on-the-fly extraction plan (pretrained)

**Status:** deferred (build when training this branch). Inputs are NOT materialized to disk.

**Input contract:** raw waveform @ **32 kHz**. CNN14 computes its own log-mel internally
(window 1024, hop 320, 64 mel bins, fmin 50, fmax 14000) with its own normalization — so it does
**not** use the Step-6 train stats and is **not** fed the 22050 set raw.

**Flow (in the training DataLoader):**
1. Load the Step-5 normalized 22050 window (from `windows.csv`).
2. `wav32 = pretrained_extractors.panns_input(y)`  → float32 waveform @ 32 kHz.
3. Feed `wav32` (batched) to the model from `build_panns_model(ckpt)`.

**Model / fine-tuning:** load pretrained `Cnn14_mAP=0.431.pth`, keep the 2048-d embedding trunk,
replace `fc_audioset` with a 9-way head (`build_panns_model`). Fine-tune trunk + head (or freeze
trunk for a linear probe first).

**Deps:** `pip install torch torchlibrosa panns-inference` and download `Cnn14_mAP=0.431.pth`.

**Noise experiments:** add noise to the 22050 window first, THEN call `panns_input` — identical code path to clean.
