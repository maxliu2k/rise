# AST (Audio Spectrogram Transformer) — on-the-fly extraction plan (pretrained)

**Status:** deferred (build when training this branch). Inputs are NOT materialized to disk
(1024×128 per window × 9623 ≈ 5 GB — extract on the fly instead).

**Input contract:** AST's OWN `ASTFeatureExtractor` — **16 kHz** in, 128 mel bins, AST's own
mean/var normalization. Do not hand-roll the mel params; do not use the Step-6 train stats.

**Flow (in the training DataLoader):**
1. Load the Step-5 normalized 22050 window.
2. `feat = pretrained_extractors.build_ast_extractor()` (once).
3. `x = pretrained_extractors.ast_input(y, feat)` → `input_values` shape (1, 1024, 128).

**Model / fine-tuning:** `build_ast_model()` = `ASTForAudioClassification.from_pretrained(AST_MODEL,
num_labels=9, ignore_mismatched_sizes=True)` — **fine-tune** the pretrained model, not from scratch.
Extraction chosen **on-the-fly** (extractor stays in the training loop) rather than precomputed.

**Deps:** `pip install torch transformers` (downloads `MIT/ast-finetuned-audioset-10-10-0.4593`).

**Noise experiments:** add noise to the 22050 window first, THEN call `ast_input` — identical code path to clean.
