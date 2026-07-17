"""On-the-fly input extractors for the PRETRAINED models: PANNs CNN14, AST, MERT.

These are intentionally NOT run during preprocessing. Each pretrained model carries its own
feature extractor / expected sample rate and its own normalization, so:
  * they do NOT use the Step-6 train stats,
  * they are NOT fed the 22050 Hz set raw,
  * materializing their inputs to disk would be many GB and is unnecessary — instead the window
    is resampled + processed on the fly inside the training DataLoader.

Everything starts from the SAME Step-5 normalized 22050 Hz windows (features derived from
work/windows via windows.csv), so clean/noisy stay comparable: for noise experiments, add noise
to the 22050 window FIRST, then call these exact functions.

Deps (install only when training these branches):
    pip install torch torchaudio transformers          # AST, MERT
    pip install panns-inference torchlibrosa            # PANNs CNN14
"""
import numpy as np, librosa
from instrument_robustness.config import SR, AST_SR, AST_MODEL, MERT_SR, MERT_MODEL, TARGET_LABELS

N_CLASSES = len(TARGET_LABELS)


def _resample(y, target_sr):
    return librosa.resample(y, orig_sr=SR, target_sr=target_sr) if target_sr != SR else y


# ----------------------------------------------------------------------------- PANNs CNN14
# Raw waveform @ 32 kHz; the model computes its OWN log-mel internally
# (window 1024, hop 320, 64 mel bins, fmin 50, fmax 14000). Fine-tune: swap the 527-way head.
PANNS_SR = 32000

def panns_input(y):
    """22050 window -> float32 waveform @ 32 kHz (what Cnn14.forward expects, shape (samples,))."""
    return _resample(y, PANNS_SR).astype(np.float32)

def build_panns_model(ckpt_path):
    import torch, torch.nn as nn
    from panns_inference.models import Cnn14
    model = Cnn14(sample_rate=PANNS_SR, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model.fc_audioset = nn.Linear(2048, N_CLASSES)   # new 9-way head; embedding trunk pretrained
    return model


# ----------------------------------------------------------------------------- AST
# AST's own ASTFeatureExtractor: 16 kHz in, 128 bins, its own mean/var normalization.
def build_ast_extractor():
    from transformers import ASTFeatureExtractor
    return ASTFeatureExtractor.from_pretrained(AST_MODEL)

def ast_input(y, extractor):
    """22050 window -> AST input_values (1, 1024, 128) via AST's own extractor @ 16 kHz."""
    y16 = _resample(y, AST_SR)
    return extractor(y16, sampling_rate=AST_SR, return_tensors="pt")["input_values"]

def build_ast_model():
    from transformers import ASTForAudioClassification
    return ASTForAudioClassification.from_pretrained(
        AST_MODEL, num_labels=N_CLASSES, ignore_mismatched_sizes=True)  # fine-tune, not from scratch


# ----------------------------------------------------------------------------- MERT
# MERT consumes raw waveform @ 24 kHz via its own processor; 13 transformer layers of hidden states.
def build_mert_processor():
    from transformers import Wav2Vec2FeatureExtractor
    return Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL, trust_remote_code=True)

def mert_input(y, processor):
    """22050 window -> MERT input_values via its processor @ 24 kHz."""
    y24 = _resample(y, MERT_SR)
    return processor(y24, sampling_rate=MERT_SR, return_tensors="pt")["input_values"]

def build_mert_model():
    from transformers import AutoModel
    # DECISION (documented): frozen-feature probe first — freeze MERT, mean-pool time, learn a
    # weighted sum over the 13 layers + linear head. Switch to fine-tuning only if the probe plateaus.
    return AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True)
