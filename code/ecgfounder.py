"""ECGFounder (single-lead) for the frontier, self-contained: weights from the Hugging Face Hub,
preprocessing as in the authors' code, embeddings, and a fine-tuning loop sized for a free Colab T4.

    from ecgfounder import load_model, embed, finetune, predict
    fm = load_model('cuda')                                  # downloads 1_lead_ECGFounder.pth (370 MB) once
    E, logits = embed(fm, raw_uv_list)                       # (n, 1024), (n, 150)
    model = finetune(raw_uv_list, y, train_idx, val_idx, n_classes=4, epochs=5)    # end to end, mixed precision
    probs = predict(model, raw_uv_list)                      # (n, n_classes)

Raw inputs are one lead at 300 Hz in microvolts (the packs' *_data). A 10 s window is used for AF and sex.
For apnea, pass the 60 s minute and windows=6: the six 10 s embeddings are mean-and-SD pooled before the head.
net1d.py and the weights are MIT-licensed by PKUDigitalHealth (see LICENSE-ECGFounder).
"""
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch, medfilt, resample_poly

from net1d import Net1D

FS_IN, FS_FM, SECONDS = 300, 500, 10
N_FM = FS_FM * SECONDS
HERE = Path(__file__).resolve().parent
TASKS = (HERE / 'ecgfounder_tasks.txt').read_text().splitlines() if (HERE / 'ecgfounder_tasks.txt').exists() else []


def filter_bandpass(signal, fs):
    """From ECGFounder util.py: 50 Hz notch, 0.67-40 Hz bandpass, median baseline removal."""
    b, a = iirnotch(50, 30, fs)
    out = np.stack([filtfilt(b, a, c) for c in signal])
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    out = np.stack([filtfilt(b, a, c) for c in out])
    k = int(0.4 * fs) + 1
    k += 1 - k % 2
    return out - np.stack([medfilt(c, kernel_size=k) for c in out])


def preprocess(x):
    """300 Hz raw (uV) -> (1, 5000) float32: first 10 s, resampled to 500 Hz, filtered, z-scored."""
    return preprocess_many([x])[0]


def preprocess_many(sigs, windows=1):
    """List of raw signals -> (n, 1, 5000); with windows=w each signal is split into w pieces -> (n, w, 1, 5000).
    Batched (the filters run on a 2-D array), identical to the authors' per-signal code to 1e-5."""
    if windows > 1:
        pieces = [p for s in sigs for p in np.split(np.asarray(s, dtype=float), windows)]
        return preprocess_many(pieces).reshape(len(sigs), windows, 1, N_FM)
    n_in = FS_IN * SECONDS
    lens = [min(len(s), n_in) for s in sigs]
    x = np.stack([np.pad(np.asarray(s, dtype=float)[:n_in], (0, n_in - n)) for s, n in zip(sigs, lens)]) / 1000.0
    x = resample_poly(x, 5, 3, axis=1)[:, :N_FM]
    for i, n in enumerate(lens):                      # a short recording is zero beyond its resampled end, as in the authors' code
        x[i, int(np.ceil(n * 5 / 3)):] = 0.0
    b, a = iirnotch(50, 30, FS_FM)
    x = filtfilt(b, a, x, axis=1)
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=FS_FM)
    x = filtfilt(b, a, x, axis=1)
    k = int(0.4 * FS_FM) + 1
    k += 1 - k % 2
    x = x - np.stack([medfilt(c, kernel_size=k) for c in x])
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    return x[:, None, :].astype(np.float32)


def load_model(device='cuda', weights=None):
    """The 1-lead ECGFounder with its 150-way head; returns (logits, 1024-d features) on forward."""
    weights = weights or os.environ.get('ECGFOUNDER_WEIGHTS')
    if weights is None:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download('PKUDigitalHealth/ECGFounder', '1_lead_ECGFounder.pth')
    model = Net1D(in_channels=1, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                  m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False,
                  use_bn=False, use_do=False, n_classes=150, return_features=True)
    state = torch.load(weights, map_location=device, weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def embed(model, sigs, bs=256, chunk=2048):
    """Frozen embeddings and the 150 statement logits for raw 300 Hz uV signals (first 10 s).
    Works through the signals `chunk` at a time, so memory stays small on Colab."""
    device = next(model.parameters()).device
    E, L = [], []
    model.eval()
    for j in range(0, len(sigs), chunk):
        X = preprocess_many(sigs[j:j + chunk])
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == 'cuda'):
            for i in range(0, len(X), bs):
                logits, f = model(torch.from_numpy(X[i:i + bs]).to(device))
                E.append(f.float().cpu().numpy())
                L.append(logits.float().cpu().numpy())
    return np.concatenate(E), np.concatenate(L)


class ECGFounderClassifier(nn.Module):
    """ECGFounder with a new head. windows>1 pools the mean and SD of the window features first."""

    def __init__(self, n_classes, windows=1, device='cuda', weights=None):
        super().__init__()
        self.net = load_model(device, weights)
        self.net.dense = nn.Identity()
        self.windows = windows
        self.head = nn.Linear(2048 if windows > 1 else 1024, n_classes).to(device)

    def forward(self, x):
        if self.windows > 1:
            b, w = x.shape[:2]
            f = self.net(x.reshape(b * w, 1, -1))[1].reshape(b, w, -1)
            f = torch.cat([f.mean(1), f.std(1)], 1)
        else:
            f = self.net(x)[1]
        return self.head(f)


def predict(model, sigs=None, X=None, bs=64):
    """Class probabilities; pass raw signals (sigs) or preprocessed inputs (X)."""
    device = next(model.parameters()).device
    if X is None:
        X = preprocess_many(sigs, model.windows)
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == 'cuda'):
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + bs]).to(device)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def finetune(sigs, y, train_idx, val_idx, n_classes, windows=1, epochs=3, bs=32, lr=1e-4, device='cuda',
             score=None, X=None, weights=None, verbose=True):
    """Fine-tune end to end with AdamW and mixed precision; keep the epoch with the best validation score
    (default: macro F1 for >2 classes, AUROC for 2). Returns the model. Sized for a T4: bs 32 with
    windows=1 (about 6 GB), bs 8 with windows=6."""
    from sklearn.metrics import f1_score, roc_auc_score
    if score is None:
        score = (lambda yt, p: roc_auc_score(yt, p[:, 1])) if n_classes == 2 else (lambda yt, p: f1_score(yt, p.argmax(1), average='macro'))
    if X is None:
        X = preprocess_many(sigs, windows)
    y = np.asarray(y)
    model = ECGFounderClassifier(n_classes, windows, device, weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=device == 'cuda')
    crit = nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)
    best, t0 = (-1, None), time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            with torch.autocast(device, enabled=device == 'cuda'):
                loss = crit(model(torch.from_numpy(X[b]).to(device)), torch.from_numpy(y[b]).long().to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        s = score(y[val_idx], predict(model, X=X[val_idx], bs=bs))
        if verbose:
            print(f'epoch {ep + 1}/{epochs}: validation {s:.3f}  ({time.time() - t0:.0f} s)', flush=True)
        if s > best[0]:
            best = (s, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])
    return model
