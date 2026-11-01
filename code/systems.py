"""The three baseline systems on the three tasks, with every setting in one place.

    python systems.py --data data --out submissions [rf resnet fm]     writes one submission per system

The notebook calls the same functions, so the nine points on the leaderboard are what this file
produces, on a free Colab T4 in about ten minutes (settings in SETTINGS). ECGFounder fine-tuning is in
ecgfounder.finetune for anyone who wants to go further.
Tasks: alivecor (N/A/O/~ on hidden AliveCor; the same model gives P(AF) on PTB-XL), sex (PTB-XL ->
Chapman), apnea (Apnea-ECG minutes -> UCDDB minutes).
"""
import argparse
import copy
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from frontier_kit import CLASSES, alivecor_split, write_submission
from lab2_checks import reference_crop as crop
from model import resnet1d18

FS = 300
A = CLASSES.index('A')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SETTINGS = {
    'resnet': {'alivecor': dict(epochs=20, batch_size=128, length=10 * FS),
               'sex': dict(epochs=3, batch_size=128, length=10 * FS),
               'apnea': dict(epochs=2, batch_size=32, length=None)},
    'fm': {'alivecor': dict(C=0.01), 'sex': dict(C=0.01), 'apnea': dict(C=0.01, windows=6)},
}
f32 = lambda x: np.nan_to_num(np.asarray(x, dtype=np.float32))  # noqa: E731


def macro_f1(y, p):
    """The AF score: macro F1 of the most probable class."""
    return f1_score(y, p.argmax(1), average='macro')


def auroc(y, p):
    """The sex and apnea score: AUROC of the probability of the positive class."""
    return roc_auc_score(y, p[:, 1])


# ---------------------------------------------------------------- data
class Data:
    """The lab's AliveCor data (train/val, 85/15 stratified) and the four packs (train/val; the test sets are hidden)."""

    def __init__(self, data_dir='data'):
        d = Path(data_dir)
        load = lambda n: pickle.load(open(d / n, 'rb'))  # noqa: E731  (the lab's own files: trusted)
        ecg, feat = load('ECG.pkl'), load('ECG_feature.pkl')
        self.keys = list(ecg)
        self.y_af = np.array([CLASSES.index(ecg[k]['label']) for k in self.keys])
        self.raw_af = [np.asarray(ecg[k]['data'], dtype=float) for k in self.keys]
        self.F_af = f32(feat['features'])
        self.tr_af, self.va_af = alivecor_split(self.y_af)
        self.hid, self.ptb, self.sex, self.apn = load('alivecor_hidden.pkl'), load('ptbxl_af.pkl'), load('sex.pkl'), load('apnea.pkl')
        self.cache = d


def tasks(D):
    """Per task: (train inputs by representation, y, train idx, val idx, test inputs by representation, n_classes)."""
    sex, apn = D.sex, D.apn
    return {
        'alivecor': dict(raw=D.raw_af, F=D.F_af, y=D.y_af, tr=D.tr_af, va=D.va_af, n=4,
                         targets={'alivecor': dict(raw=D.hid['data'], F=f32(D.hid['features'])),
                                  'ptbxl_af': dict(raw=D.ptb['data'], F=f32(D.ptb['features']))}),
        'sex': dict(raw=sex['ptbxl_data'], F=f32(sex['ptbxl_features']), y=sex['ptbxl_female'],
                    tr=sex['train'], va=sex['val'], n=2,
                    targets={'sex': dict(raw=sex['chapman_data'], F=f32(sex['chapman_features']))}),
        'apnea': dict(raw=apn['apnea_ecg_data'], F=f32(apn['apnea_ecg_features']), y=apn['apnea_ecg_apnea'],
                      tr=apn['train'], va=apn['val'], n=2,
                      targets={'apnea': dict(raw=apn['ucddb_data'], F=f32(apn['ucddb_features']))}),
    }


def to_files(preds):
    """{file: probability array} -> write_submission kwargs (labels for alivecor, P(A) for ptbxl_af, P(1) otherwise)."""
    out = {}
    for name, p in preds.items():
        out[name] = p.argmax(1) if name == 'alivecor' else (p[:, A] if name == 'ptbxl_af' else p[:, 1])
    return out


# ---------------------------------------------------------------- system 1: RandomForest on the 54 features
def system_rf(D, **rf_kwargs):
    """Returns (predictions for the hidden sets, validation score per task)."""
    preds, val = {}, {}
    for name, t in tasks(D).items():
        sc = StandardScaler().fit(t['F'][t['tr']])
        m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, **rf_kwargs).fit(sc.transform(t['F'][t['tr']]), t['y'][t['tr']])
        score = macro_f1 if t['n'] > 2 else auroc
        val[name] = float(score(t['y'][t['va']], m.predict_proba(sc.transform(t['F'][t['va']]))))
        print(f'rf {name}: validation {val[name]:.3f}', flush=True)
        for file, tgt in t['targets'].items():
            preds[file] = m.predict_proba(sc.transform(tgt['F']))
    return preds, val


# ---------------------------------------------------------------- system 2: ResNet from scratch on the raw signal
def clean(sigs):
    import neurokit2 as nk
    return [nk.ecg_clean(np.asarray(s, dtype=float) / 1000.0, sampling_rate=FS).astype(np.float32) for s in sigs]


# The ResNet pipeline. The notebook's Section 2 shows these four functions cell by cell
# (build_notebooks.py copies them from here, between the "[notebook ...]" markers), with the
# student's own crop() (Task 2A) and predict_proba() (Task 2B) in place of the reference ones.

# [notebook 2.1]
def make_batch(signals, length, train):
    """In: signals, list of 1-D float arrays (cleaned ECG); length, int samples; train, bool.
    Out: float32 tensor (n, 1, length) on `device`. Each signal through crop(), then stacked."""
    windows = [crop(np.asarray(s, dtype=np.float32), length, train) for s in signals]
    windows = np.stack(windows)
    return torch.from_numpy(windows[:, None, :]).to(device)
# [end]

# [notebook 2.2]
def build_model(n_classes, lr=1e-3):
    """In: n_classes, int; lr, float.
    Out: (resnet1d18 with n_classes outputs, Adam optimizer, cross-entropy loss), seeded so a run repeats."""
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = resnet1d18(num_classes=n_classes, input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion
# [end]

# [notebook 2.3]
def predict_proba(model, signals, length, batch_size=256):
    """In: model, resnet1d18; signals, list of 1-D float arrays (cleaned ECG); length, int samples; batch_size, int.
    Out: float array (n, n_classes), class probabilities. First `length` samples, eval mode, no gradients, softmax."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            scores = model(make_batch(signals[i:i + batch_size], length, train=False))
            out.append(torch.softmax(scores, 1).cpu().numpy())
    return np.concatenate(out)
# [end]

# [notebook 2.4]
def train_resnet_classifier(signals, labels, train_idx, val_idx, n_classes,
                            length=None, epochs=20, batch_size=128):
    """In: signals, list of 1-D float arrays (cleaned ECG); labels, int array (n,); train_idx, val_idx, int arrays; n_classes, int;
    length, int samples (None: the longest); epochs, batch_size, int. Out: (trained resnet1d18, predict function, float best validation
    score: macro F1 if n_classes > 2 else AUROC). Each epoch trains on random crops, then scores validation; the best epoch is kept."""
    labels = np.asarray(labels)
    length = length or max(len(s) for s in signals)
    score = macro_f1 if n_classes > 2 else auroc
    model, optimizer, criterion = build_model(n_classes)
    predict = lambda sigs: predict_proba(model, sigs, length)  # noqa: E731

    best_score, best_state = -1.0, None
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        for i in range(0, len(order), batch_size):
            batch_idx = order[i:i + batch_size]
            optimizer.zero_grad()
            batch = make_batch([signals[j] for j in batch_idx], length, train=True)
            targets = torch.from_numpy(labels[batch_idx]).long().to(device)
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()
        val_score = score(labels[val_idx], predict([signals[j] for j in val_idx]))
        print(f'epoch {epoch + 1}/{epochs}: validation {val_score:.3f}', flush=True)
        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, predict, best_score
# [end]


def system_resnet(D, settings=None):
    settings = settings or SETTINGS['resnet']
    preds, val = {}, {}
    for name, t in tasks(D).items():
        print(f'resnet {name}: {settings[name]}', flush=True)
        _, predict, val[name] = train_resnet_classifier(clean(t['raw']), t['y'], t['tr'], t['va'], t['n'], **settings[name])
        for file, tgt in t['targets'].items():
            preds[file] = predict(clean(tgt['raw']))
    return preds, val


# ---------------------------------------------------------------- system 3: ECGFounder (frozen) + linear classifier
def fm_embeddings(model, sigs, windows=1, bs=256):
    """Frozen ECGFounder embeddings from raw signals. windows=w: split each signal into w pieces and
    return the mean and SD of their embeddings (2 x 1,024 numbers)."""
    from ecgfounder import embed
    if windows == 1:
        return embed(model, sigs, bs=bs)[0]
    pieces = [p for s in sigs for p in np.split(np.asarray(s, dtype=float), windows)]
    E = embed(model, pieces, bs=bs)[0].reshape(len(sigs), windows, -1)
    return np.concatenate([E.mean(1), E.std(1)], axis=1)


def system_fm(D, settings=None, device='cuda', weights=None):
    """ECGFounder as a frozen feature extractor: embed the raw signal once, then fit a logistic regression per task."""
    from sklearn.linear_model import LogisticRegression
    from ecgfounder import load_model
    settings = settings or SETTINGS['fm']
    model = load_model(device, weights)
    preds, val = {}, {}

    def cached(tag, sigs, windows):
        p = D.cache / f'fm_embedding_{tag}.npy'
        if not p.exists():
            t0 = time.time()
            np.save(p, fm_embeddings(model, sigs, windows))
            print(f'  embedded {tag}: {len(sigs)} items in {time.time() - t0:.0f} s', flush=True)
        return np.load(p)

    for name, t in tasks(D).items():
        st = dict(settings[name])
        w = st.pop('windows', 1)
        print(f'ecgfounder linear classifier {name}: {settings[name]}', flush=True)
        X = cached(name, t['raw'], w)
        sc = StandardScaler().fit(X[t['tr']])
        m = LogisticRegression(max_iter=3000, **st).fit(sc.transform(X[t['tr']]), t['y'][t['tr']])
        score = macro_f1 if t['n'] > 2 else auroc
        val[name] = float(score(t['y'][t['va']], m.predict_proba(sc.transform(X[t['va']]))))
        print(f'  validation {val[name]:.3f}', flush=True)
        for file, tgt in t['targets'].items():
            preds[file] = m.predict_proba(sc.transform(cached(file + '_target', tgt['raw'], w)))
    return preds, val


SYSTEMS = {'rf': ('system1_rf', 'RandomForest on the 54 Lab 1 features', system_rf),
           'resnet': ('system2_resnet', 'resnet1d18 from scratch on the raw signal', system_resnet),
           'fm': ('system3_ecgfounder', 'ECGFounder frozen embeddings + logistic regression', system_fm)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('systems', nargs='*', default=list(SYSTEMS))
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='submissions')
    ap.add_argument('--weights', default=None, help='local ECGFounder weights (default: download from the HF Hub)')
    ap.add_argument('--author', default='baseline')
    a = ap.parse_args()
    D = Data(a.data)
    for key in a.systems:
        name, notes, fn = SYSTEMS[key]
        t0 = time.time()
        preds, val = fn(D, weights=a.weights) if key == 'fm' else fn(D)
        print(f'{name}: {time.time() - t0:.0f} s', flush=True)
        write_submission(name, author=a.author, notes=notes, out_dir=a.out, **to_files(preds))
