"""Student kit for the ECG Lab Frontier: load the packs, build a submission, check it offline.
Ranked tasks: alivecor (macro F1), sex (AUROC), apnea (AUROC); rank = their geometric mean.
Every public set is training data with a validation split carved from it; the test sets are the hidden ones.

    from frontier_kit import load, write_submission, check_submission
    hid = load('alivecor_hidden')          # dict: ids, data (raw uV at 300 Hz), features, embeddings, head_logits
    ...                                    # any model you like, trained on the lab data
    write_submission('my_entry', alivecor=labels, ptbxl_af=p_af, sex=p_female, apnea=p_apnea,
                     author='team 3', notes='probe + balanced forest, averaged')
    check_submission('submissions/my_entry.zip')

Packs (put them in data/ next to your notebook, or set PACK_DIR):
  alivecor_hidden.pkl  6,528 PhysioNet 2017 recordings not in ECG.pkl, natural prevalence. Predict N / A / O / ~.
  ptbxl_af.pkl         (optional, not ranked) 4,381 PTB-XL hospital ECGs, lead I, 10 s. Predict P(AF) with your
                       AliveCor model; no PTB-XL AF labels exist for you.
  sex.pkl              PTB-XL with sex labels (ptbxl_female, train/val split by patient) and 1,000 Chapman records to predict
                       P(female).
  apnea.pkl            Apnea-ECG one-minute windows with labels (apnea_ecg_apnea, train/val split by person, 25 % of people) and 3,457 UCDDB
                       minutes to predict P(apnea). Raw minute (18,000 samples), 54 features and a 2,048-number FM summary
                       (mean and SD of its six 10 s window embeddings) per minute.
Fields: *_data (raw lead, int16 microvolts at 300 Hz), *_features (54 Lab 1 features), *_embeddings (ECGFounder,
1,024; float16), *_head_logits (its 150 statement scores; 'tasks' lists them). Everything starts from the raw signal.
"""
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PACK_DIR = Path(os.environ.get('PACK_DIR', 'data'))
CLASSES = ['N', 'A', 'O', '~']
TARGET_IDS = {'alivecor': ('alivecor_hidden', 'ids'), 'ptbxl_af': ('ptbxl_af', 'ids'),
              'sex': ('sex', 'chapman_ids'), 'apnea': ('apnea', 'ucddb_ids')}
_cache = {}


def alivecor_split(labels, val_size=0.25, seed=42):
    """The lab's split of the 2,000 AliveCor recordings: (train, val) indices, stratified by label.
    There is no test split: the test sets are the benchmark's hidden sets."""
    from sklearn.model_selection import train_test_split
    labels = np.asarray(labels)
    if np.issubdtype(labels.dtype, np.integer):            # class codes 0-3 (N, A, O, ~) -> the label strings,
        labels = np.array(CLASSES)[labels]                 # so the split is the same however the labels are encoded
    return train_test_split(np.arange(len(labels)), test_size=val_size, random_state=seed, stratify=labels)


def load(name):
    """Load a pack by name (without .pkl). Pickle runs code when loaded: open only the lab's own files."""
    if name not in _cache:
        with open(PACK_DIR / f'{name}.pkl', 'rb') as f:
            _cache[name] = pickle.load(f)
    return _cache[name]


def ids(task):
    pack, key = TARGET_IDS[task]
    return list(load(pack)[key])


def write_submission(name, alivecor=None, ptbxl_af=None, sex=None, apnea=None, author='', notes='', out_dir='submissions'):
    """Write submissions/<name>/ and submissions/<name>.zip. Pass arrays in the pack's id order:
    alivecor = labels (N/A/O/~ strings or class indices 0-3); the others = P(positive) per id."""
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'entry.json').write_text(json.dumps({'name': name, 'author': author, 'notes': notes}, indent=1))
    parts = {'alivecor': alivecor, 'ptbxl_af': ptbxl_af, 'sex': sex, 'apnea': apnea}
    for task, vals in parts.items():
        if vals is None:
            continue
        vals = np.asarray(vals)
        if task == 'alivecor':
            if np.issubdtype(vals.dtype, np.integer):
                vals = np.array(CLASSES)[vals]
            pd.DataFrame({'id': ids(task), 'label': vals}).to_csv(d / f'{task}.csv', index=False)
        else:
            pd.DataFrame({'id': ids(task), 'score': vals.astype(float)}).to_csv(d / f'{task}.csv', index=False)
    z = Path(out_dir) / f'{name}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in d.iterdir():
            zf.write(p, p.name)
    print(f'wrote {d}/ and {z}')
    return z


def check_submission(path):
    """Validate a submission directory or zip against the public ids (no labels needed)."""
    path = Path(path)
    read = (lambda n: (path / n).read_bytes() if (path / n).exists() else None) if path.is_dir() else None
    if read is None:
        zf = zipfile.ZipFile(path)
        names = {Path(i.filename).name: i.filename for i in zf.infolist()}
        read = lambda n: zf.read(names[n]) if n in names else None  # noqa: E731
    ok = True
    for task in TARGET_IDS:
        raw = read(f'{task}.csv')
        if raw is None:
            print(f'{task:9s} not included')
            continue
        df = pd.read_csv(__import__('io').BytesIO(raw), dtype={'id': str})
        want = ids(task)
        col = 'label' if task == 'alivecor' else 'score'
        problems = []
        if list(df.columns[:2]) != ['id', col]:
            problems.append(f'columns must be id,{col}')
        if df.id.duplicated().any():
            problems.append('duplicate ids')
        if set(df.id) != set(want):
            problems.append(f'{len(set(want) - set(df.id))} ids missing, {len(set(df.id) - set(want))} unknown')
        if task == 'alivecor' and not set(df[col].astype(str)) <= set(CLASSES):
            problems.append('labels must be N, A, O or ~')
        if task != 'alivecor' and not np.isfinite(df[col].astype(float)).all():
            problems.append('scores must be finite')
        print(f'{task:9s} {"ok" if not problems else "PROBLEM: " + "; ".join(problems)} ({len(df)} rows)')
        ok &= not problems
    print('ready to upload' if ok else 'fix the problems above before uploading')
    return ok
