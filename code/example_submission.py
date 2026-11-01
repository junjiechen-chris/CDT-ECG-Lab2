"""A complete submission from the lab's Route C (linear probe on ECGFounder embeddings), as a template.
Run from the lab folder (ECG.pkl etc. in data/, the four packs in data/ too):  python example_submission.py
Everything here is a choice you can change: the model, the features, the training data, the class weights.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from frontier_kit import CLASSES, load, write_submission, check_submission

f32 = lambda x: np.nan_to_num(np.asarray(x, dtype=np.float32))  # noqa: E731


def probe(X_train, y_train, X_test, C=0.01):
    sc = StandardScaler().fit(X_train)
    return LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X_train), y_train).predict_proba(sc.transform(X_test))


# AliveCor: train on all 2,000 lab recordings (the lab itself used 1,400 of them)
with open('data/ECG_fm_embedding.pkl', 'rb') as f:
    fm = pickle.load(f)
y = np.array([CLASSES.index(l) for l in fm['labels']])
hid, ptb = load('alivecor_hidden'), load('ptbxl_af')
p_hid = probe(f32(fm['embeddings']), y, f32(hid['embeddings']))
p_ptb = probe(f32(fm['embeddings']), y, f32(ptb['embeddings']))

# Sex and apnea: train on the labelled part of each pack
sex, apn = load('sex'), load('apnea')
p_sex = probe(f32(sex['ptbxl_embeddings'])[sex['train']], sex['ptbxl_female'][sex['train']], f32(sex['chapman_embeddings']))
p_apn = probe(f32(apn['apnea_ecg_embeddings'])[apn['train']], apn['apnea_ecg_apnea'][apn['train']], f32(apn['ucddb_embeddings']))

z = write_submission('probe_all2000', alivecor=p_hid.argmax(1), ptbxl_af=p_ptb[:, CLASSES.index('A')],
                     sex=p_sex[:, 1], apnea=p_apn[:, 1], author='example', notes='FM probe, trained on all 2,000')
check_submission(z)
