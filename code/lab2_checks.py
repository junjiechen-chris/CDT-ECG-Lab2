"""Checks for the four functions you write in Lab 2.

Each check runs your function on fixed inputs and compares the result with a reference implementation.
If your function passes, the check returns it and the notebook uses it from then on. If it is not
written yet, or gives a different answer, the check says what is wrong and returns the reference
instead, so that the later cells still run. Come back and fix it: the leaderboard needs your own.
"""
import numpy as np


def _done(name, problems, fn, ref):
    if not problems:
        print(f'{name}: check passed (matches the reference). The notebook uses your function from here on.')
        return fn
    print(f'{name}: NOT passed yet.')
    for p in problems:
        print('  -', p)
    print(f'  The notebook uses the reference {name} for now, so later cells still run.')
    return ref


def _call(problems, fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except NotImplementedError:
        problems.append('the function still raises NotImplementedError: write the YOUR CODE part')
    except Exception as e:  # noqa: BLE001  (report any failure, do not stop the notebook)
        problems.append(f'calling it raised {type(e).__name__}: {e}')
    return None


def _show(got, want, rows=3, label=''):
    """Print the first few rows of the student's output next to the reference."""
    fmt = lambda a: np.array2string(np.asarray(a)[:rows], precision=4, suppress_small=True, max_line_width=100)
    print(f'  {label}yours:    {fmt(got)}')
    print(f'  {label}expected: {fmt(want)}')
    got, want = np.asarray(got), np.asarray(want)
    if got.shape != want.shape:
        print(f'  {label}shape: yours {got.shape}, expected {want.shape}')
    elif np.issubdtype(got.dtype, np.number):
        print(f'  {label}max |yours - expected| = {np.abs(got - want).max():.2e}')


# ---------------------------------------------------------------- Task 2A: crop
def reference_crop(x, length, train):
    x = np.asarray(x)
    if len(x) > length:
        start = np.random.randint(0, len(x) - length + 1) if train else 0
        return x[start:start + length]
    return np.pad(x, (0, length - len(x)), 'constant')


def check_crop(crop):
    problems = []
    x = np.arange(5000, dtype=float)
    out = _call(problems, crop, x, 3000, False)
    if out is not None:
        _show(out, x[:3000], rows=8, label='train=False  ')
        if not (len(out) == 3000 and np.array_equal(np.asarray(out), x[:3000])):
            problems.append('with train=False a long signal must give its first `length` samples')
    starts = set()
    for _ in range(30):
        out = _call(problems, crop, x, 3000, True)
        if out is None:
            break
        out = np.asarray(out)
        if len(out) != 3000 or not np.array_equal(out, x[int(out[0]):int(out[0]) + 3000]):
            problems.append('with train=True a long signal must give one contiguous window of `length` samples')
            break
        starts.add(int(out[0]))
    if starts and len(starts) < 2:
        problems.append('with train=True the window should start at a random position (30 calls all started at the same sample)')
    if starts and max(starts) > 2000:
        problems.append('the window runs past the end of the signal')
    short = np.arange(1, 101, dtype=float)
    for train in (False, True):
        out = _call(problems, crop, short, 3000, train)
        if out is None:
            break
        if not train:
            _show(out, np.pad(short, (0, 2900)), rows=8, label='short signal ')
        if not (len(out) == 3000 and np.array_equal(np.asarray(out)[:100], short)
                and not np.asarray(out)[100:].any()):
            problems.append(f'a short signal must be padded with zeros at the end to `length` (train={train})')
            break
    exact = np.arange(3000, dtype=float)
    out = _call(problems, crop, exact, 3000, True)
    if out is not None and not np.array_equal(np.asarray(out), exact):
        problems.append('a signal of exactly `length` samples must come back unchanged')
    return _done('crop', list(dict.fromkeys(problems)), crop, reference_crop)


# ---------------------------------------------------------------- Task 2B: predict_proba
def reference_predict_proba(model, signals, length, batch_size=256):
    import torch
    from systems import make_batch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            scores = model(make_batch(signals[i:i + batch_size], length, train=False))
            out.append(torch.softmax(scores, 1).cpu().numpy())
    return np.concatenate(out)


def check_predict_proba(predict_proba):
    """A small untrained network on fixed random signals: your probabilities must match the reference."""
    import torch
    from model import resnet1d18
    from systems import device
    problems = []
    torch.manual_seed(1)
    model = resnet1d18(num_classes=4, input_channels=1).to(device)
    rng = np.random.default_rng(1)
    signals = [rng.standard_normal(n).astype(np.float32) for n in (3000, 4500, 1000, 3000, 2000)]
    want = reference_predict_proba(model, signals, 3000)
    got = _call(problems, predict_proba, model, signals, 3000)
    if got is not None:
        got = np.asarray(got)
        _show(got, want)
        if got.shape != want.shape:
            problems.append(f'shape {got.shape}, expected {want.shape}: one row per signal, one column per class')
        elif not np.allclose(got.sum(1), 1, atol=1e-4):
            problems.append('rows must sum to 1: apply a softmax to the scores')
        elif not np.allclose(got, want, atol=1e-4):
            problems.append('probabilities differ from the reference: use the first `length` samples (crop with '
                            'train=False, so make_batch(..., train=False)), the model in eval mode, no gradients')
    return _done('predict_proba', list(dict.fromkeys(problems)), predict_proba, reference_predict_proba)


# ---------------------------------------------------------------- Task 3A: pool_windows
def reference_pool_windows(E, windows):
    E = np.asarray(E)
    E = E.reshape(len(E) // windows, windows, -1)
    return np.concatenate([E.mean(axis=1), E.std(axis=1)], axis=1)


def check_pool_windows(pool_windows):
    problems = []
    rng = np.random.default_rng(0)
    E = rng.standard_normal((12, 5))
    out = _call(problems, pool_windows, E, 3)
    if out is not None:
        out = np.asarray(out)
        want = reference_pool_windows(E, 3)
        _show(out, want)
        if out.shape != want.shape:
            problems.append(f'shape {out.shape}, expected {want.shape}: one row per item, [mean, SD] side by side')
        elif not np.allclose(out, want, atol=1e-6):
            if np.allclose(out[:, :5], want[:, :5]):
                problems.append('the mean half is right; the SD half differs (use the population SD, ddof=0, over the windows of one item)')
            else:
                problems.append('values differ: rows i*windows ... (i+1)*windows-1 of E belong to item i; mean first, then SD')
    return _done('pool_windows', problems, pool_windows, reference_pool_windows)


# ---------------------------------------------------------------- Task 3B: linear_classifier
def reference_linear_classifier(E_train, y_train, E_eval):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(E_train)
    m = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(E_train), y_train)
    return m.predict_proba(sc.transform(E_eval))


def check_linear_classifier(linear_classifier):
    problems = []
    rng = np.random.default_rng(1)
    E_tr = rng.standard_normal((300, 20)) * rng.uniform(0.1, 50, 20)
    y_tr = rng.integers(0, 3, 300)
    E_tr[y_tr == 1, :3] += 20
    E_ev = rng.standard_normal((40, 20)) * rng.uniform(0.1, 50, 20)
    out = _call(problems, linear_classifier, E_tr, y_tr, E_ev)
    if out is not None:
        out = np.asarray(out)
        want = reference_linear_classifier(E_tr, y_tr, E_ev)
        _show(out, want)
        if out.shape != want.shape:
            problems.append(f'shape {out.shape}, expected {want.shape}: class probabilities, one row per item of E_eval')
        elif not np.allclose(out, want, atol=1e-4):
            problems.append('probabilities differ: standardise with a scaler fitted on E_train only, then LogisticRegression(C=0.01, max_iter=3000)')
    return _done('linear_classifier', problems, linear_classifier, reference_linear_classifier)
