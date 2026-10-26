import numpy as np
from scipy.stats import rankdata


def denoise(X, mu, V, k):
    """x^ = mu + V_k V_k^T (x - mu)"""
    Vk = V[:, :k]
    return mu + (X - mu) @ Vk @ Vk.T


def snr(clean, est):
    """10 log10( ||x||^2 / ||x - x^||^2 )"""
    return 10 * np.log10(np.sum(clean**2) / np.sum((clean - est) ** 2))


def snr_gain(clean, noisy, est):
    return snr(clean, est) - snr(clean, noisy)


def prd(clean, est):
    """percent RMS difference: 100 ||x - x^|| / ||x||"""
    return 100 * np.sqrt(np.sum((clean - est) ** 2) / np.sum(clean**2))


def cr(M, N, k):
    """raw M*N numbers vs stored scores (M k) + basis (N k) + mean (N)"""
    return M * N / (M * k + N * k + N)


def eckart_young(Xc, C, vals, V, k):
    """measured mean ||x~ - P x~||^2  vs  predicted ((M-1)/M) * sum_{i>k} lambda_i"""
    M = Xc.shape[0]
    Vk = V[:, :k]
    measured = np.mean(np.sum((Xc - Xc @ Vk @ Vk.T) ** 2, axis=1))
    predicted = (M - 1) / M * (np.trace(C) - vals[:k].sum())
    return measured, predicted


def spe(Xc, V, k):
    """Q statistic: ||(I - P)(x - mu)||^2 = ||x~||^2 - ||V_k^T x~||^2"""
    Z = Xc @ V[:, :k]
    return np.sum(Xc**2, axis=1) - np.sum(Z**2, axis=1)


def t2(Xc, V, vals, k):
    """Hotelling T^2: sum_i z_i^2 / lambda_i"""
    Z = Xc @ V[:, :k]
    return np.sum(Z**2 / vals[:k], axis=1)


def threshold(scores, q=99):
    return np.percentile(scores, q)


def auc(y, s):
    """Mann-Whitney: P(score_abnormal > score_normal)"""
    r = rankdata(s)
    p, n = y.sum(), (~y).sum()
    return (r[y].sum() - p * (p + 1) / 2) / (p * n)


def roc(y, s):
    """sweep threshold from high to low: (FPR, TPR) at every cut"""
    y = y.astype(bool)[np.argsort(-s)]
    return (np.r_[0, np.cumsum(~y) / (~y).sum()], np.r_[0, np.cumsum(y) / y.sum()])


def evaluate(y, scores, thr):
    y = y.astype(bool)
    pred = scores > thr
    tp, fp = np.sum(pred & y), np.sum(pred & ~y)
    fn, tn = np.sum(~pred & y), np.sum(~pred & ~y)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"auc": auc(y, scores), "precision": prec, "recall": rec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
