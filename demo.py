import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import linalg as la
import apps as ap

NORMAL, BEATS = set("NLRej"), set("NLRBAaJSVrFejnE/fQ")
PRE, POST, K = 90, 162, 30
MITDB = ("100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 122 123 124 "
         "200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 222 223 228 230 231 232 233 234").split()
NOISES = ("em", "ma", "bw")


def bandpass(x, fs, lo=0.5, hi=40):
    b, a = butter(3, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, x, axis=0)


MIRROR = "https://physionet-open.s3.amazonaws.com/{db}/1.0.0/{f}"


def fetch(db, rec, ann):
    import os, urllib.request, wfdb
    d = os.path.join("data", db)
    if os.path.exists(os.path.join(d, rec + ".dat")):
        return os.path.join(d, rec)
    os.makedirs(d, exist_ok=True)
    files = [rec + ".hea"] + [f"{rec}.{a}" for a in (ann or [])] + [rec + ".dat"]
    try:
        print(f"downloading {db}/{rec} from AWS mirror (one time)...", flush=True)
        for f in files:
            urllib.request.urlretrieve(MIRROR.format(db=db, f=f), os.path.join(d, f))
    except Exception:
        for f in files:
            if os.path.exists(os.path.join(d, f)):
                os.remove(os.path.join(d, f))
        try:
            print("mirror failed, trying PhysioNet...", flush=True)
            wfdb.dl_database(db, d, records=[rec], annotators=ann)
        except Exception:
            have = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".dat"))
            raise ValueError(f"Can't download (no internet?). Already downloaded {db}: {have or 'none'}")
    return os.path.join(d, rec)


def load(rec="119", noise="em", snr_db=6):
    import wfdb
    if rec not in MITDB:
        raise ValueError(f"Record {rec} doesn't exist in MIT-BIH. Pick one of: {' '.join(MITDB)}")
    if noise not in NOISES:
        raise ValueError(f"Noise must be one of: {' '.join(NOISES)}")
    p, q = fetch("mitdb", rec, ["atr"]), fetch("nstdb", noise, None)
    print("loaded, running maths...", flush=True)
    r, ann = wfdb.rdrecord(p), wfdb.rdann(p, "atr")
    nz = wfdb.rdrecord(q).p_signal
    fs, n = r.fs, min(len(r.p_signal), len(nz))
    clean, nz = bandpass(r.p_signal[:n], fs), bandpass(nz[:n], fs)
    a = np.sqrt(np.sum(clean**2, 0) / (np.sum(nz**2, 0) * 10 ** (snr_db / 10)))
    noisy = clean + a * nz
    keep = [(s, c) for s, c in zip(ann.sample, ann.symbol) if c in BEATS and PRE <= s < n - POST]
    r_idx = np.array([s for s, _ in keep])
    y = np.array([c not in NORMAL for _, c in keep])
    cut = lambda S: np.stack([S[s - PRE:s + POST].T.ravel() for s in r_idx])
    return cut(clean), cut(noisy), y, r_idx, fs


def show(title, d):
    print(f"\n== {title}")
    for k, v in d.items():
        print(f"   {k:<24} {v:.3e}" if isinstance(v, float) else f"   {k:<24} {v}")


def main(rec="119", noise="em", snr_db=6.0):
    Xcl, Xn, y, r_idx, fs = load(rec, noise, snr_db)
    M, N = Xn.shape

    Xc, mu = la.center(Xn)
    show("1. DATA MATRIX", {"X shape (M x N)": f"{M} x {N}", "abnormal beats": int(y.sum()),
                            "input SNR dB": float(ap.snr(Xcl, Xn)),
                            "max |col mean of X~|": float(np.abs(Xc.mean(0)).max())})

    C = la.covariance(Xc)
    vals, V, hist = la.power_iteration(C, K)
    ref_vals, _ = la.eig_reference(C)
    k95 = la.k_for_variance(ref_vals, 0.95)
    show("2-3. COVARIANCE + EIGENBASIS", {"C shape": f"{N} x {N}", "k for 95% variance": k95,
                                          "lambda2/lambda1 (conv rate)": float(ref_vals[1] / ref_vals[0]),
                                          "power-iter steps": len(hist)})

    ks = np.arange(1, K + 1)
    gains = np.array([ap.snr_gain(Xcl, Xn, ap.denoise(Xn, mu, V, k)) for k in ks])
    k = int(ks[gains.argmax()])
    show(f"PROOF CHECKS at k={k} (all ~0, cos ~1)", {n: float(v) for n, v in la.checks(C, Xc, vals, V, k).items()})

    Xhat = ap.denoise(Xn, mu, V, k)
    meas, pred = ap.eckart_young(Xc, C, vals, V, k)
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((N, k)))
    rand_err = np.mean(np.sum((Xc - Xc @ Q @ Q.T) ** 2, 1))
    show("4-5. PROJECTION -> DENOISE + COMPRESS", {
        "best k (vs clean)": k, "SNR gain dB": float(gains[k - 1]),
        "PRD noisy %": float(ap.prd(Xcl, Xn)), "PRD denoised %": float(ap.prd(Xcl, Xhat)),
        "compression ratio": float(ap.cr(M, N, k)),
        "Eckart-Young measured": float(meas), "Eckart-Young predicted": float(pred),
        "random k-subspace err": float(rand_err)})

    idx = np.flatnonzero((~y) & (r_idx < 5 * 60 * fs))
    fit, cal = idx[::2], idx[1::2]
    test = r_idx >= 5 * 60 * fs
    Xtr, mu_a = la.center(Xcl[fit])
    va, Va, _ = la.power_iteration(la.covariance(Xtr), K)
    ka = min(la.k_for_variance(la.eig_reference(la.covariance(Xtr))[0], 0.95), K)
    s_te, t_te = ap.spe(Xcl[test] - mu_a, Va, ka), ap.t2(Xcl[test] - mu_a, Va, va, ka)
    thr = ap.threshold(ap.spe(Xcl[cal] - mu_a, Va, ka), 99)
    res = ap.evaluate(y[test], s_te, thr)
    show(f"6. ANOMALY (fit {len(fit)} / calibrate {len(cal)} normal beats, k={ka}, SPE @ 99th pct)",
         {**res, "T2 auc": float(ap.auc(y[test], t_te))})

    t = (np.arange(PRE + POST) - PRE) / fs * 1000
    b = int(np.argmax(~y))
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    ev = ref_vals[:K] / ref_vals.sum()
    ax[0, 0].bar(ks, ev, color="#4c72b0")
    ax[0, 0].plot(ks, np.cumsum(ev), "o-", c="#c44e52", ms=3)
    ax[0, 0].set(title="Scree + cumulative variance", xlabel="component i", ylabel="lambda_i / tr(C)")
    for i in range(3):
        ax[0, 1].semilogy(hist[:, [0, 9, K - 1][i]], label=f"v{[1, 10, K][i]}")
    ax[0, 1].set(title="Power iteration ||Cv - lambda v|| / lambda1", xlabel="iteration")
    ax[0, 1].legend()
    for i in range(3):
        ax[0, 2].plot(V[:, i], label=f"v{i + 1}")
    ax[0, 2].axvline(PRE + POST, c="gray", ls=":")
    ax[0, 2].set(title="Eigen-beats (lead 1 | lead 2)", xlabel="feature")
    ax[0, 2].legend()
    ax[1, 0].plot(t, Xn[b, :PRE + POST], c="#bbbbbb", label="noisy")
    ax[1, 0].plot(t, Xcl[b, :PRE + POST], c="k", label="clean")
    ax[1, 0].plot(t, Xhat[b, :PRE + POST], c="#c44e52", ls="--", label=f"PCA k={k}")
    ax[1, 0].set(title="x^ = mu + V_k V_k^T (x - mu)", xlabel="ms from R-peak", ylabel="mV")
    ax[1, 0].legend()
    ax[1, 1].plot(ks, gains, "o-", c="#4c72b0", ms=3)
    ax[1, 1].axvline(k, c="#c44e52", ls=":")
    ax[1, 1].set(title="SNR gain vs k", xlabel="k", ylabel="dB")
    ax[1, 2].scatter(r_idx[test][~y[test]] / fs / 60, s_te[~y[test]], s=4, c="#4c72b0", label="normal")
    ax[1, 2].scatter(r_idx[test][y[test]] / fs / 60, s_te[y[test]], s=8, c="#c44e52", label="abnormal")
    ax[1, 2].axhline(thr, c="k", ls="--", label="99th pct threshold")
    ax[1, 2].set(yscale="log", title=f"SPE  (AUC {res['auc']:.3f}, F1 {res['f1']:.3f})", xlabel="minutes")
    ax[1, 2].legend(loc="center right", fontsize=8)
    fig.suptitle(f"MIT-BIH {rec}  +  NSTDB '{noise}' @ {snr_db} dB", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"demo_{rec}.png", dpi=130)
    print(f"\nsaved demo_{rec}.png")


if __name__ == "__main__":
    a = sys.argv[1:]
    try:
        main(a[0] if a else "119", a[1] if len(a) > 1 else "em", float(a[2]) if len(a) > 2 else 6.0)
    except ValueError as e:
        print("\nERROR:", e)
