import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import linalg as la
import apps as ap
from demo import load, PRE, POST, K, MITDB, NOISES

app = FastAPI(title="ECG PCA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
S = {}


def js(o):
    if isinstance(o, dict):
        return {k: js(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return js(o.tolist())
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(f"{o:.6g}")
    return o


def need():
    if not S:
        raise HTTPException(400, "call POST /fit first")
    return S


def kk(k):
    return int(np.clip(k, 1, K))


class FitReq(BaseModel):
    record: str = "119"
    noise: str = "em"
    snr_db: float = 6.0


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("index.html")


@app.get("/records")
def records():
    return {"records": MITDB, "noise": NOISES}


@app.post("/fit")
def fit(r: FitReq):
    try:
        Xcl, Xn, y, r_idx, fs = load(r.record, r.noise, r.snr_db)
    except ValueError as e:
        raise HTTPException(400, str(e))
    Xc, mu = la.center(Xn)
    C = la.covariance(Xc)
    vals, V, hist = la.power_iteration(C, K)
    ref = la.eig_reference(C)[0]
    ks = np.arange(1, K + 1)
    D = [ap.denoise(Xn, mu, V, k) for k in ks]
    gains = np.array([ap.snr_gain(Xcl, Xn, d) for d in D])
    prds = np.array([ap.prd(Xcl, d) for d in D])
    del D
    idx = np.flatnonzero((~y) & (r_idx < 5 * 60 * fs))
    fit_i, cal_i, test = idx[::2], idx[1::2], r_idx >= 5 * 60 * fs
    Xtr, mu_a = la.center(Xcl[fit_i])
    Ctr = la.covariance(Xtr)
    va, Va, _ = la.power_iteration(Ctr, K)
    ka = min(la.k_for_variance(la.eig_reference(Ctr)[0]), K)
    S.clear()
    S.update(Xcl=Xcl, Xn=Xn, y=y, r_idx=r_idx, fs=fs, Xc=Xc, mu=mu, C=C, vals=vals, V=V, hist=hist,
             ref=ref, gains=gains, prds=prds, cal=cal_i, test=test, mu_a=mu_a, va=va, Va=Va, ka=ka,
             k95=la.k_for_variance(ref), best_k=int(gains.argmax() + 1), req=r.model_dump())
    return js({"M": Xn.shape[0], "N": Xn.shape[1], "fs": fs, "abnormal": y.sum(), "k95": S["k95"],
               "best_k": S["best_k"], "anomaly_k": ka, "input_snr_db": ap.snr(Xcl, Xn), **S["req"]})


@app.get("/stage/matrix")
def matrix():
    s = need()
    return js({"vars": {"M": s["Xn"].shape[0], "N": s["Xn"].shape[1], "split": PRE + POST},
               "chart": {"heatmap": s["Xn"][:200, ::2], "mean_beat": s["mu"]},
               "checks": {"centered_max_mean": np.abs(s["Xc"].mean(0)).max()}})


@app.get("/stage/covariance")
def covariance():
    s = need()
    C = s["C"]
    return js({"vars": {"shape": C.shape, "trace": np.trace(C)},
               "chart": {"heatmap": C[::2, ::2]},
               "checks": {"C_symmetric": np.linalg.norm(C - C.T), "C_psd_min_eig": s["ref"][-1]}})


@app.get("/stage/eigen")
def eigen():
    s = need()
    ev = s["ref"][:K] / s["ref"].sum()
    c = la.checks(s["C"], s["Xc"], s["vals"], s["V"], s["k95"] if s["k95"] <= K else K)
    keys = ["VtV_minus_I", "eig_vs_numpy_rel", "vec_vs_numpy_min_cos", "eig_vs_svd_rel"]
    sub = np.linspace(0, len(s["y"]) - 1, min(len(s["y"]), 1500)).astype(int)
    return js({"vars": {"k95": s["k95"], "rate": s["ref"][1] / s["ref"][0], "iters": len(s["hist"])},
               "chart": {"cloud": {"z": s["Xc"][sub] @ s["V"][:, :3], "abnormal": s["y"][sub]}, "explained": ev, "cumulative": np.cumsum(ev), "eigenbeats": s["V"][:, :6].T,
                         "convergence": {f"v{i + 1}": s["hist"][:, i] for i in (0, 9, K - 1)}},
               "checks": {k: c[k] for k in keys}})


@app.get("/stage/projection")
def projection(k: int = 5, beat: int | None = None):
    s, k = need(), kk(k)
    b = int(np.argmax(~s["y"])) if beat is None else int(np.clip(beat, 0, len(s["y"]) - 1))
    x = s["Xn"][b] - s["mu"]
    Vk = s["V"][:, :k]
    px = Vk @ (Vk.T @ x)
    c = la.checks(s["C"], s["Xc"], s["vals"], s["V"], k)
    keys = ["P2_minus_P", "P_symmetric", "trace_P_minus_k", "residual_dot_Vk"]
    return js({"vars": {"k": k, "beat": b, "abnormal": s["y"][b], "split": PRE + POST},
               "chart": {"t_ms": (np.arange(PRE + POST) - PRE) / s["fs"] * 1000, "noisy": s["Xn"][b],
                         "clean": s["Xcl"][b], "denoised": s["mu"] + px, "residual": x - px},
               "checks": {k_: c[k_] for k_ in keys}})


@app.get("/stage/denoise")
def denoise(k: int = 5):
    s, k = need(), kk(k)
    M, N = s["Xn"].shape
    meas, pred = ap.eckart_young(s["Xc"], s["C"], s["vals"], s["V"], k)
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((N, k)))[0]
    rand = np.mean(np.sum((s["Xc"] - s["Xc"] @ Q @ Q.T) ** 2, 1))
    return js({"vars": {"k": k, "best_k": s["best_k"], "snr_gain_db": s["gains"][k - 1],
                        "prd_noisy": ap.prd(s["Xcl"], s["Xn"]), "prd_denoised": s["prds"][k - 1],
                        "compression_ratio": ap.cr(M, N, k)},
               "chart": {"k": np.arange(1, K + 1), "snr_gain": s["gains"], "prd": s["prds"],
                         "cr": [ap.cr(M, N, j) for j in range(1, K + 1)]},
               "checks": {"eckart_young_measured": meas, "eckart_young_predicted": pred,
                          "random_subspace_err": rand}})


@app.get("/stage/anomaly")
def anomaly(k: int | None = None, q: float = 99.0, thr: float | None = None):
    s = need()
    k = s["ka"] if k is None else kk(k)
    Xte = s["Xcl"][s["test"]] - s["mu_a"]
    yt = s["y"][s["test"]]
    spe, t2 = ap.spe(Xte, s["Va"], k), ap.t2(Xte, s["Va"], s["va"], k)
    if thr is None:
        thr = ap.threshold(ap.spe(s["Xcl"][s["cal"]] - s["mu_a"], s["Va"], k), q)
    fpr, tpr = ap.roc(yt, spe)
    st = max(1, len(fpr) // 200)
    return js({"vars": {"k": k, "q": q, "threshold": thr},
               "chart": {"minutes": s["r_idx"][s["test"]] / s["fs"] / 60, "spe": spe, "t2": t2,
                         "z2": Xte @ s["Va"][:, :2], "first_test_beat": int(np.argmax(s["test"])),
                         "abnormal": yt, "roc": {"fpr": fpr[::st], "tpr": tpr[::st]}},
               "checks": {**ap.evaluate(yt, spe, thr), "t2_auc": ap.auc(yt, t2)}})
