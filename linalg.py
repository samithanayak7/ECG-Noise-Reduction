import numpy as np


def center(X):
    """X~ = X - 1 mu^T"""
    mu = X.mean(axis=0)
    return X - mu, mu


def covariance(Xc):
    """C = X~^T X~ / (M - 1)"""
    return Xc.T @ Xc / (Xc.shape[0] - 1)


def power_iteration(C, k, iters=500, tol=1e-12, oversample=10, seed=0):
    """Top-k eigenpairs of symmetric C by block power iteration + Rayleigh-Ritz.
    Q <- qr(C Q);  Q^T C Q = S L S^T;  V = Q S.  Eigenpair i converges at rate lambda_{p+1}/lambda_i.
    hist[t, i] = ||C v_i - lambda_i v_i|| / lambda_1 at iteration t."""
    p = min(k + oversample, C.shape[0])
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((C.shape[0], p)))[0]
    hist = []
    for _ in range(iters):
        Z = C @ Q
        lam, S = np.linalg.eigh(Q.T @ Z)
        lam, S = lam[::-1], S[:, ::-1]
        V, CV = Q @ S, Z @ S
        hist.append(np.linalg.norm(CV - V * lam, axis=0)[:k] / lam[0])
        if hist[-1].max() < tol:
            break
        Q = np.linalg.qr(CV)[0]
    return lam[:k], V[:, :k], np.array(hist)


def eig_reference(C):
    """numpy eigh, sorted descending"""
    vals, V = np.linalg.eigh(C)
    return vals[::-1], V[:, ::-1]


def svd_reference(Xc):
    """X~ = U S V^T  =>  lambda_i = s_i^2 / (M - 1)"""
    s = np.linalg.svd(Xc, compute_uv=False)
    return s**2 / (Xc.shape[0] - 1)


def projector(V, k):
    """P = V_k V_k^T"""
    Vk = V[:, :k]
    return Vk @ Vk.T


def checks(C, Xc, vals, V, k):
    """Numerical proofs of every identity used. All should be ~0 (or True)."""
    Vk, P = V[:, :k], projector(V, k)
    ref_vals, ref_V = eig_reference(C)
    R = Xc - Xc @ P
    return {
        "centered_max_mean": np.abs(Xc.mean(0)).max(),
        "C_symmetric": np.linalg.norm(C - C.T),
        "C_psd_min_eig": ref_vals[-1],
        "VtV_minus_I": np.linalg.norm(Vk.T @ Vk - np.eye(k)),
        "eig_vs_numpy_rel": np.abs(vals[:k] - ref_vals[:k]).max() / ref_vals[0],
        "vec_vs_numpy_min_cos": np.abs(np.sum(Vk * ref_V[:, :k], axis=0)).min(),
        "eig_vs_svd_rel": np.abs(svd_reference(Xc)[:k] - ref_vals[:k]).max() / ref_vals[0],
        "P2_minus_P": np.linalg.norm(P @ P - P),
        "P_symmetric": np.linalg.norm(P - P.T),
        "trace_P_minus_k": np.trace(P) - k,
        "residual_dot_Vk": np.abs(R @ Vk).max(),
    }


def k_for_variance(vals_all, frac=0.95):
    """smallest k with sum_{i<=k} lambda_i / tr(C) >= frac"""
    return int(np.searchsorted(np.cumsum(vals_all) / vals_all.sum(), frac) + 1)
