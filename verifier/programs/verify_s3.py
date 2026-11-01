"""Independent check of the 3-adic basis of S (g13/K3.npy) from the definition of S, as in the precision lemma.

S is the image of the kernel of A, the 3125 x 854 integer matrix of period numerators of the 427 forms of F^3 on the
Pham cycles (prod_k (z^{(b_k+1) a_k} - z^{b_k a_k}), split over the basis (1, z) of Z[z], z = e^{2 pi i/6}). The
checks: every row of K3 is in the kernel modulo 3^15; the rows are independent modulo 3 (2271 = 3125 - 854); the
Smith valuations of A over Z_3 (rank 854, all <= 3, so the lemma gives the saturated kernel modulo 3^12); the Smith
valuations of the Gram matrix K3 Psi K3^T (the 3-adic discriminant of S used in Proposition 3.2). Products in float64
only on integers whose sums stay below 2^53 (asserted)."""
import itertools

import numpy as np
from sage.all import GF, matrix

p, KP = 3, 15
mod = p ** KP
EXACT = 2.0 ** 52
Ib = np.array(list(itertools.product(range(5), repeat=5)), dtype=np.int64)
I1 = [b for b in Ib.tolist() if (sum(b) + 5) % 6 != 0 and (sum(b) + 5) < 12]


def zmul(x, y):
    """(x0 + x1 z)(y0 + y1 z) with z^2 = z - 1."""
    return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0] + x[1] * y[1])


ZP = [(1, 0)]
for _ in range(5):
    ZP.append(zmul(ZP[-1], (0, 1)))                     # z^0 .. z^5


def zpow(e):
    return ZP[e % 6]


A = np.zeros((3125, 2 * len(I1)), dtype=np.int64)
for j, bp in enumerate(I1):
    a = [t + 1 for t in bp]
    for i, b in enumerate(Ib.tolist()):
        val = (1, 0)
        for k in range(5):
            u, w = zpow((b[k] + 1) * a[k]), zpow(b[k] * a[k])
            val = zmul(val, (u[0] - w[0], u[1] - w[1]))
        A[i, j], A[i, len(I1) + j] = val
K3 = np.asarray(kairos_data("g13/K3.npy")).astype(np.int64)
assert np.abs(K3).max() * np.abs(A).max() * 3125 < EXACT
kernel_mod_3_15 = bool(not np.any(np.rint(K3.astype(np.float64) @ A.astype(np.float64)).astype(np.int64) % mod))
rank_f3 = int(matrix(GF(3), (K3 % 3).tolist()).rank())


def smith_vals(M, K):
    md = p ** K
    M = M % md
    m, n = M.shape
    rows_alive = np.ones(m, dtype=bool)
    cols_alive = np.ones(n, dtype=bool)
    vals, v = [], 0
    while v < K:
        sub = M[np.ix_(rows_alive, cols_alive)]
        if sub.size == 0:
            break
        nz = np.argwhere(sub % (p ** (v + 1)) != 0)
        if nz.size == 0:
            v += 1
            continue
        li, lj = nz[0]
        r = np.flatnonzero(rows_alive)[li]
        c = np.flatnonzero(cols_alive)[lj]
        pv = p ** v
        uinv = pow((int(M[r, c]) // pv) % md, -1, md)
        col = M[:, c].copy()
        col[r] = 0
        f = ((col // pv) % md) * uinv % md
        nzr = np.flatnonzero(f)
        if nzr.size:
            M[nzr] = (M[nzr] - np.outer(f[nzr], M[r])) % md
        row = M[r].copy()
        row[c] = 0
        g = ((row // pv) % md) * uinv % md
        nzc = np.flatnonzero(g)
        if nzc.size:
            M[:, nzc] = (M[:, nzc] - np.outer(M[:, c], g[nzc])) % md
        rows_alive[r] = False
        cols_alive[c] = False
        vals.append(v)
    return vals


vA = smith_vals(A.copy(), 6)
counts_A = {v: vA.count(v) for v in sorted(set(vA))}
rank_A = len(vA)
max_val_A = max(vA)
precision_ok = bool(rank_A == 854 and max_val_A <= 3 and rank_f3 == 3125 - 854 and kernel_mod_3_15)

# the Gram matrix of S modulo 3^15: (K3 Psi) K3^T, the second product split in 12-bit digits to stay exact
Psi = np.rint(np.asarray(kairos_data("g13/Psi.npy"))).astype(np.int64)
KP_ = np.rint(K3.astype(np.float64) @ Psi.astype(np.float64)).astype(np.int64) % mod
B = K3.T % mod
lo, hi = B % 4096, B // 4096
assert KP_.max() * 4096.0 * 3125 < EXACT
G_lo = np.rint(KP_.astype(np.float64) @ lo.astype(np.float64)).astype(np.int64) % mod
G_hi = np.rint(KP_.astype(np.float64) @ hi.astype(np.float64)).astype(np.int64) % mod
Gram = (G_hi * 4096 + G_lo) % mod
gram_symmetric = bool(np.array_equal(Gram, Gram.T))
vG = smith_vals(Gram, KP)
counts_G = {v: vG.count(v) for v in sorted(set(vG))}
result = {"kernel_mod_3_15": kernel_mod_3_15, "rank_f3": rank_f3, "rank_A": rank_A, "max_val_A": max_val_A,
          "vals_A": str(counts_A), "precision_ok": precision_ok, "gram_symmetric": gram_symmetric,
          "gram_rank": len(vG), "vals_gram": str(counts_G), "v3_disc_S": int(sum(vG))}
