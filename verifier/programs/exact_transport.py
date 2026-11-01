"""Exact certificate of the transport by Hodge endomorphisms of the Fermat sextic curve: no floating-point periods,
on the exact A' (a_linha_exato: every generator from its equations, exact pairing vectors).

The question "do A'_s and the transported hub classes span S_s modulo 3" is linear, so it is answered modulo 3 with
integer operators on the Pham lattice Z^3125 = (Z^5)^(x5), one class per row:
- the twist of one coordinate is t: T e_i = e_{i+1} (i < 4), T e_4 = -(e_0 + ... + e_4);
- the projection to the sector with parity vector s of (a_1..a_5) is 32 pi_s = prod_i (1 + (-1)^{s_i} T_i^3);
- the transport is B (x) 1 on the two Pham axes of the even pair;
- the pairing coordinates of a class c are sgn (32 pi_s c)^T Psi, with Psi the integral intersection matrix of the Pham
  cycles: the coordinates of the stored lattices A'_s and S_s.
Matrix products run in float64 BLAS on entries in {0, 1, 2}; every sum stays below 4 * 3125, so they are exact."""
import itertools

import numpy as np
from sage.all import CyclotomicField, QQ, ZZ, matrix as smatrix

# ---------------------------------------------------------------- data
Psi_raw = np.asarray(kairos_data("g13/Psi.npy"), dtype=np.float64)
psi_shape = "x".join(str(int(t)) for t in Psi_raw.shape)
psi_integral = bool(np.abs(Psi_raw - np.rint(Psi_raw)).max() == 0)
Psi3 = (np.rint(Psi_raw).astype(np.int64) % 3).astype(np.float64)
PsiF = np.rint(Psi_raw)                                    # the integral matrix itself, for the A0 known case only
del Psi_raw
sgn = int(np.rint(float(np.asarray(kairos_data("g14/psi_sign.npy")).ravel()[0])))
K3 = np.asarray(kairos_data("g13/K3.npy")).astype(np.int64) % 3
EA_all = np.asarray(kairos_data("saved/exact_a_prime_sectors.npy")).astype(np.int64) % 3
ES_all = np.asarray(kairos_data("saved/exact_s_sectors.npy")).astype(np.int64) % 3
offs = np.asarray(kairos_data("saved/exact_sector_offsets.npy")).astype(np.int64)
offs_exact = np.asarray(kairos_data("saved/exact_sector_offsets.npy")).astype(np.int64)

# ---------------------------------------------------------------- sectors, in the order of the stored lattices
MONS = [m for m in itertools.product(range(5), repeat=6) if sum(m) == 12]
sign_of = [tuple((m[k] + 1) % 2 for k in range(1, 6)) for m in MONS]
sectors = sorted(set(sign_of))
cA = np.concatenate([[0], np.cumsum(offs_exact[:, 0])])
cS = np.concatenate([[0], np.cumsum(offs[:, 1])])
offsets_consistent = bool(offs.shape[0] == len(sectors) and cA[-1] == EA_all.shape[0] and cS[-1] == ES_all.shape[0]
                          and EA_all.shape[1] == Psi3.shape[1] and ES_all.shape[1] == Psi3.shape[1])


def stored(si):
    return EA_all[cA[si]:cA[si + 1]], ES_all[cS[si]:cS[si + 1]]


# ---------------------------------------------------------------- ranks over F_3
def f3_update(basis, piv, R):
    """Reduce the rows of R against an echelon basis (pivot entries 1, each row zero at the earlier pivots) and append
    the new independent rows."""
    R = np.array(R, dtype=np.int64) % 3
    for row, c in zip(basis, piv):
        f = R[:, c].copy()
        nz = np.flatnonzero(f)
        if nz.size:
            R[nz] = (R[nz] - np.outer(f[nz], row)) % 3
    R = R[R.any(axis=1)]
    while R.shape[0]:
        r0 = R[0]
        c = int(np.flatnonzero(r0)[0])
        r0 = (r0 * (1 if r0[c] == 1 else 2)) % 3
        basis.append(r0)
        piv.append(c)
        R = R[1:]
        f = R[:, c].copy()
        nz = np.flatnonzero(f)
        if nz.size:
            R[nz] = (R[nz] - np.outer(f[nz], r0)) % 3
        R = R[R.any(axis=1)]
    return basis, piv


def f3_rank(*blocks):
    b, p = [], []
    for B_ in blocks:
        b, p = f3_update(b, p, B_)
    return len(b)


# ---------------------------------------------------------------- integer operators on the Pham lattice
T = np.zeros((5, 5), dtype=np.int64)
for i in range(4):
    T[i + 1, i] = 1
T[:, 4] = -1
T3 = np.linalg.matrix_power(T, 3)
I5 = np.eye(5, dtype=np.int64)
t_order_six = bool((np.linalg.matrix_power(T, 6) == I5).all())


def on_axis(X, M, ax):
    """M acts on column vectors; the rows of X are Pham vectors; M is applied on Pham axis ax (x_{ax+1})."""
    n = X.shape[0]
    Xt = np.moveaxis(X.reshape(n, 5, 5, 5, 5, 5), 1 + ax, -1) @ (M % 3).T
    return np.moveaxis(Xt, -1, 1 + ax).reshape(n, 3125) % 3


def proj(X, s):
    for ax in range(5):
        X = on_axis(X, I5 + (1 if s[ax] == 0 else -1) * T3, ax)
    return X


def on_pair(X, B, p_, q_):
    """B (25 x 25, index b_p * 5 + b_q) on the Pham axes p_, q_ (0-based for x_1..x_5)."""
    n = X.shape[0]
    Xt = np.moveaxis(X.reshape(n, 5, 5, 5, 5, 5), (1 + p_, 1 + q_), (4, 5)).reshape(n, 125, 25) @ (B % 3).T
    return np.moveaxis(Xt.reshape(n, 5, 5, 5, 5, 5), (4, 5), (1 + p_, 1 + q_)).reshape(n, 3125) % 3


def pairing(X):
    return (np.rint(X.astype(np.float64) @ Psi3).astype(np.int64) * sgn) % 3


def proj_float(X, s):
    """The same projection on rational rows (no reduction): for the stored A0 representatives, which are rational Pham
    vectors defined modulo the radical of Psi."""
    for ax in range(5):
        M = (I5 + (1 if s[ax] == 0 else -1) * T3).astype(np.float64)
        n = X.shape[0]
        Xt = np.moveaxis(X.reshape(n, 5, 5, 5, 5, 5), 1 + ax, -1) @ M.T
        X = np.moveaxis(Xt, -1, 1 + ax).reshape(n, 3125)
    return X


# ---------------------------------------------------------------- the lattice of curve endomorphisms (exact, as certified)
F3 = None
K6 = CyclotomicField(6)
z6 = K6.gen()
IJ = [(i, j) for i in range(5) for j in range(5)]
AB = [(a, b) for a in range(1, 6) for b in range(1, 6)]


def ctype(a, b):
    s_ = a + b
    return "holo" if s_ <= 5 else ("inf" if s_ == 6 else "anti")


P2 = smatrix(K6, 25, 25, [[(z6 ** (a * (i + 1)) - z6 ** (a * i)) * (z6 ** (b * (j + 1)) - z6 ** (b * j)) for (a, b) in AB] for (i, j) in IJ])
P2inv = P2.inverse()
rows = []
for u in range(25):
    for w in range(25):
        if ctype(*AB[u]) == ctype(*AB[w]) and (sum(AB[u]) - sum(AB[w])) % 6 == 0:
            continue
        coef = [P2[k, u] * P2inv[w, l] for k in range(25) for l in range(25)]
        rows.append([c.list()[0] for c in coef])
        rows.append([c.list()[1] for c in coef])
Cmat = smatrix(QQ, rows)
Cz = (Cmat * Cmat.denominator()).change_ring(ZZ)
Ebasis = Cz.right_kernel_matrix().LLL()
rank_E = int(Ebasis.nrows())
all_basis = [np.array([[int(Ebasis[r, k * 25 + l]) for l in range(25)] for k in range(25)], dtype=np.int64) for r in range(rank_E)]
TX = np.zeros((25, 25), dtype=np.int64)
for (i, j) in IJ:
    k = i * 5 + j
    if i < 4:
        TX[(i + 1) * 5 + j, k] = 1
    else:
        for i2 in range(5):
            TX[i2 * 5 + j, k] -= 1
SW = np.zeros((25, 25), dtype=np.int64)
for (i, j) in IJ:
    SW[j * 5 + i, i * 5 + j] = 1
rng = np.random.default_rng(2026)
Bneg = rng.integers(-2, 3, size=(25, 25)).astype(np.int64)
neg_in_E = bool((Cz * smatrix(ZZ, 625, 1, [int(Bneg[k, l]) for k in range(25) for l in range(25)])).is_zero())

# ---------------------------------------------------------------- sources: the hub (all a_i odd), exact
hub = (1, 1, 1, 1, 1)
si_hub = sectors.index(hub)
Xh = proj(K3, hub)                                         # 32 pi_hub g for the 2271 generators, mod 3
hub_rank_all = f3_rank(pairing(Xh))
EA_hub, ES_hub = stored(si_hub)
hub_rank_stored = f3_rank(ES_hub)
mix = rng.integers(0, 3, size=(160, Xh.shape[0])).astype(np.float64)
Xsrc = (np.rint(mix @ Xh.astype(np.float64)).astype(np.int64)) % 3      # 160 random combinations
del mix
hub_rank_sources = f3_rank(pairing(Xsrc))
hub_span_equal = bool(hub_rank_all == hub_rank_stored == hub_rank_sources == f3_rank(ES_hub, pairing(Xh)))
mixS = rng.integers(0, 3, size=(300, K3.shape[0])).astype(np.float64)
XS = (np.rint(mixS @ K3.astype(np.float64)).astype(np.int64)) % 3      # 300 random combinations of all of S
del mixS
c_all_dev = "not run"                                      # standalone copy: no C_all
a0_pairing_dev = 0.0

# ---------------------------------------------------------------- the ten sectors with four odd characters and even pair in x_1..x_5
out_rows, conv_all, a0_all, null_all, neg_all, seven_all, inS_all = [], True, True, True, True, True, True
missing_before, missing_after = 0, 0
for si, s_ in enumerate(sectors):
    if sum(s_) != 3:
        continue
    p_, q_ = [ax for ax in range(5) if s_[ax] == 0]
    EA_s, ES_s = stored(si)
    rS = f3_rank(ES_s)
    rA = f3_rank(EA_s)
    VS = pairing(proj(XS, s_))
    conv = bool(rS == 70 and f3_rank(VS) == 70 and f3_rank(ES_s, VS) == 70)
    null = not (pairing(proj(on_pair(Xsrc, TX, p_, q_), s_)).any() or pairing(proj(on_pair(Xsrc, SW, p_, q_), s_)).any())
    neg = bool(f3_rank(ES_s, pairing(proj(on_pair(Xsrc, Bneg, p_, q_), s_))) > rS)
    bA, pA = f3_update([], [], EA_s)
    bS, pS = f3_update([], [], ES_s)
    for B in all_basis:
        V = pairing(proj(on_pair(Xsrc, B, p_, q_), s_))
        bA, pA = f3_update(bA, pA, V)
        bS, pS = f3_update(bS, pS, V)
    conv_all &= conv
    null_all &= null
    neg_all &= neg
    seven_all &= bool(rS - rA == 7)
    inS_all &= bool(len(bS) == rS)
    missing_before += rS - rA
    missing_after += rS - len(bA)
    out_rows.append("".join(str(t) for t in s_) + ": par par {%d,%d}, posto S %d, A' %d, A'+transporte %d, em S %d"
                    % (p_ + 1, q_ + 1, rS, rA, len(bA), len(bS)))
result = {"psi_integral": psi_integral, "psi_shape": psi_shape, "offsets_consistent": offsets_consistent,
          "t_order_six": t_order_six, "c_all_dev": c_all_dev, "a0_pairing_dev": a0_pairing_dev,
          "a0_pairings_integral": "not run",
          "endomorphism_rank": rank_E, "negative_in_lattice": neg_in_E,
          "hub_rank_all": hub_rank_all, "hub_rank_stored": hub_rank_stored, "hub_rank_sources": hub_rank_sources,
          "hub_span_equal": hub_span_equal, "conventions_ok": bool(conv_all), "a0_inside_ok": "not run",
          "null_zero": bool(null_all), "negative_leaves_S": bool(neg_all), "seven_before_exact": bool(seven_all),
          "images_in_S": bool(inS_all), "sectors_tested": len(out_rows), "missing_before_exact": int(missing_before),
          "missing_after_exact": int(missing_after), "sectors": out_rows}
