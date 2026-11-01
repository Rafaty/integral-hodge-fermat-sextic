"""Exact A' for the Fermat sextic fourfold X6, from the equations of the surfaces to the 3-adic sector lattices,
with no floating point in any period.

For a complete intersection Z = {f = 0} with F = sum f_i g_i, the Villaflor functional Phi_Z(a) (the coefficient of
det Jac(f, g) at the monomial complementary to x^(a-1)) is exact over L = Q(zeta12, 2^(1/3)). The Fourier coefficients
of the Pham coordinates of [Z] are chat(a) = Phi_Z(a) / (kappa alpha(a) prod_{k>=1} (zeta6^{a_k} - 1)): alpha is the
algebraic part of the Gamma normalisation (prod_k Gamma(a_k/6) sin(pi a_0/6) zeta12^S / pi = pi^2 alpha(a), by
reflection and duplication), kappa = 3/2, checked by -6^7 lambda_a rho(a) rho(-a) = 1 on every Hodge character
(lambda_a the eigenvalue of Psi = V + V^T, V = (1 - shift)^{x5}). The pairing vector v = Psi c, c = G^{x5} chat with
G = F1^{-1}, must be an exact integer vector; the sector lattices are v^T (32 pi_s) modulo 3^12. Generators: the seeded
recipe of save_a_prime (1771 combinations of the 22680 planes and (1,2,3), (2,2,3) surfaces, the 1440 cone classes
Phi(Gamma)_0, the pairs of six-tangent conics under S6 and under the 2880 twisted permutations), with the canonical
sign of every determinant and [Z]_0 = [Z] - deg(Z)[P0] for the true plane P0. Products of float64 matrices are used
only on integers whose sums stay below 2^53 (asserted), so they are exact."""
import itertools

import numpy as np
from sage.all import CyclotomicField, PolynomialRing, QQ, ZZ, matrix, identity_matrix, vector

lap = []
p, K = 3, 12
mod = p ** K
EXACT = 2.0 ** 52


def mark(name):
    lap.append(name)


# ---------------------------------------------------------------- fields and characters
K0 = CyclotomicField(12)
eta = K0.gen()
z6 = eta ** 2
L = K0.extension(PolynomialRing(K0, "t").gen() ** 3 - 2, "lam")
lam = L.gen()
w = L(eta ** 4)
I_ = L(eta ** 3)
sqrt3 = L(eta + eta ** 11)
Rx = PolynomialRing(L, "x0,x1,x2,x3,x4,x5")
x = Rx.gens()
F6 = sum(v ** 6 for v in x)
ETAS = [eta ** k for k in range(1, 12, 2)]
CUBE = {1: [lam * w ** k for k in range(3)], -1: [-lam * w ** k for k in range(3)]}
MONS = [m for m in itertools.product(range(5), repeat=6) if sum(m) == 12]
NH = len(MONS)
mon_index = {m: i for i, m in enumerate(MONS)}
A6 = [tuple(v + 1 for v in m) for m in MONS]
pos = np.array([sum((a[k] - 1) * 5 ** (5 - k) for k in range(1, 6)) for a in A6])
neg = np.array([mon_index[tuple(4 - v for v in m)] for m in MONS])
a_of = np.array(A6, dtype=np.int64)


def alpha(a):
    n = [sum(1 for v in a if v == j) for j in range(6)]
    q, r = divmod(-n[1] + 3 * n[4] + 4 * n[5], 3)
    e3 = n[1] - n[4] - n[5]
    s = {1: L(1) / 2, 2: sqrt3 / 2, 3: L(1), 4: sqrt3 / 2, 5: L(1) / 2}[a[0]]
    return L(2) ** q * lam ** r * L(3) ** (e3 // 2) * (sqrt3 if e3 % 2 else 1) * s * L(eta) ** (sum(a[1:]) % 12)


def prodz(a):
    out = L(1)
    for v in a[1:]:
        out *= L(z6) ** v - 1
    return out


KAPPA = QQ(3) / 2
rho = [1 / (KAPPA * alpha(a) * prodz(a)) for a in A6]
rho_e, rho_u = [], []
for r_ in rho:
    lst = r_.list()
    nz_ = [j for j in range(3) if lst[j] != 0]
    rho_e.append(nz_[0])
    rho_u.append(lst[nz_[0]])
rho_pure = all(len([j for j in range(3) if r_.list()[j] != 0]) == 1 for r_ in rho)

F1 = matrix(K0, 5, 5, [[z6 ** (a * b) for b in range(5)] for a in range(1, 6)])
G = F1.inverse()
V1 = np.eye(5, dtype=np.int64)
for j in range(4):
    V1[j, j + 1] = -1
M1 = G.transpose() * matrix(K0, V1.tolist()) * G
mu = {a: M1[a - 1, 5 - a] for a in range(1, 6)}
anti_ok = all(M1[i, j] == 0 for i in range(5) for j in range(5) if i + j != 4)


def lam_of(a):
    p1, p2 = K0(1), K0(1)
    for v in a[1:]:
        p1 *= mu[v]
        p2 *= mu[6 - v]
    return p1 + p2


kappa_identity = bool(anti_ok and all(-(6 ** 7) * L(lam_of(a)) * rho[i] * rho[neg[i]] == 1 for i, a in enumerate(A6)))


def zpair(k):
    c = K0(k).list()
    assert c[1] == 0 and c[3] == 0
    return c[0], c[2]


Gr = np.array([[int(zpair(6 * G[i, j])[0]) for j in range(5)] for i in range(5)], dtype=np.int64)
Gi = np.array([[int(zpair(6 * G[i, j])[1]) for j in range(5)] for i in range(5)], dtype=np.int64)
GROWTH = (2 * int((np.abs(Gr) + np.abs(Gi)).sum(axis=1).max())) ** 5 * 64

Psi = np.asarray(kairos_data("g13/Psi.npy"))
Vk = V1
for _ in range(4):
    Vk = np.kron(Vk, V1)
psi_kronecker = bool(np.array_equal(np.rint(Psi).astype(np.int64), Vk + Vk.T))
del Vk
mark("constants")

# ---------------------------------------------------------------- exact functionals
bad = {"lam": 0, "odd": 0, "den": 0}


def functional(fs, gs):
    det = matrix(Rx, [[f.derivative(v) for v in x] for f in list(fs) + list(gs)]).det()
    out = {}
    for mon, c in det.dict().items():
        mon = tuple(mon)
        if max(mon) <= 4:
            out[mon_index[tuple(4 - v for v in mon)]] = c
    return out


def chat2(Phi):
    """2 chat over (1, zeta6), two integer arrays; counts the coefficients outside (1/2) Z[zeta6]."""
    re = np.zeros(NH, dtype=np.int64)
    im = np.zeros(NH, dtype=np.int64)
    for i, c in Phi.items():
        lst = c.list()
        e = rho_e[i]
        j0 = (-e) % 3
        if any(lst[j] != 0 for j in range(3) if j != j0):
            bad["lam"] += 1
            continue
        cc = (rho_u[i] * lst[j0] * (2 if j0 + e == 3 else 1)).list()
        if cc[1] != 0 or cc[3] != 0:
            bad["odd"] += 1
            continue
        r0, r2 = 2 * QQ(cc[0]), 2 * QQ(cc[2])
        if r0.denominator() != 1 or r2.denominator() != 1:
            bad["den"] += 1
            continue
        re[i], im[i] = int(r0), int(r2)
    return re, im


def villaflor_self(Phi):
    """Primitive self-intersection by the residue pairing, -6^-7 sum Phi(a) Phi(-a), exact."""
    s = L(0)
    for i, c in Phi.items():
        j = int(neg[i])
        if j in Phi:
            s += c * Phi[j]
    return -s / 6 ** 7


def ideal_key(fs):
    return tuple(sorted(str(f / f.lc()) for f in fs))


# planes, in the order of the bridge: pairings, then the three odd exponents
def pairings():
    res = []

    def rec(rest, acc):
        if not rest:
            res.append(tuple(acc))
            return
        for b in rest[1:]:
            rec([v for v in rest if v not in (rest[0], b)], acc + [(rest[0], b)])
    rec(list(range(6)), [])
    return res


PL = [tuple((a, b, k) for (a, b), k in zip(pr, ks)) for pr in pairings() for ks in itertools.product(range(1, 12, 2), repeat=3)]


def plane_fg(pl):
    fs, gs = [], []
    for (a, b, k) in pl:
        e = L(eta) ** k
        fs.append(x[a] - e * x[b])
        gs.append(sum(e ** j * x[a] ** (5 - j) * x[b] ** j for j in range(6)))
    return fs, gs


def p1_fg(a, b, c, d, e, f, e1, eps, lam_, iota):
    fs = (x[a] - L(e1) * x[b], x[c] ** 2 - lam_ * x[d] * x[e], x[d] ** 3 + eps * x[e] ** 3 - iota * x[f] ** 3)
    gs = (sum(L(e1) ** j * x[a] ** (5 - j) * x[b] ** j for j in range(6)),
          x[c] ** 4 + lam_ * x[c] ** 2 * x[d] * x[e] + lam_ ** 2 * x[d] ** 2 * x[e] ** 2,
          x[d] ** 3 + eps * x[e] ** 3 + iota * x[f] ** 3)
    return fs, gs


def p2_fg(a, b, c, d, e, f, lam1, eps1, lam2, eps2, iota):
    A_ = x[b] ** 3 + eps1 * x[c] ** 3
    B_ = x[e] ** 3 + eps2 * x[f] ** 3
    fs = (x[a] ** 2 - lam1 * x[b] * x[c], x[d] ** 2 - lam2 * x[e] * x[f], A_ - iota * B_)
    gs = (x[a] ** 4 + lam1 * x[a] ** 2 * x[b] * x[c] + lam1 ** 2 * x[b] ** 2 * x[c] ** 2,
          x[d] ** 4 + lam2 * x[d] ** 2 * x[e] * x[f] + lam2 ** 2 * x[e] ** 2 * x[f] ** 2, A_ + iota * B_)
    return fs, gs


SURF = [plane_fg(pl) for pl in PL]
NP = len(SURF)
P1keys, P2keys = {}, {}
for (a, b) in itertools.combinations(range(6), 2):
    rest = [v for v in range(6) if v not in (a, b)]
    for c in rest:
        rest2 = [v for v in rest if v != c]
        for (d, e) in itertools.permutations(rest2, 2):
            f = [v for v in rest2 if v not in (d, e)][0]
            for e1 in ETAS:
                for eps in (1, -1):
                    for lam_ in CUBE[eps]:
                        for iota in (I_, -I_):
                            fs, gs = p1_fg(a, b, c, d, e, f, e1, eps, lam_, iota)
                            key = ideal_key(fs)
                            if key not in P1keys:
                                P1keys[key] = len(SURF)
                                SURF.append((fs, gs))
NP1 = len(SURF) - NP
for a in range(6):
    for (b, c) in itertools.combinations([v for v in range(6) if v != a], 2):
        rest = [v for v in range(6) if v not in (a, b, c)]
        for d in rest:
            e, f = [v for v in rest if v != d]
            if a > d:
                continue
            for eps1 in (1, -1):
                for lam1 in CUBE[eps1]:
                    for eps2 in (1, -1):
                        for lam2 in CUBE[eps2]:
                            for iota in (I_, -I_):
                                fs, gs = p2_fg(a, b, c, d, e, f, lam1, eps1, lam2, eps2, iota)
                                key = ideal_key(fs)
                                if key not in P2keys:
                                    P2keys[key] = len(SURF)
                                    SURF.append((fs, gs))
NS_ = len(SURF)
deg = np.concatenate([np.ones(NP), np.full(NP1, 6.0), np.full(NS_ - NP - NP1, 12.0)]).astype(np.int64)
identity_all = all(sum(f * g for f, g in zip(fs, gs)) == F6 for fs, gs in SURF)
mark("enumeration")

# ---------------------------------------------------------------- the cone classes Phi(Gamma) (as in save_a_prime)
G252 = kairos_data("x2int_char0_G252.json")
LL = [tuple(v) for v in G252["line_labels"]]
CL = [tuple(v) for v in G252["curve_labels"]]
Cl = matrix(ZZ, kairos_data("x2int_lambda_basis.json")["C"])
Sat = matrix(ZZ, kairos_data("x2int_N_basis.json")["Sat"])


def canon_line(lab):
    a, b, cc, dd, k1, k2 = lab
    q1 = (a, b, k1 % 12) if a < b else (b, a, (-k1) % 12)
    q2 = (cc, dd, k2 % 12) if cc < dd else (dd, cc, (-k2) % 12)
    if q1[0] > q2[0]:
        q1, q2 = q2, q1
    return ("L", q1[0], q1[1], q2[0], q2[1], q1[2], q2[2])


def canon_curve(lab):
    l, j, i, kk, m, r = lab
    m %= 6
    r %= 12
    if i > kk:
        i, kk = kk, i
        if m % 2 == 1:
            r = (r + 6) % 12
    return ("C", l, j, i, kk, m, r)


cidx = {canon_line(lab): n for n, lab in enumerate(LL)}
cidx.update({canon_curve(lab): 108 + n for n, lab in enumerate(CL)})


def sig(lab, line):
    if line:
        a, b, cc, dd, k1, k2 = lab
        k1 += 4 * ((a == 3) - (b == 3))
        k2 += 4 * ((cc == 3) - (dd == 3))
        return canon_line((a, b, cc, dd, k1 % 12, k2 % 12))
    l, j, i, kk, m, r = lab
    return canon_curve((l, j, i, kk, (m + 4 * (3 in (l, i, kk))) % 6, r))


perm252 = [cidx[sig(lab, True)] for lab in LL] + [cidx[sig(lab, False)] for lab in CL]
coords = Sat.change_ring(QQ).solve_left(Cl.change_ring(QQ)).change_ring(ZZ)
piv = coords.transpose().pivots()
Sg = coords.matrix_from_rows(list(piv)).change_ring(QQ).solve_right(coords.matrix_from_rows([perm252[i] for i in piv]).change_ring(QQ)).change_ring(ZZ)
one = identity_matrix(ZZ, 86)
Iinv = matrix(ZZ, (Sg - one).left_kernel().basis_matrix())
Nn = (one + Sg + Sg ** 2).echelon_form()
Nn = Nn.matrix_from_rows([i for i in range(86) if not Nn.row(i).is_zero()])
H, U = coords.hermite_form(transformation=True)
rk = H.rank()
Hr, Ur = H.matrix_from_rows(range(rk)), U.matrix_from_rows(range(rk))
Kb = matrix(ZZ, coords.left_kernel().basis_matrix()).LLL()
KbQ = Kb.change_ring(QQ)
Gram_inv = (KbQ * KbQ.transpose()).inverse()


def small_coeffs(y):
    c = Hr.change_ring(QQ).solve_left(y.change_ring(QQ))
    n = c.change_ring(ZZ) * Ur
    for _ in range(3):
        t = (n.change_ring(QQ) * KbQ.transpose()) * Gram_inv
        n = n - vector(ZZ, [round(ti) for ti in t]) * Kb
    assert n * coords == y
    return [int(t) for t in n]


ZI = ZZ ** Iinv.nrows()
co = Iinv.change_ring(QQ).solve_left(Nn.change_ring(QQ)).change_ring(ZZ)
Qm = ZI / ZI.span(co.rows())
NS_gens = [small_coeffs(3 * (vector(ZZ, g.lift()) * Iinv)) for g in Qm.gens()]


def p1_key(a, b, c, d, e, f, e1, eps, lam_, iota):
    return P1keys[ideal_key((x[a] - L(e1) * x[b], x[c] ** 2 - lam_ * x[d] * x[e], x[d] ** 3 + eps * x[e] ** 3 - iota * x[f] ** 3))]


def p2_key(a, b, c, d, e, f, lam1, eps1, lam2, eps2, iota):
    return P2keys[ideal_key((x[a] ** 2 - lam1 * x[b] * x[c], x[d] ** 2 - lam2 * x[e] * x[f],
                             x[b] ** 3 + eps1 * x[c] ** 3 - iota * (x[e] ** 3 + eps2 * x[f] ** 3)))]


def cone_rows(a, b, c, eps_Q, il_Q):
    lam_Q = CUBE[eps_Q][il_Q]
    Rr = sorted(set(range(6)) - {a, b, c})
    plus = p1_key(Rr[0], Rr[1], a, b, c, Rr[2], eta, eps_Q, lam_Q, I_)
    minus = p1_key(Rr[0], Rr[1], a, b, c, Rr[2], eta, eps_Q, lam_Q, -I_)
    rows = []
    for lab in LL:
        a_, b_, cc, dd, k1, k2 = lab
        pairs = [(a_, b_, k1), (cc, dd, k2)]
        (pp, _, k), = [q for q in pairs if q[1] == 3]
        (rr, ss, kp), = [q for q in pairs if q[1] != 3]
        rows.append(((1, p1_key(Rr[rr], Rr[ss], a, b, c, Rr[pp], eta ** kp, eps_Q, lam_Q, L(eta ** ((-3 * k) % 12)))),))
    for lab in CL:
        l, j, i, kk, m, r_ = lab
        if j == 3:
            rows.append(((3, p2_key(a, b, c, Rr[l], Rr[i], Rr[kk], lam_Q, eps_Q, lam * L(eta ** (2 * m)), 1 if m % 2 == 0 else -1, L(eta ** r_))),))
        else:
            rows.append(((3, plus), (3, minus)))
    return rows


cones = [(a, b, c, e, il) for a in range(6) for (b, c) in itertools.combinations([v for v in range(6) if v != a], 2) for e in (1, -1) for il in range(3)]
CONE_ROWS = [cone_rows(*cn) for cn in cones]
need = {0}
for rows in CONE_ROWS:
    for row in rows:
        for _, gi in row:
            need.add(gi)
rng_s = np.random.default_rng(3)
sample = sorted(set(int(t) for t in rng_s.choice(NP, 40, replace=False)) | set(int(t) for t in NP + rng_s.choice(NP1, 40, replace=False))
                | set(int(t) for t in NP + NP1 + rng_s.choice(NS_ - NP - NP1, 40, replace=False)))
need |= set(sample)
mark("cone structure")

# ---------------------------------------------------------------- A0: exact classes of the 22680 surfaces, combined as before
rng = np.random.default_rng(11)
R = rng.integers(0, 3, size=(1771, NS_)).astype(np.int64)
store = {}
acc_re = np.zeros((1771, NH), dtype=np.int64)
acc_im = np.zeros((1771, NH), dtype=np.int64)
self_vf = {}
for g, (fs, gs) in enumerate(SURF):
    Phi = functional(fs, gs)
    re, im = chat2(Phi)
    supp = np.flatnonzero(re | im)
    acc_re[:, supp] += np.outer(R[:, g], re[supp])
    acc_im[:, supp] += np.outer(R[:, g], im[supp])
    if g in need:
        store[g] = (supp, re[supp], im[supp])
    if g in sample or g == 0:
        self_vf[g] = villaflor_self(Phi)
mark("functionals of the surfaces")


def dense(g):
    re = np.zeros(NH, dtype=np.int64)
    im = np.zeros(NH, dtype=np.int64)
    supp, r_, i_ = store[g]
    re[supp], im[supp] = r_, i_
    return re, im


P0re, P0im = dense(0)
dR = R @ deg
acc_re -= np.outer(dR, P0re)
acc_im -= np.outer(dR, P0im)
rows_even = bool(not np.any(acc_re % 2) and not np.any(acc_im % 2))
A0_rows = (acc_re // 2, acc_im // 2)
del acc_re, acc_im
self_expected = {0: QQ(125) / 6}
for g in sample:
    self_expected[g] = QQ(125) / 6 if g < NP else (QQ(60) if g < NP + NP1 else QQ(96))
villaflor_self_ok = all(self_vf[g] == self_expected[g] for g in self_vf)

# cone rows, true convention [Phi(Gamma)]_0, and the degree of each Phi(Gamma)
cone_re = np.zeros((4 * len(CONE_ROWS), NH), dtype=np.int64)
cone_im = np.zeros((4 * len(CONE_ROWS), NH), dtype=np.int64)
cone_deg = []
col = 0
for rows in CONE_ROWS:
    for n in NS_gens:
        Dtot = 0
        for ni, row in zip(n, rows):
            if ni:
                for coef, gi in row:
                    supp, r_, i_ = store[gi]
                    cone_re[col, supp] += ni * coef * r_
                    cone_im[col, supp] += ni * coef * i_
                    Dtot += ni * coef * int(deg[gi])
        cone_re[col] -= Dtot * P0re
        cone_im[col] -= Dtot * P0im
        cone_deg.append(Dtot)
        col += 1
cone_div18 = bool(not np.any(cone_re % 18) and not np.any(cone_im % 18))
cone_deg_integral = all(d % 9 == 0 for d in cone_deg)
cone_deg = [d // 9 for d in cone_deg]
cone_rows_exact = (cone_re // 18, cone_im // 18)
n_cone_deg_not_div3 = sum(1 for d in cone_deg if d % 3)
mark("cone classes")

# ---------------------------------------------------------------- pairs of six-tangent conics (as in save_a_prime)
Rq = PolynomialRing(QQ, "a,b,g0,g1,g2,g3,g4,g5,g6", order="degrevlex")
a_, b_, *gg = Rq.gens()
Sef = PolynomialRing(Rq, "e,f")
e_, f_ = Sef.gens()
Sd = PolynomialRing(Sef, "d")
d_ = Sd.gen()
q_ = d_ ** 2 + a_ * (e_ ** 2 + f_ ** 2) + b_ * e_ * f_
g_ = d_ * (gg[0] * e_ ** 2 + gg[1] * e_ * f_ + gg[2] * f_ ** 2) + gg[3] * e_ ** 3 + gg[4] * e_ ** 2 * f_ + gg[5] * e_ * f_ ** 2 + gg[6] * f_ ** 3
rem = (d_ ** 6 + e_ ** 6 + f_ ** 6 + g_ ** 2) % q_
J = Rq.ideal([c for part in rem.list() for c in Sef(part).coefficients() if c != 0]).saturation(Rq.ideal(4 * a_ ** 2 - b_ ** 2))[0]
Labs = L.absolute_field("c")
from_abs, to_abs = Labs.structure()
RqA = Rq.change_ring(Labs)
pts = [{i: from_abs(pt[RqA.gen(i)]) for i in range(9)} for pt in RqA.ideal([RqA(t) for t in J.gens()]).variety()]
reps, seen = [], set()
for pt in pts:
    key = (str(pt[0]), str(pt[1]))
    if key not in seen:
        seen.add(key)
        reps.append(pt)
new_reps = [pt for pt in reps if L(pt[0]) != 0]


def conic_data(pt, v):
    X, Y, Z = x[v[0]], x[v[1]], x[v[2]]
    Q = X ** 2 + L(pt[0]) * (Y ** 2 + Z ** 2) + L(pt[1]) * Y * Z
    Gc = X * (L(pt[2]) * Y ** 2 + L(pt[3]) * Y * Z + L(pt[4]) * Z ** 2) + L(pt[5]) * Y ** 3 + L(pt[6]) * Y ** 2 * Z + L(pt[7]) * Y * Z ** 2 + L(pt[8]) * Z ** 3
    return Q, Gc


def divide(num, Q, v):
    Ref = PolynomialRing(L, "y1,y2")
    y1, y2 = Ref.gens()
    Tq = PolynomialRing(Ref, "y0")
    y0 = Tq.gen()
    to_T = lambda P: sum(c * y0 ** m[v[0]] * y1 ** m[v[1]] * y2 ** m[v[2]] for m, c in P.dict().items())
    qt, rt = to_T(num).quo_rem(to_T(Q))
    back = lambda P: sum(c * x[v[0]] ** k * x[v[1]] ** i * x[v[2]] ** j for k, coef in enumerate(P.list()) for (i, j), c in coef.dict().items())
    return Rx(back(qt)) if qt != 0 else Rx(0), Rx(back(rt)) if rt != 0 else Rx(0)


V1c, V2c = (0, 1, 2), (3, 4, 5)
base_re, base_im, base_fg, n_identity, base_self = [], [], [], 0, []
for p1 in new_reps:
    Q1, G1 = conic_data(p1, V1c)
    r1, rr1 = divide(sum(x[i] ** 6 for i in V1c) + G1 ** 2, Q1, V1c)
    for p2 in new_reps:
        Q2, G2 = conic_data(p2, V2c)
        r2, rr2 = divide(sum(x[i] ** 6 for i in V2c) + G2 ** 2, Q2, V2c)
        fs = (Q1, Q2, G1 - I_ * G2)
        gs = (r1, r2, -(G1 + I_ * G2))
        n_identity += int(rr1 == 0 and rr2 == 0 and sum(f * h for f, h in zip(fs, gs)) == F6)
        Phi = functional(fs, gs)
        if len(base_self) < 6:
            base_self.append(villaflor_self(Phi))
            base_fg.append((fs, gs, Phi))
        re, im = chat2(Phi)
        base_re.append(re)
        base_im.append(im)
base_re, base_im = np.array(base_re), np.array(base_im)
n_base = base_re.shape[0]
pairs_self_ok = all(s_ == 96 for s_ in base_self)
mark("conic pairs")

# the group: (mu_3)^6 twists, then a permutation of the coordinates; exact action on functionals
perms = []
for A_ in itertools.combinations(range(6), 3):
    B_ = tuple(v for v in range(6) if v not in A_)
    for pa in itertools.permutations(A_):
        for pb in itertools.permutations(B_):
            perms.append(pa + pb)
newidx_of = {pm: np.array([mon_index[tuple(m[pm.index(k)] for k in range(6))] for m in MONS]) for pm in perms}


def perm_sign(pm):
    return -1 if sum(1 for i in range(6) for j in range(i + 1, 6) if pm[i] > pm[j]) % 2 else 1


def rot(re, im, j):
    for _ in range(j % 6):
        re, im = -im, re + im
    return re, im


# rho(a) depends on a_0 only through h(a_0) = sin(pi a_0/6) zeta12^(18-a_0) / (zeta6^a_0 - 1); moving the monomial
# multiplies chat by h(a_0)/h(a''_0), an element of Z[zeta6]
SIN = {1: L(1) / 2, 2: sqrt3 / 2, 3: L(1), 4: sqrt3 / 2, 5: L(1) / 2}
H0 = {t: SIN[t] * L(eta) ** ((18 - t) % 12) / (L(z6) ** t - 1) for t in range(1, 6)}
RAT_re, RAT_im = np.zeros((6, 6), dtype=np.int64), np.zeros((6, 6), dtype=np.int64)
for t in range(1, 6):
    for t2 in range(1, 6):
        q_ = (H0[t] / H0[t2]).list()
        assert q_[1] == 0 and q_[2] == 0
        c0, c2 = zpair(q_[0])
        RAT_re[t, t2], RAT_im[t, t2] = int(c0), int(c2)


def act(re, im, pm, k):
    """chat of pm(t_k(Z)): the functional at the base monomial times zeta6^(2 k.a) (the twist), times sign(pm), moved
    to the permuted monomial; then the change of rho, h(a_0)/h(a''_0)."""
    ph = (2 * (a_of @ np.array(k))) % 6
    Fr, Fi = re.copy(), im.copy()
    for j in range(1, 6):
        cols = ph == j
        if cols.any():
            Fr[..., cols], Fi[..., cols] = rot(re[..., cols], im[..., cols], j)
    nid = newidx_of[pm]
    cr, ci = RAT_re[a_of[:, 0], a_of[nid, 0]], RAT_im[a_of[:, 0], a_of[nid, 0]]
    Fr, Fi = cr * Fr - ci * Fi, cr * Fi + ci * Fr + ci * Fi
    Pr, Pi = np.zeros_like(Fr), np.zeros_like(Fi)
    Pr[..., nid] = perm_sign(pm) * Fr
    Pi[..., nid] = perm_sign(pm) * Fi
    return Pr, Pi


# known case: the action formula against a direct determinant of the moved surface
fs, gs, Phi_b = base_fg[1]
re_b, im_b = chat2(Phi_b)
action_exact = True
for pm_t, k_t in ((perms[437], (1, 2, 0, 1, 0, 2)), (perms[5], (0, 0, 0, 0, 0, 0)), (perms[611], (2, 0, 1, 1, 2, 0))):
    sub = [L(w) ** ((3 - k_t[i]) % 3) * x[pm_t[i]] for i in range(6)]
    re_d, im_d = chat2(functional([f(*sub) for f in fs], [h(*sub) for h in gs]))
    re_f, im_f = act(re_b, im_b, pm_t, k_t)
    action_exact = action_exact and bool(np.array_equal(re_d, re_f) and np.array_equal(im_d, im_f))
scalar_twist_trivial = all(not np.any((2 * (a_of @ np.array((j,) * 6))) % 6) for j in (1, 2))

Rr_dummy = rng.integers(-1, 2, size=(900, 8640))            # keeps the stream of save_a_prime (its rule classes)
B0 = rng.integers(0, 3, size=(200, n_base)).astype(np.float64)
bc_re, bc_im = B0 @ base_re, B0 @ base_im
base_deg = 12.0 * B0.sum(axis=1)
assert max(np.abs(bc_re).max(), np.abs(bc_im).max()) < EXACT


def accumulate(elements, n_rows, seed):
    rng_ = np.random.default_rng(seed)
    Ar, Ai, dacc = np.zeros((n_rows, NH)), np.zeros((n_rows, NH)), np.zeros(n_rows)
    for pm, k in elements:
        Pr, Pi = act(bc_re, bc_im, pm, k)
        Rp = rng_.integers(0, 3, size=(n_rows, Pr.shape[0])).astype(np.float64)
        Ar += Rp @ Pr
        Ai += Rp @ Pi
        dacc += Rp @ base_deg
    assert max(np.abs(Ar).max(), np.abs(Ai).max()) < EXACT
    Ar = np.rint(Ar).astype(np.int64) - np.outer(np.rint(dacc).astype(np.int64), P0re)
    Ai = np.rint(Ai).astype(np.int64) - np.outer(np.rint(dacc).astype(np.int64), P0im)
    return Ar, Ai


rng_k = np.random.default_rng(23)
control = [(pm, (0, 0, 0, 0, 0, 0)) for pm in perms]
twisted = [(pm, tuple(int(t) for t in rng_k.integers(0, 3, size=6))) for pm in perms for _ in range(2)]
more = [(pm, tuple(int(t) for t in rng_k.integers(0, 3, size=6))) for pm in perms for _ in range(2)]
Ar0, Ai0 = accumulate(control, 600, 5)
Ar1, Ai1 = accumulate(twisted, 900, 7)
Ar2, Ai2 = accumulate(more, 900, 11)
Ar12, Ai12 = np.concatenate([Ar1, Ar2]), np.concatenate([Ai1, Ai2])
pair_even = bool(not any(np.any(t % 2) for t in (Ar0, Ai0, Ar12, Ai12)))
stage_rows = {"A0": A0_rows, "cones": cone_rows_exact, "pairs": (Ar0 // 2, Ai0 // 2), "twisted": (Ar12 // 2, Ai12 // 2)}
mark("pair rows")


# ---------------------------------------------------------------- exact pairing vectors and sector lattices
def mode(X, M, k):
    return np.moveaxis(np.tensordot(X, M, axes=([1 + k], [1])), -1, 1 + k)


flags = {"imag_zero": True, "divisible": True, "object_used": False}


def reset_flags():
    flags["imag_zero"], flags["divisible"] = True, True


def pairing_rows(Cr, Ci, with_c=False):
    n = Cr.shape[0]
    big = int(max(np.abs(Cr).max(initial=0), np.abs(Ci).max(initial=0)))
    dt = np.int64 if big * GROWTH < 2 ** 62 else object
    flags["object_used"] = flags["object_used"] or dt is object
    Xr = np.zeros((n, 3125), dtype=dt)
    Xi = np.zeros((n, 3125), dtype=dt)
    Xr[:, pos], Xi[:, pos] = Cr, Ci
    Xr, Xi = Xr.reshape((n,) + (5,) * 5), Xi.reshape((n,) + (5,) * 5)
    gr, gi, vv = Gr.astype(dt), Gi.astype(dt), V1.astype(dt)
    for k in range(5):
        a1, a2, b1, b2 = mode(Xr, gr, k), mode(Xi, gi, k), mode(Xi, gr, k), mode(Xr, gi, k)
        Xr, Xi = a1 - a2, b1 + b2 + a2
    flags["imag_zero"] = flags["imag_zero"] and not np.any(Xi)
    v1, v2 = Xr, Xr
    for k in range(5):
        v1, v2 = mode(v1, vv, k), mode(v2, vv.T, k)
    W = (v1 + v2).reshape(n, 3125)
    flags["divisible"] = flags["divisible"] and not np.any(W % 6 ** 5)
    v = (W // 6 ** 5).astype(np.int64)
    return (v, Xr.reshape(n, 3125)) if with_c else v


T = np.zeros((5, 5))
for i in range(4):
    T[i + 1, i] = 1.0
T[:, 4] = -1.0
T3 = T @ T @ T
PM = {0: np.eye(5) + T3, 1: np.eye(5) - T3}
sectors = sorted(set(tuple(a[k] % 2 for k in range(1, 6)) for a in A6))
n_in_sector = {s: sum(1 for a in A6 if tuple(a[k] % 2 for k in range(1, 6)) == s) for s in sectors}
odd_of = {s: sum(s) + sum(s) % 2 for s in sectors}


def sector_rows(v, s):
    assert np.abs(v).max(initial=0) * 6 ** 5 < EXACT
    X = v.astype(np.float64).reshape((v.shape[0],) + (5,) * 5)
    for k in range(5):
        X = np.moveaxis(np.tensordot(X, PM[s[k]], axes=([1 + k], [0])), -1, 1 + k)
    return np.rint(X).reshape(v.shape[0], 3125).astype(np.int64)


def smith_vals(A):
    A = A % mod
    m, n = A.shape
    rows_alive = np.ones(m, dtype=bool)
    cols_alive = np.ones(n, dtype=bool)
    vals = []
    v = 0
    while v < K:
        sub_ = A[np.ix_(rows_alive, cols_alive)]
        if sub_.size == 0:
            break
        nz = np.argwhere(sub_ % (p ** (v + 1)) != 0)
        if nz.size == 0:
            v += 1
            continue
        li, lj = nz[0]
        r = np.flatnonzero(rows_alive)[li]
        c = np.flatnonzero(cols_alive)[lj]
        pv = p ** v
        uinv = pow((int(A[r, c]) // pv) % mod, -1, mod)
        colv = A[:, c].copy()
        colv[r] = 0
        f = ((colv // pv) % mod) * uinv % mod
        nzr = np.flatnonzero(f)
        if nzr.size:
            A[nzr] = (A[nzr] - np.outer(f[nzr], A[r])) % mod
        row = A[r].copy()
        row[c] = 0
        gq = ((row // pv) % mod) * uinv % mod
        nzc = np.flatnonzero(gq)
        if nzc.size:
            A[:, nzc] = (A[:, nzc] - np.outer(A[:, c], gq[nzc])) % mod
        rows_alive[r] = False
        cols_alive[c] = False
        vals.append(v)
    return vals


def echelon(Ain):
    Arow = Ain % mod
    alive = Arow.any(axis=1)
    rows_out = []
    while alive.any():
        sub_ = Arow[alive]
        vmin = None
        for v_ in range(K):
            nzv = np.argwhere(sub_ % (3 ** (v_ + 1)) != 0)
            if nzv.size:
                vmin, (li, c) = v_, nzv[0]
                break
        if vmin is None:
            break
        r = np.flatnonzero(alive)[li]
        pv = 3 ** vmin
        u = pow((int(Arow[r, c]) // pv) % mod, -1, mod)
        colv = Arow[:, c].copy()
        colv[r] = 0
        colv[~alive] = 0
        f = ((colv // pv) % mod) * u % mod
        nz = np.flatnonzero(f)
        if nz.size:
            Arow[nz] = (Arow[nz] - np.outer(f[nz], Arow[r])) % mod
        rows_out.append(Arow[r].copy())
        alive[r] = False
        alive &= Arow.any(axis=1)
    if not rows_out:
        return np.zeros((0, Ain.shape[1]), dtype=np.int64)
    return np.array(rows_out, dtype=np.int64)


# S from its integral basis
K3 = np.asarray(kairos_data("g13/K3.npy"))
assert np.abs(K3).max() * 2 * 3125 < EXACT
vS = np.rint(K3.astype(np.float64) @ Psi.astype(np.float64)).astype(np.int64)
del K3
eS = {s: echelon(sector_rows(vS, s) % mod) for s in sectors}
listS = {s: sorted(smith_vals(eS[s])) for s in sectors}
valS = {s: sum(listS[s]) for s in sectors}
s_full_rank = all(len(eS[s]) == n_in_sector[s] for s in sectors)
del vS
mark("S")

eA = {s: np.zeros((0, 3125), dtype=np.int64) for s in sectors}
missing, ranks, all_rows, max_abs_v = {}, {}, [], 0
reset_flags()
v_P0, c_P0 = pairing_rows(P0re[None, :] * 1, P0im[None, :] * 1, with_c=True)    # 2 [P0] (P0re = 2 chat_P0)
p0_even = bool(not np.any(v_P0 % 2) and flags["imag_zero"] and flags["divisible"])
v_P0 = v_P0 // 2
cone_float_leaves, stage_integral = [], {}
for name in ("A0", "cones", "pairs", "twisted"):
    reset_flags()
    Cr, Ci = stage_rows[name]
    for s0 in range(0, Cr.shape[0], 256):
        v = pairing_rows(Cr[s0:s0 + 256], Ci[s0:s0 + 256])
        max_abs_v = max(max_abs_v, int(np.abs(v).max()))
        all_rows.append(v.astype(np.int32) if np.abs(v).max() < 2 ** 31 else v)
        for s in sectors:
            eA[s] = echelon(np.concatenate([eA[s], sector_rows(v, s) % mod], axis=0))
        if name == "cones":
            vf = v + 2 * np.array(cone_deg[s0:s0 + 256])[:, None] * v_P0
            for s in sectors:
                Pf = sector_rows(vf, s) % mod
                if sorted(smith_vals(echelon(np.concatenate([eS[s], Pf], axis=0)))) != listS[s]:
                    cone_float_leaves.append("".join(str(t) for t in s))
    stage_integral[name] = bool(flags["imag_zero"] and flags["divisible"])
    missing[name] = {s: sum(smith_vals(eA[s])) - valS[s] for s in sectors}
    ranks[name] = {s: len(eA[s]) for s in sectors}
    mark("stage " + name)
cone_float_leaves = sorted(set(cone_float_leaves))

inside_S = all(sorted(smith_vals(echelon(np.concatenate([eS[s], eA[s]], axis=0)))) == listS[s] for s in sectors)
a_full_rank = all(ranks["twisted"][s] == n_in_sector[s] for s in sectors)
sector_table = ["".join(str(t) for t in s) + f": {odd_of[s]} odd, {n_in_sector[s]} chars, missing "
                + "/".join(str(missing[n][s]) for n in ("A0", "cones", "pairs", "twisted")) for s in sectors]

# standalone copy: the comparison with the lattices stored by the earlier floating-point computation is cut
mark("comparison")

# the float bridge on a sample of single surfaces (its planes carry the opposite sign)
# standalone copy: no floating-point bridge (C_all) to compare the sample with
sr = np.array([dense(g)[0] for g in sample])
si = np.array([dense(g)[1] for g in sample])
reset_flags()
ve2, c62 = pairing_rows(sr, si, with_c=True)
sample_even = bool(not np.any(ve2 % 2) and flags["imag_zero"] and flags["divisible"])
ve = ve2 // 2
sgn_s = np.array([-1.0 if g < NP else 1.0 for g in sample])
float_bridge_dev, planes_flipped = "not run", "not run"
# self-intersections through Psi: c . v with c = c6 / 6^5 and the factor 2 of 2 chat on both sides
self_psi = [QQ(int(np.dot(c62[j].astype(object), ve2[j].astype(object)))) / (4 * 6 ** 5) for j in range(len(sample))]
psi_self_ok = all(self_psi[j] == self_expected[g] for j, g in enumerate(sample))
p0_self = QQ(int(np.dot(c_P0[0].astype(object), (2 * v_P0[0]).astype(object)))) / (4 * 6 ** 5)

eAll = np.concatenate([eA[s] for s in sectors], axis=0)
pA, hA = kairos_save("exact_a_prime_sectors.npy", eAll)
pO, hO = kairos_save("exact_sector_offsets.npy", np.array([[len(eA[s]), len(eS[s])] for s in sectors], dtype=np.int64))
kairos_save("exact_s_sectors.npy", np.concatenate([eS[s] for s in sectors], axis=0))       # standalone: S_s for step 3
rows_all = np.concatenate([r.astype(np.int64) for r in all_rows], axis=0)
pV, hV = kairos_save("exact_a_prime_rows.npy", rows_all.astype(np.int32) if np.abs(rows_all).max() < 2 ** 31 else rows_all)
mark("saved")


def chain(name):
    return sum(missing[name][s] for s in sectors)


def by_class(name):
    out = {}
    for s in sectors:
        out.setdefault(odd_of[s], set()).add(missing[name][s])
    return {k: sorted(v) for k, v in sorted(out.items())}


auxiliary = {"rho_pure": rho_pure, "n_surfaces": NS_, "n_planes": NP, "n_p1": NP1, "n_p2": NS_ - NP - NP1,
             "outside_detail": dict(bad), "cone_deg_integral": cone_deg_integral, "n_base_pairs": n_base,
             "pairs_self_96": pairs_self_ok, "scalar_twist_trivial": scalar_twist_trivial, "p0_pairing_even": p0_even,
             "sample_even": sample_even, "object_arithmetic": flags["object_used"], "s_full_rank": s_full_rank,
             "stages_done": lap}
result = {
    "psi_kronecker": psi_kronecker, "kappa_identity": kappa_identity, "identity_all": bool(identity_all),
    "outside_half_z_zeta6": sum(bad.values()), "a0_rows_even": rows_even, "cone_div18": cone_div18,
    "n_cone_deg_not_div3": n_cone_deg_not_div3, "pairs_even": bool(pair_even), "n_identity_pairs": n_identity,
    "action_exact": action_exact, "villaflor_self_ok": villaflor_self_ok, "psi_self_ok": psi_self_ok,
    "p0_self": str(p0_self), "pairing_integral": all(stage_integral.values()), "integral_by_stage": str(stage_integral),
    "max_abs_pairing": max_abs_v, "float_bridge_dev": float_bridge_dev, "planes_flipped_in_float": planes_flipped,
    "a_full_rank": a_full_rank, "inside_S": inside_S,
    "missing_A0": chain("A0"), "missing_cones": chain("cones"), "missing_pairs": chain("pairs"), "missing_twisted": chain("twisted"),
    "per_class": "; ".join(f"{n}: {by_class(n)}" for n in ("A0", "cones", "pairs", "twisted")),
    "same_as_stored_all": "not run", "same_as_stored_count": "not run",
    "s_same_as_stored_all": "not run", "cone_float_leaves_S": ",".join(cone_float_leaves) or "nenhum",
    "saved_exact_a_prime": f"{pA} {hA}", "saved_offsets": f"{pO} {hO}", "saved_rows": f"{pV} {hV}",
    "n_rows": int(rows_all.shape[0]), "sectors": " | ".join(sector_table), "auxiliary": str(auxiliary),
}
