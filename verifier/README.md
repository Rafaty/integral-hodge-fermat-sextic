# Verifier for "The integral Hodge conjecture for the Fermat sextic fourfold"

This folder lets a reader recompute, outside the KAIROS system, every lattice computation the proof uses. It contains
the programs that KAIROS certified (`programs/`), a small runner (`run.py`), the checks of each program
(`checks_*.json`), the SHA-256 digests of the input files (`MANIFEST.txt`) and the provenance of every program
(`PROVENANCE.md`). No computation uses rounding: float64 matrix products appear only on integers whose partial sums
stay below 2^53, and every such bound is asserted in the code.

Requirements: SageMath 10 (with its numpy) and about 8 GB of memory. The times below were measured on one core.

## What each step proves

| step | program | what it establishes | time |
|---|---|---|---|
| `s3` | `programs/verify_s3.py` | The rows of `K3.npy` are, modulo 3^12, a Z_3-basis of the saturated kernel of the period constraints, hence of S ⊗ Z_3 (Lemma 2.1): they satisfy the constraints modulo 3^15 and are independent modulo 3, and the constraint matrix has rank 854 with Smith valuations at most 3. It also gives the 3-adic discriminant of S (valuation 1282). | 4 min |
| `a_prime` | `programs/exact_a_prime.py` | Every generator of A' is computed from its equations: the Villaflor functional over Q(zeta12, 2^(1/3)), the exact bridge to Pham coordinates (Lemma 2.3) and the exact integer pairing vector. It checks the bridge: the Kronecker form of Psi, the Riemann identity on all 1751 Hodge characters, the self-intersections 125/6, 60 and 96, and the group action against direct determinants. It also checks that every class is integral and that A' ⊂ S, and computes the index chain 3^540 → 3^345 → 3^120 → 3^105 sector by sector. | 56 min |
| `transport` | `programs/exact_transport.py` | The transport of Theorem 5.6 modulo 3, with integer operators on the Pham lattice, on the A' of the previous step. In each of the ten sectors with four odd characters and even pair in x_1..x_5, A'_s and the transported hub classes span S_s modulo 3, and Nakayama's lemma gives S_s ⊗ Z_3. The other five sectors follow by symmetry (Section 7). | 4 min |

Run from this folder:

```
sage -python run.py all          # or: s3, a_prime, transport (in this order)
```

Each step prints its checks (`[ok]` or `[FAILED]`) and writes its full result to `out/result_<step>.json`. Step
`transport` reads the lattices that step `a_prime` saves in `out/saved/`.

## Input files

They are in `data/` (another folder can be given with `--data DIR`). The runner refuses a file whose SHA-256 differs
from `MANIFEST.txt`, and an input shipped as `name.xz` is decompressed on first use and then checked like the others.

- `g13/K3.npy` (shipped as `K3.npy.xz`, 3.2 MB for 57 MB): the 3-adic basis of S, 2271 rows of residues modulo 3^15,
  computed by the method of Aljovin, Movasati and Villaflor Loyola. Step `s3` checks it from the definition of S.
- `g14/psi_sign.npy`: the sign (+1) with which Psi is the intersection form.
- `x2int_char0_G252.json`, `x2int_lambda_basis.json`, `x2int_N_basis.json`: the 252 known curves of the K3 surface
  K = {u^2 + x^6 + y^6 + z^6 = 0} and the saturation NS(K), from which the quadric-cone classes are built.
- `g13/Psi.npy`, the intersection matrix of the Pham cycles, need not be supplied. If it is absent the runner builds it
  from Pham's formula, Psi = V^(x5) + (V^T)^(x5) with V = 1 - shift, and checks its digest like any input.

## Reproduction

On 1 November 2026, `sage -python run.py all` ran in a clean folder on one core. Its inputs were `K3.npy`,
`psi_sign.npy` and the three json files; Psi was built from Pham's formula. It passed every check in about 61
minutes. Its outputs are identical, byte for byte, to those of the certified run:

```
db3e43536fd6f721f2edfc113360834b2bb16f892df1c5c79378683907e5585c  saved/exact_a_prime_sectors.npy
89e57389f368663a9a5c982eea4964f0008479894b53870471294f6bab50746a  saved/exact_sector_offsets.npy
1644fa02695cc649eeda5ec649ebc62db21cecf2f9228938bf42ae0bc6c25734  saved/exact_a_prime_rows.npy
e1d2af0957567e8ed19caac126ee0be94268d3098a77844f23be9970391502ea  data/g13/Psi.npy (built from the formula)
```

## Differences from the certified runs

The programs are the certified texts. Two certified runs also compared their results with an earlier floating-point
computation, which needs 600 MB of float data that the proof does not use. Those comparisons are cut from the
standalone copies, and the transport reads the exact S_s that step `a_prime` saves instead of a stored copy.
`PROVENANCE.md` lists, for each program, the KAIROS certificate (job, number of checks, SHA-256 of the certified text)
and every change.
