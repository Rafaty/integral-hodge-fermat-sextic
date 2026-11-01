# The integral Hodge conjecture for the Fermat sextic fourfold

Rafael Santana de Jesus

This repository contains the paper and a standalone verifier for its computations. The paper proves that every
integral Hodge class of degree four on the Fermat sextic fourfold X = {x_0^6 + ... + x_5^6 = 0} in P^5 is the class of
an algebraic cycle. So the integral Hodge conjecture holds for X.

## Contents

- `paper/x6_integral_hodge.pdf` is the paper.
- `verifier/` is a standalone verifier. In plain SageMath it recomputes every lattice computation the proof uses:
  - the lattice S ⊗ Z_3 of integral primitive Hodge classes, from its definition;
  - the exact lattice A' of known algebraic classes, each generator computed from its equations, with the index chain
    3^540 → 3^345 → 3^120 → 3^105;
  - the transport by Hodge endomorphisms of the Fermat sextic curve, which closes the last 105 directions.

  `verifier/README.md` explains what each step proves.

## Verifying

```
cd verifier
sage -python run.py all
```

You need SageMath 10 and about 8 GB of memory. The run takes about one hour on one core. Each step prints its checks.
The input files are in `verifier/data/`, with their SHA-256 digests in `verifier/MANIFEST.txt`. The outputs agree, byte
for byte, with the certified run; the digests are listed in `verifier/README.md`.

## About the computations

The computations were designed and certified with KAIROS, an automated research system developed by the author.
KAIROS itself is private, and it is not needed to check the paper: the verifier runs the certified programs directly.
`verifier/PROVENANCE.md` lists, for each program, its certificate and the SHA-256 of the certified text.

## Citation

Rafael Santana de Jesus, *The integral Hodge conjecture for the Fermat sextic fourfold*, 2026,
https://github.com/Rafaty/integral-hodge-fermat-sextic. Machine-readable metadata is in `CITATION.cff`.

## License

- The paper (`paper/x6_integral_hodge.pdf`) and the data in `verifier/data/` are licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see `LICENSE-CC-BY-4.0.txt`.
- The code (`verifier/run.py` and `verifier/programs/`) is licensed under the MIT License; see `LICENSE`.

## Contact

Rafael Santana de Jesus, rafael.santanarf18@gmail.com
