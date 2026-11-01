# Provenance of the verifier programs

Each program in `programs/` is the text certified by KAIROS, with the cuts listed below. The certificates (job,
checks, SHA-256 of the certified program text) are:

| step | KAIROS task | job | checks | time | SHA-256 of the certified program |
|---|---|---|---|---|---|
| `s3` | `base_3adica_S` | j20261101-030814-5c533640 | 6 of 6 | 213 s | `c8a1baf54fdfaa40477c43c64e7b7dee1a0b47b4d5d8e49505f5468af0c6a07b` |
| `a_prime` | `a_linha_exato` | j20261101-030311-882581ff | 17 of 17 | 3356 s | `89ccb1843737fd7436ddea4a4bddb760e047328854b88816e558b5ddb6501010` |
| `transport` | `transporte_exato_v3` | j20261101-040126-0f2bcafd | 9 of 9 | 210 s | `42c9dd5521483715ea1673b9d051b49a1bff3a40eed307f553f7a4be4c74b310` |

All three ran in the conversation 20261030-210605-8e6df4 of the research problem research:20261019-030536-176da7.

## `programs/verify_s3.py`

SHA-256 of the standalone copy: `c8a1baf54fdfaa40477c43c64e7b7dee1a0b47b4d5d8e49505f5468af0c6a07b`.

- identical to the certified program.

## `programs/exact_a_prime.py`

SHA-256 of the standalone copy: `fb5a0d4adc9bf787c6c5e3c45435c1c43acd1efabb71fcae3762621f5d214573`.

- comparison of A'_s and S_s with the lattices stored by the floating-point computation: cut
- comparison of a sample of surfaces with the floating-point bridge C_all: cut
- the two outputs of that comparison are reported as 'not run'
- added: the exact S_s are saved too, for the transport step
- the outputs of the cut comparison are reported as 'not run'

## `programs/exact_transport.py`

SHA-256 of the standalone copy: `f009ff8cc7d3108473079c59601a39b4c735022978ec8ef84380e7260590a4f3`.

- the floating-point Pham coordinates C_all are not read
- A'_s is read from the output of step a_prime
- S_s is read from the output of step a_prime (exact, from K3) instead of the stored copy
- the sector offsets of S_s come from step a_prime
- the sector offsets of A'_s come from step a_prime
- the control 'planes and complete intersections of C_all fall in A'_s' is cut (in step a_prime they are generators of A')
- (same control, inside the loop)
- (same control)
- (same control: output 'not run')
- (same control: output 'not run')

The checks of the cut comparisons (`s_guardado`, `a0_dentro`, `a0_inteiros`) are not part of the standalone checks.
