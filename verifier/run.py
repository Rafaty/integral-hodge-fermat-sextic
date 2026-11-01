"""Standalone runner for the verification programs of the paper, outside KAIROS.

Each program is the one certified by KAIROS (same text), run with two small functions in place of the system's:
kairos_data(name) reads an input file from the data folder (decompressing name.xz on first use) and checks its
SHA-256 against MANIFEST.txt; kairos_save(name, array) writes an output into the out folder. The checks of each
program are evaluated at the end.

Usage (in a SageMath shell, from this folder):
    sage -python run.py s3          # the 3-adic basis of S (minutes)
    sage -python run.py a_prime     # the exact A' and the index chain (about 30 minutes)
    sage -python run.py transport   # the exact transport, on the A' just computed (minutes)
    sage -python run.py all
Options: --data DIR (default ./data), --out DIR (default ./out)."""
import argparse
import hashlib
import json
import lzma
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROGRAMS = {
    "s3": ("verify_s3.py", "checks_s3.json"),
    "a_prime": ("exact_a_prime.py", "checks_a_prime.json"),
    "transport": ("exact_transport.py", "checks_transport.json"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(data_dir):
    out = {}
    for line in (HERE / "MANIFEST.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            digest, name = line.split(None, 1)
            out[name] = digest
    return out


def build_psi(path):
    """The intersection matrix of the Pham cycles, Psi = V^(x5) + (V^T)^(x5) with V = 1 - shift (Pham), written once;
    its digest is then checked against MANIFEST.txt like any input."""
    v1 = np.eye(5, dtype=np.int64)
    for j in range(4):
        v1[j, j + 1] = -1
    v = v1
    for _ in range(4):
        v = np.kron(v, v1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, v + v.T)


def make_io(data_dir, out_dir, digests):
    def kairos_data(name):
        path = data_dir / name
        if name == "g13/Psi.npy" and not path.exists():
            build_psi(path)
        packed = path.with_name(path.name + ".xz")
        if not path.exists() and packed.exists():          # inputs may be shipped xz-compressed
            with lzma.open(packed) as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
        if not path.exists():
            path = out_dir / name
        if name in digests and sha256(path) != digests[name]:
            raise SystemExit(f"{name}: SHA-256 differs from MANIFEST.txt")
        if name.endswith(".npy"):
            return np.load(path, allow_pickle=False)
        return json.loads(path.read_text(encoding="utf-8"))

    def kairos_save(name, array):
        path = out_dir / "saved" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        return str(path), sha256(path)
    return kairos_data, kairos_save


def run(key, data_dir, out_dir):
    program, checks_file = PROGRAMS[key]
    kairos_data, kairos_save = make_io(data_dir, out_dir, manifest(data_dir))
    scope = {"kairos_data": kairos_data, "kairos_save": kairos_save, "__name__": "kairos_program"}
    source = (HERE / "programs" / program).read_text(encoding="utf-8")
    exec(compile(source, program, "exec"), scope)
    result = scope["result"]
    checks = json.loads((HERE / checks_file).read_text(encoding="utf-8"))
    ok = True
    for chk in checks:
        value = result.get(chk["output"])
        passed = value == chk["expected"]
        ok = ok and passed
        print(f"[{'ok' if passed else 'FAILED'}] {chk['name']}: {chk['output']} = {value!r} (expected {chk['expected']!r})")
    (out_dir / f"result_{key}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"{key}: {'all checks passed' if ok else 'some checks FAILED'}; result in {out_dir / f'result_{key}.json'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=sorted(PROGRAMS) + ["all"])
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["s3", "a_prime", "transport"] if args.what == "all" else [args.what]
    sys.exit(0 if all([run(k, data_dir, out_dir) for k in keys]) else 1)


if __name__ == "__main__":
    main()
