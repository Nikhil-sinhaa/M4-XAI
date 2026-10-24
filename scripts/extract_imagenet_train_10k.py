"""
Pull the 10,000 SynScore training images (benchmark_data/imagenet_train_10k.txt) out of a full ImageNet
train archive, without unpacking the rest. Accepts either form:

  * Kaggle imagenet-object-localization-challenge zip  (members ILSVRC/Data/CLS-LOC/train/<wnid>/<file>.JPEG)
  * official ILSVRC2012_img_train.tar                   (nested <wnid>.tar files, each holding the JPEGs)
  * an already-unpacked train directory                 (<wnid>/<file>.JPEG)

    .venv/bin/python scripts/extract_imagenet_train_10k.py /path/to/imagenet-object-localization-challenge.zip
    .venv/bin/python scripts/extract_imagenet_train_10k.py /path/to/ILSVRC2012_img_train.tar
    .venv/bin/python scripts/extract_imagenet_train_10k.py /path/to/train_dir

Files already present as valid JPEGs in data/imagenet_train_10k are kept. Exits non-zero if any listed
image was not found in the source.
"""

import argparse
import io
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def is_jpeg_bytes(b: bytes) -> bool:
    return b[:3] == b"\xff\xd8\xff"


def have(out: Path, rel: str) -> bool:
    p = out / rel
    try:
        with open(p, "rb") as f:
            return is_jpeg_bytes(f.read(3))
    except OSError:
        return False


def write(out: Path, rel: str, data: bytes) -> bool:
    if not is_jpeg_bytes(data):
        return False
    dest = out / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def from_zip(src, wanted, out):
    with zipfile.ZipFile(src) as z:
        names = {n for n in z.namelist() if n.endswith(".JPEG")}
        # Kaggle prefixes members with ILSVRC/Data/CLS-LOC/train/; accept any prefix ending in <wnid>/<file>
        index = {"/".join(n.split("/")[-2:]): n for n in names}
        for i, rel in enumerate(wanted, 1):
            if rel in index and write(out, rel, z.read(index[rel])):
                pass
            if i % 1000 == 0:
                print(f"  {i}/{len(wanted)} processed", flush=True)


def from_outer_tar(src, wanted, out):
    by_wnid = {}
    for rel in wanted:
        by_wnid.setdefault(rel.split("/")[0], set()).add(rel.split("/")[1])
    with tarfile.open(src) as outer:
        for m in outer:
            wnid = Path(m.name).stem
            if not m.isfile() or wnid not in by_wnid:
                continue
            inner_bytes = outer.extractfile(m).read()
            with tarfile.open(fileobj=io.BytesIO(inner_bytes)) as inner:
                for im in inner:
                    if im.isfile() and im.name in by_wnid[wnid]:
                        write(out, f"{wnid}/{im.name}", inner.extractfile(im).read())
            print(f"  {wnid} done", flush=True)


def from_dir(src, wanted, out):
    for rel in wanted:
        s = Path(src) / rel
        if s.exists() and is_jpeg_bytes(s.read_bytes()[:3]):
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, out / rel)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="Kaggle .zip, ILSVRC2012_img_train.tar, or an unpacked train directory")
    p.add_argument("--list", default=str(ROOT / "benchmark_data/imagenet_train_10k.txt"))
    p.add_argument("--out", default=str(ROOT / "data/imagenet_train_10k"))
    a = p.parse_args()

    out = Path(a.out)
    wanted_all = [l.strip() for l in open(a.list) if l.strip()]
    wanted = [r for r in wanted_all if not have(out, r)]
    print(f"{len(wanted_all) - len(wanted)} already present, {len(wanted)} to extract")

    src = Path(a.source)
    if src.is_dir():
        from_dir(src, wanted, out)
    elif src.suffix == ".zip":
        from_zip(src, wanted, out)
    elif ".tar" in src.name:
        from_outer_tar(src, wanted, out)
    else:
        sys.exit(f"don't know how to read {src}")

    missing = [r for r in wanted_all if not have(out, r)]
    print(f"extracted; {len(wanted_all) - len(missing)}/{len(wanted_all)} images in {out}")
    if missing:
        print(f"{len(missing)} still missing, e.g. {missing[:3]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
