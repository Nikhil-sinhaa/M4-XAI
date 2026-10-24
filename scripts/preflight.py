"""
Fail-fast check (and prefetch) of everything the full M4 run needs, so a multi-day job does not die
hours in on a missing file.

    .venv/bin/python scripts/preflight.py                # check data, fetch all weights / checkpoints
    .venv/bin/python scripts/preflight.py --no_synscore  # SynScore training images not required
    .venv/bin/python scripts/preflight.py --no_download  # only check what is on disk

Exit status 0 only when every requirement is met.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBLEMS = []


def problem(msg):
    print(f"  [MISSING] {msg}")
    PROBLEMS.append(msg)


def ok(msg):
    print(f"  [ok]      {msg}")


def is_jpeg(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(3) == b"\xff\xd8\xff"
    except OSError:
        return False


def check_images(name, list_file, data_dir):
    rel = [l.strip() for l in open(ROOT / list_file) if l.strip()]
    bad = [r for r in rel if not is_jpeg(ROOT / data_dir / r)]
    if bad:
        problem(f"{name}: {len(bad)} of {len(rel)} images missing/corrupt under {data_dir} (e.g. {bad[0]})")
    else:
        ok(f"{name}: all {len(rel)} images present")


def check_movies(data_dir):
    d = ROOT / data_dir
    missing = [str(d / f"{s}.jsonl") for s in ("train", "val") if not (d / f"{s}.jsonl").exists()]
    if not (d / "docs").is_dir():
        missing.append(str(d / "docs"))
    for m in missing:
        problem(f"MovieReview: {m}")
    if not missing:
        ok("MovieReview train/val/docs present")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no_synscore", action="store_true", help="do not require the 10k ImageNet-train images")
    p.add_argument("--no_download", action="store_true", help="do not fetch weights / HF checkpoints")
    a = p.parse_args()

    print("data")
    check_images("ImageNet val (M4_5000)", "benchmark_data/imagenet_val_5k.txt", "data/M4_5000")
    if not a.no_synscore:
        check_images("ImageNet train 10k (SynScore)", "benchmark_data/imagenet_train_10k.txt",
                     "data/imagenet_train_10k")
    check_movies("data/movies")

    if not a.no_download:
        print("pretrained weights")
        from m4bench.paddle_models import MAE_FINETUNED_URL, PADDLE_URLS, download
        for url in [*PADDLE_URLS.values(), MAE_FINETUNED_URL]:
            try:
                download(url)
                ok(url.rsplit("/", 1)[-1])
            except Exception as e:                                  # noqa: BLE001
                problem(f"{url}: {e}")
        print("text checkpoints (HuggingFace)")
        from huggingface_hub import snapshot_download
        from m4bench.text_models import TEXT_MODELS
        for name, cfg in TEXT_MODELS.items():
            try:
                snapshot_download(cfg["hf"], allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
                ok(cfg["hf"])
            except Exception as e:                                  # noqa: BLE001
                problem(f"{cfg['hf']}: {e}")

    if PROBLEMS:
        print(f"\npreflight FAILED: {len(PROBLEMS)} problem(s)")
        sys.exit(1)
    print("\npreflight passed")


if __name__ == "__main__":
    main()
