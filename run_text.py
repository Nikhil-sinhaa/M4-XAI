"""
M4 text benchmark: explain + evaluate (MoRF, LeRF, ABPC, INFD) on the MovieReview validation split for
every fine-tuned model (train_text.py first).

    python run_text.py --models all
    python run_text.py --models bert_base --start_layers 4     # a single GA/BT start layer

GA/BT are run for every start layer the authors ran (benchmark-nlp/run_expl.sh):
12-layer models 0 2 4 6 8 10 11, DistilBERT 0-5, BERT-large 0 4 8 12 16 20 23. Each variant is saved
as e.g. "bt_head-4"; the paper does not state which one its figures use.

Outputs (resumable):
    results/m4/text/<model>/<method>/scores.csv    per-document MoRF, LeRF, ABPC, INFD
    results/m4/text/<model>/<method>/maps/<i>.npy  per-token "sq" maps for PScore
    results/m4/text/<model>/info.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from m4bench.common import ScoreLog, get_device, seed_everything
from m4bench.text_explainers import EXPLAINERS
from m4bench.text_metrics import InfidelityCacheNLP, morf_lerf_abpc
from m4bench.text_models import TEXT_METHODS, TEXT_MODEL_NAMES, load_movies, load_text_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"], help=f"'all' or any of {TEXT_MODEL_NAMES}")
    p.add_argument("--methods", nargs="+", default=TEXT_METHODS)
    p.add_argument("--start_layers", nargs="+", type=int, default=None,
                   help="GA/BT start layers (default: the authors' sweep for the model's depth)")
    p.add_argument("--data_dir", default="data/movies")
    p.add_argument("--split", default="val", help="M4 evaluates on the validation split")
    p.add_argument("--model_root", default="models")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="results/m4/text")
    p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--sg_samples", type=int, default=50)
    p.add_argument("--sg_noise", type=float, default=0.1)
    p.add_argument("--lime_samples", type=int, default=1000)
    p.add_argument("--bt_steps", type=int, default=20)
    p.add_argument("--no_infd", action="store_true", help="skip INFD (~5 x length extra forward passes)")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad_bs", type=int, default=10)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


AUTHORS_START_LAYERS = {6: [0, 1, 2, 3, 4, 5], 12: [0, 2, 4, 6, 8, 10, 11], 24: [0, 4, 8, 12, 16, 20, 23]}


def variants(methods, start_layers):
    out = []
    for m in methods:
        if m in ("ga", "bt_head", "bt_token"):
            out += [(f"{m}-{sl}", m, sl) for sl in start_layers]
        else:
            out.append((m, m, None))
    return out


def run_model(name, texts, labels, args, device):
    tm = load_text_model(name, device, args.model_root)
    n_layers = len(tm.layers())
    vs = [(v, m, sl) for v, m, sl in variants(args.methods, args.start_layers or AUTHORS_START_LAYERS[n_layers])
          if sl is None or sl < n_layers]
    out = Path(args.out) / name
    logs = {v: ScoreLog(out / v / "scores.csv", ["MoRF", "LeRF", "ABPC", "INFD", "INFD_sq", "INFD_sum"])
            for v, _, _ in vs}
    for v, _, _ in vs:
        (out / v / "maps").mkdir(parents=True, exist_ok=True)
    kw = dict(ig_steps=args.ig_steps, sg_samples=args.sg_samples, sg_noise=args.sg_noise,
              lime_samples=args.lime_samples, bt_steps=args.bt_steps, grad_bs=args.grad_bs)

    todo = [i for i in range(len(texts)) if any(i not in logs[v].done for v, _, _ in vs)]
    print(f"[{name}] variants={[v for v, _, _ in vs]} docs={len(texts)} remaining={len(todo)}", flush=True)
    correct, t0 = 0, time.time()
    for step, idx in enumerate(todo):
        ids = tm.encode(texts[idx]).to(device)
        with torch.no_grad():
            label = int(tm.logits(ids).argmax(1))
        correct += int(label == labels[idx])
        infd = None if args.no_infd else InfidelityCacheNLP(tm, ids, device, args.bs, args.amp)
        for vi, (v, m, sl) in enumerate(vs):
            if idx in logs[v].done:
                continue
            rng = np.random.default_rng((args.seed, idx, vi))
            expl = EXPLAINERS[m](tm, ids, label, rng, start_layer=sl, **kw)
            scores, _, _ = morf_lerf_abpc(tm, ids, expl["abs"], device, args.bs, args.amp)
            # evaluate_2.py scores the "-abs", "-sq" and "-sum" reductions of IG/SG
            for col, key in (("INFD", "abs"), ("INFD_sq", "sq"), ("INFD_sum", "sum")):
                scores[col] = infd.score(expl[key]) if infd else float("nan")
            np.save(out / v / "maps" / f"{idx}.npy", expl["sq"].astype(np.float32))
            logs[v].add(idx, scores)
        rate = (step + 1) / (time.time() - t0)
        if (step + 1) % 10 == 0 or step + 1 == len(todo):
            print(f"  [{name}] {step + 1}/{len(todo)}  ETA {(len(todo) - step - 1) / rate / 3600:.1f} h", flush=True)

    info = {"model": name, "variants": [v for v, _, _ in vs], "n_docs": len(texts), "n_layers": n_layers,
            "max_len": tm.max_len, "split": args.split}
    if todo:
        info["accuracy_on_newly_evaluated"] = correct / len(todo)
    (out / "info.json").write_text(json.dumps(info, indent=2))
    del tm
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    texts, labels = load_movies(args.data_dir, args.split)
    if args.limit:
        texts, labels = texts[:args.limit], labels[:args.limit]
    for name in TEXT_MODEL_NAMES if args.models == ["all"] else args.models:
        run_model(name, texts, labels, args, device)


if __name__ == "__main__":
    main()
