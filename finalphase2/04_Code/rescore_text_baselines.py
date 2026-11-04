"""
Phase 2 extension for TEXT: re-score the saved M4 token attributions under different masking tokens.

M4's text metrics (MoRF / LeRF / ABPC, benchmark-nlp) replace the top-k / bottom-k tokens by [PAD]; the
Paddle-style attention mask then drops those tokens (m4bench/text_models.py). That is one specific choice.
Here the same saved per-token maps (results/m4/text/<model>/<variant>/maps/<doc>.npy) are evaluated with

    pad   [PAD]   M4 default ("before"); masked tokens are removed from attention
    mask  [MASK]  masked tokens stay visible as a placeholder the model saw in pre-training
    unk   [UNK]   masked tokens stay visible as an "unknown word" placeholder

Only forward passes are needed; no explainer is re-run. The perturbation schedule is M4's own
(m4bench.text_metrics.perturbation_ids, percentiles 0..100, [CLS]/[SEP] always kept).

Output per (model, variant): results/m4/text_rescore/<model>/<variant>.npy, float16, shape
(n_baselines, n_docs, 2, 101): MoRF and LeRF class-probability curves, the same layout as the vision
re-scoring, so analyze_baselines.py can read it.

    python rescore_text_baselines.py --models bert_base --variants ig --limit 5 --out /tmp/check   # quick test
    python rescore_text_baselines.py --models all --limit 200

Variants: ig, sg, lime and, for GA / BT-head / BT-token, the start layer with the best ABPC per model
(the same "best_abpc" rule as aggregate_results.py), read from results/m4/summary/text_scores.csv.
Documents whose saved map length differs from the current tokenization are left out (docs.txt lists the ones used).
Caveat: the saved maps use the squared reduction; M4's own IG/SG metric uses the absolute one.
Grey-equivalent check: re-scoring with [PAD] is compared against the stored per-document ABPC.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from m4bench.common import get_device, seed_everything
from m4bench.text_metrics import perturbation_ids
from m4bench.text_models import TEXT_MODEL_NAMES, batched_probs, load_movies, load_text_model

BASELINES = ["pad", "mask", "unk"]
SINGLE = ["ig", "sg", "lime"]
LAYERED = ["ga", "bt_head", "bt_token"]


def fill_id(tm, name: str) -> int:
    ids = {"pad": tm.tok.pad_token_id, "mask": tm.tok.mask_token_id, "unk": tm.tok.unk_token_id}
    if ids[name] is None:
        raise ValueError(f"{tm.name}: tokenizer has no {name} token")
    return int(ids[name])


def pick_variants(model: str, scores_csv: str) -> list:
    """ig / sg / lime plus, for each layered method, the start layer with the highest mean ABPC."""
    d = pd.read_csv(scores_csv)
    d = d[d.model == model]
    out = list(SINGLE)
    for m in LAYERED:
        c = d[d.method.str.startswith(m + "-")]
        out.append(c.loc[c.ABPC.idxmax(), "method"])
    return out


def aligned_docs(tm, texts, maps_dir: Path, limit: int) -> list:
    """Indices of the first `limit` documents whose saved map has exactly one value per token.
    A few Phase 1 maps are one token longer than what the current tokenizer produces (the cause was not
    found), so those documents cannot be re-scored faithfully and are left out instead of guessed."""
    return [i for i in range(limit) if len(np.load(maps_dir / f"{i}.npy")) == len(tm.encode(texts[i])[0])]


def rescore_variant(tm, texts, maps_dir: Path, baselines, device, bs, docs, amp=False):
    fills = [fill_id(tm, b) for b in baselines]
    out = np.zeros((len(baselines), len(docs), 2, 101), np.float16)
    for row, i in enumerate(docs):
        ids = tm.encode(texts[i])[0].numpy()
        expl = np.load(maps_dir / f"{i}.npy")
        assert len(expl) == len(ids), (maps_dir, i, len(expl), len(ids))
        stacks = []
        for f in fills:
            morf, lerf = perturbation_ids(ids, expl, f)
            stacks.append(np.concatenate([morf, lerf]))
        probs = batched_probs(tm, np.concatenate(stacks), device, bs, amp)
        coi = int(probs[0].argmax())                      # class of interest = prediction on the clean text
        out[:, row] = probs[:, coi].cpu().numpy().reshape(len(baselines), 2, 101)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--variants", nargs="+", default=None, help="default: ig sg lime + best GA/BT start layer")
    p.add_argument("--baselines", nargs="+", default=BASELINES, choices=BASELINES)
    p.add_argument("--maps_root", default="results/m4/text")
    p.add_argument("--scores_csv", default="results/m4/summary/text_scores.csv")
    p.add_argument("--out", default="results/m4/text_rescore")
    p.add_argument("--data_dir", default="data/movies")
    p.add_argument("--split", default="val")
    p.add_argument("--model_root", default="models")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--amp", action="store_true", help="bf16 forward passes (faster, not bit-exact)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    texts, _ = load_movies(args.data_dir, args.split)
    limit = min(args.limit, len(texts))
    for name in TEXT_MODEL_NAMES if args.models == ["all"] else args.models:
        variants = args.variants or pick_variants(name, args.scores_csv)
        tm = load_text_model(name, device, args.model_root)
        for v in variants:
            dst = Path(args.out) / name / f"{v}.npy"
            maps_dir = Path(args.maps_root) / name / v / "maps"
            docs = aligned_docs(tm, texts, maps_dir, limit)
            if dst.exists() and np.load(dst, mmap_mode="r").shape[1] == len(docs):
                print(f"[{name}/{v}] already done, skipping", flush=True)
                continue
            t0 = time.time()
            curves = rescore_variant(tm, texts, maps_dir, args.baselines, device, args.bs, docs, args.amp)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, curves)
            (dst.parent / "baselines.txt").write_text("\n".join(args.baselines))
            (dst.parent / "docs.txt").write_text("\n".join(map(str, docs)))   # same set for every variant of a model
            print(f"[{name}/{v}] {len(docs)}/{limit} docs x {len(args.baselines)} baselines in "
                  f"{(time.time() - t0) / 60:.1f} min -> {dst}", flush=True)
        del tm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
