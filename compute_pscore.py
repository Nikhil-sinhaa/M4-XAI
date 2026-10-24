"""
PScore (Eq. 5): cosine similarity between a model's explanation and the consensus (mean) of the same
method's explanations over all models, per sample, then averaged (benchmark-cv/evaluate_consensus.py).

Run after run_vision.py / run_text.py have finished the models you want in the consensus.

Vision, exactly as evaluate_consensus.py: raw maps at the method's native resolution (IG/SG 224x224
from all 9 models, GradCAM 7x7 from the 5 CNNs, GA/BT 14x14 from the 4 ViTs), squared-mean channel
reduction for IG/SG, unweighted mean over all models (the evaluated one included), cosine per image.

Text: the authors did not publish text PScore code. The same procedure is applied to the models that
share the BERT-uncased vocabulary, so tokens align (DistilBERT, BERT-B, BERT-L, ERNIE-2.0). RoBERTa is
excluded, as in Fig. 5. Documents are compared over the common prefix (BERT-L is truncated at 300
tokens), skipping [CLS] and the last position.

    python compute_pscore.py --modality vision
    python compute_pscore.py --modality text
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from m4bench.text_models import load_movies


def done_indices(scores_csv: Path) -> set:
    with open(scores_csv) as f:
        return {int(r["idx"]) for r in csv.DictReader(f)}


def cosine_to_consensus(stack: np.ndarray) -> np.ndarray:
    """stack: (n_models, D) -> cosine of each row with the row mean."""
    stack = stack.astype(np.float64)
    cons = stack.mean(0)
    num = stack @ cons
    den = np.linalg.norm(stack, axis=1) * np.linalg.norm(cons)
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def write(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "PScore"])
        w.writerows(rows)


def vision(root: Path):
    by_method = {}
    for maps in root.glob("*/*/maps_sq.npy"):
        model, method = maps.parent.parent.name, maps.parent.name
        by_method.setdefault(method, []).append(model)
    for method, models in sorted(by_method.items()):
        if len(models) < 2:
            print(f"  skip {method}: only {models}")
            continue
        models.sort()
        common = set.intersection(*(done_indices(root / m / method / "scores.csv") for m in models))
        arrs = {m: np.load(root / m / method / "maps_sq.npy", mmap_mode="r") for m in models}
        rows = {m: [] for m in models}
        for idx in sorted(common):
            cos = cosine_to_consensus(np.stack([np.asarray(arrs[m][idx], np.float32).ravel() for m in models]))
            for m, c in zip(models, cos):
                rows[m].append((idx, float(c)))
        for m in models:
            write(root / m / method / "pscore.csv", rows[m])
        print(f"  {method}: {len(models)} models, {len(common)} images -> " +
              ", ".join(f"{m}={np.mean([c for _, c in rows[m]]):.3f}" for m in models))


def text(root: Path, data_dir: str, split: str, model_root: str):
    from transformers import AutoTokenizer
    from m4bench.text_models import TEXT_MODELS, finetuned_dir
    shared = ["distilbert", "bert_base", "bert_large", "ernie_2_base"]      # BERT-uncased vocabulary
    texts, _ = load_movies(data_dir, split)
    by_method = {}
    for d in root.glob("*/*/maps"):
        model, method = d.parent.parent.name, d.parent.name
        if model in shared:
            by_method.setdefault(method, []).append(model)
    toks = {}
    for method, models in sorted(by_method.items()):
        if len(models) < 2:
            print(f"  skip {method}: only {models}")
            continue
        models.sort()
        for m in models:
            if m not in toks:
                toks[m] = AutoTokenizer.from_pretrained(finetuned_dir(m, model_root))
        common = set.intersection(*(done_indices(root / m / method / "scores.csv") for m in models))
        rows = {m: [] for m in models}
        skipped = 0
        for idx in sorted(common):
            ids = {m: toks[m](texts[idx], truncation=True, max_length=TEXT_MODELS[m]["max_len"])["input_ids"]
                   for m in models}
            L = min(len(v) for v in ids.values())
            ref = ids[models[0]][1:L - 1]
            if any(ids[m][1:L - 1] != ref for m in models):
                skipped += 1                # tokenisations disagree: cannot align this document
                continue
            stack = [np.load(root / m / method / "maps" / f"{idx}.npy")[1:L - 1] for m in models]
            for m, c in zip(models, cosine_to_consensus(np.stack(stack))):
                rows[m].append((idx, float(c)))
        for m in models:
            write(root / m / method / "pscore.csv", rows[m])
        print(f"  {method}: {models}, {len(common) - skipped} docs ({skipped} unaligned) -> " +
              ", ".join(f"{m}={np.mean([c for _, c in rows[m]]):.3f}" for m in models if rows[m]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=["vision", "text"], required=True)
    p.add_argument("--root", default=None, help="default results/m4/<modality>")
    p.add_argument("--data_dir", default="data/movies")
    p.add_argument("--split", default="val")
    p.add_argument("--model_root", default="models")
    a = p.parse_args()
    root = Path(a.root or f"results/m4/{a.modality}")
    if a.modality == "vision":
        vision(root)
    else:
        text(root, a.data_dir, a.split, a.model_root)


if __name__ == "__main__":
    main()
