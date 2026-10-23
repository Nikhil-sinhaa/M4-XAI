"""
Phase 2: Baseline Sensitivity Analysis — NLP/Text Modality
===========================================================
Uses BERT-base-uncased (pre-trained, not fine-tuned) with SST-2 sentiment
analysis data from HuggingFace datasets as a proxy for the MovieReview
dataset (which requires local files).

If a fine-tuned model at --model_dir exists, it is used instead.

Token masking baselines:
  1. pad  — [PAD] token  (original M4 paper baseline)
  2. mask — [MASK] token (BERT's native masked token, more in-distribution)
  3. unk  — [UNK] token

Attribution: Integrated Gradients
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import IntegratedGradients

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
BASELINES = ["pad", "mask", "unk"]
BASELINE_LABELS = {
    "pad":  "[PAD] Token [M4 Paper]",
    "mask": "[MASK] Token (in-dist.)",
    "unk":  "[UNK] Token",
}
BASELINE_COLORS = {
    "pad":  "#E63946",
    "mask": "#2A9D8F",
    "unk":  "#E9C46A",
}

# Sample movie-style texts for the NLP experiment
# (used as fallback when MovieReview data is unavailable)
SAMPLE_TEXTS = [
    "This film is a masterpiece of cinema with breathtaking performances and stunning visual imagery.",
    "A terrible waste of two hours. The plot makes no sense and the acting is wooden throughout.",
    "The director manages to create a genuinely moving experience that stays with you long after the credits roll.",
    "Predictable from start to finish, this formulaic thriller fails to deliver any real tension or surprise.",
    "A beautiful and thought-provoking exploration of human resilience and the bonds of family.",
    "Overwrought and melodramatic, the film mistakes loudness for emotion and spectacle for substance.",
    "Clever, funny, and surprisingly heartfelt. A rare comedy that has genuine wit and warmth.",
    "The special effects are impressive, but they cannot compensate for a script that is utterly vacuous.",
    "An intelligent thriller with razor-sharp dialogue and an ending that genuinely surprised me.",
    "Poorly paced, confusingly edited, and featuring some truly awful dialogue. A complete disappointment.",
    "The ensemble cast elevates the material, bringing nuance to what could have been stock characters.",
    "A visually stunning but emotionally hollow experience that mistakes style for substance.",
    "Witty, sharp, and deeply human. One of the best films I have seen in years.",
    "A forgettable action film that relies entirely on explosions and car chases to mask its empty plot.",
    "The documentary captures its subject with rare intimacy and intelligence.",
    "Every cliché in the book is trotted out with dreary predictability.",
    "A charming and inventive film that delights in subverting audience expectations at every turn.",
    "The pacing is glacial and the dialogue is insufferably pompous.",
    "An emotionally resonant drama anchored by a career-defining performance from the lead actress.",
    "Loud, chaotic, and thoroughly exhausting—this film mistakes noise for excitement.",
    "The screenplay crackles with sharp wit and the direction is assured and inventive.",
    "A bloated and self-indulgent mess that mistakes runtime for depth.",
    "An absolute joy from beginning to end, full of genuine laughs and touching moments.",
    "The film's ambitions far exceed its execution, resulting in a confused and unsatisfying experience.",
    "A gripping and emotionally complex study of grief and resilience.",
    "Derivative and dull, this adds nothing new to an already crowded genre.",
    "The cinematography alone makes this worth watching—every frame is a work of art.",
    "Tedious and overlong, this film mistakes torpor for profundity.",
    "A refreshingly original voice in modern cinema, this debut film announces a major new talent.",
    "Aggressively mediocre—competently made but entirely forgettable.",
    "The performances are uniformly excellent, bringing real depth to complex and troubled characters.",
    "A complete failure on every level—script, direction, and performance alike.",
    "Heartfelt and beautifully crafted, this quiet film lingers in the memory long after viewing.",
    "A cynical cash-grab that insults the intelligence of its audience at every opportunity.",
    "Sharp, funny, and unexpectedly moving. A genuine crowd-pleaser with real substance.",
    "Incoherent and frustrating, the film seems to have been edited by someone who hates narrative.",
    "The director brings a fresh and exciting perspective to familiar genre conventions.",
    "Plodding and pedestrian, this thoroughly conventional thriller holds no surprises.",
    "A luminous and deeply affecting portrait of ordinary people facing extraordinary circumstances.",
    "The worst film I have seen this year—a complete and utter disaster from start to finish.",
    "Beautifully shot and sensitively acted, this small gem of a film rewards patient viewing.",
    "An exercise in tedium disguised as art cinema—pretentious and interminable.",
    "The film's sense of humor is sharp and sophisticated without ever becoming mean-spirited.",
    "Technically impressive but spiritually empty—a film that has forgotten why stories matter.",
    "A triumphant return to form for the director—witty, wise, and wonderfully entertaining.",
    "Numbingly repetitive and bereft of ideas, this drags on for what feels like an eternity.",
    "The script is intelligent and surprising, subverting expectations at every turn.",
    "A hollow spectacle that substitutes visual excess for genuine storytelling.",
    "Touching, funny, and surprisingly profound. A film that deserves a much wider audience.",
    "Utterly joyless—a grim and punishing experience that offers no pleasure whatsoever.",
]


def load_texts_labels_from_hf():
    """Load SST-2 from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        print("Loading SST-2 from HuggingFace...")
        ds = load_dataset("sst2", split="validation")
        texts  = [row["sentence"] for row in ds]
        labels = [row["label"]    for row in ds]
        print(f"  Loaded {len(texts)} SST-2 validation samples.")
        return texts, labels
    except Exception as e:
        print(f"  [warn] Could not load SST-2: {e}")
        return None, None


def load_texts_labels(data_dir: str):
    """Load from local MovieReview data_dir, or fall back to SST-2 / built-in samples."""
    import json
    test_path = os.path.join(data_dir, "test.jsonl")
    if os.path.exists(test_path):
        print(f"Loading MovieReview from {data_dir}...")
        docs_dir = os.path.join(data_dir, "docs")
        texts, labels = [], []
        with open(test_path, "r") as f:
            for line in f:
                item = json.loads(line)
                label = 1 if item["classification"] == "POS" else 0
                doc_path = os.path.join(docs_dir, item["annotation_id"])
                try:
                    with open(doc_path, "r", errors="replace") as df:
                        texts.append(df.read().strip())
                    labels.append(label)
                except FileNotFoundError:
                    continue
        print(f"  Loaded {len(texts)} MovieReview test samples.")
        return texts, labels

    # Try SST-2
    texts, labels = load_texts_labels_from_hf()
    if texts:
        return texts, labels

    # Last resort: built-in samples
    print("Using built-in sample texts for demonstration.")
    labels = [1 if i % 2 == 0 else 0 for i in range(len(SAMPLE_TEXTS))]
    return SAMPLE_TEXTS, labels


def custom_forward(inputs_embeds, attention_mask, model):
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits


def compute_bert_ig(model, tokenizer, input_ids, attention_mask, pred_cls, device, n_steps):
    embeddings  = model.bert.embeddings.word_embeddings
    ig          = IntegratedGradients(custom_forward)
    inp_emb     = embeddings(input_ids)
    pad_emb     = embeddings(torch.full_like(input_ids, tokenizer.pad_token_id))
    attrs, _    = ig.attribute(
        inputs=inp_emb, baselines=pad_emb,
        additional_forward_args=(attention_mask, model),
        target=pred_cls, n_steps=n_steps,
        internal_batch_size=2, return_convergence_delta=True,
    )
    torch.cuda.empty_cache()
    return attrs.sum(dim=-1).squeeze(0).cpu().detach().numpy()


def get_token_id(tokenizer, baseline):
    return {"pad": tokenizer.pad_token_id,
            "mask": tokenizer.mask_token_id,
            "unk": tokenizer.unk_token_id}[baseline]


def evaluate_sample(model, tokenizer, text, device, max_len, n_steps, ig_steps):
    model.eval()
    enc   = tokenizer(text, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=max_len)
    iids  = enc["input_ids"].to(device)
    amask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out      = model(iids, attention_mask=amask)
        pred_cls = torch.argmax(out.logits, dim=1).item()
        orig_p   = torch.softmax(out.logits, dim=1)[0, pred_cls].item()

    try:
        attr_sum = compute_bert_ig(model, tokenizer, iids, amask, pred_cls, device, ig_steps)
    except Exception as e:
        print(f" [warn] IG failed: {e}")
        return None

    vlen       = int(amask.sum().item())
    valid_attr = np.abs(attr_sum[1:vlen - 1])
    n_valid    = len(valid_attr)
    if n_valid < 2:
        return None

    q    = 100.0 / n_steps
    qs   = [q * (n_steps - 1 - i) for i in range(n_steps)]
    thrs = np.percentile(valid_attr, qs)

    results = {}
    for b in BASELINES:
        mtid = get_token_id(tokenizer, b)

        morf_probs, morf_masked = [orig_p], set()
        for p in thrs:
            morf_masked.update(set(np.where(valid_attr > p)[0] + 1))
            pert = iids.clone()
            for ti in morf_masked: pert[0, ti] = mtid
            with torch.no_grad():
                prob = torch.softmax(model(pert, attention_mask=amask).logits,
                                     dim=1)[0, pred_cls].item()
            morf_probs.append(prob)

        lerf_probs, lerf_masked = [orig_p], set()
        for p in thrs[::-1]:
            lerf_masked.update(set(np.where(valid_attr < p)[0] + 1))
            pert = iids.clone()
            for ti in lerf_masked: pert[0, ti] = mtid
            with torch.no_grad():
                prob = torch.softmax(model(pert, attention_mask=amask).logits,
                                     dim=1)[0, pred_cls].item()
            lerf_probs.append(prob)

        ma = np.array(morf_probs, dtype=np.float32)
        la = np.array(lerf_probs, dtype=np.float32)
        results[b] = {
            "morf_curve": ma,
            "lerf_curve": la,
            "morf_score": float((ma[0] - ma).mean()),
            "abpc_score": float((la - ma).mean()),
        }
    return results


def plot_morf_curves(mean_curves, n_steps, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.linspace(0, 1, n_steps + 1)
    for ax, key, title in zip(
        axes, ["morf", "lerf"],
        ["MoRF Curves by Token Masking Baseline\n(BERT-base + IG)",
         "LeRF Curves by Token Masking Baseline\n(BERT-base + IG)"]
    ):
        for b in BASELINES:
            ax.plot(x, mean_curves[b][key], label=BASELINE_LABELS[b],
                    color=BASELINE_COLORS[b], linewidth=2.2,
                    marker='o' if key == "morf" else 's', markersize=4,
                    linestyle='-' if key == "morf" else '--')
        ax.set_xlabel(f"Fraction of Tokens Masked ({key.upper()})", fontsize=12)
        ax.set_ylabel("Mean Predicted Probability", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.4); ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_abpc_bar(summary, out_path):
    labels    = [BASELINE_LABELS[b] for b in BASELINES]
    morf_vals = [summary[b]["mean_morf"] for b in BASELINES]
    abpc_vals = [summary[b]["mean_abpc"] for b in BASELINES]
    colors    = [BASELINE_COLORS[b] for b in BASELINES]
    x = np.arange(len(BASELINES)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, morf_vals, w, label="MoRF ↑", color=colors, alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + w/2, abpc_vals, w, label="ABPC ↑", color=colors, alpha=0.5, edgecolor='black', hatch='//')
    for b in (bars1, bars2):
        for bar in b:
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Faithfulness Score", fontsize=12)
    ax.set_title("NLP Faithfulness Scores by Token Masking Baseline\nBERT-base + IG (Phase 2)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, axis='y', alpha=0.4)
    all_vals = morf_vals + abpc_vals
    if all_vals: ax.set_ylim(0, max(all_vals) * 1.3 + 0.02)
    ax.axhline(y=summary["pad"]["mean_morf"], color="red",
               linestyle='--', linewidth=1.2, alpha=0.7,
               label=f"M4 PAD baseline MoRF={summary['pad']['mean_morf']:.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/movies")
    p.add_argument("--model_dir",  default="models/bert-base-movies")
    p.add_argument("--n_samples",  type=int, default=50)
    p.add_argument("--max_len",    type=int, default=256)
    p.add_argument("--n_steps",    type=int, default=20)
    p.add_argument("--ig_steps",   type=int, default=50)
    p.add_argument("--output_dir", default="results/phase2")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model — fine-tuned if available, else pre-trained
    if os.path.exists(args.model_dir):
        print(f"Loading fine-tuned BERT from {args.model_dir}...")
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        model     = BertForSequenceClassification.from_pretrained(args.model_dir)
        model_tag = "fine-tuned"
    else:
        print("Fine-tuned model not found. Using bert-base-uncased (pre-trained) with SST-2 labels.")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model     = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2, ignore_mismatched_sizes=True
        )
        model_tag = "pre-trained"
    model.eval().to(device)
    print(f"Model: BERT-base-uncased ({model_tag})")

    texts, labels = load_texts_labels(args.data_dir)
    texts  = texts[:args.n_samples]
    labels = labels[:args.n_samples]
    print(f"Evaluating {len(texts)} samples, baselines={BASELINES}")

    tok_ids = {b: get_token_id(tokenizer, b) for b in BASELINES}
    print(f"Token IDs: {tok_ids}")

    all_morf = {b: [] for b in BASELINES}
    all_abpc = {b: [] for b in BASELINES}
    all_mc   = {b: [] for b in BASELINES}
    all_lc   = {b: [] for b in BASELINES}

    processed = 0
    for idx, (text, _) in enumerate(zip(texts, labels)):
        print(f"  Sample {idx+1}/{len(texts)}...", end=" ", flush=True)
        res = evaluate_sample(model, tokenizer, text, device,
                              args.max_len, args.n_steps, args.ig_steps)
        if res is None:
            print("skip")
            continue
        for b in BASELINES:
            all_morf[b].append(res[b]["morf_score"])
            all_abpc[b].append(res[b]["abpc_score"])
            all_mc[b].append(res[b]["morf_curve"])
            all_lc[b].append(res[b]["lerf_curve"])
        processed += 1
        torch.cuda.empty_cache()
        print("OK")

    if processed == 0:
        print("[error] No samples processed.")
        return

    print(f"\nProcessed {processed} samples.")
    summary = {}
    print("\n" + "="*58)
    print(f"  Phase 2 NLP Results (BERT-base {model_tag} + IG)")
    print("="*58)
    print(f"{'Baseline':<25} {'MoRF (^)':>12} {'ABPC (^)':>12}")
    print("-"*52)
    for b in BASELINES:
        mm = float(np.mean(all_morf[b]))
        ma = float(np.mean(all_abpc[b]))
        summary[b] = {"mean_morf": mm, "mean_abpc": ma}
        flag = " <- M4 paper" if b == "pad" else ""
        print(f"  {BASELINE_LABELS[b]:<25} {mm:>12.4f} {ma:>12.4f}{flag}")
    print("="*58)

    morf_mat = np.array([[all_morf[b][i] for b in BASELINES] for i in range(processed)])
    abpc_mat = np.array([[all_abpc[b][i] for b in BASELINES] for i in range(processed)])
    bss_morf = float(morf_mat.std(axis=1).mean())
    bss_abpc = float(abpc_mat.std(axis=1).mean())
    print(f"\n  BSS-MoRF: {bss_morf:.4f}  |  BSS-ABPC: {bss_abpc:.4f}")

    mean_curves = {b: {"morf": np.mean(all_mc[b], axis=0),
                        "lerf": np.mean(all_lc[b], axis=0)} for b in BASELINES}

    print("\nGenerating plots...")
    plot_morf_curves(mean_curves, args.n_steps,
                     os.path.join(args.output_dir, "nlp_morf_lerf_curves.png"))
    plot_abpc_bar(summary,
                  os.path.join(args.output_dir, "nlp_abpc_comparison.png"))

    csv_path = os.path.join(args.output_dir, "nlp_baseline_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "MoRF", "ABPC", "N", "BSS_MoRF", "BSS_ABPC",
                    "model_tag", "ig_steps", "n_steps"])
        for b in BASELINES:
            w.writerow([b, f"{summary[b]['mean_morf']:.4f}", f"{summary[b]['mean_abpc']:.4f}",
                         processed, f"{bss_morf:.4f}", f"{bss_abpc:.4f}",
                         model_tag, args.ig_steps, args.n_steps])
    print(f"  Saved: {csv_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

