"""
Fine-tune the five M4 sentiment models on MovieReview (ERASER movies) with the authors' recipe
(training_nlp/train.sh + train_model.py):

    AdamW, lr 3e-5, 10 epochs, batch 32, linear decay with 10% warm-up, weight decay 0.01
    (not on bias / LayerNorm), grad-norm clip 1.0, max_seq_len 512 (BERT-large: 300).

Training is fp32 as in Paddle; the batch of 32 is reached with gradient accumulation so it fits on an
8 GB GPU. As upstream, the model after the last epoch is saved (models/<name>-movies); validation
accuracy is logged every epoch.

    python train_text.py --models all
    python train_text.py --models bert_base --per_device_bs 8
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from m4bench.text_models import TEXT_MODEL_NAMES, TEXT_MODELS, finetuned_dir, load_movies, model_class


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"], help=f"'all' or any of {TEXT_MODEL_NAMES}")
    p.add_argument("--data_dir", default="data/movies")
    p.add_argument("--model_root", default="models")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=32, help="effective batch size")
    p.add_argument("--per_device_bs", type=int, default=None, help="default: 4 (1 for bert_large)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_one(name, args):
    cfg = TEXT_MODELS[name]
    out_dir = finetuned_dir(name, args.model_root)
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = model_class(name).from_pretrained(cfg["hf"], num_labels=2)

    def to_ds(split):
        texts, labels = load_movies(args.data_dir, split)
        enc = tok(texts, truncation=True, max_length=cfg["max_len"])
        return Dataset.from_dict({**enc, "labels": labels})

    train_ds, val_ds = to_ds("train"), to_ds("val")
    per_dev = args.per_device_bs or (1 if name == "bert_large" else 4)

    targs = TrainingArguments(
        output_dir=os.path.join(args.model_root, "_ckpt", name),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=per_dev,
        per_device_eval_batch_size=per_dev * 2,
        gradient_accumulation_steps=max(1, args.batch_size // per_dev),
        lr_scheduler_type="linear",
        warmup_steps=0.1,                     # a float < 1 is a ratio of total steps (transformers 5)
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="no",
        gradient_checkpointing=(name == "bert_large"),
        logging_steps=25,
        report_to="none",
        seed=args.seed,
    )

    def accuracy(pred):
        return {"accuracy": float((np.argmax(pred.predictions, -1) == pred.label_ids).mean())}

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=accuracy)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{name}] final val accuracy = {metrics['eval_accuracy']:.4f} -> {out_dir}", flush=True)
    del trainer, model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    set_seed(args.seed)
    for name in TEXT_MODEL_NAMES if args.models == ["all"] else args.models:
        train_one(name, args)


if __name__ == "__main__":
    main()
