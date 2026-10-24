"""
The five MovieReview sentiment models of the paper (Sec. 2.1) and the hooks the explainers need.

Upstream (PaddleNLP)                -> here (HuggingFace)
  distilbert-base-uncased           -> distilbert/distilbert-base-uncased
  bert-base-uncased / -large        -> google-bert/bert-{base,large}-uncased
  ernie-2.0-base-en                 -> nghuyong/ernie-2.0-base-en
  roberta-base                      -> FacebookAI/roberta-base

Max sequence length: 512, except BERT-large: 300 (training_nlp/train.sh, benchmark-nlp/run_expl.sh).

PaddleNLP builds the attention mask itself as (input_ids != pad_token_id) whenever no mask is passed,
which is how M4 calls the models. So replacing a token by [PAD] during MoRF/LeRF/INFD effectively
removes it. `paddle_style_mask` reproduces that here.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

TEXT_MODELS = {
    "distilbert": dict(hf="distilbert/distilbert-base-uncased", max_len=512),
    "bert_base": dict(hf="google-bert/bert-base-uncased", max_len=512),
    "bert_large": dict(hf="google-bert/bert-large-uncased", max_len=300),
    "ernie_2_base": dict(hf="nghuyong/ernie-2.0-base-en", max_len=512),
    "roberta_base": dict(hf="FacebookAI/roberta-base", max_len=512),
}
TEXT_MODEL_NAMES = list(TEXT_MODELS)
TEXT_METHODS = ["ig", "sg", "lime", "ga", "bt_head", "bt_token"]


# ─────────────────────────────────────────────────────────────────────────────
# MovieReview (ERASER)
# ─────────────────────────────────────────────────────────────────────────────

def load_movies(data_dir: str, split: str):
    """training_nlp/data.py MovieReview: document text of each annotation, label POS=1 / NEG=0."""
    texts, labels = [], []
    with open(os.path.join(data_dir, f"{split}.jsonl")) as f:
        for line in f:
            item = json.loads(line)
            with open(os.path.join(data_dir, "docs", item["annotation_id"])) as d:
                texts.append(d.read())
            labels.append(1 if item["classification"] == "POS" else 0)
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextModel:
    name: str
    net: nn.Module
    tok: object
    max_len: int

    @property
    def is_distil(self) -> bool:
        return self.net.config.model_type == "distilbert"

    @property
    def base(self) -> nn.Module:
        return self.net.base_model

    @property
    def pad_id(self) -> int:
        return self.tok.pad_token_id

    def layers(self):
        return self.base.transformer.layer if self.is_distil else self.base.encoder.layer

    def value_module(self, layer):              # v_proj
        return layer.attention.v_lin if self.is_distil else layer.attention.self.value

    def proj_weight(self, layer):               # out_proj
        return (layer.attention.out_lin if self.is_distil else layer.attention.output.dense).weight

    def norm1(self, layer):                     # Paddle post-LN norm1: input = attn_out + residual
        return layer.sa_layer_norm if self.is_distil else layer.attention.output.LayerNorm

    def encode(self, text: str) -> torch.Tensor:
        """Paddle tokenizer(text, max_seq_len=L): truncation incl. special tokens, no padding."""
        ids = self.tok(text, truncation=True, max_length=self.max_len)["input_ids"]
        return torch.tensor([ids])

    def paddle_style_mask(self, ids: torch.Tensor) -> torch.Tensor:
        return (ids != self.pad_id).long()

    def _position_ids(self, ids):
        if self.net.config.model_type != "roberta":
            return None
        # Keep positions fixed when tokens are replaced by <pad> (HF would otherwise skip pad positions).
        start = self.net.config.pad_token_id + 1
        return (torch.arange(ids.shape[1], device=ids.device) + start).expand_as(ids)

    def forward(self, ids: torch.Tensor, **kw):
        kw.setdefault("attention_mask", self.paddle_style_mask(ids))
        pos = self._position_ids(ids)
        if pos is not None:
            kw.setdefault("position_ids", pos)
        return self.net(input_ids=ids, **kw)

    def logits(self, ids: torch.Tensor, **kw) -> torch.Tensor:
        return self.forward(ids, **kw).logits


def paddle_style_roberta_cls():
    """
    PaddleNLP RobertaForSequenceClassification: logits = classifier(dropout(pooler(h[:, 0]))), with the
    tanh pooler of RobertaModel. HuggingFace's RobertaClassificationHead is dense/tanh/dropout on <s>
    without the pretrained pooler, so the Paddle head is rebuilt here.
    """
    from transformers import RobertaModel, RobertaPreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class PaddleStyleRobertaForSequenceClassification(RobertaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.roberta = RobertaModel(config, add_pooling_layer=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.post_init()

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None,
                    output_attentions=None, **kw):
            out = self.roberta(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                               output_attentions=output_attentions, **kw)
            logits = self.classifier(self.dropout(out.pooler_output))
            loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits, attentions=out.attentions)

    return PaddleStyleRobertaForSequenceClassification


def model_class(name: str):
    from transformers import AutoModelForSequenceClassification
    return paddle_style_roberta_cls() if name == "roberta_base" else AutoModelForSequenceClassification


def finetuned_dir(name: str, model_root: str = "models") -> str:
    return os.path.join(model_root, f"{name}-movies")


def load_text_model(name: str, device, model_root: str = "models", finetuned: bool = True) -> TextModel:
    from transformers import AutoTokenizer
    cfg = TEXT_MODELS[name]
    src = finetuned_dir(name, model_root) if finetuned else cfg["hf"]
    if finetuned and not os.path.isdir(src):
        raise FileNotFoundError(f"{src} not found. Fine-tune first: python train_text.py --models {name}")
    tok = AutoTokenizer.from_pretrained(src)
    net = model_class(name).from_pretrained(src, num_labels=2, attn_implementation="eager")
    net.eval().to(device)
    if finetuned:
        for p in net.parameters():
            p.requires_grad_(False)
    return TextModel(name, net, tok, cfg["max_len"])


@torch.no_grad()
def batched_probs(tm: TextModel, ids: np.ndarray | torch.Tensor, device, bs: int = 16, amp: bool = False):
    ids = torch.as_tensor(np.asarray(ids), device=device)
    outs = []
    for i in range(0, len(ids), bs):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            outs.append(torch.softmax(tm.logits(ids[i:i + bs]).float(), 1))
    return torch.cat(outs)
