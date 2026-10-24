"""
Text attribution methods, ported from InterpretDL interpreter/*.py (the *NLPInterpreter classes) with the
settings of benchmark-nlp/run_expl.sh. Each returns a dict of per-token vectors over the full sequence
(including [CLS]/[SEP], as upstream):
  IG / SG : {"abs": sum|e|, "sq": sum e^2, "sum": sum e} over the hidden dim
            (interpret_and_save.py; MoRF/LeRF/INFD use "abs")
  others  : {"abs": e, "sq": e, "sum": e}
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances

from .text_models import TextModel


def _reduce(e: torch.Tensor) -> dict:
    e = e.detach().float()[0]                          # (L, H)
    return {"abs": e.abs().sum(-1).cpu().numpy(), "sq": e.square().sum(-1).cpu().numpy(),
            "sum": e.sum(-1).cpu().numpy()}


def _same(v: torch.Tensor | np.ndarray) -> dict:
    v = v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.float32)
    return {"abs": v, "sq": v, "sum": v}


def _target(logits: torch.Tensor, label: int, of: str) -> torch.Tensor:
    return (torch.softmax(logits, 1) if of == "probability" else logits)[:, label].sum()


class _EmbeddingHook:
    """
    IntermediateGradientInterpreter / TransformerInterpreter hook: replace a layer's output by
    scale * output (+ noise), and make that tensor the leaf whose gradient is read.
    """

    def __init__(self, module, scale=None, noise_std=None, gen=None):
        self.scale, self.noise_std, self.gen = scale, noise_std, gen
        self.out = None
        self.h = module.register_forward_hook(self)

    def __call__(self, mod, inp, out):
        out = out.detach()
        if self.scale is not None:
            out = self.scale.view(-1, 1, 1).to(out.dtype) * out
        if self.noise_std is not None:
            out = out + torch.randn(out.shape, generator=self.gen, device=out.device) * self.noise_std
        out.requires_grad_(True)
        self.out = out
        return out

    def remove(self):
        self.h.remove()


def _word_emb(tm):
    return tm.base.embeddings.word_embeddings


@torch.no_grad()
def _word_embedding_output(tm, ids):
    return _word_emb(tm)(ids)


def integrated_gradients(tm: TextModel, ids, label, rng, ig_steps=50, grad_bs=10, **kw):
    """IntGradNLPInterpreter(steps=50, gradient_of='logit'): zero baseline on the word embeddings."""
    e = _word_embedding_output(tm, ids)
    total = torch.zeros_like(e)
    for a in torch.linspace(0, 1, ig_steps, device=ids.device).split(grad_bs):
        hook = _EmbeddingHook(_word_emb(tm), scale=a)
        try:
            logits = tm.logits(ids.expand(len(a), -1))
            (g,) = torch.autograd.grad(_target(logits, label, "logit"), hook.out)
        finally:
            hook.remove()
        total += g.sum(0, keepdim=True)
    return _reduce(total * e / ig_steps)


def smoothgrad(tm: TextModel, ids, label, rng, sg_samples=50, sg_noise=0.1, grad_bs=10, **kw):
    """
    SmoothGradNLPInterpreter(noise_amount=0.1, n_samples=50, gradient_of='probability'):
    noise ~ N(0, noise_amount * mean(word_embeddings)) added to the word-embedding output.
    The std is |0.1 * mean|, which is tiny for BERT-like embeddings (the paper's Sec. 3.2 remark).
    """
    e = _word_embedding_output(tm, ids)
    std = abs(sg_noise * float(e.mean()))
    gen = torch.Generator(device=ids.device).manual_seed(int(rng.integers(2**31)))
    total = torch.zeros_like(e)
    left = sg_samples
    while left > 0:
        n = min(grad_bs, left)
        hook = _EmbeddingHook(_word_emb(tm), noise_std=std, gen=gen)
        try:
            logits = tm.logits(ids.expand(n, -1))
            (g,) = torch.autograd.grad(_target(logits, label, "probability"), hook.out)
        finally:
            hook.remove()
        total += g.sum(0, keepdim=True)
        left -= n
    return _reduce(total / sg_samples)


def lime(tm: TextModel, ids, label, rng, lime_samples=1000, lime_bs=50, **kw):
    """
    LIMENLPInterpreter(num_samples=1000): every token position (incl. specials) is a binary feature,
    switched-off tokens become [UNK]; weighted least squares (Ridge alpha=0, intercept) with kernel
    sqrt(exp(-d^2 / 0.25^2)) on cosine distance to the all-ones row; the predicted class's coefficients.
    """
    n_feat = ids.shape[1]
    data = rng.integers(0, 2, size=(lime_samples, n_feat))
    data[0, :] = 1
    base = ids[0].cpu().numpy()
    samples = np.where(data == 1, base[None, :], tm.tok.unk_token_id)
    probs = []
    with torch.no_grad():
        for i in range(0, lime_samples, lime_bs):
            batch = torch.as_tensor(samples[i:i + lime_bs], device=ids.device)
            probs.append(torch.softmax(tm.logits(batch).float(), 1)[:, label].cpu().numpy())
    y = np.concatenate(probs)
    d = pairwise_distances(data, data[:1], metric="cosine").ravel()
    w = np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
    reg = LinearRegression(fit_intercept=True).fit(data, y, sample_weight=w)
    return _same(reg.coef_)


class _AttnRecorder:
    """Per-layer value-projection outputs and norm1 inputs (BT token mode)."""

    def __init__(self, tm: TextModel):
        self.v, self.z, self.hooks = [], [], []
        for layer in tm.layers():
            self.hooks.append(tm.value_module(layer).register_forward_hook(
                lambda m, i, o: self.v.append(o.detach())))
            self.hooks.append(tm.norm1(layer).register_forward_pre_hook(
                lambda m, i: self.z.append(i[0].detach())))
        self.projs = [tm.proj_weight(layer).detach() for layer in tm.layers()]

    def remove(self):
        for h in self.hooks:
            h.remove()


def _attn_forward_backward(tm, ids, label, of, scale=None, record_tokens=False):
    """Forward with attention maps kept in the graph, backward from the target; returns attentions."""
    hook = _EmbeddingHook(tm.base.embeddings, scale=scale)
    rec = _AttnRecorder(tm) if record_tokens else None
    try:
        n = 1 if scale is None else len(scale)
        out = tm.forward(ids.expand(n, -1), output_attentions=True)
        attns = out.attentions
        for a in attns:
            a.retain_grad()
        _target(out.logits, label, of).backward()
    finally:
        hook.remove()
        if rec:
            rec.remove()
    return attns, rec


def generic_attention(tm: TextModel, ids, label, rng, start_layer=None, **kw):
    """GANLPInterpreter(gradient_of='logit'): R += clip(grad * A, 0).mean(heads) @ R; returns R[CLS, :]."""
    attns, _ = _attn_forward_backward(tm, ids, label, "logit")
    sl = len(attns) - 1 if start_layer is None else start_layer
    s = attns[0].shape[-1]
    R = torch.eye(s, device=ids.device)
    for i, A in enumerate(attns):
        if i < sl:
            continue
        cam = (A.grad[0] * A.detach()[0]).clamp(min=0).mean(0)
        R = R + cam @ R
    return _same(R[0])


def _bt(tm: TextModel, ids, label, ap_mode, start_layer=None, bt_steps=20, grad_bs=10, **kw):
    """BTNLPInterpreter(gradient_of='logit'); integration scales the whole embedding output by alpha."""
    attns, rec = _attn_forward_backward(tm, ids, label, "logit", record_tokens=(ap_mode == "token"))
    sl = len(attns) - 1 if start_layer is None else start_layer
    s = attns[0].shape[-1]
    R = torch.eye(s, device=ids.device)
    for i, A in enumerate(attns):
        if i < sl:
            continue
        A0 = A.detach()[0]
        if ap_mode == "head":
            Ih = (A0.transpose(-1, -2) @ A.grad[0]).abs().mean((-1, -2))
            Ih = Ih / Ih.sum()
            R = R + (Ih.view(-1, 1, 1) * A0).sum(0) @ R
        else:
            vproj = rec.v[i][0] @ rec.projs[i].T
            order = vproj.norm(dim=-1) / rec.z[i][0].norm(dim=-1)
            R = R + (A0.mean(0) * order.view(1, -1)) @ R

    total = torch.zeros_like(attns[-1].detach()[0])
    for a in torch.linspace(0, 1, bt_steps, device=ids.device).split(grad_bs):
        att, _ = _attn_forward_backward(tm, ids, label, "logit", scale=a)
        total += att[-1].grad.sum(0)
    feedback = (total / bt_steps).clamp(min=0).mean(0)
    return _same(R[0] * feedback[0])


def bt_head(tm, ids, label, rng, **kw):
    return _bt(tm, ids, label, "head", **kw)


def bt_token(tm, ids, label, rng, **kw):
    return _bt(tm, ids, label, "token", **kw)


EXPLAINERS = {"ig": integrated_gradients, "sg": smoothgrad, "lime": lime,
              "ga": generic_attention, "bt_head": bt_head, "bt_token": bt_token}
