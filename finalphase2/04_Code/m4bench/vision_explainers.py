"""
Vision attribution methods, ported from InterpretDL (interpreter/*.py) with the settings in
M4_XAI_Benchmark/benchmark-cv/run_expl.sh.

Every explainer returns {"abs": 2-D map, "sq": 2-D map}:
  * "abs" is what evaluate_faithfulness.py / evaluate_2.py feed to MoRF/LeRF/INFD
    (np.abs(expl).mean(axis=(0,1)) for input-gradient maps),
  * "sq"  is what evaluate_consensus.py uses for PScore (np.square(expl).mean(axis=1)).
Maps that are already 2-D (GradCAM 7x7, GA/BT 14x14, baselines) are used as-is for both.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .vision_models import VisionModel


def _both(m: torch.Tensor) -> dict:
    a = m.detach().float().cpu().numpy()
    return {"abs": a, "sq": a}


def _input_grad_maps(e: torch.Tensor) -> dict:
    """(1, 3, H, W) input-space attribution -> channel-reduced maps."""
    e = e.detach().float()[0]
    return {"abs": e.abs().mean(0).cpu().numpy(), "sq": e.square().mean(0).cpu().numpy()}


def _prob_grads(vm: VisionModel, xs: torch.Tensor, label: int) -> torch.Tensor:
    """d softmax(f(x))[label] / dx for a batch of normalised inputs (gradient_of='probability')."""
    xs = xs.detach().requires_grad_(True)
    probs = F.softmax(vm.net(xs), dim=1)
    (g,) = torch.autograd.grad(probs[:, label].sum(), xs)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Baselines (Table 1)
# ─────────────────────────────────────────────────────────────────────────────

def constant(vm, x, label, rng, **kw):
    return _both(torch.ones(224, 224))


def random_pixel(vm, x, label, rng, **kw):
    return _both(torch.from_numpy(rng.random((224, 224), dtype=np.float32)))


def random_16(vm, x, label, rng, **kw):
    patches = torch.from_numpy(rng.random((14, 14), dtype=np.float32))
    return _both(patches.repeat_interleave(16, 0).repeat_interleave(16, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Gradient methods
# ─────────────────────────────────────────────────────────────────────────────

def ig_attribution(vm, x, label, baseline, ig_steps=50, grad_bs=25):
    """sum_{a in linspace(0,1,steps)} grad(b + a (x - b)) * (x - b) / steps   (one IG trial)."""
    alphas = torch.linspace(0, 1, ig_steps, device=x.device).view(-1, 1, 1, 1)
    grad_sum = torch.zeros_like(x)
    for a in alphas.split(grad_bs):
        grad_sum += _prob_grads(vm, baseline + a * (x - baseline), label).sum(0, keepdim=True)
    return grad_sum * (x - baseline) / ig_steps


def integrated_gradients(vm, x, label, rng, ig_steps=50, ig_trials=10, grad_bs=25, **kw):
    """
    IntGradCVInterpreter(baselines='random', num_random_trials=10, steps=50, gradient_of='probability'):
    baseline_i ~ N(0, 1) in normalised input space, result = mean_i IG_i.
    """
    gen = torch.Generator(device=x.device).manual_seed(int(rng.integers(2**31)))
    total = torch.zeros_like(x)
    for _ in range(ig_trials):
        b = torch.randn(x.shape, generator=gen, device=x.device)
        total += ig_attribution(vm, x, label, b, ig_steps, grad_bs)
    return _input_grad_maps(total / ig_trials)


def smoothgrad(vm, x, label, rng, sg_samples=100, sg_noise=0.1, grad_bs=25, **kw):
    """SmoothGradInterpreter(noise_amount=0.1, n_samples=100): mean of plain gradients (not squared)."""
    gen = torch.Generator(device=x.device).manual_seed(int(rng.integers(2**31)))
    std = sg_noise * (x.max() - x.min())
    total = torch.zeros_like(x)
    left = sg_samples
    while left > 0:
        n = min(grad_bs, left)
        noisy = x + torch.randn((n, *x.shape[1:]), generator=gen, device=x.device) * std
        total += _prob_grads(vm, noisy, label).sum(0, keepdim=True)
        left -= n
    return _input_grad_maps(total / sg_samples)


def gradcam(vm, x, label, rng, **kw):
    """
    GradCAMInterpreter: relu(mean_c(mean_hw(dP/dA) * A)) at the feature-map resolution, with the
    gradient of the softmax probability of the label (gradient_cam.py: target = sum(softmax * onehot)).
    """
    store = {}
    h = vm.gradcam_layer.register_forward_hook(lambda mod, i, o: store.__setitem__("A", o))
    try:
        xs = x.detach().requires_grad_(True)
        probs = F.softmax(vm.net(xs), dim=1)
        (g,) = torch.autograd.grad(probs[0, label], store["A"])
    finally:
        h.remove()
    cam = (g.mean((2, 3), keepdim=True) * store["A"]).mean(1)[0].clamp(min=0)
    return _both(cam)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer-specific methods (GA, BT)
# ─────────────────────────────────────────────────────────────────────────────

class _ViTRecorder:
    """Collects per-block attention maps (with grads), block inputs z, values v and W_proj."""

    def __init__(self, vm: VisionModel, need_tokens: bool):
        self.attns, self.z, self.v = [], [], []
        self.hooks = []
        for blk in vm.net.blocks:
            self.hooks.append(blk.attn.attn_drop.register_forward_hook(self._attn))
            if need_tokens:
                self.hooks.append(blk.norm1.register_forward_pre_hook(lambda m, i: self.z.append(i[0].detach())))
                self.hooks.append(blk.attn.qkv.register_forward_hook(self._value))
        self.projs = [blk.attn.proj.weight.detach() for blk in vm.net.blocks]

    def _attn(self, mod, inp, out):
        out.retain_grad()
        self.attns.append(out)

    def _value(self, mod, inp, out):
        b, s, three_c = out.shape
        self.v.append(out.detach().reshape(b, s, 3, three_c // 3)[:, :, 2])

    def remove(self):
        for h in self.hooks:
            h.remove()


def _vit_forward_backward(vm, xs, label, need_tokens=False):
    rec = _ViTRecorder(vm, need_tokens)
    try:
        xs = xs.detach().requires_grad_(True)
        probs = F.softmax(vm.net(xs), dim=1)
        probs[:, label].sum().backward()
    finally:
        rec.remove()
    return rec


def _relevance_to_map(vm, rel: torch.Tensor) -> torch.Tensor:
    """(s, s) relevance -> (14, 14) patch map, following the class-token / global-pool branches upstream."""
    if getattr(vm.net, "global_pool", False):                      # MAE: global average pooling
        row = rel[1:, :].mean(0)[1:]
    else:
        row = rel[0, 1:]
    side = int(row.numel() ** 0.5)
    return row.reshape(side, side)


def generic_attention(vm, x, label, rng, **kw):
    """GACVInterpreter: R += clip(grad * A, 0).mean(heads) @ R for blocks >= start_layer."""
    rec = _vit_forward_backward(vm, x, label)
    s = rec.attns[0].shape[-1]
    R = torch.eye(s, device=x.device)
    for i, A in enumerate(rec.attns):
        if i < vm.start_layer:
            continue
        cam = (A.grad[0] * A.detach()[0]).clamp(min=0).mean(0)
        R = R + cam @ R
    return _both(_relevance_to_map(vm, R))


def _bt(vm, x, label, ap_mode, bt_steps=20, grad_bs=20, **kw):
    """
    BTCVInterpreter (Chen et al., "Beyond Intuition"):
      attentional perception R from blocks >= start_layer ('head': head-importance-weighted attention;
      'token': mean-head attention times diag(||v W_proj|| / ||z||)),
      reasoning feedback = mean_heads(clip(mean_alpha dP/dA_last at input alpha * x, 0)),
      explanation = (R * feedback) read out at the class token (or averaged for global pooling).
    """
    rec = _vit_forward_backward(vm, x, label, need_tokens=(ap_mode == "token"))
    s = rec.attns[0].shape[-1]
    R = torch.eye(s, device=x.device)
    for i, A in enumerate(rec.attns):
        if i < vm.start_layer:
            continue
        A0 = A.detach()[0]                                              # (h, s, s)
        if ap_mode == "head":
            G0 = A.grad[0]
            Ih = (A0.transpose(-1, -2) @ G0).abs().mean((-1, -2))       # (h,)
            Ih = Ih / Ih.sum()
            attn = (Ih.view(-1, 1, 1) * A0).sum(0)
            R = R + attn @ R
        else:
            vproj = rec.v[i][0] @ rec.projs[i].T                        # v W_proj (torch Linear stores W^T)
            order = vproj.norm(dim=-1) / rec.z[i][0].norm(dim=-1)
            R = R + (A0.mean(0) * order.view(1, -1)) @ R                # A @ diag(order) @ R

    total = torch.zeros_like(rec.attns[-1].detach()[0])
    for a in torch.linspace(0, 1, bt_steps, device=x.device).split(grad_bs):
        r = _vit_forward_backward(vm, a.view(-1, 1, 1, 1) * x, label)
        total += r.attns[-1].grad.sum(0)
    feedback = (total / bt_steps).clamp(min=0).mean(0)                  # (s, s)
    return _both(_relevance_to_map(vm, R * feedback))


def bt_head(vm, x, label, rng, **kw):
    return _bt(vm, x, label, "head", **kw)


def bt_token(vm, x, label, rng, **kw):
    return _bt(vm, x, label, "token", **kw)


EXPLAINERS = {
    "constant": constant,
    "random_16": random_16,
    "random": random_pixel,
    "gradcam": gradcam,
    "ig": integrated_gradients,
    "sg": smoothgrad,
    "ga": generic_attention,
    "bt_head": bt_head,
    "bt_token": bt_token,
}

NATIVE_HW = {"constant": (224, 224), "random_16": (224, 224), "random": (224, 224),
             "ig": (224, 224), "sg": (224, 224), "gradcam": (7, 7),
             "ga": (14, 14), "bt_head": (14, 14), "bt_token": (14, 14)}
