"""
Self-checks of the M4 implementation that need no dataset and no pretrained weights
(randomly initialised models + synthetic inputs). Run before launching the long jobs:

    .venv/bin/python tests/smoke_test.py

Checks:
  * IG completeness: sum of attributions == f(x) - f(baseline) (vision and text)
  * GradCAM agrees with the pytorch-grad-cam library
  * every vision explainer runs on its architectures (CNN, ViT, MAE global-pool) with the right shape
  * metric invariants: constant map -> MoRF = LeRF = ABPC = 0, INFD has 1235 boxes, SynScore AP of a
    constant map = square area / image area (paper Table 1: 0.072), curve lengths
  * every text explainer runs on BERT / DistilBERT / RoBERTa layouts and attentions receive gradients
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m4bench.vision_explainers import EXPLAINERS as VEXP, NATIVE_HW, ig_attribution  # noqa: E402
from m4bench.vision_metrics import (InfidelityCache, infd_boxes, morf_lerf_abpc,  # noqa: E402
                                    synscore)
from m4bench.vision_models import load_vision_model  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def fake_image(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224]
    img = np.stack([(xx * 255 / 223), (yy * 255 / 223), np.full_like(xx, 90)], -1).astype(np.float32)
    img += rng.normal(0, 20, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def vision_checks():
    print("vision")
    img = fake_image()
    rng = np.random.default_rng(0)
    kw = dict(ig_steps=8, ig_trials=2, sg_samples=6, bt_steps=4, grad_bs=4)

    vm = load_vision_model("resnet50", DEV, pretrained=False)
    x = vm.normalize(torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(DEV))
    with torch.no_grad():
        label = int(vm.net(x).argmax(1))

    # IG completeness (many steps, one trial, zero-ish baseline)
    b = torch.zeros_like(x)
    ig = ig_attribution(vm, x, label, b, ig_steps=300, grad_bs=50)
    with torch.no_grad():
        p = F.softmax(vm.net(torch.cat([x, b])), 1)[:, label]
    gap = float(p[0] - p[1])
    check("IG completeness (ResNet-50)", abs(float(ig.sum()) - gap) < 0.05 * abs(gap) + 1e-4,
          f"sum={float(ig.sum()):.5f} f(x)-f(b)={gap:.5f}")

    # GradCAM vs pytorch-grad-cam
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
        ours = VEXP["gradcam"](vm, x, label, rng)["abs"]
        for p_ in vm.net.parameters():
            p_.requires_grad_(True)
        with GradCAM(model=vm.net, target_layers=[vm.net.layer4[-1]]) as cam:
            ref = cam(input_tensor=x, targets=[ClassifierOutputSoftmaxTarget(label)])[0]
        for p_ in vm.net.parameters():
            p_.requires_grad_(False)
        import cv2
        ours224 = cv2.resize(ours, (224, 224), interpolation=cv2.INTER_LINEAR)
        r = np.corrcoef(ours224.ravel(), ref.ravel())[0, 1]
        check("GradCAM matches pytorch-grad-cam", r > 0.99, f"corr={r:.4f}")
    except ImportError:
        print("  [skip] pytorch-grad-cam not installed")

    # Metric invariants
    sc, morf_p, lerf_p = morf_lerf_abpc(vm, img, np.ones((224, 224), np.float32), DEV, n_steps=100, bs=50)
    check("constant map -> MoRF=LeRF=ABPC=0", all(abs(sc[k]) < 1e-3 for k in sc), str(sc))
    check("curves have L+1 = 101 points", len(morf_p) == 101 and len(lerf_p) == 101)
    check("INFD boxes = 1235", len(infd_boxes()) == 1235)
    infd = InfidelityCache(vm, img, DEV, bs=128)
    check("INFD finite", np.isfinite(infd.score(np.random.rand(7, 7))))
    ap = synscore(np.ones((224, 224)), 10, 10)["SynScore_AP"]
    check("SynScore AP of constant map = 60^2/224^2", abs(ap - 3600 / 224 ** 2) < 1e-6, f"{ap:.4f}")

    for name in ["resnet50", "vgg16", "mobilenet", "vit_small", "vit_base", "mae_vit_base"]:
        vm = load_vision_model(name, DEV, pretrained=False)
        x = vm.normalize(torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(DEV))
        with torch.no_grad():
            label = int(vm.net(x).argmax(1))
        for m in vm.methods:
            out = VEXP[m](vm, x, label, rng, **kw)
            ok = out["abs"].shape == NATIVE_HW[m] and np.isfinite(out["abs"]).all() and np.abs(out["abs"]).sum() > 0
            check(f"{name}/{m} shape {NATIVE_HW[m]}", ok, str(out["abs"].shape))
        del vm
        torch.cuda.empty_cache()


def text_checks():
    print("text")
    from transformers import (AutoTokenizer, BertConfig, BertForSequenceClassification, DistilBertConfig,
                              DistilBertForSequenceClassification, RobertaConfig,
                              RobertaForSequenceClassification)
    from m4bench.text_explainers import EXPLAINERS as TEXP
    from m4bench.text_metrics import InfidelityCacheNLP, morf_lerf_abpc as t_morf
    from m4bench.text_models import TextModel

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    small = dict(vocab_size=tok.vocab_size, num_labels=2, attn_implementation="eager")
    nets = {
        "bert": BertForSequenceClassification(BertConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
                                                         intermediate_size=128, **small)),
        "distilbert": DistilBertForSequenceClassification(DistilBertConfig(dim=64, n_layers=3, n_heads=4,
                                                                           hidden_dim=128, **small)),
        "roberta": RobertaForSequenceClassification(RobertaConfig(hidden_size=64, num_hidden_layers=3,
                                                                  num_attention_heads=4, intermediate_size=128,
                                                                  pad_token_id=tok.pad_token_id,
                                                                  max_position_embeddings=600, **small)),
    }
    text = "a gripping , beautifully acted film that never once loses its nerve . " * 3
    rng = np.random.default_rng(0)
    for name, net in nets.items():
        net.eval().to(DEV)
        for p in net.parameters():
            p.requires_grad_(False)
        tm = TextModel(name, net, tok, 512)
        ids = tm.encode(text).to(DEV)
        with torch.no_grad():
            label = int(tm.logits(ids).argmax(1))
        for m, fn in TEXP.items():
            out = fn(tm, ids, label, rng, ig_steps=200, sg_samples=4, lime_samples=200, bt_steps=4,
                     grad_bs=50, start_layer=0)
            ok = out["abs"].shape == (ids.shape[1],) and np.isfinite(out["abs"]).all() and np.abs(out["abs"]).sum() > 0
            check(f"{name}/{m} per-token vector", ok, str(out["abs"].shape))
            if m == "ig":
                with torch.no_grad():
                    lo = tm.logits(ids)[0, label]
                    from m4bench.text_explainers import _EmbeddingHook, _word_emb
                    h = _EmbeddingHook(_word_emb(tm), scale=torch.zeros(1, device=DEV))
                    lb = tm.logits(ids)[0, label]
                    h.remove()
                gap, s = float(lo - lb), float(out["sum"].sum())
                check(f"{name}/IG completeness", abs(s - gap) < 0.05 * abs(gap) + 1e-3, f"sum={s:.4f} gap={gap:.4f}")
        sc, morf_p, lerf_p = t_morf(tm, ids, np.ones(ids.shape[1]), DEV)
        check(f"{name} MoRF/LeRF curves 101 points", len(morf_p) == 101 and len(lerf_p) == 101)
        check(f"{name} INFD finite", np.isfinite(InfidelityCacheNLP(tm, ids, DEV).score(np.random.rand(ids.shape[1]))))


if __name__ == "__main__":
    torch.manual_seed(0)
    vision_checks()
    text_checks()
    print(f"\n{len(FAILS)} failure(s)" + (f": {FAILS}" if FAILS else ""))
    sys.exit(1 if FAILS else 0)
