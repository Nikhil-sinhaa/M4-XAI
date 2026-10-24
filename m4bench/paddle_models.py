"""
PyTorch mirrors of the exact networks M4 used, so the authors' Paddle weights load 1:1.

  * ResNet-50/101/152, VGG16 : paddle.vision (2.4) == torchvision layout, weights from paddle-hapi
  * MobileNet-V3-Large       : paddle.vision.models.mobilenetv3 (own layout, mirrored below)
  * ViT-S/B/L                : M4_XAI_Benchmark/benchmark-cv/models/vit.py + PaddleClas weights
  * MAE-ViT-B                : M4_XAI_Benchmark/benchmark-cv/models/mae_vit.py (global pool + fc_norm)
                               with the official MAE fine-tuned checkpoint (the authors' mae_finetune
                               .pdparams is a conversion of it and was never published)

Paddle -> torch weight conversion: Linear weights are stored [in, out] in Paddle (transposed here),
BatchNorm `_mean/_variance` are torch `running_mean/running_var`.
"""

import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PADDLE_URLS = {
    "resnet50": "https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams",
    "resnet101": "https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams",
    "resnet152": "https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams",
    "vgg16": "https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams",
    "mobilenet": "https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_large_x1.0.pdparams",
    "vit_small": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams",
    "vit_base": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams",
    "vit_large": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams",
}
MAE_FINETUNED_URL = "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth"
WEIGHTS_DIR = Path("models/paddle_weights")


# ─────────────────────────────────────────────────────────────────────────────
# .pdparams loading (pickle of numpy arrays) without Paddle and without arbitrary code execution
# ─────────────────────────────────────────────────────────────────────────────

class _NumpyOnlyUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"), ("numpy", "dtype"), ("collections", "OrderedDict"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
        ("_codecs", "encode"),        # Python-2-era numpy pickles store raw bytes as _codecs.encode(str, "latin1")
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a .pdparams file")
        if module.endswith("multiarray"):                 # numpy.core was renamed numpy._core in numpy 2
            import importlib
            for mod in ("numpy._core.multiarray", "numpy.core.multiarray"):
                try:
                    return getattr(importlib.import_module(mod), name)
                except ImportError:
                    continue
        return super().find_class(module, name)


def load_pdparams(path: Path) -> dict:
    with open(path, "rb") as f:
        state = _NumpyOnlyUnpickler(f, encoding="latin1").load()
    return {k: v for k, v in state.items() if isinstance(v, np.ndarray)}


def download(url: str, dest_dir: Path = WEIGHTS_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url.rsplit("/", 1)[-1]
    if not dest.exists():
        torch.hub.download_url_to_file(url, str(dest), progress=True)
    return dest


def load_paddle_state(net: nn.Module, paddle_state: dict) -> None:
    """Copy a Paddle state dict into a mirrored torch module; every tensor must be matched."""
    linear_weights = {f"{n}.weight" for n, m in net.named_modules() if isinstance(m, nn.Linear)}
    new, used = {}, set()
    for k, ref in net.state_dict().items():
        if k.endswith("num_batches_tracked"):
            new[k] = ref
            continue
        pk = k.replace("running_mean", "_mean").replace("running_var", "_variance")
        if pk not in paddle_state:
            raise KeyError(f"Paddle weights lack {pk} (torch key {k})")
        v = torch.from_numpy(np.ascontiguousarray(paddle_state[pk]))
        if k in linear_weights:
            v = v.T
        if v.shape != ref.shape:
            raise ValueError(f"{k}: paddle {tuple(v.shape)} vs torch {tuple(ref.shape)}")
        new[k] = v.to(ref.dtype)
        used.add(pk)
    unused = set(paddle_state) - used
    if unused:
        raise KeyError(f"unused Paddle tensors: {sorted(unused)[:5]} ...")
    net.load_state_dict(new, strict=True)


# ─────────────────────────────────────────────────────────────────────────────
# MobileNet-V3-Large (paddle.vision.models.mobilenetv3, release/2.4)
# ─────────────────────────────────────────────────────────────────────────────

def _make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


class ConvNormActivation(nn.Sequential):
    def __init__(self, cin, cout, k, stride, padding, groups=1, norm=None, act=None):
        layers = [nn.Conv2d(cin, cout, k, stride, padding, groups=groups, bias=norm is None)]
        if norm is not None:
            layers.append(norm(cout))
        if act is not None:
            layers.append(act())
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    def __init__(self, cin, csq):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(cin, csq, 1)
        self.fc2 = nn.Conv2d(csq, cin, 1)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Hardsigmoid()      # paddle Hardsigmoid(slope=1/6, offset=0.5) == torch's

    def forward(self, x):
        s = self.scale_activation(self.fc2(self.activation(self.fc1(self.avgpool(x)))))
        return s * x


class InvertedResidual(nn.Module):
    def __init__(self, cin, cexp, cout, k, stride, use_se, act, norm):
        super().__init__()
        self.use_res_connect = stride == 1 and cin == cout
        self.use_se, self.expand = use_se, cin != cexp
        if self.expand:
            self.expand_conv = ConvNormActivation(cin, cexp, 1, 1, 0, norm=norm, act=act)
        self.bottleneck_conv = ConvNormActivation(cexp, cexp, k, stride, (k - 1) // 2, groups=cexp,
                                                  norm=norm, act=act)
        if use_se:
            self.mid_se = SqueezeExcitation(cexp, _make_divisible(cexp // 4))
        self.linear_conv = ConvNormActivation(cexp, cout, 1, 1, 0, norm=norm, act=None)

    def forward(self, x):
        y = self.expand_conv(x) if self.expand else x
        y = self.bottleneck_conv(y)
        if self.use_se:
            y = self.mid_se(y)
        y = self.linear_conv(y)
        return x + y if self.use_res_connect else y


class MobileNetV3Large(nn.Module):
    CONFIG = [  # in, kernel, expanded, out, use_se, activation, stride
        (16, 3, 16, 16, False, "relu", 1), (16, 3, 64, 24, False, "relu", 2),
        (24, 3, 72, 24, False, "relu", 1), (24, 5, 72, 40, True, "relu", 2),
        (40, 5, 120, 40, True, "relu", 1), (40, 5, 120, 40, True, "relu", 1),
        (40, 3, 240, 80, False, "hardswish", 2), (80, 3, 200, 80, False, "hardswish", 1),
        (80, 3, 184, 80, False, "hardswish", 1), (80, 3, 184, 80, False, "hardswish", 1),
        (80, 3, 480, 112, True, "hardswish", 1), (112, 3, 672, 112, True, "hardswish", 1),
        (112, 5, 672, 160, True, "hardswish", 2), (160, 5, 960, 160, True, "hardswish", 1),
        (160, 5, 960, 160, True, "hardswish", 1),
    ]

    def __init__(self, num_classes=1000):
        super().__init__()
        norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)   # paddle momentum 0.99 == torch 0.01
        acts = {"relu": nn.ReLU, "hardswish": nn.Hardswish}
        cfg = [(_make_divisible(i), k, _make_divisible(e), _make_divisible(o), se, acts[a], s)
               for i, k, e, o, se, a, s in self.CONFIG]
        self.conv = ConvNormActivation(3, cfg[0][0], 3, 2, 1, norm=norm, act=nn.Hardswish)
        self.blocks = nn.Sequential(*[InvertedResidual(i, e, o, k, s, se, a, norm) for i, k, e, o, se, a, s in cfg])
        last_in = cfg[-1][0]
        self.lastconv = ConvNormActivation(last_in, last_in * 6, 1, 1, 0, norm=norm, act=nn.Hardswish)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        last_channel = _make_divisible(1280)
        self.classifier = nn.Sequential(nn.Linear(last_in * 6, last_channel), nn.Hardswish(), nn.Dropout(0.2),
                                        nn.Linear(last_channel, num_classes))

    def forward(self, x):
        x = self.avgpool(self.lastconv(self.blocks(self.conv(x))))
        return self.classifier(torch.flatten(x, 1))


# ─────────────────────────────────────────────────────────────────────────────
# ViT (benchmark-cv/models/vit.py) and MAE-ViT (benchmark-cv/models/mae_vit.py)
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)          # hook point for GA / BT, as upstream
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = self.attn_drop(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, qk_scale, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch=16):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """global_pool=False: vit.py (class token + norm). global_pool=True: mae_vit.py (mean of patches + fc_norm)."""

    def __init__(self, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale=None, eps=1e-5,
                 num_classes=1000, global_pool=False):
        super().__init__()
        self.global_pool = global_pool
        self.patch_embed = PatchEmbed(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, eps)
                                     for _ in range(depth)])
        if global_pool:
            self.fc_norm = nn.LayerNorm(embed_dim, eps=eps)
        else:
            self.norm = nn.LayerNorm(embed_dim, eps=eps)
        self.head = nn.Linear(embed_dim, num_classes)
        # upstream _init_weights (only matters for the from-scratch SynScore models)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.fc_norm(x[:, 1:].mean(1)) if self.global_pool else self.norm(x)[:, 0]
        return self.head(x)


VIT_CONFIGS = {
    "vit_small": dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, qkv_bias=False, qk_scale=768 ** -0.5, eps=1e-5),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, eps=1e-6),
    "vit_large": dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, eps=1e-6),
    "mae_vit_base": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, eps=1e-6,
                         global_pool=True),
}


def build(name: str, num_classes: int = 1000) -> nn.Module:
    import torchvision.models as tvm
    if name in ("resnet50", "resnet101", "resnet152"):
        return getattr(tvm, name)(weights=None, num_classes=num_classes)
    if name == "vgg16":
        return tvm.vgg16(weights=None, num_classes=num_classes)
    if name == "mobilenet":
        return MobileNetV3Large(num_classes)
    return VisionTransformer(**VIT_CONFIGS[name], num_classes=num_classes)


def build_pretrained(name: str) -> nn.Module:
    net = build(name)
    if name == "mae_vit_base":
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(download(MAE_FINETUNED_URL), map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt["model"], strict=True)
    else:
        load_paddle_state(net, load_pdparams(download(PADDLE_URLS[name])))
    return net
