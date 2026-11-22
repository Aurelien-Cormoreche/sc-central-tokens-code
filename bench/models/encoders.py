"""Frozen encoders. Each one runs ONE forward pass per crop and returns both the model-card
standard embedding (`cls` readout) and the final-layer patch-token grid (for `central`).

    enc = build_encoder("uni2", cfg.models["uni2"], input_px=224, device="cuda", precision="bf16")
    out = enc.encode(crops_uint8)   # crops (B, N, N, 3) uint8 on the device
    out["cls"], out["central"]       # float32 (B, D_std), (B, D_tok)

`reference(x)` reproduces the official model-card snippet (fresh model built with the card's
exact kwargs, plain model(x)); `bench smoke` compares it with `features()` for parity.

Standard embeddings (verified against model cards / official code, 2026-11):
  uni2      CLS (1536)                      timm hf-hub:MahmoodLab/UNI2-h, 8 register tokens
  conch15   attentional pooler + LN (768)   TRIDENT create_model_from_pretrained, trunk ViT-L/16
  virchow2  CLS || mean(patch tokens) (2560) timm hf-hub:paige-ai/Virchow2, 4 register tokens
  hoptimus1 CLS (1536)                      timm hf-hub:bioptimus/H-optimus-1, 4 register tokens
  lemon     CLS (384, ViT-S/8)              timm VisionTransformer as in LEMON's model.py (MIT)
  genbio    3 x CLS (4608)                  HF AutoModel genbio-ai/genbio-pathfm, per-channel forward
"""
from __future__ import annotations

import contextlib
import json
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from bench.config import ModelCfg
from bench.models.readouts import central_flat

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _hub(repo: str, revision: Optional[str]) -> str:
    return f"hf-hub:{repo}@{revision}" if revision else f"hf-hub:{repo}"


class Encoder:
    name = "base"
    mean, std = IMAGENET

    def __init__(self, mcfg: ModelCfg, input_px: int, device: str = "cuda", precision: str = "bf16"):
        if input_px % mcfg.patch:
            raise ValueError(f"{self.name}: input {input_px} px is not a multiple of patch {mcfg.patch}")
        self.mcfg = mcfg
        self.input_px = input_px
        self.patch = mcfg.patch
        self.grid = input_px // mcfg.patch
        self.device = torch.device(device)
        self.precision = precision if self.device.type == "cuda" else "fp32"
        self.model: Optional[nn.Module] = None
        self.revision = mcfg.revision
        self._mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
        self._central = central_flat(self.grid).to(self.device)

    # ── to implement ──
    def build(self) -> nn.Module:
        raise NotImplementedError

    def features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: normalised (B, 3, N, N) -> (standard embedding (B, D), patch tokens (B, g*g, Dt))."""
        raise NotImplementedError

    def build_reference(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Official model-card forward (normalised input -> standard embedding)."""
        model = self.model
        return lambda x: model(x)

    # ── shared ──
    def load(self) -> "Encoder":
        self.model = self.build().eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    def normalize(self, crops_uint8: torch.Tensor) -> torch.Tensor:
        x = crops_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        return (x - self._mean) / self._std

    def autocast(self):
        if self.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=_DTYPES[self.precision])

    @torch.inference_mode()
    def encode(self, crops_uint8: torch.Tensor, central: bool = True) -> Dict[str, torch.Tensor]:
        x = self.normalize(crops_uint8)
        with self.autocast():
            std, tokens = self.features(x)
        g = self.grid
        if tokens.shape[1] != g * g:
            raise RuntimeError(f"{self.name}: expected {g * g} patch tokens for {self.input_px} px, got {tokens.shape[1]}")
        out = {"cls": std.float()}
        if central:
            out["central"] = tokens.index_select(1, self._central).float().mean(dim=1)
        return out

    @torch.inference_mode()
    def tokens(self, crops_uint8: torch.Tensor) -> torch.Tensor:
        """(B, g*g, Dt) float32 patch tokens (multicell readout)."""
        x = self.normalize(crops_uint8)
        with self.autocast():
            _, tokens = self.features(x)
        return tokens.float()

    @property
    def dims(self) -> Dict[str, int]:
        with torch.inference_mode():
            x = torch.zeros(1, self.input_px, self.input_px, 3, dtype=torch.uint8, device=self.device)
            out = self.encode(x)
        return {k: int(v.shape[1]) for k, v in out.items()}


# ── timm models ──

class _Timm(Encoder):
    repo = ""

    def card_kwargs(self) -> dict:
        return {}

    def build(self) -> nn.Module:
        import timm
        kw = {**self.card_kwargs(), "dynamic_img_size": True}
        return timm.create_model(_hub(self.repo, self.revision), pretrained=True, **kw)

    def build_reference(self):
        import timm
        ref = timm.create_model(_hub(self.repo, self.revision), pretrained=True, **self.card_kwargs())
        ref = ref.eval().to(self.device)
        return lambda x: ref(x)

    def _tokens(self, x):
        t = self.model.forward_features(x)
        return t, t[:, self.model.num_prefix_tokens:]


class UNI2(_Timm):
    name, repo = "uni2", "MahmoodLab/UNI2-h"

    def card_kwargs(self) -> dict:
        import timm
        return {"img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24, "init_values": 1e-5,
                "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0, "no_embed_class": True,
                "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU, "reg_tokens": 8,
                "dynamic_img_size": True}

    def features(self, x):
        t, tok = self._tokens(x)
        return t[:, 0], tok


class Virchow2(_Timm):
    name, repo = "virchow2", "paige-ai/Virchow2"

    def card_kwargs(self) -> dict:
        import timm
        return {"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU}

    def build_reference(self):
        ref_model = super().build_reference()

        def ref(x):
            out = ref_model(x)                       # (B, 1 + 4 + g*g, 1280)
            return torch.cat([out[:, 0], out[:, 5:].mean(1)], dim=-1)
        return ref

    def features(self, x):
        t, tok = self._tokens(x)
        return torch.cat([t[:, 0], tok.mean(1)], dim=-1), tok


class HOptimus1(_Timm):
    name, repo = "hoptimus1", "bioptimus/H-optimus-1"
    mean, std = (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)

    def card_kwargs(self) -> dict:
        return {"init_values": 1e-5, "dynamic_img_size": False}

    def features(self, x):
        t, tok = self._tokens(x)
        return t[:, 0], tok


# ── CONCH v1.5 (TRIDENT) ──

class Conch15(Encoder):
    name, repo = "conch15", "MahmoodLab/conchv1_5"

    def build(self) -> nn.Module:
        from huggingface_hub import hf_hub_download
        from trident.patch_encoder_models.model_zoo.conchv1_5.conchv1_5 import create_model_from_pretrained
        ckpt = hf_hub_download(self.repo, "pytorch_model_vision.bin", revision=self.revision)
        model, _ = create_model_from_pretrained(checkpoint_path=ckpt, img_size=448)
        return model

    def features(self, x):
        t = self.model.trunk.forward_features(x)                   # (B, 1 + g*g, 1024), final norm applied
        pooled = self.model.ln_contrast(self.model.attn_pool_contrast(t)[:, 0])
        return pooled, t[:, 1:]


# ── LEMON ──

class Lemon(Encoder):
    name, repo = "lemon", "aliceblondel/LEMON"

    def __init__(self, mcfg: ModelCfg, input_px: int, device: str = "cuda", precision: str = "bf16"):
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(self.repo, "mean_std.json", revision=mcfg.revision)) as fh:
            st = json.load(fh)
        self.mean, self.std = tuple(st["mean"]), tuple(st["std"])
        super().__init__(mcfg, input_px, device, precision)

    def _arch(self, dynamic: bool) -> nn.Module:
        # identical to LEMON's build_lemon_moco_backbone (num_heads=12 for both archs, as there)
        from timm.models.vision_transformer import VisionTransformer
        arch = self.mcfg.arch or "vits8"
        return VisionTransformer(img_size=40, patch_size=8, embed_dim=384 if arch == "vits8" else 768, depth=12,
                                 num_heads=12, mlp_ratio=4, qkv_bias=True,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0,
                                 dynamic_img_size=dynamic)

    def _state(self) -> dict:
        from huggingface_hub import hf_hub_download
        arch = self.mcfg.arch or "vits8"
        ckpt = torch.load(hf_hub_download(self.repo, f"lemon_{arch}.pth.tar", revision=self.revision),
                          map_location="cpu", weights_only=True)
        out = {}
        for k, v in ckpt["state_dict"].items():
            k = k.removeprefix("module.")
            if k.startswith("base_encoder.") and not k.startswith(("base_encoder.head.", "base_encoder.fc.")):
                out[k.removeprefix("base_encoder.")] = v
        return out

    def _load(self, dynamic: bool) -> nn.Module:
        m = self._arch(dynamic)
        res = m.load_state_dict(self._state(), strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"LEMON weights mismatch: missing {res.missing_keys} unexpected {res.unexpected_keys}")
        return m

    def build(self) -> nn.Module:
        return self._load(dynamic=True)

    def build_reference(self):
        ref = self._load(dynamic=False).eval().to(self.device)
        return lambda x: ref(x)

    def features(self, x):
        t = self.model.forward_features(x)
        return t[:, 0], t[:, self.model.num_prefix_tokens:]


# ── GenBio-PathFM ──

class GenBio(Encoder):
    name, repo = "genbio", "genbio-ai/genbio-pathfm"
    mean, std = (0.697, 0.575, 0.728), (0.188, 0.240, 0.187)

    def build(self) -> nn.Module:
        from transformers import AutoModel
        return AutoModel.from_pretrained(self.repo, trust_remote_code=True, revision=self.revision)

    def features(self, x):
        cls, patches = self.model.forward_with_patches(x)   # (B, 3*1536), (B, g*g, 3*1536)
        return cls, patches


# ── tiny random ViT (CPU tests / synthetic smoke only) ──

class Tiny(Encoder):
    name = "tiny"

    def build(self) -> nn.Module:
        from timm.models.vision_transformer import VisionTransformer
        torch.manual_seed(0)
        return VisionTransformer(img_size=self.mcfg.input_px, patch_size=self.patch, embed_dim=32, depth=2,
                                 num_heads=2, num_classes=0, reg_tokens=2, dynamic_img_size=True)

    def build_reference(self):
        model = self.model
        return lambda x: model(x)

    def features(self, x):
        t = self.model.forward_features(x)
        return t[:, 0], t[:, self.model.num_prefix_tokens:]


ENCODERS = {c.name: c for c in (UNI2, Conch15, Virchow2, HOptimus1, Lemon, GenBio, Tiny)}


def build_encoder(name: str, mcfg: ModelCfg, input_px: Optional[int] = None, device: str = "cuda",
                  precision: str = "bf16") -> Encoder:
    if name not in ENCODERS:
        raise KeyError(f"no encoder named {name!r} (known: {', '.join(ENCODERS)})")
    return ENCODERS[name](mcfg, input_px or mcfg.input_px, device, precision).load()


def round_to_patch(px: int, patch: int, mode: str = "up") -> int:
    import math
    f = {"up": math.ceil, "down": math.floor, "nearest": round}[mode]
    return max(1, int(f(px / patch))) * patch
