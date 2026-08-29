import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster / headless environments
import matplotlib.cm as cm
import torch
import numpy as np
import os


class AttentionCapture:
    """Captures multi-head self-attention weights from a ViT layer via a forward hook.

    The hook is registered on the `attn_drop` Dropout module of the last attention
    layer found by walking the model. That module receives `(B, heads, N, N)` as
    input directly after softmax, which is exactly what we want.

    Usage:
        capture = AttentionCapture()
        ok = capture.register_on_attn_drop(model)
        # ... run forward pass ...
        maps = capture.get_attention_for_token(token_idx, prefix_tokens, patches_per_side)
        capture.remove()
    """

    def __init__(self):
        self.weights: torch.Tensor | None = None  # (B, heads, N, N)
        self._handles: list = []

    def _hook_attn_drop(self, module, input, output):
        # attn_drop input[0] is (B, heads, N, N) after softmax
        self.weights = input[0].detach().cpu().float()

    def register_on_attn_drop(self, model: torch.nn.Module) -> bool:
        """Disable fused attention, then hook the attn_drop of the last attention layer.

        timm's Attention module uses F.scaled_dot_product_attention when
        self.fused_attn=True, which passes dropout inline and never calls the
        attn_drop Dropout module — the hook would never fire.  Setting
        fused_attn=False forces the explicit path where attn_drop is called.

        Returns True on success, False if no attn_drop was found.
        """
        # Force non-fused attention path so attn_drop is called as a module
        n_patched = 0
        for m in model.modules():
            if hasattr(m, 'fused_attn'):
                m.fused_attn = False
                n_patched += 1
        if n_patched == 0:
            print("[AttentionCapture] Note: no fused_attn attribute found — model may already use explicit attention.")

        last_drop = None
        for name, m in model.named_modules():
            if name.split(".")[-1] == "attn_drop":
                last_drop = m
        if last_drop is None:
            return False
        self._handles.append(last_drop.register_forward_hook(self._hook_attn_drop))
        return True

    def get_attention_for_token(
        self,
        token_idx: int,
        prefix_tokens: int,
        num_patches_per_side: int,
    ) -> torch.Tensor | None:
        """Return per-head attention from one query token to all spatial tokens.

        Args:
            token_idx: index of the query token in the full token sequence
                       (e.g. 0 for CLS; prefix_tokens + x*S + y for a spatial patch).
            prefix_tokens: number of non-spatial tokens at the start of the sequence.
            num_patches_per_side: S, so spatial grid is S×S.

        Returns:
            Tensor of shape (B, n_heads, S, S) or None if no weights captured yet.
        """
        if self.weights is None:
            return None
        B, n_heads, N, _ = self.weights.shape
        n_spatial = num_patches_per_side ** 2
        attn = self.weights[:, :, token_idx, prefix_tokens: prefix_tokens + n_spatial]
        return attn.reshape(B, n_heads, num_patches_per_side, num_patches_per_side)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.weights = None


# ── Visualization helpers ──────────────────────────────────────────────────────

def attention_to_heatmap(
    attn_map: torch.Tensor | np.ndarray,
    patch_size: tuple[int, int],
    colormap: str = "inferno",
) -> np.ndarray:
    """Upsample a (H_grid, W_grid) attention map to patch_size pixels.

    Returns an RGB uint8 array of shape (patch_size[1], patch_size[0], 3).
    """
    from PIL import Image as PILImage
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.numpy()
    attn_map = attn_map.astype(np.float32)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    w_out, h_out = patch_size
    heatmap_pil = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (w_out, h_out), resample=PILImage.BILINEAR
    )
    heatmap_arr = np.array(heatmap_pil) / 255.0
    rgb = (cm.get_cmap(colormap)(heatmap_arr)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def save_cell_attention_maps(
    raw_patch,
    attn_by_query: dict[str, torch.Tensor],
    cell_id: str,
    label: str,
    output_path: str,
    colormap: str = "inferno",
    alpha: float = 0.5,
):
    """Save per-head attention maps and weights for one cell.

    Output layout::

        {output_path}/{label}/{cell_id}/
            patch.png
            {query_name}/
                weights.npy          # (n_heads, H_grid, W_grid) float32
                head_00.png          # [H&E patch | attention overlay], exactly patch_size px wide
                head_01.png
                ...

    Each head_XX.png is composed directly with PIL (no matplotlib) so its pixel
    dimensions are exactly 2*W × H (two side-by-side panels at native patch resolution).

    Args:
        raw_patch: un-transformed PIL Image of the H&E patch.
        attn_by_query: mapping from query name (e.g. "cls", "top_left") to a
                       (n_heads, H_grid, W_grid) float tensor.
        cell_id: used as the sub-directory name and figure title.
        label: cell-type label, used as the top-level directory.
        output_path: root output directory.
        colormap: matplotlib colormap for the heatmap overlay.
        alpha: opacity of the heatmap overlay (blended into the H&E).
    """
    from PIL import Image as PILImage

    cell_dir = os.path.join(output_path, label, cell_id)
    os.makedirs(cell_dir, exist_ok=True)

    patch_rgb = raw_patch.convert("RGB")
    patch_np  = np.array(patch_rgb)           # (H, W, 3) uint8
    h_px, w_px = patch_np.shape[:2]

    # Save raw patch once at its native resolution
    patch_rgb.save(os.path.join(cell_dir, "patch.png"))

    for query_name, head_maps in attn_by_query.items():
        if isinstance(head_maps, torch.Tensor):
            head_maps = head_maps.cpu().float()
        head_maps_np = np.array(head_maps) if not isinstance(head_maps, np.ndarray) else head_maps

        query_dir = os.path.join(cell_dir, query_name)
        os.makedirs(query_dir, exist_ok=True)

        # Save raw attention weights
        np.save(os.path.join(query_dir, "weights.npy"), head_maps_np.astype(np.float32))

        # Save one PNG per head — composed with PIL, so output is exactly 2*W × H pixels
        n_heads = head_maps_np.shape[0]
        for hi in range(n_heads):
            heatmap_rgb = attention_to_heatmap(head_maps_np[hi], (w_px, h_px), colormap)  # (H, W, 3) uint8

            # Alpha-blend heatmap onto H&E patch
            overlay_np = (
                (1 - alpha) * patch_np.astype(np.float32)
                + alpha * heatmap_rgb.astype(np.float32)
            ).clip(0, 255).astype(np.uint8)

            # Side-by-side canvas: [raw patch | overlay], same height
            canvas = np.concatenate([patch_np, overlay_np], axis=1)  # (H, 2*W, 3)
            PILImage.fromarray(canvas).save(os.path.join(query_dir, f"head_{hi:02d}.png"))


def sample_vis_indices(labels_dataset, n_samples: int = 10, seed: int = 42) -> set[int]:
    """Stratified sampling of n_samples indices across all cell types."""
    rng = np.random.default_rng(seed)
    unique_labels = list(set(labels_dataset))
    per_class = max(1, n_samples // len(unique_labels))
    selected: set[int] = set()
    for label in unique_labels:
        mask = np.where(labels_dataset == label)[0]
        chosen = rng.choice(mask, min(per_class, len(mask)), replace=False)
        selected.update(chosen.tolist())
    return selected
