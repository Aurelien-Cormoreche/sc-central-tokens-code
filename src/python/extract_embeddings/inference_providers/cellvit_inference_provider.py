import torch
import os
from .inference_provider import InferenceProvider
from .attention_utils import AttentionCapture, save_cell_attention_maps, sample_vis_indices
from ..data.patch_dataset import PatchDataset, MultiCellPatchDataset, multicell_collate_fn
from os import PathLike
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.utils.tools import unflatten_dict
from torchvision import transforms as T
from typing import Literal, Union
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.python.code_configs import paths

'''
Inspired from the cellViT_inference git repo
'''

# CellViT-HIPT (ViT-256): CLS(0) + 16*16 spatial(1-256)
_HIPT_CLS_IDX   = 0
_HIPT_PREFIX     = 1

# CellViT-SAM: no CLS; all tokens are spatial
_SAM_CLS_IDX    = None
_SAM_PREFIX      = 0

_NUM_PATCHES_PER_SIDE = 16


class CellViTInferenceProvider(InferenceProvider):

    def __init__(self, base_model: str, patches_to_save: dict[str, tuple[int, int]] =
         {'top_left':(7,7), 'top_right':(7,8), 'bottom_left':(8,7), 'bottom_right':(8,8)}):
        super().__init__(patches_to_save)
        self.num_patches_per_side = _NUM_PATCHES_PER_SIDE
        self.base_model = base_model
        self.cache_dir = paths.CELLVIT_CACHE

    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.base_model == 'SAM':
            file_name = "CellViT-SAM-H-x40-AMP.pth"
            self.embedding_dim = 1280
        elif self.base_model == 'HIPT':
            file_name = "CellViT-256-x40-AMP.pth"
            self.embedding_dim = 384
        else:
            raise ValueError("Base model not available please provide one of ['SAM', 'HIPT']")

        model_checkpoint = torch.load(os.path.join(self.cache_dir, file_name), map_location='cpu')
        self.run_conf = unflatten_dict(model_checkpoint["config"], ".")
        self.model = self._get_model(model_type=model_checkpoint["arch"])
        self.model.load_state_dict(model_checkpoint["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)
        self.run_conf["model"]["token_patch_size"] = self.model.patch_size
        self.model_arch = model_checkpoint["arch"]
        self.mixed_precision = self.run_conf["training"].get("mixed_precision", False)
        self.patch_size = self.model.patch_size
        self._load_inference_transforms()

    # ── model-specific token config ───────────────────────────────────────────

    @property
    def _cls_idx(self) -> int | None:
        return _HIPT_CLS_IDX if self.base_model == 'HIPT' else _SAM_CLS_IDX

    @property
    def _prefix(self) -> int:
        return _HIPT_PREFIX if self.base_model == 'HIPT' else _SAM_PREFIX

    # ── shared attention helper ────────────────────────────────────────────────

    def _collect_attention_maps(
        self,
        capture: AttentionCapture,
        local_idx: int,
    ) -> dict[str, torch.Tensor]:
        maps = {}
        S = self.num_patches_per_side
        if self._cls_idx is not None:
            m = capture.get_attention_for_token(self._cls_idx, self._prefix, S)
            if m is not None:
                maps['cls'] = m[local_idx]
        for key, (x, y) in self.patches_to_save.items():
            tok_idx = self._prefix + x * S + y
            m = capture.get_attention_for_token(tok_idx, self._prefix, S)
            if m is not None:
                maps[key] = m[local_idx]
        return maps

    def _forward(self, patches: torch.Tensor):
        """Run encoder forward and return last-block token tensor."""
        with torch.inference_mode():
            if self.mixed_precision:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, z = self.model.encoder(patches)
            else:
                _, _, z = self.model.encoder(patches)
        return z[-1]  # (B, N, C)

    # ── attention-only mode ────────────────────────────────────────────────────

    def run_attention_only(
        self,
        dataset: PatchDataset,
        output_path: PathLike,
        n_samples: int = 10,
    ) -> None:
        dataset.transform = self.inference_transforms
        vis_indices = sorted(sample_vis_indices(dataset.labels_dataset, n_samples))
        subset = torch.utils.data.Subset(dataset, vis_indices)
        dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        capture = AttentionCapture()
        if not capture.register_on_attn_drop(self.model.encoder):
            raise RuntimeError(f"Could not find attn_drop in CellViT-{self.base_model} encoder.")

        print(f"Saving attention maps to: {output_path}")
        for subset_idx, (patches, labels, cell_ids) in enumerate(tqdm(dataloader, desc="Attention")):
            global_idx = vis_indices[subset_idx]
            patches = patches.to(self.device)
            self._forward(patches)  # triggers hook

            attn_by_query = self._collect_attention_maps(capture, local_idx=0)
            if not attn_by_query:
                print(f"WARNING: attention hook did not fire for sample {global_idx} — attn_drop may be bypassed")
                continue

            raw_patch   = dataset.get_raw_patch(global_idx)
            label_str   = labels[0].decode()   if isinstance(labels[0],   bytes) else str(labels[0])
            cell_id_str = cell_ids[0].decode() if isinstance(cell_ids[0], bytes) else str(cell_ids[0])
            save_cell_attention_maps(raw_patch, attn_by_query, cell_id_str, label_str, str(output_path))

        capture.remove()

    # ── standard embedding inference ──────────────────────────────────────────

    def inference(
        self,
        dataset: PatchDataset,
        output_path: PathLike,
        batch_size: int = 8,
        visualize_attention: bool = False,
        attention_output_path: PathLike | None = None,
        n_attention_samples: int = 10,
        save_cls: bool = False,
    ) -> None:
        if save_cls and self.base_model == 'SAM':
            print("[CellViTInferenceProvider] WARNING: CellViT-SAM has no CLS token — save_cls ignored.")
            save_cls = False

        self.create_output_file(output_path, num_samples=len(dataset), embedding_dim=self.embedding_dim,
                                dataset_stats=self.compute_dataset_statistics(dataset), save_cls=save_cls)
        dataset.transform = self.inference_transforms
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        vis_indices: set[int] = set()
        capture = AttentionCapture()
        if visualize_attention:
            if attention_output_path is None:
                raise ValueError("attention_output_path must be set when visualize_attention=True")
            if not capture.register_on_attn_drop(self.model.encoder):
                print(f"[AttentionCapture] WARNING: attn_drop not found in CellViT-{self.base_model} — skipping visualisation.")
                visualize_attention = False
            else:
                vis_indices = sample_vis_indices(dataset.labels_dataset, n_attention_samples)

        for batch_idx, (patches, labels, cell_ids) in enumerate(tqdm(dataloader, desc="Inference", total=len(dataloader))):
            patches = patches.to(self.device)
            outputs = self._forward(patches)

            if self.base_model == 'HIPT':
                cls_token = outputs[:, 0, :] if save_cls else None
                patch_tokens = outputs[:, 1:, :]
                B, N, C = patch_tokens.shape
                H = W = int(N ** 0.5)
                patch_tokens = patch_tokens.reshape(B, H, W, C)
            else:
                cls_token = None
                patch_tokens = outputs

            tokens_to_save = []
            for key in self.patches_to_save:
                x, y = self.patches_to_save[key]
                tokens_to_save.append(patch_tokens[:, x, y, :])
            self.save_embeddings(tokens_to_save, cell_ids, labels, output_path, start_idx=batch_idx * batch_size, cls_token=cls_token)

            if visualize_attention and capture.weights is not None:
                start = batch_idx * batch_size
                for local_idx in range(len(patches)):
                    if start + local_idx not in vis_indices:
                        continue
                    attn_by_query = self._collect_attention_maps(capture, local_idx)
                    if not attn_by_query:
                        continue
                    raw_patch   = dataset.get_raw_patch(start + local_idx)
                    label_str   = labels[local_idx].decode()   if isinstance(labels[local_idx],   bytes) else str(labels[local_idx])
                    cell_id_str = cell_ids[local_idx].decode() if isinstance(cell_ids[local_idx], bytes) else str(cell_ids[local_idx])
                    save_cell_attention_maps(raw_patch, attn_by_query, cell_id_str, label_str, str(attention_output_path))

        capture.remove()

    # ── multicell patch inference ─────────────────────────────────────────────

    def inference_multicell(
        self,
        dataset: MultiCellPatchDataset,
        output_path: PathLike,
        batch_size: int = 4,
    ) -> None:
        self.create_output_file_multicell(
            output_path,
            num_cells=dataset.total_cells,
            embedding_dim=self.embedding_dim,
            dataset_stats=self.compute_dataset_statistics_multicell(dataset),
        )
        dataset.transform = self.inference_transforms
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=multicell_collate_fn,
        )


        cell_write_idx = 0

        for patches, batch_rel_xs, batch_rel_ys, batch_cell_ids, batch_cell_labels in tqdm(dataloader, desc="Multicell Inference"):
            patches = patches.to(self.device)
            outputs = self._forward(patches)  # (B, N, C) or (B, H, W, C) depending on backbone

            # Normalise to (B, H, W, C) spatial layout
            if self.base_model == 'HIPT':
                spatial = outputs[:, 1:, :]           # remove CLS → (B, H*W, C)
            else:
                spatial = outputs                      # SAM: already (B, H, W, C) or (B, N, C)

            B, N, C = spatial.shape[:3] if spatial.dim() == 3 else (spatial.shape[0], spatial.shape[1] * spatial.shape[2], spatial.shape[3])
            H = W = int(N ** 0.5)
            spatial = spatial.reshape(B, H, W, C) if spatial.dim() == 3 else spatial

            for i in range(B):
                rel_xs      = batch_rel_xs[i]
                rel_ys      = batch_rel_ys[i]
                cell_ids    = batch_cell_ids[i]
                cell_labels = batch_cell_labels[i]

                token_cols = rel_xs // self.patch_size
                token_rows = rel_ys // self.patch_size

                cell_tokens = spatial[i, token_rows, token_cols, :]  # (N_cells, C)
                self.save_embeddings_multicell(cell_tokens, cell_ids, cell_labels, output_path, start_idx=cell_write_idx)
                cell_write_idx += len(cell_ids)

    def _load_inference_transforms(self):
        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std  = transform_settings["normalize"].get("std",  (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std  = (0.5, 0.5, 0.5)
        self.inference_transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def _get_model(self, model_type: Literal["CellViT256", "CellViTSAM"]) -> Union[CellViT256, CellViTSAM]:
        implemented_models = ["CellViT256", "CellViTSAM"]
        if model_type not in implemented_models:
            raise NotImplementedError(f"Unknown model type. Please select one of {implemented_models}")
        elif model_type == "CellViT256":
            return CellViT256(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        else:
            return CellViTSAM(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
