import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .inference_provider import InferenceProvider
from ..data.patch_dataset import PatchDataset, MultiCellPatchDataset, multicell_collate_fn
from os import PathLike
from tqdm import tqdm

# Phikon-v2 (Owkin): ViT-L/16, DINOv2-pretrained, no register tokens, @ 224x224.
# Loaded via HF `transformers` (Dinov2Model), not timm -- config.json has no
# num_register_tokens key (defaults to 0) and architectures: ["Dinov2Model"].
# Token layout after model(pixel_values=...).last_hidden_state: CLS(0) + 14*14 spatial(1-196).
_CLS_TOKEN_IDX = 0
_PREFIX_TOKENS = 1   # CLS only, no registers
_NUM_PATCHES_PER_SIDE = 14  # 224 / 16
_EMBED_DIM = 1024
_PATCH_SIZE = 16
_HF_REPO_ID = "owkin/phikon-v2"


class PhikonV2InferenceProvider(InferenceProvider):

    def __init__(self, patches_to_save: dict[str, tuple[int, int]] =
                 {'top_left': (6, 6), 'top_right': (6, 7), 'bottom_left': (7, 6), 'bottom_right': (7, 7)}):
        super().__init__(patches_to_save)
        self.num_patches_per_side = _NUM_PATCHES_PER_SIDE
        self.tokens_to_remove = _PREFIX_TOKENS
        self.embedding_dim = _EMBED_DIM
        self.patch_size = _PATCH_SIZE

    def load_model(self):
        from transformers import AutoModel

        # attn_implementation="eager" would be needed to make attention weights
        # observable via a dropout-module hook (see run_attention_only below) -- not
        # requested here, so left on the default (sdpa) for speed.
        self.model = AutoModel.from_pretrained(_HF_REPO_ID)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # Unlike UNI2-h/H-optimus-1/Virchow2, Phikon-v2's public checkpoint keeps its
        # trained mask_token (stored as "embeddings.mask_token", matched by
        # InferenceProvider.load_mask_token's suffix search) -- so this loads the
        # model's own pretraining-time mask embedding, not the zero-vector fallback.
        self.mask_token = self.load_mask_token_or_zero(_HF_REPO_ID)

        # Patches already arrive at the model's native 224x224 (see inference()'s
        # assert / ResizedCellDataset's size=224), so no resize/crop step is needed --
        # matches owkin/phikon-v2's preprocessor_config.json rescale (1/255) + normalize
        # steps (standard ImageNet mean/std).
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _patch_embed_module(self) -> torch.nn.Module:
        return self.model.embeddings.patch_embeddings

    # ── attention-only mode: not supported ─────────────────────────────────────

    def run_attention_only(self, dataset: PatchDataset, output_path: PathLike, n_samples: int = 10) -> None:
        raise NotImplementedError(
            "Attention visualisation is not supported for Phikon-v2: its HF `transformers` "
            "Dinov2Model implementation names its attention-probs dropout module "
            "'...attention.dropout', not 'attn_drop' -- AttentionCapture.register_on_attn_drop "
            "would never find it. Same situation as CONCHInferenceProvider; would need a "
            "dedicated hook (and attn_implementation='eager' at load_model()) to support this."
        )

    # ── standard embedding inference ──────────────────────────────────────────

    def inference(
        self,
        dataset: PatchDataset,
        output_path: PathLike,
        batch_size: int = 16,
        visualize_attention: bool = False,
        attention_output_path: PathLike | None = None,
        n_attention_samples: int = 10,
        save_cls: bool = False,
        save_cell: bool = False,
        save_nucleus: bool = False,
    ) -> None:
        if visualize_attention:
            raise NotImplementedError("visualize_attention is not supported for Phikon-v2 -- see run_attention_only.")
        assert dataset.x_size == 224 and dataset.y_size == 224 and dataset.offset_x == 0 and dataset.offset_y == 0, \
            f"Phikon-v2 requires 224×224 patches with no offset, got {dataset.x_size}×{dataset.y_size} offset=({dataset.offset_x},{dataset.offset_y})"
        self.check_boundary_sources(dataset, save_cell, save_nucleus)
        self.check_mask_support(dataset)

        self.create_output_file(output_path, num_samples=len(dataset), embedding_dim=self.embedding_dim,
                                dataset_stats=self.compute_dataset_statistics(dataset), save_cls=save_cls,
                                save_cell=save_cell, save_nucleus=save_nucleus)
        dataset.transform = self.transforms
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        mask_handle = self.register_mask_token_hook(dataset.mask_grid_size) if getattr(dataset, 'mask', False) else None
        try:
            for batch_idx, (patches, labels, cell_ids) in enumerate(tqdm(dataloader, desc="Inference", total=len(dataloader))):
                patches = patches.to(self.device)
                with torch.inference_mode():
                    outputs = self.model(pixel_values=patches).last_hidden_state

                patch_tokens = outputs[:, self.tokens_to_remove:]
                tokens_to_save = []
                for key in self.patches_to_save:
                    x, y = self.patches_to_save[key]
                    tokens_to_save.append(patch_tokens[:, x * self.num_patches_per_side + y])
                cls_token = outputs[:, _CLS_TOKEN_IDX] if save_cls else None

                cell_token = nucleus_token = None
                if save_cell or save_nucleus:
                    S = self.num_patches_per_side
                    spatial = patch_tokens.reshape(patch_tokens.shape[0], S, S, patch_tokens.shape[-1])
                    indices = range(batch_idx * batch_size, batch_idx * batch_size + len(patches))
                    pooled = self.pool_boundary_tokens(dataset, indices, spatial, self.patch_size, save_cell, save_nucleus)
                    cell_token = pooled.get('cell')
                    nucleus_token = pooled.get('nucleus')

                self.save_embeddings(tokens_to_save, cell_ids, labels, output_path, start_idx=batch_idx * batch_size,
                                     cls_token=cls_token, cell_token=cell_token, nucleus_token=nucleus_token)
        finally:
            if mask_handle is not None:
                mask_handle.remove()

    # ── multicell patch inference ─────────────────────────────────────────────

    def inference_multicell(
        self,
        dataset: MultiCellPatchDataset,
        output_path: PathLike,
        batch_size: int = 4,
        save_cell: bool = False,
        save_nucleus: bool = False,
    ) -> None:
        assert dataset.x_size == 224 and dataset.y_size == 224, \
            f"Phikon-v2 requires 224×224 patches for multicell inference, got {dataset.x_size}×{dataset.y_size}"
        self.check_boundary_sources(dataset, save_cell, save_nucleus)

        self.create_output_file_multicell(
            output_path,
            num_cells=dataset.total_cells,
            embedding_dim=self.embedding_dim,
            dataset_stats=self.compute_dataset_statistics_multicell(dataset),
            save_cell=save_cell,
            save_nucleus=save_nucleus,
        )
        dataset.transform = self.transforms
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=multicell_collate_fn,
        )

        S = self.num_patches_per_side
        cell_write_idx = 0

        for batch_idx, (patches, batch_rel_xs, batch_rel_ys, batch_cell_ids, batch_cell_labels) in enumerate(tqdm(dataloader, desc="Multicell Inference")):
            patches = patches.to(self.device)
            with torch.inference_mode():
                outputs = self.model(pixel_values=patches).last_hidden_state

            spatial_tokens = outputs[:, self.tokens_to_remove:]  # (B, N_spatial, D)
            spatial = spatial_tokens.reshape(spatial_tokens.shape[0], S, S, spatial_tokens.shape[-1])

            patch_indices = range(batch_idx * batch_size, batch_idx * batch_size + len(patches))
            results = self.select_multicell_tokens(dataset, patch_indices, spatial, batch_rel_xs, batch_rel_ys,
                                                     self.patch_size, self.patch_size, save_cell, save_nucleus)
            for i, res in enumerate(results):
                cell_ids, cell_labels = batch_cell_ids[i], batch_cell_labels[i]
                self.save_embeddings_multicell(res['cell_tokens'], cell_ids, cell_labels, output_path, start_idx=cell_write_idx,
                                                cell_token=res['cell_boundary'], nucleus_token=res['nucleus_boundary'])
                cell_write_idx += len(cell_ids)
