import torch
from torch.utils.data import DataLoader
from .inference_provider import InferenceProvider
from ..data.patch_dataset import PatchDataset, MultiCellPatchDataset, multicell_collate_fn
from os import PathLike
from tqdm import tqdm
import os

# CONCH vision backbone: ViT-B/16 @ 448×448
# Token layout after trunk: CLS(0) + 28*28 spatial(1-784)
# Trunk output embedding dim: 768
_CLS_PREFIX = 1
_SPATIAL_GRID_SIZE = 28   # 448 / 16 = 28
_EMBED_DIM = 768
_MODEL_IMAGE_SIZE = 448


class CONCHInferenceProvider(InferenceProvider):

    def __init__(self, patches_to_save: dict[str, tuple[int, int]] =
                 {'top_left': (13, 13), 'top_right': (13, 14), 'bottom_left': (14, 13), 'bottom_right': (14, 14)}):
        super().__init__(patches_to_save)
        self.embedding_dim = _EMBED_DIM
        self.spatial_grid = _SPATIAL_GRID_SIZE

    def load_model(self):
        from conch.open_clip_custom import create_model_from_pretrained

        self.model, self.transforms = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=os.environ['HF_TOKEN'])

        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def run_attention_only(self, dataset: PatchDataset, output_path: PathLike, n_samples: int = 10) -> None:
        raise NotImplementedError(
            "Attention visualisation is not supported for CONCH: the trunk uses a custom ViT "
            "with attentional pooling that does not expose standard attn_drop hooks."
        )

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
        assert dataset.x_size == 448 and dataset.y_size == 448 and dataset.offset_x == -224 and dataset.offset_y == -224, \
            f"CONCH requires 448×448 patches with offset (-224, -224), got {dataset.x_size}×{dataset.y_size} offset=({dataset.offset_x},{dataset.offset_y})"
        self.check_boundary_sources(dataset, save_cell, save_nucleus)
        self.create_output_file(
            output_path,
            num_samples=len(dataset),
            embedding_dim=self.embedding_dim,
            dataset_stats=self.compute_dataset_statistics(dataset),
            save_cls=save_cls,
            save_cell=save_cell,
            save_nucleus=save_nucleus,
        )
        dataset.transform = self.transforms
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        token_size = dataset.x_size // self.spatial_grid
        for batch_idx, (patches, labels, cell_ids) in enumerate(tqdm(dataloader, desc="Inference", total=len(dataloader))):
            patches = patches.to(self.device)
            with torch.inference_mode():
                # trunk returns (B, 1+784, 768): CLS at 0, spatial tokens at 1:
                features = self.model.visual.trunk(patches)

            cls_token = features[:, 0] if save_cls else None
            spatial = features[:, _CLS_PREFIX:].reshape(
                features.shape[0], self.spatial_grid, self.spatial_grid, self.embedding_dim
            )

            tokens_to_save = [spatial[:, x, y, :] for x, y in self.patches_to_save.values()]

            cell_token = nucleus_token = None
            if save_cell or save_nucleus:
                indices = range(batch_idx * batch_size, batch_idx * batch_size + len(patches))
                pooled = self.pool_boundary_tokens(dataset, indices, spatial, token_size, save_cell, save_nucleus)
                cell_token = pooled.get('cell')
                nucleus_token = pooled.get('nucleus')

            self.save_embeddings(tokens_to_save, cell_ids, labels, output_path, start_idx=batch_idx * batch_size,
                                 cls_token=cls_token, cell_token=cell_token, nucleus_token=nucleus_token)

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
        dataset.transform = self.transforms
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
            with torch.inference_mode():
                features = self.model.visual.trunk(patches)  # (B, 785, 768)

            B = features.shape[0]
            spatial = features[:, _CLS_PREFIX:].reshape(B, self.spatial_grid, self.spatial_grid, self.embedding_dim)

            for i in range(B):
                rel_xs      = batch_rel_xs[i]
                rel_ys      = batch_rel_ys[i]
                cell_ids    = batch_cell_ids[i]
                cell_labels = batch_cell_labels[i]

                # Map original-space coords → token indices, accounting for any resize
                # (the transform resizes to _MODEL_IMAGE_SIZE, so we scale accordingly)
                token_cols = torch.clamp((rel_xs * self.spatial_grid) // dataset.x_size, 0, self.spatial_grid - 1)
                token_rows = torch.clamp((rel_ys * self.spatial_grid) // dataset.y_size, 0, self.spatial_grid - 1)

                cell_tokens = spatial[i, token_rows, token_cols, :]  # (N_cells, 768)
                self.save_embeddings_multicell(cell_tokens, cell_ids, cell_labels, output_path, start_idx=cell_write_idx)
                cell_write_idx += len(cell_ids)
