import torch
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.models.layers import to_2tuple
from .inference_provider import InferenceProvider
from ..data.patch_dataset import PatchDataset, MultiCellPatchDataset, multicell_collate_fn
from os import PathLike
from tqdm import tqdm

from src.python.code_configs import paths


class ConvStem(nn.Module):
    """ConvStem patch embedding from https://arxiv.org/abs/2106.14881, used by CTransPath."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

# swin_tiny spatial layout after 4 stages (3 patch-merging steps):
#   resolution: 224 → 56×56 → 28×28 → 14×14 → 7×7  (49 tokens)
#   embed_dim:  96  → 96    → 192   → 384   → 768
#   each token covers 32×32 pixels (patch_size=4 * 2^3 from patch merging)
_SPATIAL_GRID_SIZE = 7
_EMBED_DIM = 768
_EFFECTIVE_STRIDE = 32


class CTransPathInferenceProvider(InferenceProvider):

    def __init__(self, patches_to_save: dict[str, tuple[int, int]] = {'center': (3, 3)}):
        super().__init__(patches_to_save)
        self.embedding_dim = _EMBED_DIM
        self.spatial_grid = _SPATIAL_GRID_SIZE
        self.effective_stride = _EFFECTIVE_STRIDE

    def load_model(self):
        weights_path = paths.CTRANSPATH_WEIGHTS
        # Create model matching the checkpoint structure (with default head/pool)
        # then bypass them at inference time via forward_features().
        self.model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()
        td = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(td['model'], strict=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _spatial_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Swin trunk up to the norm layer, returning (B, 49, 768) spatial tokens.

        forward_features() in this timm version applies avgpool+flatten and returns (B, 768).
        We stop before that step to keep spatial structure.
        """
        x = self.model.patch_embed(x)
        if hasattr(self.model, 'pos_drop'):
            x = self.model.pos_drop(x)
        x = self.model.layers(x)        # (B, N, C), pre-norm
        x = self.model.norm(x) 
        return x

    def run_attention_only(self, dataset: PatchDataset, output_path: PathLike, n_samples: int = 10) -> None:
        raise NotImplementedError(
            "CTransPath is a Swin Transformer — window-based attention cannot be visualised "
            "with the global ViT-style attention maps used by the other providers."
        )

    def inference(
        self,
        dataset: PatchDataset,
        output_path: PathLike,
        batch_size: int = 32,
        visualize_attention: bool = False,
        attention_output_path: PathLike | None = None,
        n_attention_samples: int = 10,
        save_cls: bool = False,
        save_cell: bool = False,
        save_nucleus: bool = False,
    ) -> None:
        if save_cls:
            print("[CTransPathInferenceProvider] WARNING: CTransPath is a Swin Transformer with no CLS token — save_cls ignored.")
            save_cls = False

        if dataset.x_size != 224 or dataset.y_size != 224:
            raise ValueError(
                f"CTransPathInferenceProvider requires 224×224 patches, got {dataset.x_size}×{dataset.y_size}."
            )
        self.check_boundary_sources(dataset, save_cell, save_nucleus)

        self.create_output_file(
            output_path,
            num_samples=len(dataset),
            embedding_dim=self.embedding_dim,
            dataset_stats=self.compute_dataset_statistics(dataset),
            save_cell=save_cell,
            save_nucleus=save_nucleus,
        )
        dataset.transform = self.transforms
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        for batch_idx, (patches, labels, cell_ids) in enumerate(tqdm(dataloader, desc="Inference", total=len(dataloader))):
            patches = patches.to(self.device)
            with torch.inference_mode():
                features = self._spatial_forward(patches)

            B, N, C = features.shape
            spatial = features.reshape(B, self.spatial_grid, self.spatial_grid, C)

            tokens_to_save = [spatial[:, x, y, :] for x, y in self.patches_to_save.values()]

            cell_token = nucleus_token = None
            if save_cell or save_nucleus:
                indices = range(batch_idx * batch_size, batch_idx * batch_size + len(patches))
                pooled = self.pool_boundary_tokens(dataset, indices, spatial, self.effective_stride, save_cell, save_nucleus)
                cell_token = pooled.get('cell')
                nucleus_token = pooled.get('nucleus')

            self.save_embeddings(tokens_to_save, cell_ids, labels, output_path, start_idx=batch_idx * batch_size,
                                 cell_token=cell_token, nucleus_token=nucleus_token)

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
                features = self._spatial_forward(patches)

            B, N, C = features.shape
            spatial = features.reshape(B, self.spatial_grid, self.spatial_grid, C)

            for i in range(B):
                rel_xs      = batch_rel_xs[i]
                rel_ys      = batch_rel_ys[i]
                cell_ids    = batch_cell_ids[i]
                cell_labels = batch_cell_labels[i]

                token_cols = torch.clamp(rel_xs // self.effective_stride, 0, self.spatial_grid - 1)
                token_rows = torch.clamp(rel_ys // self.effective_stride, 0, self.spatial_grid - 1)

                cell_tokens = spatial[i, token_rows, token_cols, :]  # (N_cells, 768)
                self.save_embeddings_multicell(cell_tokens, cell_ids, cell_labels, output_path, start_idx=cell_write_idx)
                cell_write_idx += len(cell_ids)
