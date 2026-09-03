import torch
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from .inference_provider import InferenceProvider
from .attention_utils import AttentionCapture, save_cell_attention_maps, sample_vis_indices
from ..data.patch_dataset import PatchDataset, MultiCellPatchDataset, multicell_collate_fn
from os import PathLike
from tqdm import tqdm

# H-optimus-1: ViT-g/14 with 4 register tokens @ 224×224
# Token layout after forward_features: CLS(0) + 4 registers(1-4) + 16*16 spatial(5-260)
_CLS_TOKEN_IDX = 0
_PREFIX_TOKENS = 5    # CLS + 4 registers
_NUM_PATCHES_PER_SIDE = 16  # 224 / 14 = 16
_EMBED_DIM = 1536
_PATCH_SIZE = 14


class HOptimus1InferenceProvider(InferenceProvider):

    def __init__(self, patches_to_save: dict[str, tuple[int, int]] =
                 {'top_left': (7, 7), 'top_right': (7, 8), 'bottom_left': (8, 7), 'bottom_right': (8, 8)}):
        super().__init__(patches_to_save)
        self.num_patches_per_side = _NUM_PATCHES_PER_SIDE
        self.tokens_to_remove = _PREFIX_TOKENS
        self.embedding_dim = _EMBED_DIM
        self.patch_size = _PATCH_SIZE

    def load_model(self):
        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
            ),
        ])

    # ── shared attention helper ────────────────────────────────────────────────

    def _collect_attention_maps(self, capture: AttentionCapture, local_idx: int) -> dict[str, torch.Tensor]:
        maps = {}
        S = self.num_patches_per_side
        m = capture.get_attention_for_token(_CLS_TOKEN_IDX, _PREFIX_TOKENS, S)
        if m is not None:
            maps['cls'] = m[local_idx]
        for key, (x, y) in self.patches_to_save.items():
            tok_idx = _PREFIX_TOKENS + x * S + y
            m = capture.get_attention_for_token(tok_idx, _PREFIX_TOKENS, S)
            if m is not None:
                maps[key] = m[local_idx]
        return maps

    # ── attention-only mode ────────────────────────────────────────────────────

    def run_attention_only(self, dataset: PatchDataset, output_path: PathLike, n_samples: int = 10) -> None:
        assert dataset.x_size == 224 and dataset.y_size == 224, \
            f"H-optimus-1 requires 224×224 patches, got {dataset.x_size}×{dataset.y_size}"

        dataset.transform = self.transforms
        vis_indices = sorted(sample_vis_indices(dataset.labels_dataset, n_samples))
        subset = torch.utils.data.Subset(dataset, vis_indices)
        dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        capture = AttentionCapture()
        if not capture.register_on_attn_drop(self.model):
            raise RuntimeError("Could not find attn_drop in H-optimus-1 model.")

        print(f"Saving attention maps to: {output_path}")
        for subset_idx, (patches, labels, cell_ids) in enumerate(tqdm(dataloader, desc="Attention")):
            global_idx = vis_indices[subset_idx]
            patches = patches.to(self.device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                self.model.forward_features(patches)

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
        batch_size: int = 16,
        visualize_attention: bool = False,
        attention_output_path: PathLike | None = None,
        n_attention_samples: int = 10,
        save_cls: bool = False,
        save_cell: bool = False,
        save_nucleus: bool = False,
    ) -> None:
        assert dataset.x_size == 224 and dataset.y_size == 224 and dataset.offset_x == 0 and dataset.offset_y == 0, \
            f"H-optimus-1 requires 224×224 patches with no offset, got {dataset.x_size}×{dataset.y_size} offset=({dataset.offset_x},{dataset.offset_y})"
        self.check_boundary_sources(dataset, save_cell, save_nucleus)

        self.create_output_file(output_path, num_samples=len(dataset), embedding_dim=self.embedding_dim,
                                dataset_stats=self.compute_dataset_statistics(dataset), save_cls=save_cls,
                                save_cell=save_cell, save_nucleus=save_nucleus)
        dataset.transform = self.transforms
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        vis_indices: set[int] = set()
        capture = AttentionCapture()
        if visualize_attention:
            if attention_output_path is None:
                raise ValueError("attention_output_path must be set when visualize_attention=True")
            if not capture.register_on_attn_drop(self.model):
                print("[AttentionCapture] WARNING: attn_drop not found in H-optimus-1 — skipping visualisation.")
                visualize_attention = False
            else:
                vis_indices = sample_vis_indices(dataset.labels_dataset, n_attention_samples)

        for batch_idx, (patches, labels, cell_ids) in enumerate(tqdm(dataloader, desc="Inference", total=len(dataloader))):
            patches = patches.to(self.device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                # forward_features returns the full token sequence (B, 261, 1536);
                # model() would return only the pooled CLS vector (B, 1536).
                outputs = self.model.forward_features(patches)

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

            self.save_embeddings(tokens_to_save, cell_ids, labels, output_path,
                                 start_idx=batch_idx * batch_size, cls_token=cls_token,
                                 cell_token=cell_token, nucleus_token=nucleus_token)

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
        dataset.transform = self.transforms
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=multicell_collate_fn,
        )

        num_token_cols = dataset.x_size // self.patch_size
        cell_write_idx = 0

        for patches, batch_rel_xs, batch_rel_ys, batch_cell_ids, batch_cell_labels in tqdm(dataloader, desc="Multicell Inference"):
            patches = patches.to(self.device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model.forward_features(patches)

            spatial_tokens = outputs[:, self.tokens_to_remove:]  # (B, N_spatial, D)

            for i in range(len(patches)):
                rel_xs      = batch_rel_xs[i]
                rel_ys      = batch_rel_ys[i]
                cell_ids    = batch_cell_ids[i]
                cell_labels = batch_cell_labels[i]

                token_cols    = rel_xs // self.patch_size
                token_rows    = rel_ys // self.patch_size
                token_indices = token_rows * num_token_cols + token_cols

                cell_tokens = spatial_tokens[i, token_indices]  # (N_cells, D)
                self.save_embeddings_multicell(cell_tokens, cell_ids, cell_labels, output_path, start_idx=cell_write_idx)
                cell_write_idx += len(cell_ids)
