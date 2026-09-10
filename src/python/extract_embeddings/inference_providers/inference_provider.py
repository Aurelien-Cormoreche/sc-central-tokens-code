import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from ..data.patch_dataset import PatchDataset, MultiCellPatchDataset
from ..data.xenium_boundaries import token_overlap_mask, nearest_token
from abc import ABC, abstractmethod
import json
import os
class InferenceProvider(ABC):

    def __init__(self, patches_to_save: dict):
        self.patches_to_save = patches_to_save
        # Set by load_mask_token() in providers that support dataset.mask (DINOv2-family
        # models with a learned iBOT mask token -- UNI2, H-optimus-1, VirchowV2). Stays
        # None otherwise, in which case check_mask_support() rejects dataset.mask=True.
        self.mask_token: torch.Tensor | None = None

    @abstractmethod
    def load_model(self)-> torch.nn.Module:
        pass

    @abstractmethod
    def inference(self, dataset: Dataset, output_path: str) -> None:
        pass

    @abstractmethod
    def inference_multicell(self, dataset: MultiCellPatchDataset, output_path: str, batch_size: int,
                             save_cell: bool = False, save_nucleus: bool = False) -> None:
        pass

    @abstractmethod
    def run_attention_only(self, dataset: Dataset, output_path: str, n_samples: int = 10) -> None:
        pass

    def save_embeddings(self, embeddings: list[torch.Tensor], cell_ids: torch.Tensor, cell_labels: torch.Tensor, output_path: str, start_idx: int = 0,
                         cls_token: torch.Tensor | None = None, cell_token: torch.Tensor | None = None, nucleus_token: torch.Tensor | None = None):
        embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        n = embeddings[0].shape[0]
        with h5py.File(f'{output_path}/embeddings_dataset.h5', 'a') as f:
            for i, key in enumerate(self.patches_to_save):
                f[f'embeddings_{key}'][start_idx:start_idx + n] = embeddings[i]
            if cls_token is not None and 'embeddings_cls' in f:
                f['embeddings_cls'][start_idx:start_idx + n] = cls_token.cpu().numpy()
            if cell_token is not None and 'embeddings_cell' in f:
                f['embeddings_cell'][start_idx:start_idx + n] = cell_token.cpu().numpy()
            if nucleus_token is not None and 'embeddings_nucleus' in f:
                f['embeddings_nucleus'][start_idx:start_idx + n] = nucleus_token.cpu().numpy()
            if isinstance(cell_ids[0], bytes):
                cell_ids = [cell_id.decode('utf-8') for cell_id in cell_ids]
            if isinstance(cell_labels[0], bytes):
                cell_labels = [cell_label.decode('utf-8') for cell_label in cell_labels]
            f['cell_ids'][start_idx:start_idx + n] = cell_ids
            f['cell_labels'][start_idx:start_idx + n] = cell_labels

    def save_embeddings_multicell(self, embeddings: torch.Tensor, cell_ids, cell_labels, output_path: str, start_idx: int = 0,
                                   cell_token: torch.Tensor | None = None, nucleus_token: torch.Tensor | None = None):
        emb_np = embeddings.cpu().numpy()
        n = emb_np.shape[0]
        cell_ids_dec   = [cid.decode('utf-8')   if isinstance(cid,   bytes) else cid   for cid   in cell_ids]
        cell_labels_dec = [lbl.decode('utf-8')  if isinstance(lbl,   bytes) else lbl   for lbl   in cell_labels]
        with h5py.File(f'{output_path}/embeddings_dataset.h5', 'a') as f:
            f['embeddings_cell_token'][start_idx:start_idx + n] = emb_np
            if cell_token is not None and 'embeddings_cell' in f:
                f['embeddings_cell'][start_idx:start_idx + n] = cell_token.cpu().numpy()
            if nucleus_token is not None and 'embeddings_nucleus' in f:
                f['embeddings_nucleus'][start_idx:start_idx + n] = nucleus_token.cpu().numpy()
            f['cell_ids'][start_idx:start_idx + n]              = cell_ids_dec
            f['cell_labels'][start_idx:start_idx + n]           = cell_labels_dec

    def create_output_file(self, output_path: str, num_samples: int, embedding_dim: int, dataset_stats: dict = None,
                            save_cls: bool = False, save_cell: bool = False, save_nucleus: bool = False):
        os.makedirs(output_path, exist_ok=True)

        if dataset_stats is not None:
            with open(f'{output_path}/dataset_stats.json', 'w+') as f:
                json.dump(dataset_stats, f)

        with h5py.File(f'{output_path}/embeddings_dataset.h5', 'w') as f:
            for key in self.patches_to_save:
                f.create_dataset(f'embeddings_{key}', shape=(num_samples, embedding_dim), dtype='float32')
            if save_cls:
                f.create_dataset('embeddings_cls', shape=(num_samples, embedding_dim), dtype='float32')
            if save_cell:
                f.create_dataset('embeddings_cell', shape=(num_samples, embedding_dim), dtype='float32')
            if save_nucleus:
                f.create_dataset('embeddings_nucleus', shape=(num_samples, embedding_dim), dtype='float32')
            f.create_dataset('cell_ids', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
            f.create_dataset('cell_labels', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))

    def check_boundary_sources(self, dataset: PatchDataset | MultiCellPatchDataset, save_cell: bool, save_nucleus: bool) -> None:
        """Fail fast if save_cell/save_nucleus is requested but `dataset` wasn't built
        with the matching Xenium boundary file (see PatchDataset's cells_csv_path /
        nucleus_boundaries_path / alignment_matrix_path)."""
        if save_cell and not dataset.has_boundary_source('cell'):
            raise ValueError(
                "save_cell=True requires the dataset to be built with cells_csv_path "
                "and alignment_matrix_path (see PatchDataset)."
            )
        if save_nucleus and not dataset.has_boundary_source('nucleus'):
            raise ValueError(
                "save_nucleus=True requires the dataset to be built with nucleus_boundaries_path "
                "and alignment_matrix_path (see PatchDataset)."
            )

    def load_mask_token(self, hf_repo_id: str) -> torch.Tensor | None:
        """Fetch this model's own learned mask token -- the iBOT/DINOv2 embedding
        substituted for masked patches during pretraining -- straight from the raw
        Hugging Face Hub checkpoint, for use by register_mask_token_hook().

        Matches the key by exact name ("mask_token") or by suffix (any key ending
        in ".mask_token"), since layout differs by release pipeline: MahmoodLab's /
        Bioptimus's / Paige's own hf-hub conversions (UNI2-h, H-optimus-1, Virchow2)
        strip it entirely, while a checkpoint saved via HF `transformers` itself
        (e.g. Owkin/Phikon-v2, whose Dinov2Model nests it at "embeddings.mask_token")
        keeps it under a module-prefixed name.

        timm's hf-hub loader (used by e.g. UNI2/H-optimus-1/VirchowV2's load_model())
        silently drops this weight when converting a DINOv2-format checkpoint to its
        own VisionTransformer layout (see timm.models.vision_transformer._convert_dinov2,
        which does `state_dict.pop("mask_token", None)`), so self.model never has it --
        it has to be read directly from the checkpoint file instead. hf_hub_download
        reuses the local cache timm's own pretrained=True load already populated, so
        this doesn't trigger a second download.

        Deliberately avoids huggingface_hub.list_repo_files -- unlike hf_hub_download,
        it's a pure Hub-API call with no local-cache fallback, so it fails outright on
        an offline compute node (no outbound internet, or HF_HUB_OFFLINE=1) even though
        the checkpoint is already sitting in the local cache from timm's pretrained=True
        load two lines above this call in load_model(). Instead this tries each
        candidate filename directly via hf_hub_download(local_files_only=True) (cache
        only, never touches the network) and only falls back to a normal
        hf_hub_download (which may attempt a network refresh) if nothing is cached
        under that name yet.

        Returns a flat (embedding_dim,) CPU tensor, or None (with a printed warning)
        if the checkpoint has no 'mask_token' key or couldn't be fetched -- callers
        should treat that as "this provider doesn't support dataset.mask=True" (see
        check_mask_support).
        """
        from huggingface_hub import hf_hub_download

        def _resolve(filename: str) -> str | None:
            try:
                return hf_hub_download(hf_repo_id, filename, local_files_only=True)
            except Exception:
                pass
            try:
                return hf_hub_download(hf_repo_id, filename)
            except Exception:
                return None

        path = filename = None
        for candidate in ("model.safetensors", "pytorch_model.bin"):
            resolved = _resolve(candidate)
            if resolved is not None:
                path, filename = resolved, candidate
                break
        if path is None:
            print(f"WARNING: could not resolve model.safetensors/pytorch_model.bin for {hf_repo_id} "
                  f"from the local Hub cache or network -- dataset.mask=True will not be supported "
                  f"for this model.")
            return None

        def _find_mask_token_key(keys) -> str | None:
            for k in keys:
                if k == "mask_token" or k.endswith(".mask_token"):
                    return k
            return None

        try:
            if filename.endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as f:
                    key = _find_mask_token_key(f.keys())
                    if key is None:
                        print(f"WARNING: 'mask_token' not found in {hf_repo_id}/{filename} "
                              f"(keys sample: {list(f.keys())[:20]}) -- dataset.mask=True will not be supported.")
                        return None
                    tensor = f.get_tensor(key)
            else:
                try:
                    state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
                except TypeError:
                    state_dict = torch.load(path, map_location="cpu")
                key = _find_mask_token_key(state_dict.keys())
                if key is None:
                    print(f"WARNING: 'mask_token' not found in {hf_repo_id}/{filename} "
                          f"(keys sample: {list(state_dict.keys())[:20]}) -- dataset.mask=True will not be supported.")
                    return None
                tensor = state_dict[key]
            return tensor.reshape(-1).float()
        except Exception as e:
            print(f"WARNING: failed to read mask_token from {hf_repo_id}/{filename}: {e} -- "
                  f"dataset.mask=True will not be supported for this model.")
            return None

    def load_mask_token_or_zero(self, hf_repo_id: str) -> torch.Tensor:
        """load_mask_token(hf_repo_id), falling back to a zero vector -- the standard
        "no signal" proxy used in ViT/MAE-style masking studies when a model has no
        trained mask token of its own -- if the checkpoint doesn't have one.

        Confirmed (2026-09) that UNI2-h, H-optimus-1 and VirchowV2's published
        checkpoints have all been stripped of their mask_token weight before release
        (their state dicts are already in timm-native key format -- reg_token,
        cls_token, etc. -- with no mask_token at all), so for all three of them this
        fallback is not merely defensive: it's what dataset.mask=True actually uses
        today. self.mask_token stays honestly documented either way -- see
        register_mask_token_hook, which doesn't distinguish the two cases.
        """
        token = self.load_mask_token(hf_repo_id)
        if token is None:
            print(f"[{type(self).__name__}] No mask_token in {hf_repo_id}'s published checkpoint "
                  f"-- falling back to a zero vector for dataset.mask=True.")
            token = torch.zeros(self.embedding_dim)
        return token

    def check_mask_support(self, dataset) -> None:
        """Fail fast if dataset.mask=True but this provider has no mask_token (see
        load_mask_token) -- i.e. it isn't a DINOv2-family model with a learned iBOT
        mask token, or the checkpoint fetch failed."""
        if getattr(dataset, 'mask', False) and self.mask_token is None:
            raise ValueError(
                f"{type(self).__name__} does not support dataset.mask=True -- no mask_token "
                "was loaded for this model (see the load_mask_token warning printed during "
                "load_model())."
            )

    def _patch_embed_module(self) -> torch.nn.Module:
        """Submodule whose forward output is (B, N_spatial, D) -- the pure patch-token
        grid, row-major, with no prefix (CLS/register) tokens prepended yet. Defaults
        to timm's `self.model.patch_embed` convention (UNI2/H-optimus-1/Virchow2/
        CellViT); override for a model whose patch-embedding submodule lives
        elsewhere, e.g. HF `transformers`' Dinov2Model nests it at
        `self.model.embeddings.patch_embeddings` (see PhikonV2InferenceProvider)."""
        return self.model.patch_embed

    def register_mask_token_hook(self, grid_size: int) -> "torch.utils.hooks.RemovableHandle":
        """Register a forward hook on _patch_embed_module() that overwrites the
        centred grid_size x grid_size block of patch tokens (same token-boundary
        convention as data.central_mask.central_mask_box, e.g. grid_size=3 on a
        16x16 grid -> tokens {6,7,8}x{6,7,8}) with self.mask_token, applied in
        embedding space regardless of what the dataset drew into those pixels.
        self.mask_token is the model's own learned mask token when its published
        checkpoint has one, else a zero-vector fallback (see load_mask_token_or_zero) --
        as of 2026-09 that's a zero vector for UNI2/H-optimus-1/VirchowV2, none of
        which ship a mask_token weight.

        patch_embed's output is (B, N, D) in row-major (row * S + col) order, matching
        the token-indexing convention used by patches_to_save / pool_boundary_tokens
        elsewhere in this class. Caller must call handle.remove() when done (e.g. in a
        `finally` block) -- the hook stays registered on self.model otherwise.
        """
        if self.mask_token is None:
            raise ValueError(f"{type(self).__name__}.mask_token is not set -- call check_mask_support first.")
        S = self.num_patches_per_side
        start = (S - grid_size) // 2
        rows = torch.arange(start, start + grid_size)
        cols = torch.arange(start, start + grid_size)
        idx = (rows.unsqueeze(1) * S + cols.unsqueeze(0)).reshape(-1).to(self.device)
        mask_token = self.mask_token.to(self.device)

        def _hook(module, inputs, output):
            output = output.clone()
            output[:, idx, :] = mask_token.to(output.dtype)
            return output

        return self._patch_embed_module().register_forward_hook(_hook)

    def pool_boundary_tokens(
        self,
        dataset: PatchDataset,
        indices: range,
        spatial: torch.Tensor,
        token_size: float,
        save_cell: bool,
        save_nucleus: bool,
    ) -> dict[str, torch.Tensor]:
        """Mean-pool `spatial` (B, S, S, D) over the grid tokens whose pixel footprint
        overlaps each sample's Xenium cell/nucleus boundary polygon (see
        data.xenium_boundaries.token_overlap_mask). Falls back to the single
        nearest-to-centroid token when no footprint overlaps (e.g. the boundary pokes
        outside the fixed patch crop). `indices` are the dataset-global sample indices
        for this batch, in the same order as `spatial`'s batch dimension.

        Returns {'cell': (B, D)} / {'nucleus': (B, D)} for each kind requested via
        save_cell/save_nucleus; call check_boundary_sources first to ensure `dataset`
        actually has the corresponding boundary source configured.
        """
        grid_size = spatial.shape[1]
        kinds = [k for k, want in (('cell', save_cell), ('nucleus', save_nucleus)) if want]
        pooled: dict[str, list[torch.Tensor]] = {kind: [] for kind in kinds}
        for b, idx in enumerate(indices):
            x0, y0 = dataset.origin(idx)
            for kind in kinds:
                polygon = dataset.boundary_polygon(idx, kind)
                if polygon is None:
                    print(f"WARNING: no {kind} boundary polygon for sample idx={idx} -- using a zero vector.")
                    pooled[kind].append(torch.zeros(spatial.shape[-1], device=spatial.device, dtype=spatial.dtype))
                    continue
                mask = token_overlap_mask(polygon, x0, y0, token_size, grid_size)
                if not mask.any():
                    r, c = nearest_token(polygon, x0, y0, token_size, grid_size)
                    print(f"WARNING: no {kind} token overlap for sample idx={idx} -- falling back to nearest token (row={r}, col={c}).")
                    pooled[kind].append(spatial[b, r, c])
                else:
                    mask_t = torch.from_numpy(mask).to(spatial.device)
                    pooled[kind].append(spatial[b][mask_t].mean(dim=0))
        return {kind: torch.stack(vecs, dim=0) for kind, vecs in pooled.items()}

    def create_output_file_multicell(self, output_path: str, num_cells: int, embedding_dim: int, dataset_stats: dict = None,
                                      save_cell: bool = False, save_nucleus: bool = False):
        os.makedirs(output_path, exist_ok=True)
        if dataset_stats is not None:
            with open(f'{output_path}/dataset_stats.json', 'w+') as f:
                json.dump(dataset_stats, f)
        with h5py.File(f'{output_path}/embeddings_dataset.h5', 'w') as f:
            f.create_dataset('embeddings_cell_token', shape=(num_cells, embedding_dim), dtype='float32')
            if save_cell:
                f.create_dataset('embeddings_cell', shape=(num_cells, embedding_dim), dtype='float32')
            if save_nucleus:
                f.create_dataset('embeddings_nucleus', shape=(num_cells, embedding_dim), dtype='float32')
            f.create_dataset('cell_ids',    shape=(num_cells,), dtype=h5py.string_dtype(encoding='utf-8'))
            f.create_dataset('cell_labels', shape=(num_cells,), dtype=h5py.string_dtype(encoding='utf-8'))

    def pool_boundary_tokens_multicell(
        self,
        dataset: MultiCellPatchDataset,
        patch_idx: int,
        spatial: torch.Tensor,
        token_size: float,
        save_cell: bool,
        save_nucleus: bool,
    ) -> dict[str, torch.Tensor]:
        """Like pool_boundary_tokens, but for one MultiCellPatchDataset patch holding
        several cells: `spatial` is that single patch's (S, S, D) token grid, and for
        each cell assigned to `patch_idx` this mean-pools it over the grid tokens whose
        pixel footprint overlaps that cell's Xenium cell/nucleus boundary polygon
        (already mapped into this patch's local x_size × y_size frame -- see
        MultiCellPatchDataset.boundary_polygon). Falls back to the single
        nearest-to-centroid token when no footprint overlaps.

        Returns {'cell': (N_cells, D)} / {'nucleus': (N_cells, D)} for each kind
        requested via save_cell/save_nucleus, in the same cell order as
        dataset.patch_infos[patch_idx]; call check_boundary_sources first to ensure
        `dataset` actually has the corresponding boundary source configured.
        """
        grid_size = spatial.shape[0]
        n_cells = dataset.num_cells_in_patch(patch_idx)
        kinds = [k for k, want in (('cell', save_cell), ('nucleus', save_nucleus)) if want]
        pooled: dict[str, list[torch.Tensor]] = {kind: [] for kind in kinds}
        for local_idx in range(n_cells):
            for kind in kinds:
                polygon = dataset.boundary_polygon(patch_idx, local_idx, kind)
                if polygon is None:
                    print(f"WARNING: no {kind} boundary polygon for patch_idx={patch_idx} local_idx={local_idx} -- using a zero vector.")
                    pooled[kind].append(torch.zeros(spatial.shape[-1], device=spatial.device, dtype=spatial.dtype))
                    continue
                mask = token_overlap_mask(polygon, 0, 0, token_size, grid_size)
                if not mask.any():
                    r, c = nearest_token(polygon, 0, 0, token_size, grid_size)
                    print(f"WARNING: no {kind} token overlap for patch_idx={patch_idx} local_idx={local_idx} "
                          f"-- falling back to nearest token (row={r}, col={c}).")
                    pooled[kind].append(spatial[r, c])
                else:
                    mask_t = torch.from_numpy(mask).to(spatial.device)
                    pooled[kind].append(spatial[mask_t].mean(dim=0))
        return {kind: torch.stack(vecs, dim=0) for kind, vecs in pooled.items()}

    def select_multicell_tokens(
        self,
        dataset: MultiCellPatchDataset,
        patch_indices: range,
        spatial: torch.Tensor,
        batch_rel_xs: list,
        batch_rel_ys: list,
        token_size_x: float,
        token_size_y: float,
        save_cell: bool,
        save_nucleus: bool,
    ) -> list[dict[str, torch.Tensor | None]]:
        """Per-patch cell-token selection shared by every provider's inference_multicell.

        `spatial` is the whole batch's (B, S, S, D) token grid; `patch_indices` are the
        dataset-global patch indices for this batch, in the same order as `spatial`'s
        batch dimension (and as batch_rel_xs/batch_rel_ys, the per-patch cell-centre
        arrays from multicell_collate_fn).

        For each patch, picks the grid token at each cell's (rel_x, rel_y) position
        (-> 'cell_tokens', what embeddings_cell_token is built from) and, if requested,
        mean-pools the tokens overlapping each cell's Xenium boundary polygon via
        pool_boundary_tokens_multicell (-> 'cell_boundary' / 'nucleus_boundary', for
        embeddings_cell / embeddings_nucleus).

        Returns one {'cell_tokens', 'cell_boundary', 'nucleus_boundary'} dict per patch,
        in batch order.
        """
        grid_h, grid_w = spatial.shape[1], spatial.shape[2]
        results = []
        for i, patch_idx in enumerate(patch_indices):
            token_cols = np.clip(np.asarray(batch_rel_xs[i]) // token_size_x, 0, grid_w - 1).astype(np.int64)
            token_rows = np.clip(np.asarray(batch_rel_ys[i]) // token_size_y, 0, grid_h - 1).astype(np.int64)
            cell_tokens = spatial[i, token_rows, token_cols, :]

            cell_boundary = nucleus_boundary = None
            if save_cell or save_nucleus:
                pooled = self.pool_boundary_tokens_multicell(dataset, patch_idx, spatial[i], token_size_x, save_cell, save_nucleus)
                cell_boundary = pooled.get('cell')
                nucleus_boundary = pooled.get('nucleus')

            results.append({'cell_tokens': cell_tokens, 'cell_boundary': cell_boundary, 'nucleus_boundary': nucleus_boundary})
        return results

    def compute_dataset_statistics(self, dataset: PatchDataset):
        stats = {
            'num_samples': len(dataset),
            'x_size': dataset.x_size,
            'y_size': dataset.y_size,
            'num_classes': len(set(dataset.labels_dataset)),
            'class_distribution': {label.decode('utf-8') if isinstance(label, bytes) else label: int((dataset.labels_dataset == label).sum()) for label in set(dataset.labels_dataset)}
        }
        return stats

    def compute_dataset_statistics_multicell(self, dataset: MultiCellPatchDataset):
        all_labels = [lbl for p in dataset.patch_infos for lbl in p['cell_labels']]
        all_labels_str = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in all_labels]
        unique = set(all_labels_str)
        return {
            'num_patches': len(dataset),
            'num_cells': dataset.total_cells,
            'x_size': dataset.x_size,
            'y_size': dataset.y_size,
            'num_classes': len(unique),
            'class_distribution': {lbl: all_labels_str.count(lbl) for lbl in unique},
        }