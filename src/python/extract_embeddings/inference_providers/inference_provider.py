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