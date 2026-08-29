import torch
import h5py
from torch.utils.data import Dataset
from ..data.patch_dataset import PatchDataset, MultiCellPatchDataset
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
    def inference_multicell(self, dataset: MultiCellPatchDataset, output_path: str, batch_size: int) -> None:
        pass

    @abstractmethod
    def run_attention_only(self, dataset: Dataset, output_path: str, n_samples: int = 10) -> None:
        pass

    def save_embeddings(self, embeddings: list[torch.Tensor], cell_ids: torch.Tensor, cell_labels: torch.Tensor, output_path: str, start_idx: int = 0, cls_token: torch.Tensor | None = None):
        embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        n = embeddings[0].shape[0]
        with h5py.File(f'{output_path}/embeddings_dataset.h5', 'a') as f:
            for i, key in enumerate(self.patches_to_save):
                f[f'embeddings_{key}'][start_idx:start_idx + n] = embeddings[i]
            if cls_token is not None and 'embeddings_cls' in f:
                f['embeddings_cls'][start_idx:start_idx + n] = cls_token.cpu().numpy()
            if isinstance(cell_ids[0], bytes):
                cell_ids = [cell_id.decode('utf-8') for cell_id in cell_ids]
            if isinstance(cell_labels[0], bytes):
                cell_labels = [cell_label.decode('utf-8') for cell_label in cell_labels]
            f['cell_ids'][start_idx:start_idx + n] = cell_ids
            f['cell_labels'][start_idx:start_idx + n] = cell_labels

    def save_embeddings_multicell(self, embeddings: torch.Tensor, cell_ids, cell_labels, output_path: str, start_idx: int = 0):
        emb_np = embeddings.cpu().numpy()
        n = emb_np.shape[0]
        cell_ids_dec   = [cid.decode('utf-8')   if isinstance(cid,   bytes) else cid   for cid   in cell_ids]
        cell_labels_dec = [lbl.decode('utf-8')  if isinstance(lbl,   bytes) else lbl   for lbl   in cell_labels]
        with h5py.File(f'{output_path}/embeddings_dataset.h5', 'a') as f:
            f['embeddings_cell_token'][start_idx:start_idx + n] = emb_np
            f['cell_ids'][start_idx:start_idx + n]              = cell_ids_dec
            f['cell_labels'][start_idx:start_idx + n]           = cell_labels_dec

    def create_output_file(self, output_path: str, num_samples: int, embedding_dim: int, dataset_stats: dict = None, save_cls: bool = False):
        os.makedirs(output_path, exist_ok=True)

        if dataset_stats is not None:
            with open(f'{output_path}/dataset_stats.json', 'w+') as f:
                json.dump(dataset_stats, f)

        with h5py.File(f'{output_path}/embeddings_dataset.h5', 'w') as f:
            for key in self.patches_to_save:
                f.create_dataset(f'embeddings_{key}', shape=(num_samples, embedding_dim), dtype='float32')
            if save_cls:
                f.create_dataset('embeddings_cls', shape=(num_samples, embedding_dim), dtype='float32')
            f.create_dataset('cell_ids', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
            f.create_dataset('cell_labels', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))

    def create_output_file_multicell(self, output_path: str, num_cells: int, embedding_dim: int, dataset_stats: dict = None):
        os.makedirs(output_path, exist_ok=True)
        if dataset_stats is not None:
            with open(f'{output_path}/dataset_stats.json', 'w+') as f:
                json.dump(dataset_stats, f)
        with h5py.File(f'{output_path}/embeddings_dataset.h5', 'w') as f:
            f.create_dataset('embeddings_cell_token', shape=(num_cells, embedding_dim), dtype='float32')
            f.create_dataset('cell_ids',    shape=(num_cells,), dtype=h5py.string_dtype(encoding='utf-8'))
            f.create_dataset('cell_labels', shape=(num_cells,), dtype=h5py.string_dtype(encoding='utf-8'))

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