import random

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import transforms
from .inference_provider import InferenceProvider
from ..data.patch_dataset import PatchDataset, MultiCellPatchDataset
from os import PathLike
from tqdm import tqdm


class DummyInferenceProvider(InferenceProvider):
    """Baseline provider: the "embedding" is just the flattened patch pixels.

    If use_pca is True, a PCA is fit on randomly sampled cells from the given
    datasets (via fit_pca) before inference, and embeddings are the PCA
    projection of the flattened pixels instead of the raw pixels.
    """

    def __init__(self, patches_to_save: dict[str, tuple[int, int]] = {'pixels': (0, 0)},
                 use_pca: bool = False, n_components: int = 50, n_pca_samples: int = 1000):
        super().__init__(patches_to_save)
        self.use_pca = use_pca
        self.n_components = n_components
        self.n_pca_samples = n_pca_samples
        self.pca = None

    def load_model(self):
        # No learned model — patches are (optionally PCA-reduced) flattened pixels.
        self.transforms = transforms.ToTensor()
        self.device = torch.device('cpu')

    def _flatten_batch(self, patches: torch.Tensor) -> np.ndarray:
        return patches.reshape(patches.shape[0], -1).cpu().numpy()

    def fit_pca(self, datasets: list[PatchDataset]) -> None:
        """Sample n_pca_samples random cells from each dataset and fit a PCA on flattened pixels."""
        if not self.use_pca:
            return

        pixels = []
        for dataset in datasets:
            dataset.transform = self.transforms
            n = min(self.n_pca_samples, len(dataset))
            for idx in random.sample(range(len(dataset)), n):
                patch, _, _ = dataset[idx]
                pixels.append(patch.reshape(-1).cpu().numpy())

        pixels = np.stack(pixels)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(pixels)

    def run_attention_only(self, dataset: PatchDataset, output_path: PathLike, n_samples: int = 10) -> None:
        raise NotImplementedError("DummyInferenceProvider has no attention to visualize.")

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
            print("[DummyInferenceProvider] WARNING: no CLS token available — save_cls ignored.")
        if save_cell or save_nucleus:
            print("[DummyInferenceProvider] WARNING: no spatial token grid available — save_cell/save_nucleus ignored.")

        if self.use_pca and self.pca is None:
            raise RuntimeError("use_pca=True but fit_pca(datasets) was never called.")

        embedding_dim = self.n_components if self.use_pca else 3 * dataset.x_size * dataset.y_size
        self.create_output_file(output_path, num_samples=len(dataset), embedding_dim=embedding_dim,
                                 dataset_stats=self.compute_dataset_statistics(dataset))
        dataset.transform = self.transforms
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        for batch_idx, (patches, labels, cell_ids) in enumerate(tqdm(dataloader, desc="Inference", total=len(dataloader))):
            flat = self._flatten_batch(patches)
            if self.use_pca:
                flat = self.pca.transform(flat)
            embedding = torch.from_numpy(flat).float()
            self.save_embeddings([embedding], cell_ids, labels, output_path, start_idx=batch_idx * batch_size)

    def inference_multicell(
        self,
        dataset: MultiCellPatchDataset,
        output_path: PathLike,
        batch_size: int = 4,
        save_cell: bool = False,
        save_nucleus: bool = False,
    ) -> None:
        raise NotImplementedError("DummyInferenceProvider does not support multicell inference.")
