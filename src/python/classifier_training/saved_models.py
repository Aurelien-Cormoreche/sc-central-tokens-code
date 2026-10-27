"""
Save / reload the val-selected classifier of one split, so it can be re-applied to
slides it was never evaluated on (e.g. _metadata_work/make_splits.py's E2/E3, which
reuse E1 models on other tissues/cohorts) without retraining.

Layout (repo-relative, gitignored -- see experiment.py's main for how each part is set):
    {save_model.root}/{experiment}/{train_cohorts}/{job_name}/
        model.pt         weights + architecture + train-fitted Preprocessor + class
                         names + model_info; loadable with torch.load(weights_only=True)
        model_info.yaml  the same model_info, human-readable (data/read-out, split WSIs,
                         best hparams, val/test metrics, git commit)

Usage:
    clf = load_classifier("saved_models/classification_linear/cross_cohort/...")
    probs = clf.predict_proba(raw_embeddings)   # same read-out/averaging as at train time
    labels = [clf.class_names[i] for i in probs.argmax(1)]
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import yaml

from src.python.classifier_training.trainer import MLP, Preprocessor, TrainConfig

FORMAT_VERSION = 1
_PREPROCESSING_FIELDS = ("pca_mean", "pca_components", "z_mean", "z_std")


def train_cohorts_label(train_cohorts: Optional[list[str]], split_label: str) -> str:
    """Folder name for a split's training cohorts (see experiment.py's main for where
    they come from), sorted and '+'-joined so declaration order doesn't matter. Falls
    back to split_label (split_<i> / same_wsi_split) when there are none."""
    if not train_cohorts:
        return split_label
    return "+".join(sorted(str(c) for c in train_cohorts))


def save_classifier(
    model_dir: Path,
    state_dict: dict[str, torch.Tensor],
    train_cfg: TrainConfig,
    preprocessor: Preprocessor,
    model_info: dict[str, Any],
) -> None:
    """Write model.pt + model_info.yaml. model_info must hold only builtin types
    (it is stored inside model.pt too, which torch.load(weights_only=True) requires)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = {
        name: (torch.from_numpy(np.asarray(getattr(preprocessor, name))) if getattr(preprocessor, name) is not None else None)
        for name in _PREPROCESSING_FIELDS
    }
    model_input_dim = (
        int(preprocessor.pca_components.shape[0])
        if preprocessor.pca_components is not None else int(train_cfg.input_dim)
    )
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "state_dict": state_dict,
            "architecture": {
                "embedding_dim": int(train_cfg.input_dim),
                "model_input_dim": model_input_dim,
                "hidden_dim": int(train_cfg.hidden_dim),
                "num_classes": int(train_cfg.num_classes),
                "dropout": float(train_cfg.dropout),
            },
            "class_names": list(train_cfg.class_names),
            "preprocessing": preprocessing,
            "model_info": model_info,
        },
        model_dir / "model.pt",
    )
    with open(model_dir / "model_info.yaml", "w") as f:
        yaml.dump(model_info, f, default_flow_style=False, sort_keys=False)


@dataclass
class LoadedClassifier:
    model: MLP
    preprocessor: Preprocessor
    class_names: list[str]
    model_info: dict[str, Any]
    device: str = "cpu"

    def predict_proba(self, embeddings: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        """Softmax probabilities, (N, len(class_names)), for raw (un-preprocessed)
        embeddings built with the same read-out as at train time (see
        model_info['embeddings_datasets'])."""
        x = self.preprocessor.transform(embeddings)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=self.device)
                out.append(torch.softmax(self.model(xb), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.class_names)), dtype=np.float32)

    def predict(self, embeddings: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        """Predicted class indices into class_names."""
        return self.predict_proba(embeddings, batch_size).argmax(axis=1)


def load_classifier(model_dir: Union[Path, str], device: str = "cpu") -> LoadedClassifier:
    ckpt = torch.load(Path(model_dir) / "model.pt", map_location=device, weights_only=True)
    if ckpt["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported saved-classifier format_version {ckpt['format_version']} in {model_dir}")
    arch = ckpt["architecture"]
    model = MLP(arch["model_input_dim"], arch["hidden_dim"], arch["num_classes"], arch["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    preprocessor = Preprocessor(**{
        name: (t.cpu().numpy() if t is not None else None) for name, t in ckpt["preprocessing"].items()
    })
    return LoadedClassifier(model, preprocessor, ckpt["class_names"], ckpt["model_info"], device)
