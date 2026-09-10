"""Loads the embedding-bearing cell population for one probing split -- adapted
from classifier_training/experiment.py's `_load_wsi_data`, but keeps each cell's
own cell_id / mapped cell_type (needed to join against boundary polygons and the
WSI population, and to report per-cell results) instead of encoding labels to an
int classifier target -- probing targets are continuous, there's no classifier
output space here.

Duplicated rather than imported from classifier_training on purpose --
grid_experiment.py already keeps its own near-identical `_load_wsi_data_with_coords`
independent of experiment.py's, which is this repo's established convention for
these per-pipeline loaders (keeps each pipeline's data loading independently
editable without coordinating a shared-helper signature change across all of them).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.python.utils.loading_functions import load_data_separate


def load_probe_population(
    wsi_proportions: dict[str, float],
    base_path: str,
    embeddings_datasets: list[str],
    mapping: dict,
    rng: np.random.Generator,
    wsi_cell_type_proportions: Optional[dict[str, dict[str, float]]] = None,
    consider_matching: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same WSI-proportion / per-WSI cell-type-proportion subsampling as
    classifier_training.experiment._load_wsi_data (see its docstring), but
    returns raw string cell_id/cell_type arrays instead of an int-encoded label
    tensor, and does NOT drop "Unknown" cells -- a cell's own type being
    ambiguous doesn't disqualify it as a probing target for e.g. area/
    eccentricity, or as a context-probe window centre.

    Returns (embeddings (N, D) float32, cell_ids (N,) str, cell_types_mapped (N,) str,
    wsi_names (N,) str).
    """
    wsi_names_list = list(wsi_proportions.keys())
    embeddings_parts, labels_parts, ids_parts, _wsi_origin_parts, wsi_idx_map, _matching_parts = (
        load_data_separate(base_path, embeddings_datasets, wsi_names_list, mapping, consider_matching)
    )

    emb_out, ids_out, types_out, wsi_out = [], [], [], []
    for wsi_idx, wsi_name in enumerate(wsi_idx_map):
        emb = embeddings_parts[wsi_idx]
        labels = labels_parts[wsi_idx]
        cell_ids = np.asarray(
            [c.decode('utf-8') if isinstance(c, bytes) else c for c in ids_parts[wsi_idx]]
        )

        n = len(labels)
        if n == 0:
            raise ValueError(f"WSI '{wsi_name}' contributed 0 cells -- base_path={base_path!r}")
        proportion = float(wsi_proportions[wsi_name])
        n_keep = max(1, int(n * proportion))
        selected = rng.choice(n, size=n_keep, replace=False)

        ct_props = wsi_cell_type_proportions.get(wsi_name) if wsi_cell_type_proportions else None
        if ct_props:
            wsi_labels = labels[selected]
            ct_keep: list[np.ndarray] = []
            for ct in np.unique(wsi_labels):
                ct_mask = np.where(wsi_labels == ct)[0]
                ct_prop = float(ct_props.get(ct, 1.0))
                # ct_prop == 0.0 means "exclude this cell type entirely" -- no min-1 floor,
                # same convention as classifier_training/experiment.py's _load_wsi_data.
                n_keep_ct = 0 if ct_prop <= 0 else max(1, int(len(ct_mask) * ct_prop))
                if n_keep_ct > 0:
                    ct_keep.append(selected[rng.choice(ct_mask, size=min(n_keep_ct, len(ct_mask)), replace=False)])
            selected = np.sort(np.concatenate(ct_keep)) if ct_keep else selected[:0]

        emb_out.append(emb[selected])
        ids_out.append(cell_ids[selected])
        types_out.append(labels[selected])
        wsi_out.append(np.full(len(selected), wsi_name))

    if not emb_out:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.array([], dtype=str),
            np.array([], dtype=str),
            np.array([], dtype=str),
        )
    return (
        np.concatenate(emb_out),
        np.concatenate(ids_out),
        np.concatenate(types_out),
        np.concatenate(wsi_out),
    )
