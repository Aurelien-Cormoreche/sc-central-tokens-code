import h5py
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def load_data(embeddings_file_path, embeddings_datasets, wsi_to_process, mapping, consider_matching=True):
    embeddings, cell_labels, cell_ids,wsi_origin, mapping_idx_to_wsi, _matching = load_data_separate(embeddings_file_path,
                                                                                 embeddings_datasets,
                                                                                   wsi_to_process,
                                                                                   mapping,
                                                                                   consider_matching)
    cell_labels = np.concatenate(cell_labels)
    embeddings = np.concatenate(embeddings)
    wsi_origin = np.concatenate(wsi_origin)
    cell_ids = np.concatenate(cell_ids)
    unique_cell_labels = set(cell_labels)

    return embeddings, cell_labels, cell_ids,wsi_origin, unique_cell_labels, mapping_idx_to_wsi

def load_data_separate(embeddings_file_path, embeddings_datasets, wsi_to_process, mapping=None, consider_matching=True):
    'Function to load embeddings in a parallel way'
    mapping_idx_to_wsi = []
    args = [
        (i, wsi_file, embeddings_file_path, embeddings_datasets, mapping, consider_matching)
        for i, wsi_file in enumerate(wsi_to_process)
    ]
    with ThreadPoolExecutor(max_workers=9) as executor:
        results = list(executor.map(load_unique_wsi, args))
    cell_labels = []
    embeddings = []
    wsi_origin = []
    cell_ids = []
    matching_list = []
    for cmapped_labels, emb, cell_id, origins, wsi_file, matching in results:
        mapping_idx_to_wsi.append(wsi_file)
        cell_labels.append(cmapped_labels)
        cell_ids.append(cell_id)
        embeddings.append(emb)
        wsi_origin.append(origins)
        matching_list.append(matching)

    return embeddings, cell_labels, cell_ids, wsi_origin, mapping_idx_to_wsi, matching_list

def load_unique_wsi(args):
    i, wsi_file, embeddings_file_path, embeddings_datasets, mapping, consider_matching = args
    with h5py.File(os.path.join(embeddings_file_path, wsi_file, 'embeddings_dataset.h5'), 'r') as f:

        if len(embeddings_datasets) == 0:
            embeddings_datasets = [key.split('embeddings_')[1] for key in f.keys() if key.startswith('embeddings_')]
        # Read the matching dataset (if present) regardless of consider_matching,
        # so it can be carried through even when it isn't used as a filter.
        matching = None
        if 'matching' in f:
            matching = f['matching'][()].astype(bool)
        matching_mask = matching if consider_matching else None
        cell_labels = f['cell_labels'][()]
        embeddings = np.mean(
            [f[f'embeddings_{key}'][()] for key in embeddings_datasets],
            axis=0
        )
        cell_ids = f['cell_ids'][()]
        if matching_mask is not None:
            cell_labels = cell_labels[matching_mask]
            embeddings = embeddings[matching_mask]
            cell_ids = cell_ids[matching_mask]
            matching = matching[matching_mask]
        if mapping is not None:
            cmapped_labels = np.array([
                mapping.get(label, 'Unknown')
                for label in cell_labels
            ])
        else:
            cmapped_labels = cell_labels

    return cmapped_labels, embeddings, cell_ids, np.full(len(cell_labels), i), wsi_file, matching


def load_unique_wsi_with_coords(args):
    """Like load_unique_wsi but also loads x_centroid and y_centroid."""
    i, wsi_file, embeddings_file_path, embeddings_datasets, mapping, consider_matching = args
    with h5py.File(os.path.join(embeddings_file_path, wsi_file, 'embeddings_dataset.h5'), 'r') as f:
        if len(embeddings_datasets) == 0:
            embeddings_datasets = [key.split('embeddings_')[1] for key in f.keys() if key.startswith('embeddings_')]
        matching_mask = None
        if consider_matching and 'matching' in f:
            matching_mask = f['matching'][()].astype(bool)
        cell_labels = f['cell_labels'][()]
        embeddings = np.mean(
            [f[f'embeddings_{key}'][()] for key in embeddings_datasets],
            axis=0,
        )
        x_centroid = f['x_centroid'][()]
        y_centroid = f['y_centroid'][()]
        if matching_mask is not None:
            cell_labels = cell_labels[matching_mask]
            embeddings = embeddings[matching_mask]
            x_centroid = x_centroid[matching_mask]
            y_centroid = y_centroid[matching_mask]
        cmapped_labels = np.array([mapping.get(label, 'Unknown') for label in cell_labels])
        cell_ids = f['cell_ids'][()]
        if matching_mask is not None:
            cell_ids = cell_ids[matching_mask]
    return cmapped_labels, embeddings, cell_ids, np.full(len(cell_labels), i), wsi_file, x_centroid, y_centroid


def load_data_with_coords(embeddings_file_path, embeddings_datasets, wsi_to_process, mapping, consider_matching=True):
    """Like load_data but also returns concatenated x_centroid and y_centroid arrays."""
    args = [
        (i, wsi_file, embeddings_file_path, embeddings_datasets, mapping, consider_matching)
        for i, wsi_file in enumerate(wsi_to_process)
    ]
    with ThreadPoolExecutor(max_workers=9) as executor:
        results = list(executor.map(load_unique_wsi_with_coords, args))

    mapping_idx_to_wsi = []
    cell_labels_list, embeddings_list, cell_ids_list = [], [], []
    wsi_origin_list, x_list, y_list = [], [], []

    for cmapped_labels, emb, cell_id, origins, wsi_file, x, y in results:
        mapping_idx_to_wsi.append(wsi_file)
        cell_labels_list.append(cmapped_labels)
        cell_ids_list.append(cell_id)
        embeddings_list.append(emb)
        wsi_origin_list.append(origins)
        x_list.append(x)
        y_list.append(y)

    cell_labels = np.concatenate(cell_labels_list)
    return (
        np.concatenate(embeddings_list),
        cell_labels,
        np.concatenate(cell_ids_list),
        np.concatenate(wsi_origin_list),
        set(cell_labels),
        mapping_idx_to_wsi,
        np.concatenate(x_list),
        np.concatenate(y_list),
    )
