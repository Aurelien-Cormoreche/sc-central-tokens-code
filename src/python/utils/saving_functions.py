import h5py
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os

def save_embeddings(embeddings: np.ndarray, cell_ids: np.ndarray, cell_labels: np.ndarray, output_path: str, matching: np.ndarray = None):
    os.makedirs(output_path, exist_ok=True)

    with h5py.File(f'{output_path}/embeddings_dataset.h5', 'w') as f:
        f.create_dataset('embeddings_cell', data=embeddings)
        f.create_dataset('cell_ids', data=np.array(cell_ids, dtype=h5py.string_dtype(encoding='utf-8')))
        f.create_dataset('cell_labels', data=np.array(cell_labels, dtype=h5py.string_dtype(encoding='utf-8')))
        if matching is not None:
            f.create_dataset('matching', data=np.array(matching, dtype=bool))



def save_dataframe_embeddings(df: pd.DataFrame, output_path: str):
    groups = df.groupby("wsi_origin", sort=False).indices

    selected_data = [(wsi, df.iloc[indices]["embedding", "cell_id", "cell_label"]) for wsi, indices in groups.items()]

    with ThreadPoolExecutor(max_workers=9) as executor:
        executor.map(lambda x: save_embeddings(x[1]["embedding"].tolist(), x[1]["cell_id"].tolist(), x[1]["cell_label"].tolist(), f'{output_path}/{x[0]}'), selected_data)

def save_list_embeddings(embeddings_list: list[np.ndarray], cell_ids_list: list[np.ndarray], cell_labels_list: list[np.ndarray], matching_list: list[np.ndarray], mapping_idx_to_wsi: list[str], output_path: str):
    os.makedirs(output_path, exist_ok=True)
    with ThreadPoolExecutor(max_workers=9) as executor:
        executor.map(lambda x: save_embeddings(embeddings_list[x], cell_ids_list[x], cell_labels_list[x], f'{output_path}/{mapping_idx_to_wsi[x]}', matching_list[x]), range(len(embeddings_list)))