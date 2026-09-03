import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import openslide as Openslide
from os import PathLike
from PIL import Image
from shapely.geometry import Polygon

from .xenium_boundaries import BoundaryPolygons, load_alignment_matrix_inv, XENIUM_MPP


class PatchDataset(Dataset):
    def __init__(self, wsi_path: PathLike, cells_info_path: PathLike, x_size: int = 224, y_size: int = 224, transform: callable = None,
                 offset_x: int = 0, offset_y: int = 0,
                 cells_csv_path: PathLike | None = None,
                 nucleus_boundaries_path: PathLike | None = None,
                 alignment_matrix_path: PathLike | None = None,
                 xenium_mpp: float = XENIUM_MPP):
            """
            cells_csv_path / nucleus_boundaries_path: optional Xenium cell_boundaries.csv.gz
                / nucleus_boundaries.parquet, enabling boundary_polygon(idx, 'cell'/'nucleus')
                (used by InferenceProvider.pool_boundary_tokens for the embeddings_cell /
                embeddings_nucleus mean-pooled tokens). alignment_matrix_path is required
                when either is given.
            """
            self.wsi_path = wsi_path
            self.cells_info_path = cells_info_path
            self.x_size = x_size
            self.y_size = y_size
            self.transform = transform
            self.offset_x = offset_x
            self.offset_y = offset_y
            print(f"Loading WSI from {wsi_path} and cell info from {cells_info_path}...")
            self.wsi = Openslide.OpenSlide(self.wsi_path)
            self.wsi_w, self.wsi_h = self.wsi.dimensions
            with h5py.File(cells_info_path, 'r') as f:
                self.min_x_dataset = f['x_start'][:]
                self.min_y_dataset = f['y_start'][:]
                self.cell_ids_dataset = f['cell_id'][:]
                self.labels_dataset = f['cell_type'][:]

            self._cell_boundaries: BoundaryPolygons | None = None
            self._nucleus_boundaries: BoundaryPolygons | None = None
            if cells_csv_path is not None or nucleus_boundaries_path is not None:
                if alignment_matrix_path is None:
                    raise ValueError(
                        "alignment_matrix_path is required when cells_csv_path or "
                        "nucleus_boundaries_path is set"
                    )
                M_inv = load_alignment_matrix_inv(alignment_matrix_path)
                if cells_csv_path is not None:
                    self._cell_boundaries = BoundaryPolygons(cells_csv_path, M_inv, xenium_mpp)
                if nucleus_boundaries_path is not None:
                    self._nucleus_boundaries = BoundaryPolygons(nucleus_boundaries_path, M_inv, xenium_mpp)

    def __len__(self):
        return len(self.labels_dataset)

    def has_boundary_source(self, kind: str) -> bool:
        """kind: 'cell' or 'nucleus'. Whether this dataset was built with the
        corresponding boundary file (cells_csv_path / nucleus_boundaries_path)."""
        store = self._cell_boundaries if kind == 'cell' else self._nucleus_boundaries
        return store is not None

    def boundary_polygon(self, idx: int, kind: str) -> Polygon | None:
        """kind: 'cell' or 'nucleus'. The Xenium boundary polygon for sample `idx`,
        in absolute WSI pixel coordinates (same frame as origin(idx)). None if that
        boundary source wasn't configured, or no polygon was found for this cell."""
        store = self._cell_boundaries if kind == 'cell' else self._nucleus_boundaries
        if store is None:
            return None
        return store.polygon_for(self.cell_ids_dataset[idx])

    def origin(self, idx: int) -> tuple[int, int]:
        """Absolute WSI-pixel (x, y) top-left corner of this sample's crop -- the
        same frame boundary_polygon() returns polygons in. Public alias for
        _clamped_origin, used by InferenceProvider.pool_boundary_tokens."""
        return self._clamped_origin(idx)

    def _clamped_origin(self, idx):
        """Corner of the crop, shifted (not padded) to stay within WSI bounds.

        Offsets (e.g. CellViT's -16 recentring) can push the crop past the
        WSI's top/left edge into negative coordinates, which OpenSlide's
        generic-tiff backend can't read (TIFFRGBAImageGet failed).
        """
        min_x = int(self.min_x_dataset[idx]) + self.offset_x
        min_y = int(self.min_y_dataset[idx]) + self.offset_y
        min_x = max(0, min(min_x, self.wsi_w - self.x_size))
        min_y = max(0, min(min_y, self.wsi_h - self.y_size))
        return min_x, min_y

    def __getitem__(self, idx):
        min_x, min_y = self._clamped_origin(idx)
        cell_id = self.cell_ids_dataset[idx]
        label = self.labels_dataset[idx]

        try:
            patch = self.wsi.read_region((min_x, min_y), 0, (self.x_size, self.y_size)).convert('RGB')
        except Exception as e:
            print(
                f"WARNING: failed to read region idx={idx} cell_id={cell_id} at ({min_x},{min_y}) "
                f"size=({self.x_size},{self.y_size}) from {self.wsi_path} "
                f"(wsi dims={self.wsi_w}x{self.wsi_h}): {e} -- using a black placeholder patch."
            )
            patch = Image.new('RGB', (self.x_size, self.y_size))

        if self.transform:
            patch = self.transform(patch)

        return patch, label, cell_id

    def get_raw_patch(self, idx):
        """Return the un-transformed RGB PIL Image for a given sample index."""
        min_x, min_y = self._clamped_origin(idx)
        return self.wsi.read_region((min_x, min_y), 0, (self.x_size, self.y_size)).convert('RGB')


class MultiCellPatchDataset(Dataset):
    """Tiles the WSI into large patches and groups all cells that fall inside each tile.

    x_start / y_start in the H5 file are the top-left corners of the original
    224×224 cell-centred patches, so the actual cell centre is offset by
    (cell_offset_x, cell_offset_y) — default 112 pixels each.

    __getitem__ returns:
        patch          – transformed (or raw PIL) image of size (x_size, y_size)
        cell_rel_x     – int32 array of cell-centre x positions relative to patch origin
        cell_rel_y     – int32 array of cell-centre y positions relative to patch origin
        cell_ids       – array of cell id strings
        cell_labels    – array of cell type strings
    """

    def __init__(
        self,
        wsi_path: PathLike,
        cells_info_path: PathLike,
        x_size: int = 224,
        y_size: int = 224,
        transform: callable = None,
        cell_offset_x: int = 112,
        cell_offset_y: int = 112,
    ):
        self.wsi_path = wsi_path
        self.cells_info_path = cells_info_path
        self.x_size = x_size
        self.y_size = y_size
        self.transform = transform

        print(f"Loading WSI from {wsi_path} and cell info from {cells_info_path}...")
        self.wsi = Openslide.OpenSlide(wsi_path)

        with h5py.File(cells_info_path, 'r') as f:
            x_starts = f['x_start'][:]
            y_starts = f['y_start'][:]
            all_cell_ids = f['cell_id'][:]
            all_labels = f['cell_type'][:]

        cell_cx = (x_starts + cell_offset_x).astype(np.int64)
        cell_cy = (y_starts + cell_offset_y).astype(np.int64)

        wsi_w, wsi_h = self.wsi.dimensions

        # The grid's last row/column can extend past the WSI edge, which
        # OpenSlide's generic-tiff backend can't read (TIFFRGBAImageGet
        # failed). Clamp each tile's origin to stay in-bounds; for those
        # clamped (edge) tiles this can overlap the previous tile, so cells
        # are assigned to the first tile that claims them (`assigned` mask)
        # to avoid duplicating any cell across two tiles.
        assigned = np.zeros(len(cell_cx), dtype=bool)
        self.patch_infos = []
        for py in range(0, wsi_h, y_size):
            py_clamped = max(0, min(py, wsi_h - y_size))
            for px in range(0, wsi_w, x_size):
                px_clamped = max(0, min(px, wsi_w - x_size))
                mask = (
                    (cell_cx >= px_clamped) & (cell_cx < px_clamped + x_size) &
                    (cell_cy >= py_clamped) & (cell_cy < py_clamped + y_size) &
                    ~assigned
                )
                if not np.any(mask):
                    continue
                assigned |= mask
                self.patch_infos.append({
                    'patch_x': int(px_clamped),
                    'patch_y': int(py_clamped),
                    'cell_rel_x': (cell_cx[mask] - px_clamped).astype(np.int32),
                    'cell_rel_y': (cell_cy[mask] - py_clamped).astype(np.int32),
                    'cell_ids': all_cell_ids[mask],
                    'cell_labels': all_labels[mask],
                })

        self.total_cells = sum(len(p['cell_ids']) for p in self.patch_infos)
        print(f"Found {len(self.patch_infos)} non-empty patches covering {self.total_cells} cells.")

    def __len__(self):
        return len(self.patch_infos)

    def __getitem__(self, idx):
        p = self.patch_infos[idx]
        patch = self.wsi.read_region((p['patch_x'], p['patch_y']), 0, (self.x_size, self.y_size)).convert('RGB')
        if self.transform:
            patch = self.transform(patch)
        return patch, p['cell_rel_x'], p['cell_rel_y'], p['cell_ids'], p['cell_labels']


def multicell_collate_fn(batch):
    """Collate patches into a tensor; keep per-patch cell arrays as lists (variable length)."""
    patches = torch.stack([item[0] for item in batch])
    return (
        patches,
        [item[1] for item in batch],  # cell_rel_x per patch
        [item[2] for item in batch],  # cell_rel_y per patch
        [item[3] for item in batch],  # cell_ids per patch
        [item[4] for item in batch],  # cell_labels per patch
    )