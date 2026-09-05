import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import openslide as Openslide
from os import PathLike
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import affine_transform

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

    Two independent, opt-in features on top of the plain "one non-overlapping
    x_size × y_size tile per patch" behaviour (both default to off, i.e. the
    original behaviour):

    - size_side_x / size_side_y: crop each tile at this native WSI-pixel size and
      resize it down (or up) to x_size × y_size before it's handed to the model --
      same idea as ResizedCellDataset's `size_side` vs `size`. cell_rel_x/cell_rel_y
      (and boundary_polygon(), see below) are reported in the *output* x_size ×
      y_size frame, scaled accordingly, so downstream token-index math never needs
      to know a resize happened.
    - central_size_x / central_size_y: cells near a patch's edge are backed by a
      truncated field of view (border artifacts, less surrounding context) --
      restricting extraction to a centred central_size_x × central_size_y
      sub-window of the *output* patch avoids that. Internally this steps the
      tiling grid by the (native-space) central size instead of the full tile
      size, so tiles overlap but every cell in the WSI is still claimed by
      exactly one tile's central window -- it just also gets more surrounding
      context than the `central_size=None` default.

    __getitem__ returns:
        patch          – transformed (or raw PIL) image of size (x_size, y_size)
        cell_rel_x     – int32 array of cell-centre x positions relative to patch
                         origin, in the output (x_size × y_size) frame
        cell_rel_y     – int32 array of cell-centre y positions relative to patch
                         origin, in the output (x_size × y_size) frame
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
        size_side_x: int | None = None,
        size_side_y: int | None = None,
        central_size_x: int | None = None,
        central_size_y: int | None = None,
        cells_csv_path: PathLike | None = None,
        nucleus_boundaries_path: PathLike | None = None,
        alignment_matrix_path: PathLike | None = None,
        xenium_mpp: float = XENIUM_MPP,
    ):
        """
        cells_csv_path / nucleus_boundaries_path: optional Xenium cell_boundaries.csv.gz
            / nucleus_boundaries.parquet, enabling boundary_polygon(patch_idx, local_idx,
            'cell'/'nucleus') (used by InferenceProvider.pool_boundary_tokens_multicell for
            the embeddings_cell / embeddings_nucleus mean-pooled tokens). alignment_matrix_path
            is required when either is given.
        """
        self.wsi_path = wsi_path
        self.cells_info_path = cells_info_path
        self.x_size = x_size
        self.y_size = y_size
        self.transform = transform
        self.size_side_x = size_side_x if size_side_x is not None else x_size
        self.size_side_y = size_side_y if size_side_y is not None else y_size
        self.central_size_x = central_size_x if central_size_x is not None else x_size
        self.central_size_y = central_size_y if central_size_y is not None else y_size
        if not (0 < self.central_size_x <= x_size) or not (0 < self.central_size_y <= y_size):
            raise ValueError(
                f"central_size must lie in (0, x_size] / (0, y_size], got "
                f"({self.central_size_x}, {self.central_size_y}) for x_size={x_size}, y_size={y_size}"
            )

        print(f"Loading WSI from {wsi_path} and cell info from {cells_info_path}...")
        self.wsi = Openslide.OpenSlide(wsi_path)
        wsi_w, wsi_h = self.wsi.dimensions

        with h5py.File(cells_info_path, 'r') as f:
            x_starts = f['x_start'][:]
            y_starts = f['y_start'][:]
            all_cell_ids = f['cell_id'][:]
            all_labels = f['cell_type'][:]

        cell_cx = (x_starts + cell_offset_x).astype(np.int64)
        cell_cy = (y_starts + cell_offset_y).astype(np.int64)

        # scale_x/scale_y map native (pre-resize) pixels -> output (x_size × y_size)
        # pixels; self._scale_{x,y} is its inverse, applied below and in
        # boundary_polygon(). central_native_{x,y} is the central window's size back
        # in native pixels, so the tiling grid below (which reads/crops in native WSI
        # coordinates) can be stepped by it directly.
        self._scale_x = x_size / self.size_side_x
        self._scale_y = y_size / self.size_side_y
        central_native_x = max(1, round(self.central_size_x / self._scale_x))
        central_native_y = max(1, round(self.central_size_y / self._scale_y))
        margin_native_x = (self.size_side_x - central_native_x) // 2
        margin_native_y = (self.size_side_y - central_native_y) // 2

        # Partition the WSI into a regular grid of central_native_x × central_native_y
        # cores (last row/col truncated at the WSI edge) -- every cell belongs to
        # exactly one core by construction, however the surrounding read/crop tile
        # ends up placed (see patch_x0/patch_y0 clamping below).
        col_idx = cell_cx // central_native_x
        row_idx = cell_cy // central_native_y
        num_cols = wsi_w // central_native_x + 1
        key = row_idx * num_cols + col_idx
        order = np.argsort(key, kind='stable')
        sorted_key = key[order]
        split_points = np.flatnonzero(np.diff(sorted_key)) + 1
        groups = np.split(order, split_points) if len(order) else []

        self.patch_infos = []
        for group in groups:
            g_row = int(row_idx[group[0]])
            g_col = int(col_idx[group[0]])
            core_x0 = g_col * central_native_x
            core_y0 = g_row * central_native_y

            # The read tile is clamped (shifted, not padded) to stay within WSI
            # bounds -- OpenSlide's generic-tiff backend can't read past the edge --
            # but the core partition above never shifts, so this only moves the
            # *context* around a cell, never which tile claims it.
            patch_x0 = max(0, min(core_x0 - margin_native_x, wsi_w - self.size_side_x))
            patch_y0 = max(0, min(core_y0 - margin_native_y, wsi_h - self.size_side_y))

            rel_x_native = cell_cx[group] - patch_x0
            rel_y_native = cell_cy[group] - patch_y0
            rel_x = np.clip(np.round(rel_x_native * self._scale_x), 0, x_size - 1).astype(np.int32)
            rel_y = np.clip(np.round(rel_y_native * self._scale_y), 0, y_size - 1).astype(np.int32)

            self.patch_infos.append({
                'patch_x': int(patch_x0),
                'patch_y': int(patch_y0),
                'cell_rel_x': rel_x,
                'cell_rel_y': rel_y,
                'cell_ids': all_cell_ids[group],
                'cell_labels': all_labels[group],
            })

        self.total_cells = sum(len(p['cell_ids']) for p in self.patch_infos)
        restriction = (
            "" if central_size_x is None and central_size_y is None else
            f" (central {self.central_size_x}x{self.central_size_y} of {x_size}x{y_size})"
        )
        resize = "" if self.size_side_x == x_size and self.size_side_y == y_size else \
            f" (native crop {self.size_side_x}x{self.size_side_y}, resized to {x_size}x{y_size})"
        print(f"Found {len(self.patch_infos)} non-empty patches covering {self.total_cells} cells{restriction}{resize}.")

        # ── boundary polygons, for save_cell / save_nucleus (see PatchDataset) ──
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
        return len(self.patch_infos)

    def _read_patch(self, idx: int) -> Image.Image:
        p = self.patch_infos[idx]
        patch = self.wsi.read_region((p['patch_x'], p['patch_y']), 0, (self.size_side_x, self.size_side_y)).convert('RGB')
        if (self.size_side_x, self.size_side_y) != (self.x_size, self.y_size):
            patch = patch.resize((self.x_size, self.y_size), Image.BICUBIC)
        return patch

    def num_cells_in_patch(self, idx: int) -> int:
        return len(self.patch_infos[idx]['cell_ids'])

    def has_boundary_source(self, kind: str) -> bool:
        """kind: 'cell' or 'nucleus'. Whether this dataset was built with the
        corresponding boundary file (cells_csv_path / nucleus_boundaries_path)."""
        store = self._cell_boundaries if kind == 'cell' else self._nucleus_boundaries
        return store is not None

    def boundary_polygon(self, patch_idx: int, local_idx: int, kind: str) -> Polygon | MultiPolygon | None:
        """kind: 'cell' or 'nucleus'. The Xenium boundary polygon for the local_idx-th
        cell of patch `patch_idx` (i.e. cell_ids[local_idx] / cell_rel_{x,y}[local_idx]
        for that patch), mapped into that patch's output (x_size × y_size, already
        accounting for any size_side resize) pixel frame. None if that boundary source
        wasn't configured, or no polygon was found for this cell."""
        store = self._cell_boundaries if kind == 'cell' else self._nucleus_boundaries
        if store is None:
            return None
        p = self.patch_infos[patch_idx]
        polygon = store.polygon_for(p['cell_ids'][local_idx])
        if polygon is None:
            return None
        return affine_transform(polygon, [
            self._scale_x, 0, 0, self._scale_y,
            -p['patch_x'] * self._scale_x, -p['patch_y'] * self._scale_y,
        ])

    def __getitem__(self, idx):
        p = self.patch_infos[idx]
        patch = self._read_patch(idx)
        if self.transform:
            patch = self.transform(patch)
        return patch, p['cell_rel_x'], p['cell_rel_y'], p['cell_ids'], p['cell_labels']

    def get_raw_patch(self, idx):
        """Un-transformed, output-size (x_size × y_size) RGB PIL Image for a patch index."""
        return self._read_patch(idx)


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