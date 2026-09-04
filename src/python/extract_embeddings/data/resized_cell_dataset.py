import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import openslide as Openslide
from os import PathLike
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import affine_transform

from .xenium_boundaries import BoundaryPolygons, load_alignment_matrix_inv

# Xenium instrument pixel size (µm/px) — used as the homogeneous scale factor
# in the alignment transform.  Matches XENIUM_MPP in the alignment notebook.
_XENIUM_MPP = 0.2125


class ResizedCellDataset(Dataset):
    """Crops a square box per cell from a WSI and resizes it to a fixed size.

    Cell ordering, ids, and cell-type labels come from an H5 file with the same
    layout as PatchDataset's `cells_info_path` (datasets: x_start, y_start,
    cell_id, cell_type) — x_start/y_start are the top-left corner of the cell's
    original cell_offset_x/cell_offset_y-centred H&E patch.

    Two crop modes, selected by size_side:

    - size_side=None (default): the crop is the cell's own bounding box. The
      cell's polygon vertices are looked up by cell_id in a separate CSV
      (columns: cell_id, vertex_x, vertex_y — Xenium physical coordinates in
      µm) and mapped to H&E pixel space via a 3×3 alignment matrix, and the
      square bounding box of that cell's vertices (longest axis, centred on
      the polygon centroid) is used as the crop.
    - size_side=<int>: every cell instead uses a fixed size_side × size_side
      crop, centred on (x_start + cell_offset_x, y_start + cell_offset_y) —
      i.e. the cell centre implied by the H5 patch coordinates, ignoring the
      polygon entirely. cells_csv_path / alignment_matrix_path are not needed
      in this mode.

    Either way the crop is clamped to the WSI boundary and resized to
    (size × size).

    cells_csv_path / nucleus_boundaries_path: optionally also enable
    boundary_polygon(idx, 'cell'/'nucleus') (used by
    InferenceProvider.pool_boundary_tokens for the embeddings_cell /
    embeddings_nucleus mean-pooled tokens), the same feature PatchDataset
    offers. alignment_matrix_path is required whenever either is given (as well
    as whenever size_side is None, for the vertex-based bounding box above).
    Since the crop is resized from its native `side` down to `size`,
    boundary_polygon() returns each polygon already mapped into the resized
    (size × size) frame -- i.e. scaled by size/side -- and origin() always
    returns (0, 0) to match; pool_boundary_tokens' token grid math (which lays
    a token_size × token_size grid over [0, size) using the model's native
    patch pixel size as token_size) then lines up correctly without needing
    to know about the resize at all.

    Returns:
        patch    – PIL Image (or transformed tensor) of size (size × size)
        label    – cell type, as stored in the H5 file
        cell_id  – cell identifier, as stored in the H5 file
    """

    def __init__(
        self,
        wsi_path: PathLike,
        cells_info_path: PathLike,
        cells_csv_path: PathLike | None = None,
        nucleus_boundaries_path: PathLike | None = None,
        alignment_matrix_path: PathLike | None = None,
        size: int = 224,
        size_side: int | None = None,
        transform: callable = None,
        xenium_mpp: float = _XENIUM_MPP,
        cell_offset_x: int = 112,
        cell_offset_y: int = 112,
    ):
        self.size = size
        self.size_side = size_side
        self.transform = transform
        self.xenium_mpp = xenium_mpp
        self.cell_offset_x = cell_offset_x
        self.cell_offset_y = cell_offset_y
        # interface parity with PatchDataset, consumed by InferenceProvider
        self.x_size = size
        self.y_size = size
        self.offset_x = 0
        self.offset_y = 0

        # ── open WSI ──────────────────────────────────────────────────────────
        self.wsi = Openslide.OpenSlide(str(wsi_path))
        self.wsi_w, self.wsi_h = self.wsi.dimensions

        # ── load cell ids / labels / patch corners from the H5 file ────────────
        print(f"Loading WSI from {wsi_path} and cell info from {cells_info_path}...")
        with h5py.File(cells_info_path, 'r') as f:
            self.min_x_dataset = f['x_start'][:]
            self.min_y_dataset = f['y_start'][:]
            self.cell_ids_dataset = f['cell_id'][:]
            self.labels_dataset = f['cell_type'][:]

        # ── vertex-based bounding box (only needed when size_side is None) ─────
        self.cells_by_id = None
        self.M_inv = None
        needs_alignment = size_side is None or cells_csv_path is not None or nucleus_boundaries_path is not None
        if needs_alignment:
            if alignment_matrix_path is None:
                raise ValueError(
                    "alignment_matrix_path is required when size_side is None, or when "
                    "cells_csv_path / nucleus_boundaries_path is set"
                )
            self.M_inv = load_alignment_matrix_inv(alignment_matrix_path)

        if size_side is None:
            if cells_csv_path is None:
                raise ValueError("cells_csv_path is required when size_side is None")
            df = pd.read_csv(cells_csv_path)
            df['cell_id'] = df['cell_id'].astype(str)
            self.cells_by_id = df.groupby('cell_id')

        # ── boundary polygons, for save_cell / save_nucleus (see PatchDataset) ──
        self._cell_boundaries: BoundaryPolygons | None = None
        self._nucleus_boundaries: BoundaryPolygons | None = None
        if cells_csv_path is not None:
            self._cell_boundaries = BoundaryPolygons(cells_csv_path, self.M_inv, xenium_mpp)
        if nucleus_boundaries_path is not None:
            self._nucleus_boundaries = BoundaryPolygons(nucleus_boundaries_path, self.M_inv, xenium_mpp)

        print(f"Loaded {len(self.cell_ids_dataset)} cells from {cells_info_path}")

    def __len__(self) -> int:
        return len(self.cell_ids_dataset)

    def _box_from_vertices(self, cell_id_raw) -> tuple[int, int, int]:
        cell_id_str = cell_id_raw.decode('utf-8') if isinstance(cell_id_raw, bytes) else str(cell_id_raw)
        cell_id_str = cell_id_str.split('_')[-1]
        vertices = self.cells_by_id.get_group(cell_id_str)

        # Homogeneous Xenium coords: [xen_x, xen_y, xenium_mpp] per vertex
        xen_hom = np.column_stack([
            vertices['vertex_x'].to_numpy(dtype=float),
            vertices['vertex_y'].to_numpy(dtype=float),
            np.full(len(vertices), self.xenium_mpp),
        ])                                                # (N_vertices, 3)

        # Apply inverse alignment: result columns are [hne_px*ps, hne_py*ps, ps]
        pred_hom = (self.M_inv @ xen_hom.T).T              # (N_vertices, 3)
        pixel_x = pred_hom[:, 0] / self.xenium_mpp
        pixel_y = pred_hom[:, 1] / self.xenium_mpp

        min_px, max_px = pixel_x.min(), pixel_x.max()
        min_py, max_py = pixel_y.min(), pixel_y.max()
        cx = (min_px + max_px) / 2
        cy = (min_py + max_py) / 2
        side = max(max_px - min_px, max_py - min_py, 1.0)

        x0 = int(round(cx - side / 2))
        y0 = int(round(cy - side / 2))
        return x0, y0, int(round(side))

    def _fixed_box(self, min_x: int, min_y: int) -> tuple[int, int, int]:
        side = self.size_side
        cx = min_x + self.cell_offset_x
        cy = min_y + self.cell_offset_y
        return cx - side // 2, cy - side // 2, side

    def _crop_box(self, idx: int) -> tuple[int, int, int]:
        """Return the clamped (x0, y0, side) crop box for a given sample index."""
        cell_id = self.cell_ids_dataset[idx]
        min_x = int(self.min_x_dataset[idx])
        min_y = int(self.min_y_dataset[idx])

        if self.size_side is None:
            x0, y0, side = self._box_from_vertices(cell_id)
        else:
            x0, y0, side = self._fixed_box(min_x, min_y)

        # Clamp so the box stays within WSI bounds
        x0 = max(0, min(x0, self.wsi_w - side))
        y0 = max(0, min(y0, self.wsi_h - side))
        return x0, y0, side

    def has_boundary_source(self, kind: str) -> bool:
        """kind: 'cell' or 'nucleus'. Whether this dataset was built with the
        corresponding boundary file (cells_csv_path / nucleus_boundaries_path)."""
        store = self._cell_boundaries if kind == 'cell' else self._nucleus_boundaries
        return store is not None

    def origin(self, idx: int) -> tuple[int, int]:
        """Always (0, 0) -- boundary_polygon() already maps polygons into the
        resized (size × size) patch frame, so pool_boundary_tokens' token grid
        (laid out over [0, size)) lines up without an origin offset."""
        return 0, 0

    def boundary_polygon(self, idx: int, kind: str) -> Polygon | MultiPolygon | None:
        """kind: 'cell' or 'nucleus'. The Xenium boundary polygon for sample `idx`,
        mapped into this sample's resized (size × size) patch frame -- i.e. the
        WSI-pixel polygon shifted by this crop's (x0, y0) and scaled by size/side,
        matching origin()'s (0, 0). None if that boundary source wasn't configured,
        or no polygon was found for this cell."""
        store = self._cell_boundaries if kind == 'cell' else self._nucleus_boundaries
        if store is None:
            return None
        polygon = store.polygon_for(self.cell_ids_dataset[idx])
        if polygon is None:
            return None
        x0, y0, side = self._crop_box(idx)
        scale = self.size / side
        return affine_transform(polygon, [scale, 0, 0, scale, -x0 * scale, -y0 * scale])

    def __getitem__(self, idx: int):
        cell_id = self.cell_ids_dataset[idx]
        label = self.labels_dataset[idx]
        x0, y0, side = self._crop_box(idx)

        patch = self.wsi.read_region((x0, y0), 0, (side, side)).convert('RGB')
        patch = patch.resize((self.size, self.size), Image.BICUBIC)

        if self.transform is not None:
            patch = self.transform(patch)

        return patch, label, cell_id

    def get_raw_patch(self, idx: int):
        """Return the un-transformed, resized RGB PIL Image for a given sample index."""
        x0, y0, side = self._crop_box(idx)
        patch = self.wsi.read_region((x0, y0), 0, (side, side)).convert('RGB')
        return patch.resize((self.size, self.size), Image.BICUBIC)
