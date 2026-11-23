"""Level-0 WSI region reads in extraction space.

Extraction space = native level-0 pixels scaled by s (s = 1 unless resolution.resample_below_mpp is set;
then slides finer than it are area-resampled to resolution.target_mpp, s < 1). Extraction
pixel i covers native [i/s, (i+1)/s). With s == 1 a read is a plain level-0 read_region, so a
crop taken from a region is pixel-identical to reading that crop directly.

One WSIReader per process: DataLoader workers each open their own handle (OpenSlide handles
are not fork-safe and stay in an error state after a failed read, hence reopen-and-retry).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from bench.utils import LOG


class ReadError(RuntimeError):
    pass


class WSIReader:
    def __init__(self, path: str, scale: float = 1.0, black_max: int = 8):
        self.path = str(path)
        self.scale = float(scale)
        self.black_max = black_max
        self._slide = None
        self._pid: Optional[int] = None
        self.width = self.height = 0
        self._open()

    # ── handle management ──
    def _open(self) -> None:
        import openslide
        self._slide = openslide.OpenSlide(self.path)
        self._pid = os.getpid()
        self.width, self.height = self._slide.dimensions

    def _handle(self):
        if self._slide is None or self._pid != os.getpid():
            self._open()
        return self._slide

    def close(self) -> None:
        if self._slide is not None:
            try:
                self._slide.close()
            finally:
                self._slide = None

    @property
    def ext_width(self) -> int:
        return int(math.floor(self.width * self.scale + 1e-9))

    @property
    def ext_height(self) -> int:
        return int(math.floor(self.height * self.scale + 1e-9))

    # ── native reads ──
    def read_native_rgba(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Level-0 RGBA (h, w, 4) uint8; one reopen-and-retry on failure, then ReadError."""
        last: Optional[Exception] = None
        for attempt in range(2):
            try:
                img = self._handle().read_region((int(x), int(y)), 0, (int(w), int(h)))
                return np.asarray(img)
            except Exception as e:  # openslide.OpenSlideError, OSError
                last = e
                LOG.warning("read failed (%s, attempt %d) %s @(%d,%d,%d,%d): %s",
                            os.path.basename(self.path), attempt + 1, type(e).__name__, x, y, w, h, e)
                self.close()
        raise ReadError(f"{self.path} @({x},{y},{w},{h}): {last}")

    @staticmethod
    def rgba_to_rgb(rgba: np.ndarray, fill: int = 0) -> np.ndarray:
        """Drop alpha. Transparent pixels (outside the scanned area) become `fill`."""
        rgb = np.ascontiguousarray(rgba[..., :3])
        if rgba.shape[-1] == 4:
            transparent = rgba[..., 3] == 0
            if transparent.any():
                rgb = rgb.copy()
                rgb[transparent] = fill
        return rgb

    # ── extraction-space reads ──
    def read_ext(self, x0: int, y0: int, w: int, h: int, pad_value: int = 255,
                 allow_outside: bool = False) -> np.ndarray:
        """RGB uint8 (h, w, 3) of extraction-space box [x0, x0+w) x [y0, y0+h).

        With allow_outside, parts beyond the image are padded with pad_value (multicell tiles);
        otherwise a box leaving the image raises (single-cell crops never do: border cells are
        excluded by the manifest)."""
        ew, eh = self.ext_width, self.ext_height
        inside = x0 >= 0 and y0 >= 0 and x0 + w <= ew and y0 + h <= eh
        if not inside and not allow_outside:
            raise ReadError(f"box ({x0},{y0},{w},{h}) leaves the extraction image ({ew}x{eh})")
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x0 + w, ew), min(y0 + h, eh)
        out = np.full((h, w, 3), pad_value, dtype=np.uint8)
        if cx1 <= cx0 or cy1 <= cy0:
            return out
        sub = self._read_ext_inside(cx0, cy0, cx1 - cx0, cy1 - cy0)
        out[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = sub
        return out

    def _read_ext_inside(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        s = self.scale
        if s == 1.0:
            return self.rgba_to_rgb(self.read_native_rgba(x0, y0, w, h), fill=255)
        from PIL import Image
        # native float box covered by these extraction pixels, read on the enclosing integer box
        fx0, fy0, fx1, fy1 = x0 / s, y0 / s, (x0 + w) / s, (y0 + h) / s
        nx0, ny0 = int(math.floor(fx0)), int(math.floor(fy0))
        nx1 = min(int(math.ceil(fx1)), self.width)
        ny1 = min(int(math.ceil(fy1)), self.height)
        rgb = self.rgba_to_rgb(self.read_native_rgba(nx0, ny0, nx1 - nx0, ny1 - ny0), fill=255)
        img = Image.fromarray(rgb)
        box = (fx0 - nx0, fy0 - ny0, min(fx1, nx1) - nx0, min(fy1, ny1) - ny0)
        # BOX = area averaging over each output pixel's (fractional) native footprint
        return np.asarray(img.resize((w, h), resample=Image.BOX, box=box))


def scale_for(mpp: Optional[float], resample_below: Optional[float], target: float) -> float:
    """Extraction scale s: 1 (stored resolution) unless resampling is on and the slide is finer than resample_below."""
    if resample_below is None or mpp is None or not math.isfinite(mpp):
        return 1.0
    return mpp / target if mpp < resample_below else 1.0
