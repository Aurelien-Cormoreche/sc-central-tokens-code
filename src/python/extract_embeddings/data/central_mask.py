from PIL import Image


def central_mask_box(width: int, height: int, token_size: int, grid_size: int) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of the centred grid_size x grid_size block of
    token_size x token_size model tokens within a width x height image -- e.g.
    grid_size=3, token_size=14 covers the central 3x3 ViT patch tokens of a
    224x224 (16x16-token) image.

    Aligned to token boundaries (multiples of token_size), so it lines up with
    the model's own patchification of the image rather than the image's
    geometric centre -- for a token grid with an even side (as here: 16), the
    "central" grid_size-wide window is not perfectly symmetric, so ties are
    broken by taking the block starting at token index (num_tokens - grid_size) // 2,
    which for the common 16-token/3-wide case yields tokens {6, 7, 8}.
    """
    tokens_x = max(width // token_size, grid_size)
    tokens_y = max(height // token_size, grid_size)
    start_tx = (tokens_x - grid_size) // 2
    start_ty = (tokens_y - grid_size) // 2
    x0 = start_tx * token_size
    y0 = start_ty * token_size
    x1 = min(x0 + grid_size * token_size, width)
    y1 = min(y0 + grid_size * token_size, height)
    return x0, y0, x1, y1


def apply_central_mask(
    patch: Image.Image,
    token_size: int,
    grid_size: int,
    mask_value: tuple[int, int, int],
) -> Image.Image:
    """Return a copy of `patch` with its centred grid_size x grid_size block of
    token_size x token_size tokens filled with `mask_value` (see
    central_mask_box). Callers apply this post-resize (on the final,
    model-input-sized image) so the masked region always covers the central
    tokens of the model's actual input grid, whatever native crop / resize
    produced `patch`.
    """
    x0, y0, x1, y1 = central_mask_box(patch.width, patch.height, token_size, grid_size)
    masked = patch.copy()
    masked.paste(Image.new('RGB', (x1 - x0, y1 - y0), mask_value), (x0, y0))
    return masked
