"""Image loading, EXIF correction, HEIC handling, downscaling, base64 encoding."""

import base64
import io

from PIL import Image, ImageOps

from pillow_heif import register_heif_opener

# Register HEIC/HEIF support with Pillow so we can open iPhone photos directly.
register_heif_opener()


MAX_EDGE_PX = 2048


def encode_image_as_base64(path: str, max_edge: int = MAX_EDGE_PX) -> str:
    """
    Open the image at `path`, apply EXIF orientation, convert to RGB,
    downscale so the longest edge is at most `max_edge` pixels, encode
    as JPEG, and return the base64 string (no data-URL prefix).

    HEIC/HEIF files are supported via pillow-heif.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")

        longest = max(im.size)
        if longest > max_edge:
            scale = max_edge / float(longest)
            new_size = (int(im.size[0] * scale), int(im.size[1] * scale))
            im = im.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")