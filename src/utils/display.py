from io import BytesIO
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image

from src.types import Array, Color, PIL_Image
from src.utils.draw_bbox import draw_bboxes


def open_image(url: str) -> PIL_Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def resize_with_pad(
    pil_img: PIL_Image, shape: Tuple[int, int], background_color: Color = (0, 0, 0)
) -> PIL_Image:
    width, height = pil_img.size
    final_width, final_height = shape
    ratio = min(final_width / width, final_height / height)
    result = Image.new(pil_img.mode, shape, background_color)
    pil_img = pil_img.resize((int(ratio * width), int(ratio * height)))
    width, height = pil_img.size
    result.paste(pil_img, ((final_width - width) // 2, (final_height - height) // 2))
    return result


def load_crop(
    url: str,
    x1y1x2y2: Tuple[int, int, int, int],
    shape: Optional[Tuple[int, int]] = None,
) -> Array:
    image = open_image(url)
    crop = image.crop(x1y1x2y2)
    if shape is not None:
        crop = resize_with_pad(crop, shape=shape)
    return np.array(crop)


def load_and_annotate_image(
    url: str, image_annotations: pd.Series, color_map: Dict[str, Color]
) -> Array:
    image = np.array(open_image(url))
    boxes = image_annotations[["x1", "y1", "x2", "y2"]].values
    box_colors = [
        color_map[category_name] for category_name in image_annotations["category_name"]
    ]
    image = draw_bboxes(image, boxes, box_colors, image_annotations["category_name"])
    return image
