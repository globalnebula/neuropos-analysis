# neurofibro_progressor/utils.py

from PIL import Image
import base64
import io

def load_image(path: str, size=(256, 256)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return img

def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
