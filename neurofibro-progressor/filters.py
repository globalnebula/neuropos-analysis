# neurofibro_progressor/filters.py

from PIL import Image, ImageDraw, ImageFilter
import random

def apply_nf1_filter(image: Image.Image, stage: float) -> Image.Image:
    """
    Simulates NF1 by applying random deformations, moles, bumps to the skin.
    Stage controls severity.
    """
    img = image.convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Add random "tumor" spots on the skin
    num_bumps = int(20 + stage * 80)  # More bumps as disease progresses
    for _ in range(num_bumps):
        x = random.randint(50, img.width - 50)
        y = random.randint(50, img.height - 50)
        r = random.randint(4, int(10 + 20 * stage))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(139, 69, 19, int(150 + 100 * stage)))

    blurred = img.filter(ImageFilter.GaussianBlur(radius=1 + 3 * stage))

    return blurred.convert("RGB")
