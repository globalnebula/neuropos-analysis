import torch
import numpy as np
import PIL.Image
from torchvision import transforms
import sys
import os

sys.path.append('C:\\Users\\kunal\\Desktop\\Vivitsu\\neurofibro-progressor\\stylegan2-ada-pytorch')
import legacy
import dnnlib

from diffusers import StableDiffusionInpaintPipeline
from torchvision.transforms.functional import to_pil_image, to_tensor

# Load StyleGAN2 Generator
def load_generator(network_pkl_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(network_pkl_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G

# Encode real image into latent space (requires e4e encoder)
def encode_image_to_latent(image_path, encoder):
    image = PIL.Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(next(encoder.parameters()).device)
    with torch.no_grad():
        latent = encoder(img_tensor)
    return latent

# Generate image from latent
def generate_image(G, latent):
    with torch.no_grad():
        img = G.synthesis(latent, noise_mode='const')
    img = (img.clamp(-1, 1) + 1) * (255 / 2)
    return img.permute(0, 2, 3, 1).squeeze().cpu().numpy().astype(np.uint8)

# Load Stable Diffusion Inpainting Pipeline
def load_sd_inpainter():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return pipe.to(device)

# Inpaint and enhance image using Stable Diffusion
def enhance_with_sd(image_np, prompt="realistic face, skin clarity"):
    inpainter = load_sd_inpainter()
    image = PIL.Image.fromarray(image_np).resize((512, 512)).convert("RGB")

    # Simulate a degraded mask (e.g., missing part)
    mask = PIL.Image.new("L", image.size, 0)
    mask.paste(255, (200, 200, 312, 312))  # fake region to be inpainted

    result = inpainter(prompt=prompt, image=image, mask_image=mask).images[0]
    return result.resize((256, 256))

# Main disease progression generator
def generate_progression(G, base_latent, direction, steps=5, alpha_range=(0.0, 1.0)):
    images = []
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)

    for alpha in alphas:
        progressed_latent = base_latent + alpha * direction
        raw_img = generate_image(G, progressed_latent)
        enhanced_img = enhance_with_sd(raw_img)
        images.append(np.array(enhanced_img))

    return images

# Save progression as a GIF
def save_progression_gif(images, output_path="progression.gif", duration=500):
    import imageio
    imageio.mimsave(output_path, images, duration=duration / 1000)

# Example usage
if __name__ == "__main__":
    G = load_generator("C:\\Users\\kunal\\Desktop\\Vivitsu\\neurofibro-progressor\\stylegan2-ada-pytorch\\pretrained\\stylegan2-ffhq-config-f.pkl")

    # Replace this with your actual encoder
    from stylegan2_ada_pytorch.e4e_encoder import E4EEncoder
    encoder_model = e4e_encoder.E4EEncoder()

    latent = encode_image_to_latent("face.jpg", encoder_model)

    # Replace with a fine-tuned latent direction vector
    disease_direction = torch.randn_like(latent) * 0.1  # Placeholder

    images = generate_progression(G, latent, disease_direction, steps=6)
    save_progression_gif(images, "neurofibro_progression.gif")