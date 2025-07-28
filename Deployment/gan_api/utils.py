import torch
from torchvision.utils import save_image

def generate_image(model, latent_dim, label, device, output_path="generated.png"):
    """Generate an image for a given label using the CGAN generator"""
    z = torch.randn(1, latent_dim, device=device)
    y = torch.tensor([label], device=device)

    with torch.no_grad():
        img = model(z, y)
        img = (img.clamp(-1, 1) + 1) / 2  # Scale to [0,1]

    save_image(img, output_path)
    return output_path
