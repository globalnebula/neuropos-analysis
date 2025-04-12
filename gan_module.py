import torch
import torch.nn as nn
import torchvision.utils as vutils

# Generator
class DiseaseProgressionGenerator(nn.Module):
    def __init__(self, z_dim=100, condition_dim=1, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + condition_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, img_channels * 64 * 64),
            nn.Tanh()
        )
        self.img_channels = img_channels

    def forward(self, z, time_step):
        # Concatenate noise and time step
        input = torch.cat((z, time_step), dim=1)
        out = self.net(input)
        return out.view(-1, self.img_channels, 64, 64)

# Discriminator
class DiseaseProgressionDiscriminator(nn.Module):
    def __init__(self, img_channels=3, condition_dim=1):
        super().__init__()
        self.img_flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(img_channels * 64 * 64 + condition_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.img_channels = img_channels

    def forward(self, img, time_step):
        img_flat = self.img_flatten(img)
        x = torch.cat((img_flat, time_step), dim=1)
        return self.net(x)


def generate_progression_images(generator, z_dim=100):
    z = torch.randn(1, z_dim)
    time_steps = torch.tensor([[0.0], [0.5], [1.0]])

    generator.eval()
    images = []

    with torch.no_grad():
        for t in time_steps:
            img = generator(z, t.unsqueeze(0))
            images.append(img)

    # Save or show output
    grid = vutils.make_grid(torch.cat(images, dim=0), normalize=True)
    vutils.save_image(grid, "progression_output.png")
    print("Saved generated progression images to 'progression_output.png'")
