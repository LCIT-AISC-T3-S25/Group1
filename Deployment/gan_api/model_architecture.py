import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=4, img_channels=3, feature_dim=64):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)

        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, feature_dim * 8 * 4 * 4),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (feature_dim * 8, 4, 4)),

            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim, feature_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim // 2, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_emb(labels)
        x = torch.cat([noise, label_embed], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, label_dim=4, img_channels=3, feature_dim=64):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)

        self.model = nn.Sequential(
            nn.Conv2d(img_channels + 1, feature_dim // 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim // 2, feature_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(feature_dim * 8 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        label_embed = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_map = label_embed.expand(-1, -1, images.size(2), images.size(3))
        x = torch.cat([images, label_map[:, 0:1, :, :]], dim=1)
        return self.model(x)