"""Loss functions: pixel (L1), VGG-19 perceptual (relu5_4), and adversarial (BCE-with-logits).

Generator loss = lambda_pixel * L1 + lambda_percep * VGG + lambda_adv * BCE
Default weights match the manuscript (Eq. 26): lambda_pixel=1.0, lambda_percep=0.01, lambda_adv=0.005.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # features[:35] == up to relu5_4
        self.vgg_features = nn.Sequential(*list(vgg.features.children())[:35]).eval()
        for p in self.vgg_features.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        # inputs are in [-1, 1] (Tanh output); map to [0, 1] for VGG
        sr_norm = (sr + 1) / 2.0
        hr_norm = (hr + 1) / 2.0
        return F.l1_loss(self.vgg_features(sr_norm), self.vgg_features(hr_norm))


class HybridLossEvaluator:
    def __init__(self, lambda_pixel=1.0, lambda_percep=1e-2, lambda_adv=5e-3, device="cpu"):
        self.lambda_pixel = lambda_pixel
        self.lambda_percep = lambda_percep
        self.lambda_adv = lambda_adv
        self.l1_loss = nn.L1Loss().to(device)
        self.perceptual_loss = VGGPerceptualLoss().to(device)
        self.bce_loss = nn.BCEWithLogitsLoss().to(device)

    def generator_loss(self, sr, hr, d_logits_fake):
        loss_pixel = self.l1_loss(sr, hr)
        loss_percep = self.perceptual_loss(sr, hr)
        real_labels = torch.ones_like(d_logits_fake)
        loss_adv = self.bce_loss(d_logits_fake, real_labels)
        total = (self.lambda_pixel * loss_pixel
                 + self.lambda_percep * loss_percep
                 + self.lambda_adv * loss_adv)
        return total, loss_pixel, loss_percep, loss_adv

    def discriminator_loss(self, d_logits_real, d_logits_fake):
        real = self.bce_loss(d_logits_real, torch.ones_like(d_logits_real))
        fake = self.bce_loss(d_logits_fake, torch.zeros_like(d_logits_fake))
        return real + fake
