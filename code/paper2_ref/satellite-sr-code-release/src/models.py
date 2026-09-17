"""Model definitions for the Hybrid Fusion GAN with Transformer attention.

Generator  : Shallow conv -> 16 x RRDB -> Transformer encoder -> SARB -> HFF -> 2x PixelShuffle (x4)
Discriminator : 8-layer global convolutional classifier (Table 4 of the paper).

The architecture and hyper-parameters here match the manuscript:
  RRDB blocks      = 16   (nf=64, gc=32, residual scaling 0.2)
  Transformer      = 4 layers, 8 heads, embed_dim 256, MLP ratio 4, ReLU
  Upscale factor   = 4    (two x2 PixelShuffle stages)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.RDB1 = DenseBlock(nf, gc)
        self.RDB2 = DenseBlock(nf, gc)
        self.RDB3 = DenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class TransformerContextEncoder(nn.Module):
    def __init__(self, in_channels=64, embed_dim=256, num_heads=8, num_layers=4, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.num_patches = img_size * img_size
        self.in_proj = nn.Linear(in_channels, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.0, activation="relu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        flat_x = x.flatten(2).transpose(1, 2)          # B, H*W, C
        feat_seq = self.in_proj(flat_x) + self.pos_embedding
        tx_out = self.transformer_encoder(feat_seq)
        out = self.out_proj(tx_out).transpose(1, 2).view(B, C, H, W)
        return out


class SARB(nn.Module):
    """Self-Attention Refinement Block (channel + spatial attention)."""
    def __init__(self, channels=64, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False))
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        avg_c = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)
        max_c = F.adaptive_max_pool2d(x, (1, 1)).view(B, C)
        channel_att = self.sigmoid(self.mlp(avg_c) + self.mlp(max_c)).view(B, C, 1, 1)
        x = x * channel_att
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * spatial_att


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, gc=32,
                 num_blocks=16, embed_dim=256, num_heads=8, num_layers=4, img_size=64,
                 use_transformer=True, use_sarb=True, use_hff=True):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_sarb = use_sarb
        self.use_hff = use_hff
        self.shallow_conv = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.rrdb_stack = nn.Sequential(*[RRDB(nf, gc) for _ in range(num_blocks)])
        if use_transformer:
            self.transformer = TransformerContextEncoder(nf, embed_dim, num_heads, num_layers, img_size)
        if use_sarb:
            self.sarb = SARB(nf)
        # HFF fuses concat[RRDB, refined]; if disabled, a 1:1 conv keeps channel count
        self.hff_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1) if use_hff else None
        self.up_conv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)
        self.up_conv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)
        self.final_conv = nn.Conv2d(nf, out_channels, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        f0 = self.lrelu(self.shallow_conv(x))
        f_rrdb = self.rrdb_stack(f0)
        f_res = f_rrdb + f0
        if self.use_transformer:
            f_tf = self.transformer(f_res)
            f_fuse = f_tf + f_rrdb
        else:
            f_fuse = f_res
        f_refined = self.sarb(f_fuse) if self.use_sarb else f_fuse
        if self.hff_conv is not None:
            f_hff = self.hff_conv(torch.cat([f_rrdb, f_refined], dim=1))
        else:
            f_hff = f_refined
        out = self.lrelu(self.pixel_shuffle1(self.up_conv1(f_hff)))
        out = self.lrelu(self.pixel_shuffle2(self.up_conv2(out)))
        out = self.tanh(self.final_conv(out))
        return out


class Discriminator(nn.Module):
    """Eight-layer global convolutional discriminator (matches Table 4)."""
    def __init__(self, in_channels=3, hr_size=256):
        super().__init__()

        def conv_block(in_f, out_f, stride=1, use_bn=True):
            layers = [nn.Conv2d(in_f, out_f, 3, stride, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(in_channels, 64, 1, use_bn=False),
            conv_block(64, 64, 2),
            conv_block(64, 128, 1),
            conv_block(128, 128, 2),
            conv_block(128, 256, 1),
            conv_block(256, 256, 2),
            conv_block(256, 512, 1),
            conv_block(512, 512, 2),
        )
        reduced = hr_size // 16  # four stride-2 layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Sequential(nn.Linear(reduced * reduced * 512, 1024),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.dense2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    g, d = Generator(), Discriminator()
    print(f"Generator parameters     : {count_parameters(g):,}")
    print(f"Discriminator parameters : {count_parameters(d):,}")
