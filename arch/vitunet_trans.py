import torch
import torch.nn as nn
from timm import create_model

class AdvancedViTEncoder(nn.Module):
    def __init__(self, img_size=224, num_channels=3, embed_dim=512):
        super(AdvancedViTEncoder, self).__init__()
        self.backbone = create_model('vit_tiny_patch16_224', pretrained=True)
        self.backbone.reset_classifier(0)
        self.fc = nn.Linear(self.backbone.num_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

class TransformerAttention3D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super(TransformerAttention3D, self).__init__()
        self.layer_norm = nn.LayerNorm(channels)
        self.multihead_attn = nn.MultiheadAttention(channels, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = x.view(B, C, -1).transpose(1, 2)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + attn_output
        x = x + self.mlp(x)
        x = x.transpose(1, 2).view(B, C, D, H, W)
        return x

class ResidualDenseBlock3D(nn.Module):
    def __init__(self, channels, growth_rate=32):
        super(ResidualDenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(nn.Sequential(
                nn.Conv3d(channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
                nn.BatchNorm3d(growth_rate),
                nn.ReLU(inplace=True)
            ))
        self.conv1x1 = nn.Conv3d(channels + 4 * growth_rate, channels, kernel_size=1)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            inputs.append(layer(torch.cat(inputs, dim=1)))
        out = self.conv1x1(torch.cat(inputs, dim=1))
        return out + x

class AdvancedUNet3DDecoder(nn.Module):
    def __init__(self, embed_dim=512, out_channels=1):
        super(AdvancedUNet3DDecoder, self).__init__()

        self.upconv1 = self.upconv_block(embed_dim, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv3 = self.upconv_block(128, 64)
        self.upconv4 = self.upconv_block(64, 32)

        self.transformer1 = TransformerAttention3D(256)
        self.transformer2 = TransformerAttention3D(128)
        self.transformer3 = TransformerAttention3D(64)

        self.dense_block1 = ResidualDenseBlock3D(256)
        self.dense_block2 = ResidualDenseBlock3D(128)
        self.dense_block3 = ResidualDenseBlock3D(64)

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_connections):
        x = self.upconv1(x) + skip_connections[0]
        x = self.transformer1(x)
        x = self.dense_block1(x)

        x = self.upconv2(x) + skip_connections[1]
        x = self.transformer2(x)
        x = self.dense_block2(x)

        x = self.upconv3(x) + skip_connections[2]
        x = self.transformer3(x)
        x = self.dense_block3(x)

        x = self.upconv4(x)

        return self.final_conv(x)

class ViTUNet3D(nn.Module):
    def __init__(self, img_size=224, num_channels=3, embed_dim=512, out_channels=1):
        super(ViTUNet3D, self).__init__()
        self.encoder = AdvancedViTEncoder(img_size, num_channels, embed_dim)
        self.decoder = AdvancedUNet3DDecoder(embed_dim, out_channels)

        self.reshape_fc = nn.Linear(embed_dim, 512 * 4 * 4 * 4)
        self.skip1_fc = nn.Linear(embed_dim, 256 * 8 * 8 * 8)
        self.skip2_fc = nn.Linear(embed_dim, 128 * 16 * 16 * 16)
        self.skip3_fc = nn.Linear(embed_dim, 64 * 32 * 32 * 32)

    def forward(self, x):
        features = self.encoder(x)

        skip1 = self.skip1_fc(features).view(features.size(0), 256, 8, 8, 8)
        skip2 = self.skip2_fc(features).view(features.size(0), 128, 16, 16, 16)
        skip3 = self.skip3_fc(features).view(features.size(0), 64, 32, 32, 32)

        features = self.reshape_fc(features)
        features = features.view(features.size(0), 512, 4, 4, 4)

        return self.decoder(features, [skip1, skip2, skip3]).squeeze(1)


input_image = torch.randn(1, 3, 224, 224)
model = ViTUNet3D()

output_3d = model(input_image)
print(output_3d.shape)