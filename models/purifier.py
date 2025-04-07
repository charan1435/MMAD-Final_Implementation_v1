import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Generator from the UNet-based architecture
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Downsampling path
        self.down1 = ResidualBlock(features, features * 2)         # 64 -> 128
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = ResidualBlock(features * 2, features * 4)     # 128 -> 256
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = ResidualBlock(features * 4, features * 8)     # 256 -> 512
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(features * 8, features * 16),            # 512 -> 1024
            SelfAttention(features * 16),                          # Self-attention at bottleneck
            ResidualBlock(features * 16, features * 16)            # 1024 -> 1024
        )

        # Upsampling path
        self.up1 = UpBlock(features * 16, features * 8, features * 8)  # 1024 + 512 -> 512
        self.up2 = UpBlock(features * 8, features * 4, features * 4)   # 512 + 256 -> 256
        self.up3 = UpBlock(features * 4, features * 2, features * 2)   # 256 + 128 -> 128
        self.up4 = UpBlock(features * 2, features, features)           # 128 + 64 -> 64

        # Refinement block with detail enhancement
        self.refine = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Multiple output branches for different aspects of the image
        # Main output (general structure)
        self.output_main = nn.Sequential(
            nn.Conv2d(features // 2, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

        # Detail enhancement branch
        self.output_detail = nn.Sequential(
            nn.Conv2d(features // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # Edge enhancement branch (uses smaller kernel for fine details)
        self.output_edge = nn.Sequential(
            nn.Conv2d(features // 2, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

        # Pre-compute the Laplacian kernel for sharpening
        self.register_buffer('laplacian_kernel', torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, x):
        # Store input shape
        input_shape = x.shape

        # Initial features
        x1 = self.initial(x)

        # Downsampling
        x2 = self.down1(x1)
        x2_pool = self.pool1(x2)

        x3 = self.down2(x2_pool)
        x3_pool = self.pool2(x3)

        x4 = self.down3(x3_pool)
        x4_pool = self.pool3(x4)

        # Bottleneck
        bottleneck = self.bottleneck(x4_pool)

        # Upsampling with skip connections
        u1 = self.up1(bottleneck, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        # Refinement
        features = self.refine(u4)

        # Generate outputs from different branches
        out_main = self.output_main(features)
        out_detail = self.output_detail(features)
        out_edge = self.output_edge(features)

        # Combine outputs with different weights
        # Main output provides the base structure
        # Detail output enhances fine details
        # Edge output preserves sharp edges
        combined = out_main + 0.3 * out_detail + 0.15 * out_edge

        # Scale from [-1, 1] to [0, 1]
        output = (combined + 1) / 2

        # Apply local contrast enhancement
        output = self.enhance_local_contrast(output)

        # Apply sharpening for better edge definition
        output = self.apply_sharpening(output)

        # Ensure output has correct dimensions
        if output.shape != input_shape:
            output = F.interpolate(output, size=(input_shape[2], input_shape[3]),
                                  mode='bilinear', align_corners=True)

        return output

    def enhance_local_contrast(self, img, kernel_size=3, amount=0.2):
        # Apply local contrast enhancement using unsharp masking
        padding = kernel_size // 2
        blurred = F.avg_pool2d(img, kernel_size=kernel_size, stride=1, padding=padding)

        # Original + amount * (original - blurred)
        enhanced = img + amount * (img - blurred)

        # Clamp values to [0, 1]
        return torch.clamp(enhanced, 0, 1)

    def apply_sharpening(self, img, amount=0.2):
        # Expand kernel for multi-channel input
        kernel = self.laplacian_kernel.expand(img.size(1), 1, 3, 3).contiguous()

        # Apply Laplacian filter to get edges
        edge_map = F.conv2d(img, kernel, padding=1, groups=img.size(1))

        # Add edges to the original image to sharpen
        sharpened = img + amount * edge_map

        # Clamp values to [0, 1]
        return torch.clamp(sharpened, 0, 1)

# Residual Block with attention
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)

        # Attention mechanism
        self.attention = DualAttention(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply attention
        out = self.attention(out)

        # Skip connection
        out += self.skip(residual)
        out = self.relu(out)

        return out

# Up-sampling block with skip connection
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Add attention for better feature refinement
        self.attention = DualAttention(out_channels)

    def forward(self, x, skip):
        # Upsample
        x = self.upsampling(x)
        x = self.conv(x)
        x = self.relu(x)

        # Ensure compatible dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Apply conv block
        x = self.conv_block(x)

        # Apply attention
        x = self.attention(x)

        return x

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Fixed dimension calculation
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Flatten spatial dimensions for attention
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)

        # Compute attention map
        attention = F.softmax(torch.bmm(proj_query, proj_key), dim=-1)

        # Apply attention to values
        proj_value = self.value(x).view(batch_size, channels, -1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Residual connection
        out = self.gamma * out + x

        return out

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Ensure reduced_channels is at least 1
        reduced_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)

        return x * self.sigmoid(y)

# Dual Attention Module (Channel + Spatial)
class DualAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class Conv2DSpectralNorm(nn.Module):
    """Conv2D layer with spectral normalization for stable GAN training"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)

class DiscriminatorBlock(nn.Module):
    """Basic block for the discriminator with spectral normalization"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.downsample = downsample

        # Main path with spectral norm
        self.conv1 = Conv2DSpectralNorm(in_channels, out_channels, kernel_size=3,
                                        stride=stride, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = Conv2DSpectralNorm(out_channels, out_channels, kernel_size=3,
                                       stride=1, padding=1)

        # Skip connection if dimensions change
        if in_channels != out_channels or stride != 1:
            self.shortcut = Conv2DSpectralNorm(in_channels, out_channels,
                                              kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

        # Downsampling option
        self.pool = nn.AvgPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
        out = self.relu(out)

        if self.downsample:
            out = self.pool(out)

        return out

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization for stable training"""
    def __init__(self, in_channels=3, base_features=64, n_layers=4):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            Conv2DSpectralNorm(in_channels, base_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Discriminator blocks with increasing features and downsampling
        features = base_features
        blocks = []
        for i in range(n_layers - 1):
            next_features = min(features * 2, 512)  # Cap at 512 features
            blocks.append(
                DiscriminatorBlock(features, next_features, stride=1, downsample=(i < n_layers-2))
            )
            features = next_features

        self.blocks = nn.Sequential(*blocks)

        # Final 1x1 conv for classification
        self.final = nn.Sequential(
            Conv2DSpectralNorm(features, 1, kernel_size=3, stride=1, padding=1),
        )

        # Self-attention at mid-level features
        self.attention = SelfAttention(base_features * 4)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)

        # Apply blocks
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)

            # Apply attention at a middle layer
            if i == 1:
                x = self.attention(x)

        # Final prediction
        x = self.final(x)

        return x, features