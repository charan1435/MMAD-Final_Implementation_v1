import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import math

class AdversarialClassifier(nn.Module):
    """Hybrid model for classifying MRI images as clean or adversarially attacked"""
    def __init__(self, num_classes=4, feature_dim=768):
        super().__init__()
        
        # Create the hybrid architecture components
        self.transformer = VisionTransformer(feature_dim=feature_dim)
        self.cnn = CNNFeatureExtractor(feature_dim=feature_dim)
        self.snn = SimpleSNNComponent(feature_dim=feature_dim)
        
        # Feature fusion - attention weights
        self.fusion_attn = nn.Sequential(
            nn.Linear(feature_dim * 3, 3),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Auxiliary classifiers
        self.transformer_classifier = nn.Linear(feature_dim, num_classes)
        self.cnn_classifier = nn.Linear(feature_dim, num_classes)
        self.snn_classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        # Extract features from each component
        transformer_features = self.transformer(x)
        cnn_features = self.cnn(x)
        snn_features = self.snn(x)
        
        # Auxiliary predictions
        transformer_logits = self.transformer_classifier(transformer_features)
        cnn_logits = self.cnn_classifier(cnn_features)
        snn_logits = self.snn_classifier(snn_features)
        
        # Concatenate features for attention weighting
        concat_features = torch.cat([transformer_features, cnn_features, snn_features], dim=1)
        
        # Calculate attention weights
        attn_weights = self.fusion_attn(concat_features)
        
        # Apply attention weights
        stacked_features = torch.stack([transformer_features, cnn_features, snn_features], dim=1)
        fused_features = (stacked_features * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        if self.training:
            aux_logits = {
                'transformer': transformer_logits,
                'cnn': cnn_logits,
                'snn': snn_logits
            }
            return {
                'logits': logits,
                'aux_logits': aux_logits, 
                'attention_weights': attn_weights,
                'fused_features': fused_features
            }
        
        return logits

# Vision Transformer module
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 hidden_dim=256, feature_dim=768,
                 num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()

        # Compute number of patches
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, hidden_dim,
                                         kernel_size=patch_size, stride=patch_size)

        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches + 1, hidden_dim))

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Projection to feature_dim
        self.projection = nn.Linear(hidden_dim, feature_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize patch embedding like a nn.Linear
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)

        # Initialize position embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        # Create patch embeddings
        x = self.patch_embedding(x)  # (B, hidden_dim, grid_size, grid_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, hidden_dim)

        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, n_patches + 1, hidden_dim)

        # Add position embeddings
        x = x + self.pos_embedding

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply layer norm
        x = self.norm(x)

        # Take class token representation
        x = x[:, 0]

        # Project to feature_dim
        x = self.projection(x)

        return x

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # MLP block
        mlp_hidden_dim = hidden_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Multi-head attention with residual connection
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x

# SqueezeExcitation module
class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=768):
        super().__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection to feature_dim
        self.projection = nn.Linear(512, feature_dim)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        # First block with possible downsampling
        layers.append(self._make_block(in_channels, out_channels, stride, downsample))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, stride=1, downsample=None):
        layers = []
        # Conv + BN + ReLU
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Conv + BN
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        # SE attention
        layers.append(SqueezeExcitation(out_channels))

        # Downsample residual if needed
        if downsample is not None:
            return ResidualBlock(nn.Sequential(*layers), downsample)
        else:
            return ResidualBlock(nn.Sequential(*layers))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Feature layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Project to feature_dim
        x = self.projection(x)

        return x

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, layers, downsample=None):
        super().__init__()
        self.layers = layers
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.layers(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Simplified SNN Component
class SimpleSNNComponent(nn.Module):
    def __init__(self, in_channels=3, feature_dim=768):
        super().__init__()

        # CNN frontend for feature extraction
        self.frontend = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Conv2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Conv3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),

            # Flatten
            nn.Flatten()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        # Extract features with CNN frontend
        x = self.frontend(x)

        # Apply fully connected layers
        x = self.fc(x)

        return x