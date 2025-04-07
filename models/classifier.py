import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

class AdversarialClassifier(nn.Module):
    """Model for classifying MRI images as clean or adversarially attacked"""
    def __init__(self, num_classes=4, feature_dim=768):
        super().__init__()
        
        # Initialize with pretrained ResNet50 as base
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Replace final fully connected layer
        backbone_output_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original fc layer
        
        # Enhanced classification head with attention
        self.attention = SelfAttentionPooling(backbone_output_features)
        
        # Classification layers with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(backbone_output_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Auxiliary classifier for feature regularization
        self.aux_classifier = nn.Linear(backbone_output_features, num_classes)
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply attention pooling
        attended_features = self.attention(features)
        
        # Get primary classification logits
        logits = self.classifier(attended_features)
        
        # During training, also compute auxiliary classification
        if self.training:
            aux_logits = self.aux_classifier(features)
            return logits, aux_logits
        
        return logits

class SelfAttentionPooling(nn.Module):
    """Self-attention pooling layer for global feature weighting"""
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim // 8)
        self.key = nn.Linear(in_dim, in_dim // 8)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Project queries and keys
        query = self.query(x)
        key = self.key(x)
        
        # Compute attention scores
        scores = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5), dim=-1)
        
        # Apply attention to values
        values = self.value(x)
        out = self.gamma * torch.matmul(scores, values) + x
        
        return out

class EnhancedClassifier(nn.Module):
    """A more sophisticated classifier combining CNN and Vision Transformer features"""
    def __init__(self, num_classes=4):
        super().__init__()
        
        # CNN backbone (ResNet50)
        self.cnn_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn_backbone.fc = nn.Identity()  # Remove final fc layer
        cnn_features = 2048  # ResNet50 output features
        
        # Vision Transformer components
        self.transformer_encoder = TransformerEncoderBlock(
            hidden_dim=cnn_features,
            num_heads=8,
            mlp_dim=cnn_features*4,
            dropout=0.1,
            num_layers=2
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(cnn_features, cnn_features),
            nn.LayerNorm(cnn_features),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cnn_features, cnn_features//2),
            nn.LayerNorm(cnn_features//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(cnn_features//2, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)
        
        # Add batch dimension for transformer if needed
        if cnn_features.dim() == 2:
            cnn_features = cnn_features.unsqueeze(1)
            
        # Apply transformer encoder
        transformer_features = self.transformer_encoder(cnn_features)
        
        # Squeeze back if necessary
        if transformer_features.dim() == 3:
            transformer_features = transformer_features.squeeze(1)
            
        # Feature fusion
        fused_features = self.fusion(transformer_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits

class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with self-attention"""
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout=0.1, num_layers=1):
        super().__init__()
        
        # Stack multiple transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and MLP"""
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + residual
        
        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x