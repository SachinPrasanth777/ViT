import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        residual = x
        x = self.attention(self.layer_norm(x), self.layer_norm(x), self.layer_norm(x))[0] + residual
        x = self.mlp(self.layer_norm2(x)) + x
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_classes, transformer_units):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer_layers = nn.Sequential(
            *[Transformer(embed_dim, num_heads, mlp_dim) for _ in range(transformer_units)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x + self.pos_embedding)
        x = self.transformer_layers(x)
        return self.mlp_head(x[:, 0])
