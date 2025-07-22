import torch
import torch.nn as nn
from einops import rearrange

class ViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, dim=64, depth=6, heads=8, mlp_dim=128, num_classes=10):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by patch size"
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size  
        self.patch_to_emb = nn.Linear(patch_dim, dim)
        self.pos_em = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_tok = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                batch_first=True,
            ),
            num_layers=depth,
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p, c=C)
        x = self.patch_to_emb(x)
        cls_tok = self.cls_tok.expand(B, -1, -1)
        x = torch.cat((cls_tok, x), dim=1)
        x = x + self.pos_em[:, :x.size(1), :]
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])