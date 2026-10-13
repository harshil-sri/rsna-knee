from typing import Any
import torch
import torch.nn as nn
import timm

class GatedAttentionMIL(nn.Module):
    def __init__(self, in_features: int = 384, hidden_dim: int = 128, num_classes: int = 12):
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        v = self.attention_v(h)
        u = self.attention_u(h)
        a = self.attention_weights(v * u)
        weights = torch.softmax(a, dim=1)
        m = torch.sum(weights * h, dim=1)
        logits = self.classifier(m)
        return logits

class RSNAViTMIL(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        in_chans: int = 5,
        num_classes: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone: Any = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=224,
            num_classes=0,
        )
        raw_dim = getattr(self.backbone, "embed_dim", 384)
        embed_dim = int(raw_dim) if isinstance(raw_dim, (int, float)) else 384
        self.dropout = nn.Dropout(dropout)
        self.mil = GatedAttentionMIL(in_features=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, c, h, w_img = x.shape
        x_reshaped = x.view(b * w, c, h, w_img)
        feats: torch.Tensor = self.backbone.forward_features(x_reshaped)
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        patch_tokens = feats[:, prefix_tokens:]
        embeds = patch_tokens.mean(dim=1)
        embeds = self.dropout(embeds)
        embeds = embeds.view(b, w, -1)
        logits = self.mil(embeds)
        return logits
