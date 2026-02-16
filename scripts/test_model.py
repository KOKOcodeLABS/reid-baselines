import torch
from src.models.baseline import Baseline

# Dummy batch
x = torch.randn(8, 3, 256, 128)

# Assume Market-1501 (751 train IDs)
model = Baseline(num_classes=751, pretrained=True)

logits, feat = model(x)

print("Logits shape:", logits.shape)
print("Feature shape:", feat.shape)

# Test embedding mode
emb = model(x, return_embedding=True)
print("Embedding shape:", emb.shape)
print("Embedding norm:", emb.norm(dim=1))