import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.registry import get_dataset

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt



transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor()
])

DatasetClass = get_dataset("market1501")

dataset = DatasetClass(
    root="data/clean/market1501",
    split="train",
    transform=transform
)

print("Dataset size:", len(dataset))

loader = DataLoader(dataset, batch_size=8, shuffle=True)

images, pids, camids = next(iter(loader))

print("Batch shape:", images.shape)
print("PIDs:", pids)
print("CamIDs:", camids)

import torchvision.utils as vutils

grid = vutils.make_grid(images, nrow=4)

plt.imshow(grid.permute(1, 2, 0))
plt.title("Random Training Batch")
plt.axis("off")

plt.show(block=True)
