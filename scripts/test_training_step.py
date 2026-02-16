import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.registry import get_dataset
from src.models.baseline import Baseline
from src.losses.cross_entropy import build_cross_entropy
from src.losses.triplet import BatchHardTripletLoss
from src.samplers.pk_sampler import PKSampler



P = 4
K = 4
batch_size = P * K
lr = 3e-4

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

num_classes = dataset.num_classes()

print("Number of classes:", num_classes)

sampler = PKSampler(dataset, P=P, K=K)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler
)


model = Baseline(num_classes=num_classes, pretrained=False)


ce_loss = build_cross_entropy()
triplet_loss = BatchHardTripletLoss(margin=0.3)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


images, pids, camids = next(iter(loader))

print("Batch shape:", images.shape)
print("Unique PIDs in batch:", len(torch.unique(pids)))

print("\nRunning multiple optimization steps...\n")

for step in range(5):
    logits, feat = model(images)

    loss_ce = ce_loss(logits, pids)
    loss_tri = triplet_loss(feat, pids)

    loss = loss_ce + loss_tri

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}: "
          f"CE={loss_ce.item():.4f} | "
          f"Triplet={loss_tri.item():.4f} | "
          f"Total={loss.item():.4f}")

scheduler.step()

print("\nTraining step test completed successfully.")