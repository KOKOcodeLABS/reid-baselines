import os
from torch.utils.data import Dataset
from PIL import Image


class BaseReIDDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.samples = []
        self._build_index()
        self._relabel()   # <-- MAKE SURE THIS IS HERE

    def _build_index(self):
        split_dir = os.path.join(self.root, self.split)

        for pid in sorted(os.listdir(split_dir)):
            pid_dir = os.path.join(split_dir, pid)

            for img_name in os.listdir(pid_dir):
                img_path = os.path.join(pid_dir, img_name)

                camid = self._parse_camid(img_name)

                self.samples.append(
                    {
                        "img_path": img_path,
                        "pid": int(pid),
                        "camid": camid
                    }
                )

    def _relabel(self):
        unique_pids = sorted(set(sample["pid"] for sample in self.samples))

        self.pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}

        for sample in self.samples:
            sample["pid"] = self.pid2label[sample["pid"]]

    def num_classes(self):
        return len(self.pid2label)

    def _parse_camid(self, filename):
        return -1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["img_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, sample["pid"], sample["camid"]