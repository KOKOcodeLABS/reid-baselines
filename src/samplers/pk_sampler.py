import random
from torch.utils.data import Sampler
from collections import defaultdict


class PKSampler(Sampler):
    def __init__(self, dataset, P=16, K=4):
        self.dataset = dataset
        self.P = P
        self.K = K

        self.index_dict = defaultdict(list)

        for idx, sample in enumerate(dataset.samples):
            pid = sample["pid"]
            self.index_dict[pid].append(idx)

        self.pids = list(self.index_dict.keys())

    def __iter__(self):
        batch = []

        random.shuffle(self.pids)

        for pid in self.pids:
            idxs = self.index_dict[pid]

            if len(idxs) >= self.K:
                selected = random.sample(idxs, self.K)
            else:
                selected = random.choices(idxs, k=self.K)

            batch.extend(selected)

            if len(batch) >= self.P * self.K:
                yield from batch[: self.P * self.K]
                batch = []

    def __len__(self):
        return len(self.dataset)