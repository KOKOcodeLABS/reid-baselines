import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        # Compute pairwise distance matrix
        dist = torch.cdist(embeddings, embeddings, p=2)

        N = embeddings.size(0)

        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = ~is_pos

        dist_ap = []
        dist_an = []

        for i in range(N):
            pos = dist[i][is_pos[i]]
            neg = dist[i][is_neg[i]]

            # hardest positive
            dist_ap.append(pos.max().unsqueeze(0))

            # hardest negative
            dist_an.append(neg.min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)

        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss