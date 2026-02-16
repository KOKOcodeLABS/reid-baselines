import torch
import numpy as np


def compute_distance_matrix(qf, gf):
    return torch.cdist(qf, gf, p=2)


def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    distmat = distmat.cpu().numpy()
    q_pids = q_pids.cpu().numpy()
    g_pids = g_pids.cpu().numpy()
    q_camids = q_camids.cpu().numpy()
    g_camids = g_camids.cpu().numpy()

    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)

    all_cmc = []
    all_ap = []

    for i in range(num_q):
        q_pid = q_pids[i]
        q_cam = q_camids[i]

        order = indices[i]

        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
        keep = ~remove

        matches = (g_pids[order] == q_pid)[keep]

        if not np.any(matches):
            continue

        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        # AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        precision = tmp_cmc / (np.arange(len(matches)) + 1)
        ap = (precision * matches).sum() / num_rel
        all_ap.append(ap)

    if len(all_cmc) == 0:
        raise RuntimeError("No valid queries")

    all_cmc = np.mean(all_cmc, axis=0)
    mAP = np.mean(all_ap)

    return all_cmc, mAP