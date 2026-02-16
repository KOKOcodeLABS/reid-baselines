import torch
from src.metrics.reid_metrics import compute_distance_matrix, evaluate_rank


def extract_features(model, loader):
    model.eval()

    features = []
    pids = []
    camids = []

    with torch.no_grad():
        for imgs, pid, camid in loader:
            emb = model(imgs, return_embedding=True)
            features.append(emb)
            pids.append(pid)
            camids.append(camid)

    features = torch.cat(features)
    pids = torch.cat(pids)
    camids = torch.cat(camids)

    return features, pids, camids


def evaluate(model, query_loader, gallery_loader):
    qf, q_pids, q_camids = extract_features(model, query_loader)
    gf, g_pids, g_camids = extract_features(model, gallery_loader)

    distmat = compute_distance_matrix(qf, gf)

    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)

    return {
        "mAP": mAP,
        "Rank-1": cmc[0],
        "Rank-5": cmc[4]
    }