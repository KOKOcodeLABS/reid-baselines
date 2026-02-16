def train_one_epoch(model, loader, optimizer, ce_loss, triplet_loss):
    model.train()

    total_loss = 0
    total_ce = 0
    total_tri = 0

    for imgs, pids, camids in loader:
        logits, feat = model(imgs)

        loss_ce = ce_loss(logits, pids)
        loss_tri = triplet_loss(feat, pids)

        loss = loss_ce + loss_tri

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_tri += loss_tri.item()

    return {
        "loss": total_loss / len(loader),
        "ce": total_ce / len(loader),
        "triplet": total_tri / len(loader)
    }