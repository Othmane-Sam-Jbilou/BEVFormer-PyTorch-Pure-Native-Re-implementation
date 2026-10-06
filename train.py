import time
import os
import torch
import numpy as np
from data_loader import NuScenesFrameLoader
from Model import BEVformer
from torch.utils.data import  DataLoader

class BEVIoUMetric:

    def __init__(self, num_classes=5, threshold=0.5, eps=1e-6):
        self.num_classes = num_classes
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.total_inter = np.zeros(self.num_classes, dtype=np.float64)
        self.total_union = np.zeros(self.num_classes, dtype=np.float64)

    def update(self, logits, targets):
        preds = (torch.sigmoid(logits) > self.threshold).float()

        preds_flat = preds.view(preds.size(0), preds.size(1), -1).cpu().numpy()
        targets_flat = targets.view(targets.size(0), targets.size(1), -1).cpu().numpy()

        intersection = (preds_flat * targets_flat).sum(axis=-1).sum(axis=0)
        union = (preds_flat + targets_flat > 0).astype(np.float32).sum(axis=-1).sum(axis=0)

        self.total_inter += intersection
        self.total_union += union

    def compute(self):
        valid_classes = self.total_union > 0
        iou_per_class = np.zeros(self.num_classes, dtype=np.float64)
        iou_per_class[valid_classes] = self.total_inter[valid_classes] / self.total_union[valid_classes]

        # Only average over classes that actually appeared
        mean_iou = np.mean(iou_per_class[valid_classes]) if np.any(valid_classes) else 0.0
        return mean_iou, iou_per_class

def train_one_epoch(model, dataloader, accum_steps, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    history_BEV = None #history bev start as none then update as the training go from sample to the next

    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to('cuda', non_blocking=True)
        B = images.shape[0]
        BEV_queries = torch.zeros((B, 200, 200, model.emb_dim), device='cuda')

        K = batch['K'].cuda(non_blocking=True)
        R = batch['R'].cuda(non_blocking=True)
        t = batch['t'].cuda(non_blocking=True)
        R_curr2prev = batch['R_curr2prev'].cuda(non_blocking=True)
        t_curr2prev = batch['t_curr2prev'].cuda(non_blocking=True)
        has_prev = batch['has_prev'].cuda(non_blocking=True)
        if history_BEV is None:
            has_prev = torch.zeros_like(has_prev)

        bev_gt = batch['channels'].cuda(non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            history, logits = model(history_BEV, BEV_queries, images, has_prev, R_curr2prev, t_curr2prev, K, R, t)
            loss = criterion(logits, bev_gt) / accum_steps

        scaler.scale(loss).backward()

        history_BEV = history.detach()

        unscaled_loss = loss.item() * accum_steps
        running_loss += unscaled_loss

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (batch_idx + 1) % 5 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} Loss: {unscaled_loss:.4f}")

    return running_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, metric):
    model.eval()
    running_loss = 0.0
    history_BEV = None
    metric.reset()

    print('validating...')
    for batch in dataloader:
        images = batch['images'].to('cuda', non_blocking=True)
        B = images.shape[0]
        BEV_queries = torch.zeros((B, 200, 200, model.emb_dim), device='cuda')

        K = batch['K'].cuda(non_blocking=True)
        R = batch['R'].cuda(non_blocking=True)
        t = batch['t'].cuda(non_blocking=True)
        R_curr2prev = batch['R_curr2prev'].cuda(non_blocking=True)
        t_curr2prev = batch['t_curr2prev'].cuda(non_blocking=True)
        has_prev = batch['has_prev'].cuda(non_blocking=True)
        if history_BEV is None:
            has_prev = torch.zeros_like(has_prev)

        bev_gt = batch['channels'].cuda(non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            history, logits = model(history_BEV, BEV_queries, images, has_prev, R_curr2prev, t_curr2prev, K, R, t)
            loss = criterion(logits, bev_gt)

        history_BEV = history.detach()

        running_loss += loss.item()
        metric.update(logits, bev_gt)

    mean_iou, class_ious = metric.compute()
    return running_loss / len(dataloader), mean_iou, class_ious


if __name__ == '__main__':

    batch_size = 1 #used only 1 batch so that history Bev can be updated without skipping a timeframe.
    accum_steps = 4 #accumulate to make up for the 1 batch size
    epochs = 20
    lr = 2e-4
    num_classes = 5

    dataset = NuScenesFrameLoader()
    testset = NuScenesFrameLoader(is_test=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    model = BEVformer(freeze_backbone=False).cuda()

    optimizer = torch.optim.AdamW(
[
            {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()), "lr": 1e-5},
            {"params": model.encoder.parameters(), "lr": 2e-4},
            {"params": model.decoder.parameters(), "lr": 2e-4},
        ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    pos_weight = torch.tensor([10.0, 20.0, 20.0, 20.0, 20.0], device='cuda').view(1, 5, 1, 1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    metric = BEVIoUMetric(num_classes=num_classes)

    best_miou = 0.0

    print('starting training')
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, loader, accum_steps, criterion, optimizer, scaler)
        val_loss, mean_iou, class_ious = validate(model, val_loader, criterion, metric)
        scheduler.step()
        elapsed = time.time() - start_time

        print(f"\nEpoch [{epoch}/{epochs}] ({elapsed:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  mIoU: {mean_iou * 100:.2f}%")
        for cls_idx, iou in enumerate(class_ious):
            print(f"    - Class {cls_idx} IoU: {iou * 100:.2f}%")

        # Save Best Model Checkpoint
        if mean_iou > best_miou:
            best_miou = mean_iou
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/bevformer_best.pth")
            print("  [Saved Best Checkpoint!]")