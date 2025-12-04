import os
import time

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.bright_dataset import BRIGHTDataset
from model.loss import CombinedLoss
from model.u_net import UNet
from model.utils import (DEVICE, DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE, LEARNING_RATE,
                         NUM_EPOCHS, INPUT_SIZE, NUM_CLASSES, NUM_WORKERS, WEIGHT_DECAY,
                         PIN_MEMORY, crop_tensor)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    intersection_total = np.zeros(NUM_CLASSES)
    union_total = np.zeros(NUM_CLASSES)

    pbar = tqdm(dataloader, desc='[TRAIN] Processing Batch', ncols=110)

    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        masks_cropped = crop_tensor(masks, outputs)

        loss = criterion(outputs, masks_cropped)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        for c in range(NUM_CLASSES):
            pred_mask = (preds == c)
            true_mask = (masks_cropped == c)

            inter = (pred_mask & true_mask).sum().item()
            union = (pred_mask | true_mask).sum().item()

            intersection_total[c] += inter
            union_total[c] += union

        iou_per_class = intersection_total / (union_total + 1e-8)
        running_miou = iou_per_class.mean()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'mIoU': f'{running_miou:.4f}'})

    epoch_loss = running_loss / len(dataloader)
    final_iou_per_class = intersection_total / (union_total + 1e-8)
    final_miou = final_iou_per_class.mean()

    return epoch_loss, final_miou


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    class_predictions = np.zeros(NUM_CLASSES, dtype=np.int64)
    intersection = np.zeros(NUM_CLASSES, dtype=np.int64)
    union = np.zeros(NUM_CLASSES, dtype=np.int64)

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            masks_cropped = crop_tensor(masks, outputs)

            loss = criterion(outputs, masks_cropped)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            for c in range(NUM_CLASSES):
                mask_c = (masks_cropped == c)
                pred_c = (preds == c)

                class_predictions[c] += pred_c.sum().item()

                intersection[c] += (pred_c & mask_c).sum().item()
                union[c] += (pred_c | mask_c).sum().item()

    iou_per_class = intersection / (union + 1e-8)
    mean_iou = iou_per_class.mean()

    print("-" * 65)
    print(f"{'VALIDATION REPORT':^65}")
    print("-" * 65)
    print(f"Mean IoU (All Classes): {mean_iou:.4f}")
    print("-" * 65)
    print(f"{'Class ID':<10} | {'IoU':<10} | {'Pixel Count':<15} | {'Distribution':<10}")
    print("-" * 65)

    total_preds = class_predictions.sum()
    for c in range(NUM_CLASSES):
        pct = 100.0 * class_predictions[c] / (total_preds + 1e-8)
        print(f"{c:<10} | {iou_per_class[c]:.4f}     | {class_predictions[c]:<15,} | {pct:.1f}%")
    print("-" * 65)

    return running_loss / len(dataloader), mean_iou


def main():
    CHANNEL_MASK = '111111'
    in_channels = CHANNEL_MASK.count('1')

    weights_dir = f"{CHECKPOINT_DIR}/ic_{in_channels}_oc_{NUM_CLASSES}_bs_{BATCH_SIZE}_is_{INPUT_SIZE}_lr_{LEARNING_RATE}"

    print(f"[INFO] Initializing training sequence on device: {DEVICE}")
    print(f"[CONFIG] Dataset Root:   {DATA_DIR}")
    print(f"[CONFIG] Checkpoint Dir: {weights_dir}")
    print(f"[CONFIG] Mask:           {CHANNEL_MASK}")
    print(f"[CONFIG] Input Channels: {in_channels}")

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

    post_event_dir = os.path.join(DATA_DIR, 'post-event')
    all_images = [f for f in os.listdir(post_event_dir) if f.endswith('.tif')]

    train_size = 2000
    np.random.seed(1810)
    np.random.shuffle(all_images)
    train_names = all_images[:train_size]
    val_names = all_images[train_size:train_size + 50]

    print(f"[DATA] Partitioning complete: {len(train_names)} training samples, {len(val_names)} validation samples.")

    train_dataset = BRIGHTDataset(DATA_DIR, train_names, target_size=INPUT_SIZE, channel_mask=CHANNEL_MASK,
                                  augmentations=train_transform, mode='train')
    val_dataset = BRIGHTDataset(DATA_DIR, val_names, target_size=INPUT_SIZE, channel_mask=CHANNEL_MASK, mode='val')

    print(f"[DATA] Training Set Size: {len(train_dataset)}")
    print(f"[DATA] Validation Set Size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=min(8, BATCH_SIZE),
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = UNet(in_channels=in_channels, out_channels=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Architecture: UNet (In: {in_channels}, Out: {NUM_CLASSES})")
    print(f"[MODEL] Parameters:   {total_params:,}")

    class_weights = torch.FloatTensor([0.25, 1.0, 2.0, 2.0]).to(DEVICE)
    criterion = CombinedLoss(weight=class_weights, lovasz_weight=0.75).to(DEVICE)
    print(f"[LOSS] CombinedLoss initialized (CE + 0.75*Lovasz)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    os.makedirs(weights_dir, exist_ok=True)

    best_iou = 0.0
    patience_counter = 0
    max_patience = 15

    print("STARTING TRAINING PROTOCOL")

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch [{epoch + 1}/{NUM_EPOCHS}]')

        train_loss, train_miou = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_miou = validate_epoch(model, val_loader, criterion, DEVICE)

        print(f"[METRICS] Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
        print(f"[METRICS] Val Loss:   {val_loss:.4f} | Val mIoU:   {val_miou:.4f}")
        print(f"[SYSTEM]  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        if val_miou > best_iou:
            best_iou = val_miou
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_miou,
                'train_loss': train_loss,
                'train_iou': train_miou,
            }
            torch.save(checkpoint, os.path.join(weights_dir, 'best_model_miou.pth'))
            print(f"[CHECKPOINT] Best model saved (Improved All-Class mIoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            print(f"[STATUS] Patience counter: {patience_counter}/{max_patience}")

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_miou,
            }
            torch.save(checkpoint, os.path.join(weights_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
            print(f"[CHECKPOINT] Periodic state saved (Epoch {epoch + 1})")

        if patience_counter >= max_patience:
            print(f"\n[WARNING] Early stopping triggered. No improvement for {max_patience} epochs.")
            break

    total_time = time.time() - start_time
    print("TRAINING COMPLETE")
    print(f"Total Duration: {total_time / 60:.2f} minutes")
    print(f"Best mIoU:      {best_iou:.4f}")


if __name__ == '__main__':
    main()
