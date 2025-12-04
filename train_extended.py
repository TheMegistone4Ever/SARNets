import csv
import itertools
import os
import time

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.bright_dataset import BRIGHTDataset
from model.loss import CombinedLoss, FocalLoss, LovaszSoftmaxLoss
from model.u_net import UNet
from model.utils import (DEVICE, DATA_DIR, CHECKPOINT_DIR, NUM_WORKERS,
                         WEIGHT_DECAY, PIN_MEMORY, crop_tensor)

EXP_MAP_FILE = "grid_search_map.csv"
RESULTS_FILE = "grid_search_metrics.csv"

CHANNEL_MASKS = ["100000", "111000", "100111", "111111", "000111"]
NUM_CLASSES = 4
MAX_EPOCHS = 100
PATIENCE = 15

BATCH_SIZES = [1, 2, 4, 8]
INPUT_SIZES = [372, 512, 1024]
LEARNING_RATES = [3e-5, 1e-4, 3e-4]

LOSS_CONFIGS = [
    ("LovaszSoftmaxLoss", None),
    ("FocalLoss", None),
    ("CombinedLoss_0.25", 0.25),
    ("CombinedLoss_0.50", 0.50),
    ("CombinedLoss_0.75", 0.75),
    ("CombinedLoss_1.00", 1.00),
]

print(f"Total combinations to try:"
      f" {len(BATCH_SIZES) * len(INPUT_SIZES) * len(LEARNING_RATES) * len(LOSS_CONFIGS) * len(CHANNEL_MASKS)}")


def get_loss_function(name, lovasz_weight, class_weights, device):
    if name == "LovaszSoftmaxLoss":
        return LovaszSoftmaxLoss().to(device)
    elif name == "FocalLoss":
        return FocalLoss(alpha=class_weights).to(device)
    elif "CombinedLoss" in name:
        return CombinedLoss(weight=class_weights, lovasz_weight=lovasz_weight).to(device)
    else:
        raise ValueError(f"Unknown loss: {name}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    intersection_total = np.zeros(NUM_CLASSES)
    union_total = np.zeros(NUM_CLASSES)

    pbar = tqdm(dataloader, desc="  Train Loop", leave=False, ncols=100)

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
            intersection_total[c] += (pred_mask & true_mask).sum().item()
            union_total[c] += (pred_mask | true_mask).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    iou_per_class = intersection_total / (union_total + 1e-8)
    mean_iou = iou_per_class.mean()

    metrics = {
        "loss": running_loss / len(dataloader),
        "miou": mean_iou
    }
    for c in range(NUM_CLASSES):
        metrics[f"iou_class_{c}"] = iou_per_class[c]

    return metrics


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    intersection_total = np.zeros(NUM_CLASSES)
    union_total = np.zeros(NUM_CLASSES)

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
                pred_mask = (preds == c)
                true_mask = (masks_cropped == c)
                intersection_total[c] += (pred_mask & true_mask).sum().item()
                union_total[c] += (pred_mask | true_mask).sum().item()

    iou_per_class = intersection_total / (union_total + 1e-8)
    mean_iou = iou_per_class.mean()

    metrics = {
        "loss": running_loss / len(dataloader),
        "miou": mean_iou
    }
    for c in range(NUM_CLASSES):
        metrics[f"iou_class_{c}"] = iou_per_class[c]

    return metrics


def main():
    accum_id = 0

    if not os.path.exists(EXP_MAP_FILE):
        with open(EXP_MAP_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["exp_id", "input_size", "batch_size", "lr", "loss_name", "lovasz_weight", "channel_mask"])

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ["exp_id", "epoch", "timestamp", "duration",
                      "val_loss", "val_miou"] + [f"val_iou_{c}" for c in range(NUM_CLASSES)] + \
                     ["train_loss", "train_miou"] + [f"train_iou_{c}" for c in range(NUM_CLASSES)]
            writer.writerow(header)

    grid = list(itertools.product(BATCH_SIZES, INPUT_SIZES, LEARNING_RATES, LOSS_CONFIGS, CHANNEL_MASKS))

    post_event_dir = os.path.join(DATA_DIR, 'post-event')
    all_images = [f for f in os.listdir(post_event_dir) if f.endswith('.tif')]
    train_size = 2000
    np.random.seed(1810)
    np.random.shuffle(all_images)
    train_names = all_images[:train_size]
    val_names = all_images[train_size:train_size + 50]

    for batch_size, input_size, lr, (loss_name, lovasz_weight), mask in tqdm(grid, desc="Grid Search"):
        with open(EXP_MAP_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([accum_id, input_size, batch_size, lr, loss_name, lovasz_weight, mask])

        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

        train_dataset = BRIGHTDataset(DATA_DIR, train_names, target_size=input_size, channel_mask=mask,
                                      augmentations=train_transform, mode='train')
        val_dataset = BRIGHTDataset(DATA_DIR, val_names, target_size=input_size, channel_mask=mask, mode='val')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size // 2), shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        in_channels = mask.count('1')
        model = UNet(in_channels=in_channels, out_channels=NUM_CLASSES).to(DEVICE)

        class_weights = torch.FloatTensor([0.25, 1.0, 2.0, 2.0]).to(DEVICE)
        criterion = get_loss_function(loss_name, lovasz_weight, class_weights, DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        best_miou = 0.0
        patience_counter = 0

        weights_dir = os.path.join(CHECKPOINT_DIR, f"exp_{accum_id}")
        os.makedirs(weights_dir, exist_ok=True)

        print(f"\n[START] Exp {accum_id}: Mask={mask}, BS={batch_size}, Size={input_size}, LR={lr}, Loss={loss_name}")

        for epoch in range(MAX_EPOCHS):
            start_time = time.time()

            train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)

            duration = time.time() - start_time
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            scheduler.step()

            print(f"Exp {accum_id} | Ep {epoch + 1}/{MAX_EPOCHS} | "
                  f"Train: L={train_metrics['loss']:.4f} IoU={train_metrics['miou']:.4f} | "
                  f"Val: L={val_metrics['loss']:.4f} IoU={val_metrics['miou']:.4f} | "
                  f"T={duration:.1f}s")

            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                row = [accum_id, epoch + 1, timestamp, duration,
                       val_metrics['loss'], val_metrics['miou']] + \
                      [val_metrics[f'iou_class_{c}'] for c in range(NUM_CLASSES)] + \
                      [train_metrics['loss'], train_metrics['miou']] + \
                      [train_metrics[f'iou_class_{c}'] for c in range(NUM_CLASSES)]
                writer.writerow(row)

            if val_metrics['miou'] > best_miou:
                best_miou = val_metrics['miou']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(weights_dir, 'best_model.pth'))
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        accum_id += 1

        del model
        del optimizer
        del criterion
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
