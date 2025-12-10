import csv
import itertools
import os
import shutil
import time

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.bright_dataset import BRIGHTDataset
from model.loss import CombinedLoss, FocalLoss, LovaszSoftmaxLoss
from model.u_net import UNet
from model.utils import (DEVICE, DATA_DIR, CHECKPOINT_DIR,
                         WEIGHT_DECAY, PIN_MEMORY, crop_tensor)

EXP_MAP_FILE = "grid_search_map.csv"
RESULTS_FILE = "grid_search_metrics.csv"
TEMP_DIR = "temp_results"

CHANNEL_MASKS = ["100000", "111000", "111111", "000111"]
NUM_CLASSES = 4
MAX_EPOCHS = 100
PATIENCE = 15

BATCH_SIZES = [2, 8]
INPUT_SIZES = [372]
LEARNING_RATES = [3e-5, 3e-4]

LOSS_CONFIGS = [
    ("LovaszSoftmaxLoss", None),
    ("FocalLoss", None),
    ("CombinedLoss_0.25", 0.25),
    ("CombinedLoss_0.75", 0.75),
]

NUM_DL_WORKERS = os.cpu_count()


def get_loss_function(name, lovasz_weight, class_weights, device):
    if name == "LovaszSoftmaxLoss":
        return LovaszSoftmaxLoss().to(device)
    elif name == "FocalLoss":
        return FocalLoss(alpha=class_weights).to(device)
    elif "CombinedLoss" in name:
        return CombinedLoss(weight=class_weights, lovasz_weight=lovasz_weight).to(device)
    else:
        raise ValueError(f"Unknown loss: {name}")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch_idx):
    model.train()
    running_loss = 0.0
    intersection_total = np.zeros(NUM_CLASSES)
    union_total = np.zeros(NUM_CLASSES)

    pbar = tqdm(dataloader, desc=f"\t\t- Train Epoch {epoch_idx}", leave=False)

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

        current_loss = loss.item()
        running_loss += current_loss

        preds = torch.argmax(outputs, dim=1)

        for c in range(NUM_CLASSES):
            pred_mask = (preds == c)
            true_mask = (masks_cropped == c)
            intersection_total[c] += (pred_mask & true_mask).sum().item()
            union_total[c] += (pred_mask | true_mask).sum().item()

        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    iou_per_class = intersection_total / (union_total + 1e-8)
    mean_iou = iou_per_class.mean()

    metrics = {
        "loss": running_loss / len(dataloader),
        "miou": mean_iou
    }
    for c in range(NUM_CLASSES):
        metrics[f"iou_class_{c}"] = iou_per_class[c]

    return metrics


def validate_epoch(model, dataloader, criterion, device, epoch_idx):
    model.eval()
    running_loss = 0.0
    intersection_total = np.zeros(NUM_CLASSES)
    union_total = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc=f"\t\t- Val Epoch {epoch_idx}", leave=False):
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


def update_map_status(exp_id):
    rows = []
    if not os.path.exists(EXP_MAP_FILE):
        return

    with open(EXP_MAP_FILE, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    updated = False
    for row in rows:
        if int(row[0]) == exp_id:
            row[-1] = "True"
            updated = True
            break

    if updated:
        with open(EXP_MAP_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)


def run_experiment(args):
    exp_id, config, train_names, val_names = args
    batch_size, input_size, lr, (loss_name, lovasz_weight), mask = config

    print(
        f"\nState: Running Exp {exp_id} | BS={batch_size} | Size={input_size} | LR={lr} | Loss={loss_name} | Mask={mask}")

    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_file = os.path.join(TEMP_DIR, f"exp_{exp_id}.csv")

    if os.path.exists(temp_file):
        os.remove(temp_file)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

    train_dataset = BRIGHTDataset(DATA_DIR, train_names, target_size=input_size, channel_mask=mask,
                                  augmentations=train_transform, mode="train")
    val_dataset = BRIGHTDataset(DATA_DIR, val_names, target_size=input_size, channel_mask=mask, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_DL_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size // 2), shuffle=False,
                            num_workers=NUM_DL_WORKERS, pin_memory=PIN_MEMORY)

    in_channels = mask.count("1")
    model = UNet(in_channels=in_channels, out_channels=NUM_CLASSES).to(DEVICE)

    class_weights = torch.FloatTensor([0.25, 1.0, 2.0, 2.0]).to(DEVICE)
    criterion = get_loss_function(loss_name, lovasz_weight, class_weights, DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_miou = 0.0
    patience_counter = 0

    weights_dir = os.path.join(CHECKPOINT_DIR, f"exp_{exp_id}")
    os.makedirs(weights_dir, exist_ok=True)

    results_buffer = []

    epoch_pbar = tqdm(range(MAX_EPOCHS), desc=f"\t- Exp {exp_id} Progress", leave=False)

    for epoch in epoch_pbar:
        start_time = time.time()

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch + 1)
        val_metrics = validate_epoch(model, val_loader, criterion, DEVICE, epoch + 1)

        duration = time.time() - start_time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        scheduler.step()

        row = [exp_id, epoch + 1, timestamp, duration,
               val_metrics["loss"], val_metrics["miou"]] + \
              [val_metrics[f"iou_class_{c}"] for c in range(NUM_CLASSES)] + \
              [train_metrics["loss"], train_metrics["miou"]] + \
              [train_metrics[f"iou_class_{c}"] for c in range(NUM_CLASSES)]

        results_buffer.append(row)

        epoch_pbar.set_postfix({
            "Val mIoU": f"{val_metrics['miou']:.4f}",
            "Train Loss": f"{train_metrics['loss']:.4f}"
        })

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f" Early stopping at epoch {epoch + 1}")
            break

    with open(temp_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results_buffer)

    update_map_status(exp_id)

    del model
    del optimizer
    del criterion
    del train_loader
    del val_loader
    torch.cuda.empty_cache()


def merge_results():
    print("Merging results...")
    all_rows = []

    if os.path.exists(TEMP_DIR):
        for filename in os.listdir(TEMP_DIR):
            if filename.endswith(".csv"):
                filepath = os.path.join(TEMP_DIR, filename)
                with open(filepath, mode="r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        all_rows.append(row)

    all_rows.sort(key=lambda x: (int(x[0]), int(x[1])))

    header = ["exp_id", "epoch", "timestamp", "duration",
              "val_loss", "val_miou"] + [f"val_iou_{c}" for c in range(NUM_CLASSES)] + \
             ["train_loss", "train_miou"] + [f"train_iou_{c}" for c in range(NUM_CLASSES)]

    with open(RESULTS_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    print("Merge complete.")


def main():
    os.makedirs(TEMP_DIR, exist_ok=True)

    grid = list(itertools.product(BATCH_SIZES, INPUT_SIZES, LEARNING_RATES, LOSS_CONFIGS, CHANNEL_MASKS))

    post_event_dir = os.path.join(DATA_DIR, "post-event")
    all_images = [f for f in os.listdir(post_event_dir) if f.endswith(".tif")]
    train_size = 2000
    np.random.seed(1810)
    np.random.shuffle(all_images)
    train_names = all_images[:train_size]
    val_names = all_images[train_size:train_size + 50]

    resume_mode = os.path.exists(EXP_MAP_FILE)
    tasks_to_run = []

    last_exp_id = -1

    if not resume_mode:
        print("Starting new grid search. Creating map file...")
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

        with open(EXP_MAP_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["exp_id", "input_size", "batch_size", "lr", "loss_name", "lovasz_weight", "channel_mask",
                             "is_finished"])
            for i, config in enumerate(grid):
                writer.writerow([i, config[1], config[0], config[2], config[3][0], config[3][1], config[4], "False"])
                tasks_to_run.append((i, config))
    else:
        print("Resuming from existing map file...")
        with open(EXP_MAP_FILE, mode="r") as f:
            reader = csv.reader(f)
            header = next(reader)
            for i, row in enumerate(reader):
                is_finished = row[-1]
                exp_id = int(row[0])

                if is_finished == "True":
                    last_exp_id = exp_id
                    continue

                partial_res_file = os.path.join(TEMP_DIR, f"exp_{exp_id}.csv")
                if os.path.exists(partial_res_file):
                    os.remove(partial_res_file)
                    print(f"Resetting incomplete experiment {exp_id}")

                tasks_to_run.append((exp_id, grid[i]))

    print(f"Total combinations: {len(grid)}")
    print(f"Last completed experiment ID: {last_exp_id}" if last_exp_id >= 0 else "No completed experiments found.")
    print(f"Tasks remaining to run: {len(tasks_to_run)}")

    if len(tasks_to_run) == 0:
        print("All tasks completed.")
        merge_results()
        return

    final_tasks = []
    for exp_id, config in tasks_to_run:
        final_tasks.append((exp_id, config, train_names, val_names))

    for task_args in tqdm(final_tasks, desc="Grid Search Total Progress"):
        run_experiment(task_args)

    merge_results()


if __name__ == "__main__":
    main()
