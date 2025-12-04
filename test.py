import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.bright_dataset import BRIGHTDataset
from model.u_net import UNet
from model.utils import (DEVICE, DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
                         INPUT_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, crop_tensor)

COLOR_MAP_RGB = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (255, 255, 0),
    3: (255, 0, 0)
}


def calculate_metrics(preds, targets, num_classes=4):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    correct = (preds == targets).sum()
    total = targets.size if hasattr(targets, 'size') else len(targets)
    overall_acc = 100.0 * correct / total

    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection[c] = int((pred_c & target_c).sum())
        union[c] = int((pred_c | target_c).sum())

    iou_per_class = intersection / (union + 1e-8)
    mean_iou = iou_per_class.mean()

    return {
        'overall_acc': overall_acc,
        'iou_per_class': iou_per_class,
        'mean_iou': mean_iou,
        'intersection': intersection,
        'union': union
    }


def normalize_sar_for_display(sar_image):
    p2, p98 = np.percentile(sar_image, (2, 98))
    sar_clipped = np.clip(sar_image, p2, p98)
    sar_norm = ((sar_clipped - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)
    return sar_norm


def mask_to_rgb(mask, color_map):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        rgb[mask == class_id] = color

    return rgb


def create_overlay(sar_image, mask_rgb, mask_indices, alpha=0.6):
    if len(sar_image.shape) == 2:
        sar_rgb = cv2.cvtColor(sar_image, cv2.COLOR_GRAY2RGB)
    else:
        sar_rgb = sar_image

    output = sar_rgb.copy().astype(np.float32)
    mask_float = mask_rgb.astype(np.float32)

    is_foreground = (mask_indices > 0)
    output[is_foreground] = (alpha * mask_float[is_foreground] +
                             (1 - alpha) * output[is_foreground])

    return output.astype(np.uint8)


def test_model(model, dataloader, device, in_channels, save_results=False, results_dir=None):
    model.eval()

    total_intersection = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    if save_results and results_dir:
        os.makedirs(results_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, masks, names) in enumerate(tqdm(dataloader, desc='[TEST] Processing', ncols=100)):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            masks_cropped = crop_tensor(masks, outputs)
            images_cropped = crop_tensor(images, outputs)

            preds = torch.argmax(outputs, dim=1)

            batch_metrics = calculate_metrics(preds.cpu().numpy(),
                                              masks_cropped.cpu().numpy(),
                                              NUM_CLASSES)

            total_intersection += batch_metrics['intersection']
            total_union += batch_metrics['union']
            total_correct += (preds == masks_cropped).sum().item()
            total_pixels += int(np.prod(masks_cropped.shape))

            if save_results and results_dir:
                for i in range(len(names)):
                    name = names[i].replace('_post_disaster.tif', '')

                    if in_channels == 6:
                        sar_tensor = images_cropped[i, 3:6]
                        vis_label = "SAR Post-Event"
                    else:
                        sar_tensor = images_cropped[i, 0:min(3, in_channels)]
                        vis_label = "SAR Image"

                    sar_raw = sar_tensor.mean(dim=0).cpu().numpy()
                    sar_norm = normalize_sar_for_display(sar_raw)

                    pred_mask = preds[i].cpu().numpy()
                    gt_mask = masks_cropped[i].cpu().numpy()

                    pred_rgb = mask_to_rgb(pred_mask, COLOR_MAP_RGB)
                    gt_rgb = mask_to_rgb(gt_mask, COLOR_MAP_RGB)

                    overlay_pred = create_overlay(sar_norm, pred_rgb, pred_mask, alpha=0.5)

                    sar_bgr = cv2.cvtColor(sar_norm, cv2.COLOR_GRAY2BGR)
                    gt_bgr = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
                    pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
                    overlay_bgr = cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR)

                    combined = np.hstack([sar_bgr, gt_bgr, pred_bgr, overlay_bgr])

                    target_h = INPUT_SIZE
                    scale = target_h / combined.shape[0]
                    target_w = int(combined.shape[1] * scale)

                    combined_resized = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale_font = 0.5
                    thick = 1
                    text_color = (255, 255, 255)

                    panel_w = target_w // 4

                    labels = [vis_label, 'Ground Truth', 'Prediction', 'Overlay']
                    for idx, lbl in enumerate(labels):
                        cv2.putText(combined_resized, lbl, (idx * panel_w + 10, 25), font, scale_font, text_color,
                                    thick)

                    cv2.imwrite(os.path.join(results_dir, f'{name}_result.png'), combined_resized)

    overall_acc = 100.0 * total_correct / (total_pixels + 1e-8)
    iou_per_class = total_intersection / (total_union + 1e-8)
    mean_iou = iou_per_class.mean()

    return {
        'overall_acc': overall_acc,
        'iou_per_class': iou_per_class,
        'mean_iou': mean_iou
    }


def main():
    CHANNEL_MASK = '111111'
    IN_CHANNELS = CHANNEL_MASK.count('1')

    WEIGHTS_FOLDER = f"{CHECKPOINT_DIR}/ic_{IN_CHANNELS}_oc_{NUM_CLASSES}_bs_{BATCH_SIZE}_is_{INPUT_SIZE}_lr_{LEARNING_RATE}"
    results_dir = f"{RESULTS_DIR}/ic_{IN_CHANNELS}_oc_{NUM_CLASSES}_bs_{BATCH_SIZE}_is_{INPUT_SIZE}_lr_{LEARNING_RATE}"

    print("-" * 60)
    print("BRIGHT MODEL EVALUATION")
    print("-" * 60)
    print(f"[CONFIG] Device:         {DEVICE}")
    print(f"[CONFIG] Mask:           {CHANNEL_MASK}")
    print(f"[CONFIG] In Channels:    {IN_CHANNELS}")
    print(f"[CONFIG] Weights Folder: {WEIGHTS_FOLDER}")
    print(f"[CONFIG] Results Dir:    {results_dir}")

    post_event_dir = os.path.join(DATA_DIR, 'post-event')
    if not os.path.exists(post_event_dir):
        print(f"[ERROR] Data directory not found: {post_event_dir}")
        return

    all_images = [f for f in os.listdir(post_event_dir) if f.endswith('.tif')]
    print(f"[DATA] Found {len(all_images)} total images.")

    test_dataset = BRIGHTDataset(DATA_DIR, all_images, target_size=INPUT_SIZE,
                                 channel_mask=CHANNEL_MASK, mode='val')

    test_size = min(50, len(test_dataset))
    random.seed(42)
    test_indices = random.sample(range(len(test_dataset)), test_size)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    print(f"[DATA] Running inference on {test_size} random samples.")

    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    model = UNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)

    model_path = os.path.join(WEIGHTS_FOLDER, 'best_model_miou.pth')

    if not os.path.exists(model_path):
        print(f"[WARN] Best mIoU model not found. Trying best accuracy model...")
        model_path = os.path.join(WEIGHTS_FOLDER, 'best_model_acc.pth')

    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint not found at: {WEIGHTS_FOLDER}")
        return

    print(f"[MODEL] Loading weights from: {os.path.basename(model_path)}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    metrics = test_model(model, test_loader, DEVICE, IN_CHANNELS,
                         save_results=True, results_dir=results_dir)

    print("\n" + "-" * 60)
    print("TEST REPORT")
    print("-" * 60)
    print(f"Overall Accuracy:       {metrics['overall_acc']:.2f}%")
    print(f"Mean IoU (All Classes): {metrics['mean_iou']:.4f}")
    print("-" * 60)
    print(f"{'Class Name':<15} | {'IoU':<10}")
    print("-" * 60)

    class_names = ['0: No Damage', '1: Minor', '2: Major', '3: Destroyed']
    for i, name in enumerate(class_names):
        print(f"{name:<15} | {metrics['iou_per_class'][i]:.4f}")

    print("-" * 60)
    print(f"[INFO] Visualizations saved to: {results_dir}")
    print("-" * 60)


if __name__ == '__main__':
    main()
