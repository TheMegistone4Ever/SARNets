import os
from collections import Counter

import cv2
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")


def get_all_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]


def compute_histogram(img, bins=256):
    if len(img.shape) == 2:
        hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
        return [hist]
    else:
        hists = [cv2.calcHist([img], [i], None, [bins], [0, 256]) for i in range(3)]
        return hists


def add_gradient_fill(ax, x, y, color, label):
    line, = ax.plot(x, y, color=color, lw=1.5, label=label)

    rgb_color = mcolors.to_rgb(color)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gradient_cmap",
        [(0, (*rgb_color, 0.0)), (1, (*rgb_color, 0.4))]
    )

    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    xmin, xmax, ymin, ymax = ax.axis()
    img = ax.imshow(gradient, aspect="auto", origin="lower", cmap=cmap, extent=(xmin, xmax, ymin, ymax))

    path = ax.fill_between(x, y, 0, facecolor="none", edgecolor="none")
    img.set_clip_path(path.get_paths()[0], transform=ax.transData)


def process_targets(folder, output_dir):
    images = get_all_images(folder)
    if not images:
        return

    class_counts = Counter()
    for img_path in tqdm(images, desc="Processing Targets"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        unique, counts = np.unique(img, return_counts=True)
        class_counts.update(dict(zip(unique, counts)))

    sorted_classes = sorted(class_counts.keys())
    sorted_counts = [class_counts[k] for k in sorted_classes]

    plt.figure(figsize=(6, 4), dpi=300)
    plt.bar(sorted_classes, sorted_counts, color="gray", alpha=0.8)
    plt.title("Target Class Distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Pixel Count")
    plt.xticks(sorted_classes)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "target_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def process_images(folder, name, output_dir):
    images = get_all_images(folder)
    if not images:
        return

    all_hists = []
    for img_path in tqdm(images, desc=f"Processing {name}"):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        hists = compute_histogram(img)
        all_hists.append(hists)

    num_channels = len(all_hists[0])
    avg_hists = [np.mean([h[c] for h in all_hists], axis=0) for c in range(num_channels)]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    if name == "pre-event":
        colors = ["#E63946", "#52B69A", "#0077B6"]
        labels = ["Red Channel", "Green Channel", "Blue Channel"]
        for i, color in enumerate(colors):
            if i < len(avg_hists):
                y_vals = avg_hists[i].ravel()
                x_vals = np.arange(len(y_vals))
                add_gradient_fill(ax, x_vals, y_vals, color, labels[i])
        ax.set_title(f"RGB Intensity Distribution - {name}")

    else:
        y_vals = avg_hists[0].ravel()
        x_vals = np.arange(len(y_vals))
        add_gradient_fill(ax, x_vals, y_vals, "black", "SAR Intensity")
        ax.set_title(f"SAR Intensity Distribution - {name}")

    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 255)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{name}_histogram.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    BASE_PATH = "dataset/bright_dataset"
    OUTPUT_DIR = "plots"

    DATASETS = {
        "post-event": os.path.join(BASE_PATH, "post-event"),
        "pre-event": os.path.join(BASE_PATH, "pre-event"),
        "target": os.path.join(BASE_PATH, "target")
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, folder in DATASETS.items():
        if not os.path.exists(folder):
            continue

        if name == "target":
            process_targets(folder, OUTPUT_DIR)
        else:
            process_images(folder, name, OUTPUT_DIR)


if __name__ == "__main__":
    main()
