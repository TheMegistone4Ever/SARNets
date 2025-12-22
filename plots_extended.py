import os
from concurrent.futures import ProcessPoolExecutor
from traceback import print_exc
from typing import cast, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.optimize import curve_fit

DEEPSEEK_PALETTE = ["#3b6cf5", "#a4c2ff", "#b8b8b8", "#e0e0e0", "#d1e3ff"]
DARK_ACCENT = "#2c3e50"
HATCHES = ["//", "", "\\\\", "///"]

plt.rcParams["figure.dpi"] = 250
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=DEEPSEEK_PALETTE)


def load_data():
    df_metrics = pd.read_csv("grid_search_metrics.csv")
    df_map = pd.read_csv("grid_search_map.csv")
    df_map["channel_mask"] = df_map["channel_mask"].astype(str).apply(lambda x: x.zfill(6))
    df_map["mask_ones"] = df_map["channel_mask"].apply(lambda x: x.count("1"))
    df = pd.merge(df_metrics, df_map, on="exp_id")
    df["gflops"] = df["epoch"] * df["mask_ones"] * (df["input_size"] ** 2) * df["batch_size"] * 1e-9
    return df


def prepare_dirs():
    os.makedirs("plots/by_experiment", exist_ok=True)


def plot_1_dynamics(df):
    plt.figure(figsize=(12, 10))
    cmap = LinearSegmentedColormap.from_list("deepseek_grad", [DEEPSEEK_PALETTE[0], DEEPSEEK_PALETTE[4]])
    norm = Normalize(vmin=df["exp_id"].min(), vmax=df["exp_id"].max())

    ax1 = plt.subplot(2, 1, 1)
    for eid in df["exp_id"].unique():
        sub = df[df["exp_id"] == eid]
        ax1.plot(sub["epoch"], sub["val_miou"], color=cmap(norm(eid)), alpha=.6, linewidth=1.5)
    ax1.set_title("Динаміка val mIoU")
    ax1.set_ylabel("mIoU")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    for eid in df["exp_id"].unique():
        sub = df[df["exp_id"] == eid]
        ax2.plot(sub["epoch"], sub["val_loss"], color=cmap(norm(eid)), alpha=.6, linewidth=1.5)
    ax2.set_title("Динаміка val Loss")
    ax2.set_xlabel("Епоха")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/1_dynamics_history.png", dpi=250)
    plt.close()


def plot_2_bars(df):
    best = df.loc[df.groupby("exp_id")["val_miou"].idxmax()]
    last = df.groupby("exp_id")["epoch"].max().reset_index()
    data = pd.merge(best, last, on="exp_id", suffixes=("_best", "_max"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

    ax1.bar(data["exp_id"].astype(str), data["epoch_max"],
            color=DEEPSEEK_PALETTE[0], edgecolor="black", hatch="//", alpha=0.9)
    ax1.set_title("Кількість епох до зупинки")
    ax1.set_ylabel("Епохи")
    ax1.tick_params(axis="x", rotation=90, labelsize=7)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    sorted_data = data.sort_values("val_miou", ascending=False)
    ax2.bar(sorted_data["exp_id"].astype(str), sorted_data["val_miou"],
            color=DEEPSEEK_PALETTE[1], edgecolor="black", alpha=0.9)
    ax2.set_title("Найкращий mIoU (відсортовано за значенням)")
    ax2.set_ylabel("mIoU")
    ax2.set_xlabel("ID експерименту")
    ax2.tick_params(axis="x", rotation=90, labelsize=7)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig("plots/2_experiment_bars.png", dpi=250)
    plt.close()


def plot_3_scatter(df):
    best = df.loc[df.groupby("exp_id")["val_miou"].idxmax()]
    plt.figure(figsize=(10, 8))
    plt.scatter(best["val_miou"], best["val_loss"],
                c=best["exp_id"], cmap="Blues", alpha=.8, edgecolors="black", s=70, linewidth=0.5)
    plt.title("Best mIoU vs Loss")
    plt.xlabel("Best val mIoU")
    plt.ylabel("Loss")
    plt.grid(True, alpha=.3, linestyle="--")
    plt.savefig("plots/3_best_miou_loss_scatter.png", dpi=250)
    plt.close()


def plot_4_overfitting(df):
    best = df.loc[df.groupby("exp_id")["val_miou"].idxmax()]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    x = np.arange(len(best))
    width = .35

    ax1.bar(x - width / 2, best["val_miou"], width, label="Val mIoU",
            color=DEEPSEEK_PALETTE[0], edgecolor="black", hatch="//")
    ax1.bar(x + width / 2, best["train_miou"], width, label="Train mIoU",
            color=DEEPSEEK_PALETTE[4], edgecolor="black")
    ax1.set_title("Порівняння Val vs Train mIoU")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    diff = best["train_miou"] - best["val_miou"]
    ax2.bar(x, diff, color=DEEPSEEK_PALETTE[2], edgecolor="black")
    ax2.set_title("Різниця (Train - Val) mIoU")
    ax2.set_ylabel("Delta mIoU")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/4_overfitting_analysis.png", dpi=250)
    plt.close()


def plot_5_individual(df):
    for eid in df["exp_id"].unique():
        path = f"plots/by_experiment/exp_{eid}"
        os.makedirs(path, exist_ok=True)
        sub = df[df["exp_id"] == eid].sort_values("epoch")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(sub["epoch"], sub["train_loss"], color=DEEPSEEK_PALETTE[2], linestyle="--", label="Train Loss")
        ax1.plot(sub["epoch"], sub["val_loss"], color=DEEPSEEK_PALETTE[0], linestyle="-", linewidth=2, label="Val Loss")
        ax1.set_title(f"Loss History - Exp {eid}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(sub["epoch"], sub["train_miou"], color=DEEPSEEK_PALETTE[2], linestyle="--", label="Train mIoU")
        ax2.plot(sub["epoch"], sub["val_miou"], color=DEEPSEEK_PALETTE[0], linestyle="-", linewidth=2, label="Val mIoU")
        ax2.set_title(f"mIoU History - Exp {eid}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{path}/1_history_metrics.png", dpi=250)
        plt.close()

        best_row = sub.loc[sub["val_miou"].idxmax()]
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = ["No Damage", "Minor", "Major", "Destroyed"]
        v_ious = [best_row[f"val_iou_{i}"] for i in range(4)]
        t_ious = [best_row[f"train_iou_{i}"] for i in range(4)]
        x = np.arange(4)

        ax.bar(x - .2, v_ious, .4, label="Val IoU",
               color=DEEPSEEK_PALETTE[0], edgecolor="black", hatch="\\\\")
        ax.bar(x + .2, t_ious, .4, label="Train IoU",
               color=DEEPSEEK_PALETTE[4], edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_title(f"IoU по класах - Exp {eid}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path}/2_class_iou_bars.png", dpi=250)
        plt.close()


def plot_6_ablation(df):
    best = df.loc[df.groupby("exp_id")["val_miou"].idxmax()]
    configs = [("lr", "LR"), ("loss_name", "Loss"), ("channel_mask", "Mask"), ("batch_size", "BS"),
               ("input_size", "Size")]

    for i, (col, title) in enumerate(configs, 1):
        plt.figure(figsize=(10, 6))
        unique_vals = sorted(best[col].unique())
        data_to_plot = [best[best[col] == val]["val_miou"] for val in unique_vals]

        box = plt.boxplot(data_to_plot, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))

        for idx, patch in enumerate(box["boxes"]):
            color = DEEPSEEK_PALETTE[idx % len(DEEPSEEK_PALETTE)]
            hatch = HATCHES[idx % len(HATCHES)]
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_hatch(hatch)
            patch.set_alpha(0.8)

        plt.xticks(range(1, len(unique_vals) + 1), unique_vals)
        plt.title(f"Залежність mIoU від {title}")
        plt.ylabel("Best val mIoU")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.savefig(f"plots/6_{i}_{col}_dependency.png", dpi=250)
        plt.close()


def scaling_func(x, a, k, l):
    return a * np.exp(-k * (x + l))


def plot_8_scaling(df, target_metric=lambda x: 1. - x):
    plt.figure(figsize=(12, 10))
    for eid in df["exp_id"].unique():
        sub = df[df["exp_id"] == eid].sort_values("gflops")
        plt.plot(sub["gflops"], target_metric(sub["val_miou"]), color=DEEPSEEK_PALETTE[2], alpha=.15, lw=.5)
        try:
            popt, _ = curve_fit(scaling_func, sub["gflops"], target_metric(sub["val_miou"]), p0=[.8, .01, 1],
                                bounds=(0, [1.5, 2, 10000]), maxfev=10000)
            plt.plot(sub["gflops"], scaling_func(sub["gflops"], *popt), color=DEEPSEEK_PALETTE[2], alpha=.3, lw=1)
        except (Exception,):
            pass

    unique_gflops = np.sort(df["gflops"].unique())
    mean_err = [target_metric(df[df["gflops"] == g]["val_miou"]).mean() for g in unique_gflops]
    best_err = [target_metric(df[df["gflops"] == g]["val_miou"]).min() for g in unique_gflops]
    worst_err = [target_metric(df[df["gflops"] == g]["val_miou"]).max() for g in unique_gflops]

    curves = [
        (mean_err, "Середня", DEEPSEEK_PALETTE[0]),
        (best_err, "Найкраща", "#27ae60"),
        (worst_err, "Найгірша", "#c0392b")
    ]

    for data, lbl, clr in curves:
        try:
            popt, _ = curve_fit(scaling_func, unique_gflops, data, p0=[.8, .01, 1], bounds=(0, [1.5, 2, 10000]),
                                maxfev=10000)
            plt.plot(unique_gflops, scaling_func(unique_gflops, *popt), color=clr, lw=3, linestyle="--",
                     label=f"{lbl}: $A e^{{-k(x+l)}}$")
            plt.scatter(unique_gflops, data, color=clr, s=30, zorder=5, edgecolor="white")
            plt.text(unique_gflops[-1], scaling_func(unique_gflops[-1], *popt), f" A={popt[0]:.2f}, k={popt[1]:.4f}",
                     color=clr, fontsize=9, fontweight="bold", va="center")
        except (Exception,):
            pass

    plt.title("Закон масштабування: Помилка від GFLOPS", fontsize=14)
    plt.xlabel("GFLOPS (log)")
    plt.ylabel("Error Rate (log)")
    plt.legend()
    plt.grid(True, which="both", alpha=.15)
    plt.tight_layout()
    plt.savefig("plots/8_scaling_law_analysis.png", dpi=250)
    plt.close()


def plot_7_parallel_coordinates(df):
    best = df.loc[df.groupby("exp_id")["val_miou"].idxmax()].copy()
    cols = ["lr", "batch_size", "input_size", "loss_name", "channel_mask", "val_miou"]
    labels = ["LR", "Batch", "Size", "Loss", "Mask", "mIoU"]
    plot_df = best[cols].copy()

    cat_mappings = dict()
    for col in ["loss_name", "channel_mask"]:
        cat_mappings[col] = sorted(plot_df[col].unique())
        plot_df[col] = plot_df[col].apply(lambda x: cat_mappings[col].index(x))

    fig, axes = plt.subplots(1, len(cols) - 1, sharey=False, figsize=(18, 8), gridspec_kw={"wspace": 0})
    norm = Normalize(vmin=plot_df["val_miou"].min(), vmax=plot_df["val_miou"].max())

    colormap = LinearSegmentedColormap.from_list("blue_grad", [DEEPSEEK_PALETTE[2], DEEPSEEK_PALETTE[0]])

    for i, ax in enumerate(axes):
        col_left, col_right = cols[i], cols[i + 1]
        y_left_min, y_left_max = plot_df[col_left].min(), plot_df[col_left].max()
        y_right_min, y_right_max = plot_df[col_right].min(), plot_df[col_right].max()

        def scale(val, v_min_in, v_max_in):
            return (val - v_min_in) / (v_max_in - v_min_in + 1e-8)

        for idx in plot_df.index:
            y_l = scale(plot_df.loc[idx, col_left], y_left_min, y_left_max)
            y_r = scale(plot_df.loc[idx, col_right], y_right_min, y_right_max)
            ax.plot([0, 1], [y_l, y_r], color=colormap(norm(plot_df.loc[idx, "val_miou"])), alpha=.5, lw=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(-.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if i > 0: ax.spines["left"].set_visible(False)
        ax.set_xticks(list())
        ax.set_yticks(list())

        for side, col_name, v_min, v_max in [(0, col_left, y_left_min, y_left_max),
                                             (1, col_right, y_right_min, y_right_max)]:
            if side == 1 and i < len(axes) - 1: continue
            vals = cat_mappings[col_name] if col_name in cat_mappings else np.linspace(v_min, v_max, 5)
            for v in vals:
                y_pos = scale(cat_mappings[col_name].index(v) if col_name in cat_mappings else v, v_min, v_max)
                ax.text(side, y_pos, f" {v}" if side == 1 else f"{v} ", ha="left" if side == 1 else "right",
                        va="center", fontsize=8)
        ax.text(0, 1.1, labels[i], ha="center", va="bottom", fontweight="bold", fontsize=10)
        if i == len(axes) - 1: ax.text(1, 1.1, labels[i + 1], ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.suptitle("Паралельні координати гіперпараметрів (mIoU кольором)", fontsize=15, y=1.02)
    plt.savefig("plots/7_parallel_coordinates.png", dpi=250, bbox_inches="tight")
    plt.close()


def plot_9_loss_stability(df):
    best = df.loc[df.groupby("exp_id")["val_miou"].idxmax()]
    losses = sorted(best["loss_name"].unique())
    data = [best[best["loss_name"] == ln]["val_miou"].values for ln in losses]

    plt.figure(figsize=(12, 7))
    v_parts = plt.violinplot(data, showmeans=True, showextrema=True)

    bodies = cast(List, cast(object, v_parts["bodies"]))
    for i, pc in enumerate(bodies):
        pc.set_facecolor(DEEPSEEK_PALETTE[i % len(DEEPSEEK_PALETTE)])
        pc.set_edgecolor("black")
        pc.set_hatch(HATCHES[i % len(HATCHES)])
        pc.set_alpha(.8)

    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = v_parts[partname]
        vp.set_edgecolor(DARK_ACCENT)
        vp.set_linewidth(1.5)

    plt.xticks(range(1, len(losses) + 1), losses, rotation=15)
    plt.title("Стабільність mIoU для функцій втрат")
    plt.ylabel("Best val mIoU")
    plt.grid(axis="y", linestyle="--", alpha=.5)
    plt.tight_layout()
    plt.savefig("plots/9_loss_stability_violin.png", dpi=250)
    plt.close()


def plot_10_top_10_class_iou(df):
    best = df.loc[df.groupby("exp_id")["val_miou"].idxmax()].sort_values("val_miou", ascending=False).head(10)
    classes = ["No Damage", "Minor", "Major", "Destroyed"]
    x, width = np.arange(len(best)), .2

    plt.figure(figsize=(16, 8))
    for i in range(4):
        plt.bar(x + i * width, best[f"val_iou_{i}"], width, label=classes[i],
                color=DEEPSEEK_PALETTE[i % 4],
                edgecolor="black",
                hatch=HATCHES[i % 4])

    plt.xticks(x + width * 1.5, [f"Exp {int(eid)}" for eid in best["exp_id"]], rotation=30)
    plt.title("Покласова IoU для ТОП-10 моделей")
    plt.ylabel("IoU Score")
    plt.legend()
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig("plots/10_top_10_class_iou.png", dpi=250)
    plt.close()


def plot_11_grouped_bars(df):
    best = df.loc[df.groupby("exp_id")["val_miou"].idxmax()]
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    def plot_grouped(ax, x_col, hue_col, y_col, title, x_label):
        grouped = best.groupby([x_col, hue_col])[y_col].mean().unstack()
        x = np.arange(len(grouped.index))
        n_bars = len(grouped.columns)
        width = .8 / n_bars
        base_offset = - (width * n_bars) / 2 + width / 2

        for i, col in enumerate(grouped.columns):
            offset = base_offset + i * width
            ax.bar(x + offset, grouped[col], width, label=f"{hue_col}: {col}",
                   color=DEEPSEEK_PALETTE[i % len(DEEPSEEK_PALETTE)],
                   edgecolor="black",
                   hatch=HATCHES[i % len(HATCHES)])

        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        ax.set_xlabel(x_label)
        y_min = max(0, grouped.min().min() - .05)
        y_max = min(1., grouped.max().max() + .05)
        ax.set_ylim(y_min, y_max)
        ax.legend(title=hue_col, loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(axis="y", alpha=.3)

    plot_grouped(axes[0], "batch_size", "input_size", "val_miou",
                 "mIoU від BS та Size (групування по BS)", "Batch Size")
    plot_grouped(axes[1], "batch_size", "lr", "val_miou",
                 "mIoU від BS та LR (групування по BS)", "Batch Size")
    plot_grouped(axes[2], "batch_size", "mask_ones", "val_miou",
                 "mIoU від BS та Channel Count (групування по BS)", "Batch Size")
    plt.tight_layout()
    plt.savefig("plots/11_grouped_params.png", dpi=250, bbox_inches="tight")
    plt.close()


def plot_12_interaction_heatmap(df):
    best = df.loc[df.groupby("exp_id")["val_miou"].idxmax()]
    pivot = best.pivot_table(index="loss_name", columns="channel_mask", values="val_miou", aggfunc="mean")
    plt.figure(figsize=(12, 8))

    im = plt.imshow(pivot, cmap="Blues")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)

    mean_val = pivot.values.mean()
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            text_color = "white" if val > mean_val else "black"
            plt.text(j, i, f"{val:.3f}", ha="center", va="center",
                     color=text_color, fontweight="bold")

    plt.title("Взаємодія: Loss Function vs Mask (Mean mIoU)")
    plt.colorbar(im, label="mIoU")
    plt.tight_layout()
    plt.savefig("plots/12_interaction_heatmap.png", dpi=250)
    plt.close()


def run_plot_safe(func, df):
    try:
        print(f"Start:\t{func.__name__}")
        func(df)
        print(f"Done:\t{func.__name__}")
        return True
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        print_exc()
        return False


def main():
    prepare_dirs()
    df = load_data()

    tasks = [
        plot_5_individual,

        plot_1_dynamics,
        plot_2_bars,
        plot_3_scatter,
        plot_4_overfitting,
        plot_6_ablation,
        plot_7_parallel_coordinates,
        plot_8_scaling,
        plot_9_loss_stability,
        plot_10_top_10_class_iou,
        plot_11_grouped_bars,
        plot_12_interaction_heatmap,
    ]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_plot_safe, func, df) for func in tasks]
        for f in futures:
            if not f.result():
                print("One of the plotting tasks failed.")


if __name__ == "__main__":
    main()
