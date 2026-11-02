"""
Collect every experiment's outputs from a run directory and produce:

  <output-dir>/RESULTS.md      all results tables + figure links
  <assets-dir>/*.png           charts and sample images for the README
  README.md (optional)         the block between
                                   <!-- RESULTS:START --> and <!-- RESULTS:END -->
                               is replaced with the same tables + figures,
                               so the README never needs hand-copied numbers

Missing experiments are shown as "not run yet" rather than failing, so you
can run this after any subset of steps and re-run it as more finish.

Expected layout under --run-dir (the folder names the notebook uses):
  baseline_full/ few_shot_k20/ semi_supervised_k20/   38-class classifier runs
  baseline_apple_only/                                 4-class classifier
  cross_dataset_eval/                                  PlantVillage -> PlantDoc
  incremental/                                         class-incremental results
  unet/                                                U-Net segmentation
  gan_cedar_rust/                                      DCGAN samples
  gradcam_boxes/                                       Grad-CAM visualisations

Example (Colab):
    !python -m src.utils.generate_report --run-dir {RUN_DIR} \
        --output-dir {RUN_DIR}/report --assets-dir {RUN_DIR}/report/assets \
        --readme README.md
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Palette: validated categorical slots + a single-hue blue ramp for magnitude.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#eef4fc", "#86b6ef", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
NOT_RUN = "not run yet"


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "—"


def bar_chart(labels, series, title, path, ylabel="Score", ylim=(0, 1.22)):
    """Grouped bars, thin, value labels on top (few bars, so direct labels are fine)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    n = len(series)
    width = 0.7 / n
    for s_i, (name, values, color) in enumerate(series):
        xs = [i + (s_i - (n - 1) / 2) * width for i in range(len(labels))]
        bars = ax.bar(xs, values, width=width * 0.92, color=color, label=name)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # headroom above 1.0 is for the legend
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    if n > 1:
        ax.legend(frameon=False, loc="upper right", ncol=n)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- sections

CLASSIFIER_RUNS = [
    ("Full supervised", "baseline_full", "all", BLUE),
    ("Few-shot", "few_shot_k20", "20", ORANGE),
    ("Semi-supervised", "semi_supervised_k20", "20 + pseudo-labels", AQUA),
]


def classifier_section(run_dir, assets, figs):
    rows, labels, accs, f1s, curves = [], [], [], [], []
    n_classes = None
    for label, folder, k, color in CLASSIFIER_RUNS:
        hist = load_json(run_dir / folder / "history.json")
        classes = load_json(run_dir / folder / "classes.json")
        if classes:
            n_classes = len(classes)
        if not hist:
            rows.append(f"| {label} | {k} | {NOT_RUN} | — | — |")
            continue
        best = max(hist, key=lambda h: h["macro_f1"])
        cfg = load_json(run_dir / folder / "config.json") or {}
        if cfg.get("mode") in ("few_shot", "semi_supervised") and cfg.get("k_shot"):
            k = str(cfg["k_shot"]) + (" + pseudo-labels" if cfg["mode"] == "semi_supervised" else "")
        done = f"{len(hist)}/{cfg['epochs']}" if "epochs" in cfg else str(len(hist))
        rows.append(f"| {label} | {k} | {fmt(best['accuracy'])} | {fmt(best['macro_f1'])} | {done} |")
        labels.append(label)
        accs.append(best["accuracy"])
        f1s.append(best["macro_f1"])
        curves.append((label, [h["epoch"] for h in hist], [h["macro_f1"] for h in hist], color))

    if labels:
        bar_chart(labels, [("Accuracy", accs, BLUE), ("Macro-F1", f1s, ORANGE)],
                  "Validation scores by training mode", assets / "classifier_modes.png")
        figs.append(("classifier_modes.png", "Validation accuracy and macro-F1 by training mode"))

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for label, xs, ys, color in curves:
            ax.plot(xs, ys, color=color, linewidth=2, label=label)
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=TEXT_2, va="center")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation macro-F1")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title("Validation macro-F1 per epoch", loc="left")
        if len(curves) > 1:
            ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(assets / "training_curves.png", dpi=160)
        plt.close(fig)
        figs.append(("training_curves.png", "Validation macro-F1 per epoch"))

    title = f"{n_classes}-class PlantVillage" if n_classes else "PlantVillage"
    table = ("| Mode | Labeled images / class | Val accuracy | Val macro-F1 | Epochs run |\n"
             "|------|------------------------|--------------|--------------|------------|\n"
             + "\n".join(rows))
    return title, table


def cross_dataset_section(run_dir, assets, figs):
    folder = run_dir / "cross_dataset_eval"
    r = load_json(folder / "cross_dataset_results.json")
    if not r:
        return ("| Test set | Accuracy | Macro-F1 |\n|---|---|---|\n"
                f"| PlantVillage (in-distribution) | {NOT_RUN} | — |\n| PlantDoc (field photos) | {NOT_RUN} | — |")
    pv, pd_ = r["plantvillage_in_distribution"], r["plantdoc_out_of_distribution"]
    bar_chart(["PlantVillage\n(lab photos)", "PlantDoc\n(field photos)"],
              [("Accuracy", [pv["accuracy"], pd_["accuracy"]], BLUE),
               ("Macro-F1", [pv["macro_f1"], pd_["macro_f1"]], ORANGE)],
              "Same model, lab vs. field photos", assets / "cross_dataset.png")
    figs.append(("cross_dataset.png", "Accuracy drop from lab photos to real field photos"))
    for name in ("confusion_plantvillage.png", "confusion_plantdoc.png"):
        if (folder / name).exists():
            shutil.copy(folder / name, assets / name)
            figs.append((name, name.replace("_", " ").replace(".png", "")))
    return ("| Test set | Accuracy | Macro-F1 |\n|---|---|---|\n"
            f"| PlantVillage (in-distribution) | {fmt(pv['accuracy'])} | {fmt(pv['macro_f1'])} |\n"
            f"| PlantDoc (field photos) | {fmt(pd_['accuracy'])} | {fmt(pd_['macro_f1'])} |\n\n"
            f"Generalization gap: **{r['accuracy_gap']:.3f}** accuracy, **{r['macro_f1_gap']:.3f}** macro-F1. "
            f"*{r.get('note', '')}*")


def incremental_section(run_dir, assets, figs):
    r = load_json(run_dir / "incremental" / "incremental_results.json")
    if not r:
        return ("| Strategy | Final avg. accuracy | Avg. forgetting (lower is better) |\n|---|---|---|\n"
                f"| Naive fine-tuning | {NOT_RUN} | — |\n| Replay + distillation | {NOT_RUN} | — |")
    names = {"naive": "Naive fine-tuning", "replay_distill": "Replay + distillation"}
    present = [k for k in ("naive", "replay_distill") if k in r]

    bar_chart([names[k] for k in present],
              [("Final avg. accuracy", [r[k]["final_average_accuracy"] for k in present], BLUE),
               ("Avg. forgetting", [r[k]["average_forgetting"] for k in present], ORANGE)],
              "Class-incremental learning: accuracy vs. forgetting", assets / "incremental.png")
    figs.append(("incremental.png", "Final accuracy and forgetting for each strategy"))

    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.6), squeeze=False)
    for ax, k in zip(axes[0], present):
        m = r[k]["accuracy_matrix"]
        t = len(m)
        vals = [[(v if v is not None and v == v else float("nan")) for v in row] for row in m]
        ax.imshow(vals, cmap=BLUE_RAMP, vmin=0, vmax=1)
        for i in range(t):
            for j in range(t):
                v = vals[i][j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="#ffffff" if v > 0.6 else TEXT)
        ax.set_xticks(range(t))
        ax.set_yticks(range(t))
        ax.set_xticklabels([f"Task {j}" for j in range(t)])
        ax.set_yticklabels([f"After {i}" for i in range(t)])
        ax.set_title(names[k], loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Accuracy on each task after each training stage", x=0.02, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(assets / "incremental_matrix.png", dpi=160)
    plt.close(fig)
    figs.append(("incremental_matrix.png", "Per-task accuracy matrix"))

    rows = "\n".join(
        f"| {names[k]} | {fmt(r[k]['final_average_accuracy'])} | {fmt(r[k]['average_forgetting'])} |"
        for k in present)
    return ("| Strategy | Final avg. accuracy | Avg. forgetting (lower is better) |\n|---|---|---|\n" + rows)


def segmentation_section(run_dir, assets, figs):
    r = load_json(run_dir / "unet" / "seg_results.json")
    if (run_dir / "unet" / "seg_samples.png").exists():
        shutil.copy(run_dir / "unet" / "seg_samples.png", assets / "seg_samples.png")
        figs.append(("seg_samples.png", "U-Net leaf segmentation: input, ground truth, prediction"))
    if not r:
        return f"| Val IoU | Val Dice |\n|---|---|\n| {NOT_RUN} | — |"
    return (f"| Val IoU | Val Dice | Train / val images |\n|---|---|---|\n"
            f"| {fmt(r['val_iou'])} | {fmt(r['val_dice'])} | {r.get('n_train', '?')} / {r.get('n_val', '?')} |")


def samples_section(run_dir, assets, figs):
    gan = run_dir / "gan_cedar_rust" / "real_vs_generated.png"
    if gan.exists():
        shutil.copy(gan, assets / "gan_real_vs_generated.png")
        figs.append(("gan_real_vs_generated.png", "DCGAN: real Cedar apple rust (top) vs generated (bottom)"))
    cams = sorted((run_dir / "gradcam_boxes").glob("*.png"))[:3] if (run_dir / "gradcam_boxes").exists() else []
    for i, c in enumerate(cams):
        shutil.copy(c, assets / f"gradcam_{i}.png")
        figs.append((f"gradcam_{i}.png", f"Grad-CAM heatmap and pseudo bounding box ({c.stem})"))


def precision_note(run_dir):
    """Which runs trained with --amp. A run whose outputs carry no "amp"
    field predates the flag, and that code only trained in fp32."""
    runs = [(folder, load_json(run_dir / folder / "config.json"))
            for _, folder, _, _ in CLASSIFIER_RUNS + [("", "baseline_apple_only", "", "")]]
    runs.append(("unet", load_json(run_dir / "unet" / "seg_results.json")))
    inc = load_json(run_dir / "incremental" / "incremental_results.json")
    runs.append(("incremental", (inc or {}).get("config", {}) if inc else None))
    amp = [name for name, cfg in runs if cfg is not None and cfg.get("amp")]
    fp32 = [name for name, cfg in runs if cfg is not None and not cfg.get("amp")]
    if not amp and not fp32:
        return None
    if not amp:
        return "*Training precision: fp32 for every run above.*"
    parts = [f"mixed precision (AMP, fp16) for {', '.join(f'`{n}`' for n in amp)}"]
    if fp32:
        parts.append(f"fp32 for {', '.join(f'`{n}`' for n in fp32)}")
    return f"*Training precision: {'; '.join(parts)}. Evaluation always runs in fp32.*"


# ---------------------------------------------------------------- assemble

def build_markdown(sections, figs, asset_prefix):
    fig_lookup = {name: caption for name, caption in figs}

    def img(name):
        return f"![{fig_lookup[name]}]({asset_prefix}{name})\n" if name in fig_lookup else ""

    cls_title, cls_table = sections["classifier"]
    md = [
        f"### Classifier: {cls_title}\n", cls_table, "",
        img("classifier_modes.png"), img("training_curves.png"),
        "### Cross-dataset generalization (Apple classes, PlantVillage → PlantDoc)\n",
        sections["cross"], "", img("cross_dataset.png"),
        img("confusion_plantvillage.png"), img("confusion_plantdoc.png"),
        "### Class-incremental few-shot learning (Apple classes)\n",
        sections["incremental"], "", img("incremental.png"), img("incremental_matrix.png"),
        "### Leaf segmentation (U-Net, Apple classes)\n",
        sections["segmentation"], "", img("seg_samples.png"),
    ]
    if sections.get("precision"):
        md += [sections["precision"], ""]
    extras = [img(n) for n, _ in figs if n.startswith(("gan_", "gradcam_"))]
    if extras:
        md += ["### Sample outputs\n"] + extras
    return "\n".join(line for line in md if line is not None)


def update_readme(readme_path, block):
    text = readme_path.read_text()
    if START not in text or END not in text:
        print(f"WARNING: {readme_path} has no {START} / {END} markers; README not updated.")
        return False
    new = re.sub(re.escape(START) + r".*?" + re.escape(END),
                 START + "\n" + block.replace("\\", "\\\\") + "\n" + END, text, flags=re.S)
    readme_path.write_text(new)
    return True


def run(args):
    run_dir, out_dir = Path(args.run_dir), Path(args.output_dir)
    assets = Path(args.assets_dir) if args.assets_dir else out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    figs = []
    sections = {
        "classifier": classifier_section(run_dir, assets, figs),
        "cross": cross_dataset_section(run_dir, assets, figs),
        "incremental": incremental_section(run_dir, assets, figs),
        "segmentation": segmentation_section(run_dir, assets, figs),
        "precision": precision_note(run_dir),
    }
    samples_section(run_dir, assets, figs)

    results_md = build_markdown(sections, figs, asset_prefix=f"{assets.name}/"
                                if assets.parent == out_dir else f"{assets}/")
    (out_dir / "RESULTS.md").write_text("# Results\n\n" + results_md)
    print(results_md)
    print(f"\nWrote {out_dir / 'RESULTS.md'} and {len(figs)} figure(s) to {assets}")

    if args.readme:
        readme_block = build_markdown(sections, figs, asset_prefix="assets/")
        if update_readme(Path(args.readme), readme_block):
            print(f"Updated results block in {args.readme} (figures referenced from assets/).")
            print("Copy the figures into the repo's assets/ folder before committing.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Same as RUN_DIR in the notebook")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--assets-dir", default=None, help="Where figures go (default: <output-dir>/assets)")
    p.add_argument("--readme", default=None, help="README.md to update between the RESULTS markers")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
