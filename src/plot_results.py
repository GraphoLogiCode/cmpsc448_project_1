"""Draw the figures that compare the classifiers.

    python src/plot_results.py

Reads   results/baseline.json and every results/runs/<run>/{metrics.json, history.json}
Writes  results/figures/
            accuracy_by_input.png        test accuracy for prompt only / answer only / prompt + answer (RQ1, RQ2)
            accuracy_by_task.png         test accuracy per task type, answer only
            confusion_matrices.png       which model is mistaken for which, answer only
            accuracy_held_out_task.png   a task seen in training against the same task held out (RQ3),
                                         drawn once runs with test_tasks exist
            training_<run>.png           loss and accuracy per epoch, for every run with a history.json

A new run appears in the figures as soon as its metrics.json exists; nothing here needs editing.
When several runs share a classifier and an input, the one with the best validation accuracy is drawn.
The numbers behind every figure are printed by `python src/evaluate.py`.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

import evaluate

ROOT = Path(__file__).resolve().parents[1]

# Chart colours. One colour per classifier, the same in every figure.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CLASSIFIER_COLOURS = {"TF-IDF baseline": "#2a78d6", "CNN": "#eb6834", "LSTM": "#1baf7a"}
EXTRA_COLOURS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("blues", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
INPUTS = [("prompt", "Prompt only"), ("answer", "Answer only"), ("both", "Prompt + answer")]
BAR_WIDTH, BAR_GAP, BAR_RADIUS = 18.0, 1.5, 3.0      # in points

plt.rcParams.update({"font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 8,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK_2, "ytick.labelcolor": MUTED, "text.color": INK})


def colour_of(classifier, classifiers):
    if classifier in CLASSIFIER_COLOURS:
        return CLASSIFIER_COLOURS[classifier]
    others = sorted(c for c in classifiers if c not in CLASSIFIER_COLOURS)
    return EXTRA_COLOURS[others.index(classifier) % len(EXTRA_COLOURS)]


def ordered(classifiers):
    known = [c for c in CLASSIFIER_COLOURS if c in classifiers]
    return known + sorted(c for c in classifiers if c not in CLASSIFIER_COLOURS)


def best_runs(results, held_out=False):
    """(classifier, input) -> the run with the best validation accuracy. held_out picks the RQ3 runs instead."""
    best = {}
    for r in results:
        if bool(r.get("test_tasks")) != held_out:
            continue
        key = (r["classifier"], r["input"]) if not held_out else (r["classifier"], r["input"], tuple(r["test_tasks"]))
        if key not in best or r["val"]["accuracy"] > best[key]["val"]["accuracy"]:
            best[key] = r
    return best


def style_axes(ax, y_max=100):
    ax.set_ylim(0, y_max * 1.08)
    ax.set_yticks(range(0, y_max + 1, 20))
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=8)


def titles(fig, title, subtitle):
    fig.text(0.02, 0.965, title, fontsize=11, fontweight="semibold", color=INK, va="top")
    fig.text(0.02, 0.895, subtitle, fontsize=8.5, color=INK_2, va="top")


def legend(fig, labels, colours, y=0.80):
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in colours]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.012, y + 0.05), ncol=len(labels), frameon=False,
               fontsize=8, labelcolor=INK_2, handlelength=0.9, handleheight=0.9, columnspacing=1.4)


def grouped_bars(ax, groups, series, values, colours, chance=None, label_chance=True):
    """Bars with a rounded top. values[s][g] is a share between 0 and 1, or None for a missing bar.

    The chance level is drawn as a line; its label goes in the margin to the right of the axes.
    """
    ax.set_xlim(-0.5, len(groups) - 0.5)
    style_axes(ax)
    box = ax.get_position()
    width_pt = box.width * ax.figure.get_figwidth() * 72
    height_pt = box.height * ax.figure.get_figheight() * 72
    x_per_pt = (ax.get_xlim()[1] - ax.get_xlim()[0]) / width_pt
    y_per_pt = (ax.get_ylim()[1] - ax.get_ylim()[0]) / height_pt
    bar_pt = min(BAR_WIDTH, (width_pt / len(groups) * 0.8 - (len(series) - 1) * BAR_GAP) / len(series))
    step = (bar_pt + BAR_GAP) * x_per_pt
    for s, name in enumerate(series):
        for g in range(len(groups)):
            v = values[name][g]
            if v is None:
                continue
            top = v * 100
            x0 = g + (s - (len(series) - 1) / 2) * step - bar_pt * x_per_pt / 2
            x1 = x0 + bar_pt * x_per_pt
            rx, ry = BAR_RADIUS * x_per_pt, min(BAR_RADIUS * y_per_pt, top)
            path = MplPath([(x0, 0), (x0, top - ry), (x0, top), (x0 + rx, top), (x1 - rx, top), (x1, top),
                            (x1, top - ry), (x1, 0), (x0, 0)],
                           [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
                            MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY])
            ax.add_patch(PathPatch(path, facecolor=colours[name], edgecolor="none", zorder=3))
            ax.text((x0 + x1) / 2, top + 2.5 * y_per_pt, "100" if top >= 99.95 else f"{top:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK_2, zorder=4)
    if chance is not None:
        ax.axhline(chance * 100, color=INK_2, linewidth=0.8, zorder=2)
        if label_chance:
            ax.annotate(f"chance\n{chance * 100:.1f}", xy=(1, chance * 100), xycoords=("axes fraction", "data"),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color=INK_2,
                        linespacing=1.1, annotation_clip=False)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("Test accuracy (%)", fontsize=8, color=MUTED)


def bar_figure(path, title, subtitle, groups, series, values, colours, chance):
    fig = plt.figure(figsize=(6.4, 3.7))
    many = len(series) > 1
    ax = fig.add_axes([0.085, 0.11, 0.84, 0.60 if many else 0.70])
    grouped_bars(ax, groups, series, values, colours, chance)
    titles(fig, title, subtitle)
    if many:
        legend(fig, series, [colours[s] for s in series])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def accuracy_by_input(results, out):
    best = best_runs(results)
    series = ordered({c for c, _ in best})
    colours = {c: colour_of(c, series) for c in series}
    values = {c: [best[(c, m)]["test"]["accuracy"] if (c, m) in best else None for m, _ in INPUTS] for c in series}
    any_run = next(iter(best.values()))
    subtitle = f"Share of {any_run['test']['rows']} test answers attributed to the right model"
    if len(series) == 1:
        subtitle += f", {series[0]}"
    bar_figure(out / "accuracy_by_input.png", "Test accuracy by input", subtitle, [label for _, label in INPUTS],
               series, values, colours, any_run["chance"])


def accuracy_by_task(results, out):
    best = {c: r for (c, m), r in best_runs(results).items() if m == "answer"}
    if not best:
        return
    series = ordered(set(best))
    colours = {c: colour_of(c, series) for c in series}
    tasks = sorted(next(iter(best.values()))["test"]["accuracy_per_task"])
    values = {c: [best[c]["test"]["accuracy_per_task"].get(t) for t in tasks] for c in series}
    subtitle = "Answer only, test split, by the task type of the prompt"
    if len(series) == 1:
        subtitle += f", {series[0]}"
    bar_figure(out / "accuracy_by_task.png", "Test accuracy by task", subtitle, [t.capitalize() for t in tasks],
               series, values, colours, next(iter(best.values()))["chance"])


def accuracy_held_out_task(results, out):
    """RQ3: accuracy on a task when it was in the training data, against the same task held out."""
    seen = {c: r for (c, m), r in best_runs(results).items() if m == "answer"}
    held = {(c, t[0]): r for (c, m, t), r in best_runs(results, held_out=True).items() if m == "answer" and len(t) == 1}
    classifiers = ordered({c for c, _ in held if c in seen})
    if not classifiers:
        return False
    tasks = sorted({t for _, t in held})
    fig = plt.figure(figsize=(max(4.9, 2.9 * len(classifiers) + 1.1), 3.9))
    left, gap, right = 0.6 / fig.get_figwidth(), 0.25 / fig.get_figwidth(), 0.5 / fig.get_figwidth()
    width = (1 - left - right - gap * (len(classifiers) - 1)) / len(classifiers)
    series = ["Task seen in training", "Task held out"]
    for i, c in enumerate(classifiers):
        ax = fig.add_axes([left + i * (width + gap), 0.105, width, 0.50])
        values = {series[0]: [seen[c]["test"]["accuracy_per_task"].get(t) for t in tasks],
                  series[1]: [held[(c, t)]["test"]["accuracy"] if (c, t) in held else None for t in tasks]}
        grouped_bars(ax, [t.capitalize() for t in tasks], series, values,
                     {series[0]: MUTED, series[1]: colour_of(c, classifiers)}, seen[c]["chance"],
                     label_chance=i == len(classifiers) - 1)
        ax.set_title(c, fontsize=9, color=INK, loc="left", pad=10)
        if i:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    titles(fig, "Accuracy on a task the classifier never trained on", "Answer only, test split")
    legend(fig, [series[0]] + [f"{series[1]} ({c})" for c in classifiers],
           [MUTED] + [colour_of(c, classifiers) for c in classifiers])
    fig.savefig(out / "accuracy_held_out_task.png", dpi=200)
    plt.close(fig)
    return True


def confusion_matrices(results, out):
    best = {c: r for (c, m), r in best_runs(results).items() if m == "answer"}
    if not best:
        return
    series = ordered(set(best))
    panel, space = 2.1, 0.35                                    # panel width and the space between panels, in inches
    fig = plt.figure(figsize=(max(4.7, 0.75 + (panel + space) * len(series)), 3.45))
    left, gap, width = 0.75 / fig.get_figwidth(), space / fig.get_figwidth(), panel / fig.get_figwidth()
    for i, c in enumerate(series):
        r = best[c]
        names = [n.split(":")[0] for n in r["models"]]
        matrix = r["test"]["confusion_matrix"]
        shares = [[v / max(sum(row), 1) for v in row] for row in matrix]
        ax = fig.add_axes([left + i * (width + gap), 0.16, width, 0.56])
        ax.pcolormesh(shares, cmap=BLUES, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
        ax.invert_yaxis()
        for y, row in enumerate(matrix):
            for x, v in enumerate(row):
                ink = "white" if shares[y][x] > 0.55 else INK
                ax.text(x + 0.5, y + 0.44, str(v), ha="center", va="center", fontsize=10, fontweight="semibold", color=ink)
                ax.text(x + 0.5, y + 0.72, f"{shares[y][x]:.0%}", ha="center", va="center", fontsize=7.5, color=ink)
        ax.set_xticks([k + 0.5 for k in range(len(names))], names, fontsize=8)
        ax.set_yticks([k + 0.5 for k in range(len(names))], names if i == 0 else [], fontsize=8)
        ax.tick_params(length=0, labelcolor=INK_2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("Predicted model", fontsize=8, color=MUTED)
        if i == 0:
            ax.set_ylabel("True model", fontsize=8, color=MUTED)
        ax.set_title(f"{c}  ·  {r['test']['accuracy']:.1%} correct", fontsize=9, color=INK, loc="left", pad=6)
    titles(fig, "Which model is mistaken for which",
           "Answer only, test split. Cells show the count and the share of the row.")
    fig.savefig(out / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def training_curves(results_dir, out):
    """One figure per run that has a history.json: a list of epochs with train_loss, val_loss, train_acc, val_acc."""
    written = []
    for history_file in sorted((results_dir / "runs").glob("*/history.json")):
        run = history_file.parent.name
        history = json.loads(history_file.read_text(encoding="utf-8"))
        metrics_file = history_file.parent / "metrics.json"
        classifier = json.loads(metrics_file.read_text(encoding="utf-8"))["classifier"] if metrics_file.exists() else run
        colour = colour_of(classifier, [classifier])
        epochs = [h["epoch"] for h in history]
        best = max(history, key=lambda h: (h["val_acc"], -h["epoch"]))
        fig = plt.figure(figsize=(7.2, 3.3))
        panels = [("Loss", "train_loss", "val_loss", 1.0), ("Accuracy (%)", "train_acc", "val_acc", 100.0)]
        for i, (label, train_key, val_key, scale) in enumerate(panels):
            ax = fig.add_axes([0.075 + i * 0.5, 0.14, 0.40, 0.56])
            ax.plot(epochs, [h[train_key] * scale for h in history], color=MUTED, linewidth=1.5, solid_capstyle="round")
            ax.plot(epochs, [h[val_key] * scale for h in history], color=colour, linewidth=1.5, solid_capstyle="round")
            ax.plot([best["epoch"]], [best[val_key] * scale], marker="o", markersize=6.5, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
            # The label hangs off the marker towards the middle of the plot, so it never leaves the axes.
            late = best["epoch"] > (epochs[0] + epochs[-1]) / 2
            ax.annotate(f"best epoch {best['epoch']}: {best[val_key] * scale:.{3 if scale == 1 else 1}f}",
                        (best["epoch"], best[val_key] * scale), textcoords="offset points",
                        xytext=(-7 if late else 7, 9 if i == 0 else -13), ha="right" if late else "left",
                        fontsize=7.5, color=INK_2)
            ax.yaxis.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0, labelsize=8)
            ax.set_xticks(epochs if len(epochs) <= 12 else epochs[::max(1, len(epochs) // 10)])
            ax.set_xlabel("Epoch", fontsize=8, color=MUTED)
            ax.set_ylabel(label, fontsize=8, color=MUTED)
            ax.margins(x=0.04, y=0.14)
            if i == 0:
                ax.set_ylim(bottom=0)
        titles(fig, f"Training of {classifier} ({run})",
               "Per epoch. The marked epoch has the best validation accuracy.")
        handles = [plt.Line2D([0], [0], color=MUTED, linewidth=1.5), plt.Line2D([0], [0], color=colour, linewidth=1.5)]
        fig.legend(handles, ["Training", "Validation"], loc="upper left", bbox_to_anchor=(0.012, 0.85), ncol=2,
                   frameon=False, fontsize=8, labelcolor=INK_2, handlelength=1.6, columnspacing=1.4)
        fig.savefig(out / f"training_{run}.png", dpi=200)
        plt.close(fig)
        written.append(f"training_{run}.png")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    results = evaluate.collect(args.results_dir)
    if not results:
        raise SystemExit("nothing to draw: run src/baseline.py or train a model first")
    out = args.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    accuracy_by_input(results, out)
    accuracy_by_task(results, out)
    confusion_matrices(results, out)
    written = ["accuracy_by_input.png", "accuracy_by_task.png", "confusion_matrices.png"]
    if accuracy_held_out_task(results, out):
        written.append("accuracy_held_out_task.png")
    written += training_curves(args.results_dir, out)
    print(f"wrote {len(written)} figures to {out}:")
    for name in written:
        print("  ", name)


if __name__ == "__main__":
    main()
