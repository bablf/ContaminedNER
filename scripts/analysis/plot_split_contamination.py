from pathlib import Path
from typing import Any, Optional, TypeVar

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from paper_dataset_contamination.utils import InputDataset, parse_dataset
from with_argparse import with_argparse

ENTITY_T = tuple[str, str]
T = TypeVar("T")


mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}"
    r"\usepackage[usenames,dvipsnames]{color}"
    r"\usepackage[dvipsnames]{xcolor}"
    r"\usepackage{amssymb}"
)


def index_to_color(info: tuple[Any, int, int], offset: int):
    cmap, colors_per_category, start_pos = info
    pos = start_pos + colors_per_category - (offset + 1)
    return cmap.colors[pos]


def plot_contamination_level(
    col,
    axis: Axis,
    col_x,
    scale,
    bar_width,
    train_cmap,
    train_contamination,
    train_doc_contamination,
    test_cmap,
    test_contamination,
    test_doc_contamination,
    *,
    with_label,
    dataset,
    ours,
):
    train_bar = axis.barh(
        col_x,
        1.0 * scale,
        height=bar_width,
        color=index_to_color(train_cmap, 0),
    )

    arrow_length = bar_width * 0.75
    if col == 3:
        axis.annotate(
            r"$\textbf{whole split}$", (0.5, col_x - bar_width / 2), (0.5, col_x - bar_width / 2 - arrow_length),
            size="large", color="black", va="center", ha="center",
            arrowprops=dict(
                arrowstyle="-[,widthB=7.4,angleB=0"
            ),
        )

    train_bar_cont = axis.barh(
        col_x,
        train_contamination * scale,
        height=bar_width,
        color=index_to_color(train_cmap, 2),
    )

    if col == 2:
        tick_length = 0.2
        x = 0.5 + (train_contamination) * 0.5
        width = 7.4 * (1 - train_contamination)

        axis.annotate(
            r"$\textbf{clean}$", (x, col_x + bar_width * 0.5), (x, col_x + 0.65 * bar_width + arrow_length),
            size="large", color="black", va="center", ha="center",
            arrowprops=dict(
                arrowstyle=f"-[,widthB={width},angleB=0,lengthB={tick_length}", color="black"
            ),
        )

    axis.barh(
        col_x,
        train_contamination * scale,
        height=bar_width,
        hatch="//",
        edgecolor="white",
        color="none",
    )
    test_bar = axis.barh(
        col_x, -1.0 * scale, height=bar_width, color=index_to_color(test_cmap, 0)
    )

    if col == 2:
        tick_length = 0.2
        x = -0.5 + (1 - test_contamination) * 0.5
        width = 7.4 * (test_contamination)

        axis.annotate(
            r"$\textbf{contaminated}$", (x, col_x + bar_width * 0.5), (x, col_x + 0.65 * bar_width + arrow_length),
            size="large", color="black", va="center", ha="center",
            arrowprops=dict(
                arrowstyle=f"-[,widthB={width},angleB=0,lengthB={tick_length}", color="black"
            ),
        )

    test_bar_cont = axis.barh(
        col_x,
        -test_contamination * scale,
        height=bar_width,
        color=index_to_color(test_cmap, 3),
    )

    if col == 7:
        width = 7.4 * (1 - test_contamination)
        x = -0.5 - test_contamination * 0.5
        tick_length = 0.2
        axis.annotate(
            "$\\mathcal{D}_{conta}$", (x, col_x - bar_width / 2), (x, col_x - 0.5 * bar_width - arrow_length),
            size="large", color="black", va="center", ha="center",
            arrowprops=dict(
                arrowstyle=f"-[,widthB={width},angleB=0,lengthB={tick_length}"
            ),
        )

    axis.barh(
        col_x,
        -test_contamination * scale,
        height=bar_width,
        hatch="\\\\",
        edgecolor="white",
        color="none",
    )
    if with_label:
        axis.bar_label(
            train_bar,
            label_type="edge",
            labels=[dataset.train_size],
            padding=2,
            fontsize=7,
        )
        if ours:
            axis.bar_label(
                train_bar,
                label_type="center",
                labels=["ours"],
                padding=2,
                fontsize=7,
            )
        if 0 < train_contamination < 0.9:
            axis.bar_label(
                train_bar_cont,
                label_type="edge",
                labels=[int(train_contamination * dataset.train_size)],
                fontsize=7,
                padding=2,
            )
        axis.bar_label(
            test_bar,
            label_type="edge",
            labels=[dataset.test_size],
            padding=2,
            fontsize=7,
        )
        if ours:
            axis.bar_label(
                test_bar,
                label_type="center",
                labels=["ours"],
                padding=2,
                fontsize=7,
            )

        if 0 < test_contamination < 0.9:
            axis.bar_label(
                test_bar_cont,
                label_type="edge",
                labels=[int(test_contamination * dataset.test_size)],
                fontsize=7,
                padding=2,
            )

def parse_color_range(inp: str) -> tuple[Any, int, int]:
    print(inp)
    args = inp.split(":", 2)
    if len(args) < 3:
        raise ValueError()
    return mpl.colormaps[args[0]], int(args[1]), int(args[2])


@with_argparse(
    datasets=parse_dataset,
    mincut_datasets=parse_dataset,
    test_cmap=parse_color_range,
    train_cmap=parse_color_range,
    legend_cmap=parse_color_range,
    mincut_cmap=parse_color_range,
)
def plot_dataset_contamination(
    datasets: list[InputDataset],
    mincut_datasets: Optional[list[InputDataset]] = None,
    train_cmap: tuple[Any, int, int] = "tab20c:4:8",
    test_cmap: tuple[Any, int, int] = "tab20b:4:12",
    legend_cmap: tuple[Any, int, int] = "tab20c:4:15",
    mincut_cmap: tuple[Any, int, int] = "tab20b:4:7",
    bar_width: float = 0.3,
    with_label: bool = False,
    distance_between_bars: float = 0.95,
    scale: float = 1.0,
    y_scale: float = 1.0,
    step_scale: float = 0.25,
    filename: Optional[Path] = None,
):
    mincut_datasets = mincut_datasets or list()
    if len(mincut_datasets) > 0 and len(mincut_datasets) != len(datasets):
        raise ValueError()

    datasets.sort(key=lambda x: x.name.lower(), reverse=True)
    mincut_datasets.sort(key=lambda x: x.name.lower(), reverse=True)

    fig: Figure
    fig, axis = plt.subplots(1, 1, layout="constrained", figsize=(5.4, 3.8))

    x_offsets = np.arange(0, len(datasets) * distance_between_bars, distance_between_bars)
    x_offsets[-2:] += 0.9 * distance_between_bars
    x_offsets[-3:] += 0.9 * distance_between_bars
    x_offsets *= y_scale
    using_mincut = len(mincut_datasets) > 0
    bar_scale = 2 if using_mincut else 1
    bar_offset = bar_width / 4 if using_mincut else 0
    bar_width /= bar_scale

    contamination_levels = []
    for col, dataset in enumerate(datasets):
        train_contamination = dataset.compute_contamination("train")
        test_contamination = dataset.compute_contamination("test")
        contamination_levels.append((train_contamination, test_contamination))

        train_doc_contamination = dataset.compute_contamination(
            "train", whole_doc_contaminated=True
        )
        test_doc_contamination = dataset.compute_contamination(
            "test", whole_doc_contaminated=True
        )

        col_x = x_offsets[col] + bar_offset
        plot_contamination_level(
            col,
            axis,
            col_x,
            scale,
            bar_width,
            train_cmap,
            train_contamination,
            train_doc_contamination,
            test_cmap,
            test_contamination,
            test_doc_contamination,
            with_label=with_label,
            dataset=dataset,
            ours=False,
        )

    # axis.axvline(color="k")
    dataset_names = [f"{i}\%" for i in range(0, 110, 10)]
    axis.set(
        xlim=(-scale, scale),
        ylim=(-0.5, max(x_offsets) + 1),
        # xticks=x_offsets,  # xlabel=dataset_names,
        #        ylim=(-1.1, 1.1), yticks=np.arange(-1.0, 1.25, 0.25), # ylabel=[f"{abs(elem * 100):.0f}" for elem in np.arange(-1.0, 1.25, 0.25)]
        # ylim=(-1.25, 1.75),
    )

    leg1 = axis.legend(
        handles=[
            mpatches.Patch(
                facecolor=index_to_color(legend_cmap, offset=1),
                alpha=1.0,
                label="Partial sample contamination",
                edgecolor="white",
                hatch="\\\\",
            ),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.4),
        frameon=False,
        fontsize=12,
    )
    leg2 = axis.legend(
        handles=[
            mpatches.Patch(
                facecolor=index_to_color(legend_cmap, offset=0),
                alpha=1.0,
                label="Clean samples",
                edgecolor="white",
            ),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.4),
        frameon=False,
        fontsize=12,
    )
    # leg2.remove()
    leg1._legend_box.width = 340
    axis.add_artist(leg1)
    # fig.gca().add_artist(leg1)
    #    leg2.remove()
    leg1._legend_box._children.extend(leg2._legend_box._children)
    # leg1._legend_box.stale = True
    axis.add_artist(leg1)

    axis.plot(0, x_offsets[1] + 0.75, "o", color="black", ms=1.5)
    axis.plot(0, x_offsets[1] + 1, "o", color="black", ms=1.5)
    axis.plot(0, x_offsets[1] + 1.25, "o", color="black", ms=1.5)
    # assert False, leg1._legend_box._children

    print([f"{abs(elem * 100):.0f}" for elem in np.arange(-1.0, 1.25, 0.25)])

    axis.set_title(r"\textbf{Creating Splits with \% Contamination}\vspace{1em}{}")
    axis.set_xlabel(r"\textbf{Contamination Rate}", size=9, labelpad=7.5)

    for col, (x_offset, ds_name, (train_contamination, test_contamination)) in enumerate(zip(x_offsets, dataset_names, contamination_levels)):
        special_offset = 0.0 if col != 2 * distance_between_bars else 0.15
        axis.text(
            0,
            x_offset + distance_between_bars / 2 - distance_between_bars / 16 + special_offset,
            f"\\textbf{{{round(train_contamination * 100, 0):.0f}\\%}}",
            ha="center",
            va="center",
        )
    axis.text(
        -0.5,
        max(x_offsets) + distance_between_bars / 2 + distance_between_bars / 8,
        r"\textbf{Test Split}",
        ha="center",
        va="center",
    )
    axis.text(
        0.5,
        max(x_offsets) + distance_between_bars / 2 + distance_between_bars / 8,
        r"\textbf{Train Split}",
        ha="center",
        va="center",
    )

    # axis.set_yticks(ticks=x_offsets, labels=dataset_names)
    axis.set_yticks(ticks=[])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_linewidth(2)
    axis.tick_params(width=2)

    x_ticks = np.arange(-4 * step_scale, 5 * step_scale, step_scale)
    display_x_ticks = np.arange(-1.0, 1.25, 0.25)
    axis.set_xticks(
        ticks=x_ticks,
        labels=[
            f"\\textbf{{{abs(display_x_ticks[pos] * 100):.0f}\\%}}"
            for pos, elem in enumerate(x_ticks)
        ],
    )
    fig.savefig(filename or "contamination.pdf")


if __name__ == "__main__":
    plot_dataset_contamination()
