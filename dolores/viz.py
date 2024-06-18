import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

col_green = "#228833"
col_red = "#EE6677"
col_purp = "#AA3377"
col_blue = "#66CCEE"
col_yellow = "#CCBB44"
col_indigo = "#4477AA"
col_grey = "#BBBBBB"
colorpal = [col_blue, col_green, col_red, col_purp, col_yellow, col_indigo, col_grey]


def qqplot(
    results_list,
    edges=False,
    hist_plot=False,
    colors="",
    legend_labels="",
    save_to_file="",
    size=None,
):
    """
    Q-Q plot of p-values
    :param hist_plot: whether to plot histogram or Q-Q plot
    :param size: size of figure
    :param results_list: list of results
    :param edges: whether plotting p-values for edges (two-sided) or clades (one-sided)
    :param colors: colours to plot each set of results, in order
    :param legend_labels: labels for each set of results, in order
    :param save_to_file: filename if saving plot to file
    :return:
    """
    if not hist_plot:
        if size is None:
            size = (5, 4.5)
        plt.figure(figsize=size)
    if not colors:
        colors = colorpal
    names = []
    for i, results in enumerate(results_list):
        if edges:
            y = results.q
        else:
            y = [1 - q for q in results.q]
        if hist_plot:
            if size is None:
                size = (3.5, 3.5)
            plt.figure(figsize=size)
            plt.hist(y, color=colors[i], alpha=1, density=True)
            names.append(results.name)
        else:
            y = np.sort(y)
            plt.plot(
                [i / (1 + len(y)) for i in range(1, len(y) + 1)],
                y,
                color=colors[i],
                lw=3,
            )
            names.append(results.name)
    if not hist_plot:
        plt.plot([0, 1], [0, 1], color="black", ls="--")
        plt.xlabel("Theoretical quantiles")
        plt.ylabel("Empirical quantiles")
    else:
        plt.xlim((0, 1))
        plt.xlabel("p")
        plt.tight_layout()
    if legend_labels:
        names = legend_labels
    lines = [Line2D([0], [0], color=c, alpha=1) for c in colors[0 : len(names)]]
    if hist_plot:
        plt.legend(lines, names, loc="lower right")
    if save_to_file:
        plt.savefig(save_to_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def outliers_plot(
    results,
    outliers_threshold=0.05,
    save_to_file="",
    size=(3, 3),
    xticks=None,
    yticks=None,
    equal_axes=True,
):
    """
    Outliers plot
    :param yticks:
    :param xticks:
    :param results:
    :param outliers_threshold:
    :param save_to_file:
    :param size:
    :param equal_axes:
    :return:
    """
    inds_to_plot = [i for i in range(results.num_)]
    num = results.num
    fig, ax = plt.subplots(1, 1, figsize=size)
    log10sf = np.sort(
        np.array([results.log10sf[i] for i in range(results.num_) if i in inds_to_plot])
    )
    num_ = len(log10sf)
    inds = np.where(log10sf < np.log(outliers_threshold / num) / np.log(10))[0]
    ax.scatter(
        [-np.log(i / (1 + num_)) / np.log(10) for i in range(1, num_ + 1)],
        [-l for l in log10sf],
        color=col_blue,
        s=4,
    )
    ax.scatter(
        -np.log((inds + 1) / (1 + num_)) / np.log(10),
        [-l for l in (log10sf[inds])],
        color=col_red,
        s=6,
    )

    ax.plot([0, 100], [0, 100], color="black", ls="--")
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_ylim([min(yticks), max(yticks)])
    else:
        ax.set_ylim([0, 1.05 * np.max(-log10sf)])
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xlim([min(xticks), max(xticks)])
    else:
        ax.set_xlim([0, 1.05 * np.max(-log10sf)])
    ax.set_xlabel("Expected -log10($p$-values)")
    ax.set_ylabel("Observed -log10($p$-values)")

    if not equal_axes:
        ax.set_xlim([0, -np.log(1 / (1 + num_)) / np.log(10) * 1.05])

    plt.tight_layout()
    if save_to_file:
        plt.savefig(save_to_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def pvalues_plot(
    results,
    threshold=0.05,
    save_to_file="",
    size=None,
):
    if size is None:
        size = (8, 3.5)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)

    C = -np.log10(threshold / results.num)
    ymax = -min(results.log10sf)
    ymin = min(results.log10sf_event)
    x1 = [results.start[results.ids[i]] for i in range(results.num_)]
    x2 = [results.end[results.ids[i]] for i in range(results.num_)]
    xx = [
        (results.start[results.ids[i]] + results.end[results.ids[i]]) / 2
        for i in range(results.num_)
    ]
    c = [col_grey] * results.num_
    for i in range(results.num_):
        if -results.log10sf[i] >= C and -results.log10sf_event[i] >= C:
            c[i] = "black"
    ax.scatter(
        x=xx,
        y=[-y for y in results.log10sf],
        color=c,
        s=6,
        zorder=-20,
    )
    ax.scatter(
        x=xx,
        y=results.log10sf_event,
        color=c,
        s=6,
        zorder=-20,
    )
    ax.hlines(
        xmin=x1,
        xmax=x2,
        y=[-y for y in results.log10sf],
        color=c,
    )
    ax.hlines(
        xmin=x1,
        xmax=x2,
        y=results.log10sf_event,
        color=c,
    )
    ax.vlines(
        x=xx,
        ymin=results.log10sf_event,
        ymax=[-y for y in results.log10sf],
        color=c,
        linewidth=1,
        zorder=10,
    )

    xmin = min(results.start)
    xmax = max(results.end)
    ax.hlines(
        xmin=xmin,
        xmax=xmax,
        y=[
            -np.log10(threshold / results.num),
        ],
        color=col_red,
        linewidth=1.5,
        linestyle=":",
        zorder=10,
    )
    ax.hlines(
        xmin=xmin,
        xmax=xmax,
        y=[
            np.log10(threshold / results.num),
        ],
        color=col_red,
        linewidth=1.5,
        linestyle=":",
        zorder=10,
    )
    ticks_ = [y for y in ax.get_yticks()]
    ticks = [-y for y in ax.get_yticks() if y < 0] + [
        y for y in ax.get_yticks() if y >= 0
    ]
    ax.set_yticks(ticks_)
    ax.set_yticklabels([str(int(y)) for y in ticks])

    labs = [
        str(int(round((xmin - 1e6 + i * 1e6) / 1e6, 0)))
        for i in range(1, int(xmax / 1e6))
    ]
    ax.set_xticks(
        ticks=[xmin - 1e6 + i * 1e6 for i in range(1, int(xmax / 1e6))],
        labels=labs,
    )

    ax.hlines(
        y=0,
        xmin=xmin,
        xmax=xmax,
        color="black",
        linewidth=1.0,
        zorder=10,
    )
    ax.set_ylabel("-log10($\it{p}$-value)")
    ax.set_xlabel("Genome position, Mb")

    plt.tight_layout()
    if save_to_file != "":
        plt.savefig(save_to_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()
