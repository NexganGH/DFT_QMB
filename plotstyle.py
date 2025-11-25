import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (7, 5),
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.2,
        "axes.grid": False,
        "grid.linestyle": "--",
        "grid.color": "gray",
        "font.size": 12,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
