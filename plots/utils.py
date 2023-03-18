import json
from typing import Iterable

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def reorder(l: Iterable, index: Iterable):
    return [l[i] for i in index]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


cdict1 = {
    "red": (
        (0.0, 0.11764705882352941, 0.11764705882352941),
        (1.0, 0.9607843137254902, 0.9607843137254902),
    ),
    "green": (
        (0.0, 0.5333333333333333, 0.5333333333333333),
        (1.0, 0.15294117647058825, 0.15294117647058825),
    ),
    "blue": (
        (0.0, 0.8980392156862745, 0.8980392156862745),
        (1.0, 0.3411764705882353, 0.3411764705882353),
    ),
    "alpha": ((0.0, 1, 1), (0.5, 1, 1), (1.0, 1, 1)),
}  # #1E88E5 -> #ff0052


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])

    return pl_colorscale


RED_BLUE = matplotlib_to_plotly(LinearSegmentedColormap("RedBlue", cdict1), 255)


def get_minmax(values):
    # trim the color range, but prevent the color range from collapsing
    vmin = np.nanpercentile(values, 5)
    vmax = np.nanpercentile(values, 95)
    if vmin == vmax:
        vmin = np.nanpercentile(values, 1)
        vmax = np.nanpercentile(values, 99)
        if vmin == vmax:
            vmin = np.min(values)
            vmax = np.max(values)
    if vmin > vmax:  # fixes rare numerical precision issues
        vmin = vmax
    return vmin, vmax


def get_colors(metrics: np.ndarray, vmin, vmax):
    colors = np.array((metrics - vmin) / (vmax - vmin) * 255, dtype=np.int32)
    colors[colors >= 255] = 254
    colors[colors < 0] = 0
    return reorder(np.array(RED_BLUE)[:, 1], colors)


DEFAULT_LAYOUT = dict(
    # Black Style
    # paper_bgcolor="#222222",
    # plot_bgcolor="#222222",
    # font=dict(color="#fff"),
    autosize=True,
)
DEFAULT_CONFIG = {"scrollZoom": False, "displayModeBar": False}
