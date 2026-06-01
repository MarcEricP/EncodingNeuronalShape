import numpy as np
from matplotlib.ticker import FuncFormatter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dendrogenesis import project_style

project_style.set_style()


def format_hr_x_axis(ax, t0_h):
    t0_min = 60 * t0_h

    xmin, xmax = ax.get_xlim()
    start = np.ceil((xmin - t0_min) / 60) * 60 + t0_min
    ticks = np.arange(start, xmax + 1e-9, 60)

    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda m, pos: f"{int(round(t0_h + (m) / 60)):02d}")
    )
    return ax
