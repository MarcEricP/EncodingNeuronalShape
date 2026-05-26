import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from dendrotree.tree_structure import swc_rx, tree_plot, tree_operations
import tqdm


# Requires: ffmpeg installed in PATH

# When running this module as a script to generate mp4 afterwards
BASE_DIRS = [
    # os.path.join(os.getcwd(), "simulation_result", "Palavalli"),
    # os.path.join(os.getcwd(), "simulation_result", "Arfima"),
    # os.path.join(os.getcwd(), "simulation_result", "Arfima_msd_fit_exponent"),
    # os.path.join(os.getcwd(), "simulation_result", "Arfima_other_simu_param"),
    os.path.join(os.getcwd(), "simulation_result", "Arfima_MSD"),
]

# Default parameters for the graphics
FPS = 10                # frames per second of the output video
DPI = 200               # resolution of the figure used for the movie
CODEC = "libx264"       # H.264 for broad compatibility
PIX_FMT = "yuv420p"     # ensures compatibility with most players
FIGSIZE = (6, 6)        # inches; square and tight by default
LINEWIDTH = 1.5         # line width when drawing trees

# Timestamp/scale bar settings
MIN_PER_FRAME = 1       # one SWC = 1 minute
TIMESTAMP_FMT = "t = {t:d} min"
TIMESTAMP_FONTSIZE = 10
TIMESTAMP_XY_AX = (0.01, 0.99)  # axes coords (upper-left)
SCALE_BAR_LENGTH_UM = 10.0
SCALE_BAR_HEIGHT_FRAC = 0.012   # bar thickness as fraction of data y-range
SCALE_BAR_MARGIN_FRAC = 0.05    # margins from left/bottom as fraction of axes range
SCALE_BAR_LABEL = "10 µm"
SCALE_BAR_FONTSIZE = 9

num_re = re.compile(r"(\d+)\.swc$", re.IGNORECASE)

def numeric_key(path):
    """Sort by the trailing integer in filenames like 0.swc, 1.swc, 299.swc.
    Falls back to lexicographical order if no number is present."""
    name = os.path.basename(path)
    m = num_re.search(name)
    return int(m.group(1)) if m else name

def load_and_center_positions(swc_text):
    """
    Parse SWC, find root, center positions on root, and apply the same
    rotation/flip as in your snippet:
        pos' = -1j * ((x - root_x) + 1j*(y - root_y))
    Returns:
        positions: dict[node] -> complex
        graph: the rx_graph (so we can plot it later)
        root_node: the graph root index
    """
    rx_graph = swc_rx.swc2rx(swc_text)
    root_node = tree_operations.find_root(rx_graph)
    root_x, root_y = rx_graph[root_node]["position"][0], rx_graph[root_node]["position"][1]

    positions = {}
    for node in rx_graph.node_indices():
        x, y, _ = rx_graph[node]["position"]
        z = (x - root_x) + 1j * (y - root_y)
        z_rot = -1j * z
        positions[node] = z_rot

    return positions, rx_graph, root_node

def compute_global_bounds(swc_paths, margin=0.05):
    """
    Read all SWCs, compute centered/rotated positions and return
    consistent x/y limits to avoid flicker between frames.
    """
    xs, ys = [], []
    for p in swc_paths:
        with open(p, "r") as f:
            swc_str = f.read()
        positions, _, _ = load_and_center_positions(swc_str)
        for z in positions.values():
            xs.append(z.real)
            ys.append(z.imag)
    if not xs or not ys:
        return (-1, 1), (-1, 1)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # add a bit of margin
    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0: dx = 1.0
    if dy == 0: dy = 1.0

    xmin -= margin * dx
    xmax += margin * dx
    ymin -= margin * dy
    ymax += margin * dy
    return (xmin, xmax), (ymin, ymax)

def add_timestamp(ax, frame_idx, min_per_frame=1):
    """
    Draw an upper-left timestamp in axes coordinates.
    """
    t_min = int(frame_idx * min_per_frame)
    ax.text(
        TIMESTAMP_XY_AX[0], TIMESTAMP_XY_AX[1],
        TIMESTAMP_FMT.format(t=t_min),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=TIMESTAMP_FONTSIZE,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
        zorder=1000,
    )

def add_scale_bar(ax, xlim, ylim, length_um=SCALE_BAR_LENGTH_UM, label=SCALE_BAR_LABEL):
    """
    Add a crisp rectangular scale bar of given length in micrometers at lower-left.
    Uses a Rectangle patch to avoid rounded line caps.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0 or dy <= 0:
        return

    # If the requested bar is wider than the current field, shrink to 20% width
    max_length = 0.9 * dx
    bar_len = min(length_um, max_length)
    if bar_len <= 0:
        return

    # Position with margins
    x0 = xmin + SCALE_BAR_MARGIN_FRAC * dx
    y0 = ymin + SCALE_BAR_MARGIN_FRAC * dy
    bar_h = max(SCALE_BAR_HEIGHT_FRAC * dy, 1e-6)  # avoid zero height

    # Draw a filled rectangle (crisp edges), no antialiasing
    rect = Rectangle(
        (x0, y0), width=bar_len, height=bar_h,
        linewidth=0.0, edgecolor=None, facecolor="black",
        antialiased=False, zorder=1000
    )
    ax.add_patch(rect)

    # Label centered above the bar
    ax.text(
        x0 + bar_len / 2.0, y0 + 1.6 * bar_h,
        label,
        ha="center", va="bottom",
        fontsize=SCALE_BAR_FONTSIZE,
        color="black",
        zorder=1001,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7),
    )

def draw_frame(ax, swc_text, xlim, ylim, frame_idx):
    """
    Draw a single frame: recenters/rotates like above and plots via tree_plot,
    then overlays timestamp and scale bar.
    """
    # Parse & center
    positions, rx_graph, _ = load_and_center_positions(swc_text)

    # Overwrite positions in the graph (so tree_plot uses the transformed coords)
    for node, z in positions.items():
        rx_graph[node]["position"] = z.real + 1j*z.imag

    ax.cla()
    ax = tree_plot.plot_position(
        rx_graph,
        ax,
        "position",
        color_mode="tip_id",
        # linewidth=LINEWIDTH,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()

    # Overlays
    add_timestamp(ax, frame_idx, MIN_PER_FRAME)
    add_scale_bar(ax, xlim, ylim, SCALE_BAR_LENGTH_UM, SCALE_BAR_LABEL)

def collect_swc_folders(base_dirs):
    """
    Walk base dirs; for every folder that contains at least one .swc,
    return a mapping folder -> sorted list of swc paths.
    """
    folder_map = {}
    for base in base_dirs:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            swcs = [os.path.join(root, f) for f in files if f.lower().endswith(".swc")]
            if swcs:
                swcs.sort(key=numeric_key)
                folder_map[root] = swcs
    return folder_map

def make_movie_for_folder(folder, swc_paths):
    """
    Create an MP4 using all SWCs in `folder` as frames.
    The output is saved next to the folder as: <folder_name>_swc_sequence.mp4
    (If you prefer it inside the folder, change output_path accordingly.)
    """
    if len(swc_paths) == 1:
        print(f"[WARN] Only one SWC in {folder}; movie will have a single frame.")

    # Compute consistent axes across frames
    xlim, ylim = compute_global_bounds(swc_paths)

    # Prepare figure/writer
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    metadata = {"title": os.path.basename(folder), "artist": "dendrotree"}
    writer = animation.FFMpegWriter(
        fps=FPS,
        metadata=metadata,
        codec=CODEC,
        extra_args=["-pix_fmt", PIX_FMT],
    )

    parent_dir = os.path.dirname(folder)
    folder_tag = os.path.basename(folder.rstrip(os.sep))
    output_path = os.path.join(parent_dir, f"{folder_tag}_swc_sequence.mp4")

    print(f"[INFO] Writing {output_path} ({len(swc_paths)} frames at {FPS} fps)")

    with writer.saving(fig, output_path, DPI):
        for i, swc_file in tqdm.tqdm(enumerate(swc_paths)):
            with open(swc_file, "r") as f:
                swc_str = f.read()
            draw_frame(ax, swc_str, xlim, ylim, frame_idx=i)
            writer.grab_frame()

    plt.close(fig)
    print(f"[OK] Saved: {output_path}")

if __name__ == "__main__":
    folder_to_swcs = collect_swc_folders(BASE_DIRS)
    if not folder_to_swcs:
        print("[INFO] No SWC files found under the configured BASE_DIRS.")
    for folder, swc_paths in folder_to_swcs.items():
        try:
            make_movie_for_folder(folder, swc_paths)
        except Exception as e:
            print(f"[ERROR] Failed on folder {folder}: {e}")
