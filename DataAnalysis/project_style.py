import matplotlib as mpl
from matplotlib.figure import Figure

MM_PER_INCH = 25.4


def mm_to_in(mm):
    return mm / MM_PER_INCH


# ---- Choose final widths ----
SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183

JOURNAL_RCPARAMS = {
    # Fonts: pick something installed on your assembly machine(s)
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    # Final-size typography (tuned to survive print reduction)
    "font.size": 6.0,
    "axes.labelsize": 6.0,
    "axes.titlesize": 6.0,#6.5,
    "xtick.labelsize": 6.0,#5.5,
    "ytick.labelsize": 6.0,#5.5,
    "legend.fontsize": 6.0,#5.5,
    # Keep text as text (editable) in vector exports
    "pdf.fonttype": 42,  # crucial for editable text in many PDF workflows
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # keeps SVG text as text (nice for Inkscape)
    # Strokes that won't disappear in print
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.markersize": 4.0,
    # Cleaner export boxes
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Patch defaults (e.g., fill_between edges)
    "patch.linewidth": 0.1,
}

_SAVEFIG_PATCHED = False
_ORIG_FIGURE_SAVEFIG = None


def _patch_figure_savefig_square_axes():
    global _SAVEFIG_PATCHED, _ORIG_FIGURE_SAVEFIG
    if _SAVEFIG_PATCHED:
        return

    _ORIG_FIGURE_SAVEFIG = Figure.savefig

    def _savefig_with_square_axes(fig, *args, **kwargs):
        for ax in fig.axes:
            try:
                ax.set_box_aspect(1)
            except Exception:
                pass
        try:
            if not fig.get_constrained_layout():
                fig.tight_layout()
        except Exception:
            pass
        return _ORIG_FIGURE_SAVEFIG(fig, *args, **kwargs)

    Figure.savefig = _savefig_with_square_axes
    _SAVEFIG_PATCHED = True


def set_style():
    mpl.rcParams.update(JOURNAL_RCPARAMS)
    _patch_figure_savefig_square_axes()


def fig_single_col(height_mm=55):
    return (mm_to_in(SINGLE_COL_MM), mm_to_in(height_mm))


def fig_double_col(height_mm=80):
    return (mm_to_in(DOUBLE_COL_MM), mm_to_in(height_mm))
