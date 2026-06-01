import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import sys
import json
import shutil
from pathlib import Path


from dendrogenesis import project_style

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from scipy import stats

from dendrogenesis import load_tracks
from dendrogenesis.useful_stats import slope_intercept_with_robust_and_cluster, compare_two_series

project_style.set_style()

BASE_DIR = "/mnt/c/Users/menir/Documents/0000NeuronsData/clean_movies_1_min_fail_test"
DATASET_PATH = os.path.join(BASE_DIR, "branch_tracking", "all_branches_length.pickle")

# Output
WORKING_DIR = os.getcwd()
RESULT_DIR = os.path.join(WORKING_DIR, "DataAnalysis", "Result", "5_increments_alpha_sigma_mu")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
RAW_DATA_DIR = os.path.join(RESULT_DIR, "raw_data")
CACHED_DATASET_PATH = os.path.join(RAW_DATA_DIR, "all_branches_length.pickle")
RAW_JSON_PATH = os.path.join(RAW_DATA_DIR, "results.json")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

PREFERRED_ORDER = ["class_I", "class_IV", "class_IV_MT_inh"]
CLASS_LABEL = {"class_I": "WT cI", "class_IV": "WT cIV", "class_IV_MT_inh": "MT inh cIV"}
CLASS_COLOR = {"class_I": "tab:green", "class_IV": "tab:pink", "class_IV_MT_inh": "#353535"}

basic_figsize = (project_style.mm_to_in(65), project_style.mm_to_in(65))

# Increment binning 
BIN_SIZE_MIN = 50
N_BINS = 6
START_HOUR_AEL = 16.0
MIN_INCR_PER_BIN = 3

# Track filtering
MIN_LEN = 0
MAX_INCR = 3  

# MSD/alpha estimation 
TAU_MIN_FIT = 1
TAU_MAX_FIT = 60
TAU_MAX_HARD = 60#200
MIN_POINTS_FIT = 6
MIN_LEN_MSD_FIT = 60

#time in trajectories after 16hAEL + 300min are not used in the analysis 
ALPHA_MAX_TIME_MIN = 300

#minimum total increments per acquisition to compute global drift
MIN_TOTAL_INCR_FOR_DRIFT = 10


def save_all_ext(fig, path_wo_ext, dpi=300):
    for ext in [".png", ".svg", ".pdf"]:
        fig.savefig(f"{path_wo_ext}{ext}", dpi=dpi)

def hour_label_from_float(h):
    hh = int(np.floor(h))
    mm = int(np.round((h - hh) * 60.0))
    if mm == 60:
        hh += 1
        mm = 0
    return f"{hh:02d}h{mm:02d}"

def format_hours_axis(ax, xvals):
    ax.set_xticks(xvals)
    ax.set_xticklabels([hour_label_from_float(x) for x in xvals], rotation=0)
    return ax

def bin_centers_hours(start_hour=START_HOUR_AEL, bin_size_min=BIN_SIZE_MIN, n_bins=N_BINS):
    centers = start_hour + (np.arange(n_bins) + 0.5) * (bin_size_min / 60.0)
    return centers


def _write_df_markdown(df, path):
    cols = [str(c) for c in df.columns]
    with open(path, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in df.itertuples(index=False, name=None):
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")

def _write_df_csv(df, path):
    df.to_csv(path, index=False)


def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _dataset_to_branch_records(dataset):
    rows = []
    for cls, by_acq in dataset.items():
        for acq, by_branch in by_acq.items():
            for branch, rec in by_branch.items():
                rows.append(
                    {
                        "neuron_class": str(cls),
                        "name": str(acq),
                        "branch_num": str(branch),
                        "length_time": _to_jsonable(np.asarray(rec.get("time", []), float)),
                        "length_length": _to_jsonable(np.asarray(rec.get("length", []), float)),
                        "length_is_contact": _to_jsonable(np.asarray(rec.get("is_contact", []), bool)),
                    }
                )
    return rows

def p_to_star(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."

def add_bracket_star_data(ax, x1, x2, y, p):
    star = p_to_star(p)
    ax.plot([x1, x1, x2, x2], [y, y + 0.03, y + 0.03, y],
            color="k", lw=1, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.03, star, ha="center", va="bottom")#, fontsize=12)

def extract_level_slope_effects_from_model(model):
    names = list(model.model.exog_names)
    idx = {nm: i for i, nm in enumerate(names)}
    level_term = "C(series)[T.S2]"
    cand1 = "a:C(series)[T.S2]"
    cand2 = "C(series)[T.S2]:a"
    slope_term = cand1 if cand1 in idx else (cand2 if cand2 in idx else None)

    out = {
        "level_diff": np.nan,
        "level_ci95_low": np.nan,
        "level_ci95_high": np.nan,
        "slope_diff": np.nan,
        "slope_ci95_low": np.nan,
        "slope_ci95_high": np.nan,
    }
    if level_term in idx:
        tt = model.t_test(level_term)
        ci = np.asarray(tt.conf_int(alpha=0.05), dtype=float).ravel()
        out["level_diff"] = float(np.asarray(tt.effect).item())
        if ci.size >= 2:
            out["level_ci95_low"] = float(ci[0])
            out["level_ci95_high"] = float(ci[1])
    if slope_term and slope_term in idx:
        tt = model.t_test(slope_term)
        ci = np.asarray(tt.conf_int(alpha=0.05), dtype=float).ravel()
        out["slope_diff"] = float(np.asarray(tt.effect).item())
        if ci.size >= 2:
            out["slope_ci95_low"] = float(ci[0])
            out["slope_ci95_high"] = float(ci[1])
    return out

def extract_slope_stats_from_model(model, term="a"):
    tt = model.t_test(f"{term} = 0")
    slope = float(np.asarray(tt.effect).item())
    slope_se = float(np.asarray(tt.sd).item())
    ci95 = np.asarray(tt.conf_int(alpha=0.05), dtype=float).ravel()
    slope_ci95_low = float(ci95[0]) if ci95.size >= 2 else np.nan
    slope_ci95_high = float(ci95[1]) if ci95.size >= 2 else np.nan
    p_two_sided = float(np.asarray(tt.pvalue).item())
    if slope < 0:
        p_decrease = p_two_sided / 2.0
    else:
        p_decrease = 1.0 - p_two_sided / 2.0
    return slope, slope_se, slope_ci95_low, slope_ci95_high, p_two_sided, p_decrease

def build_bins_per_acquisition(dataset, bin_size_min=BIN_SIZE_MIN, n_bins=N_BINS):
    """
    Returns:
      bins[cls][acq][b] = list of 1-min increments (length[t+1]-length[t]) whose step starts in bin b
    """
    bins = {}
    for cls in dataset.keys():
        bins[cls] = {}
        for acq in dataset[cls].keys():
            bins[cls][acq] = {b: [] for b in range(n_bins)}
            for branch in dataset[cls][acq].keys():
                time = np.asarray(dataset[cls][acq][branch]["time"])
                length = np.asarray(dataset[cls][acq][branch]["length"])
                is_contact = np.asarray(dataset[cls][acq][branch]["is_contact"])

                # exclude any branch with contact at any time
                if np.any(is_contact):
                    continue
                if time.size < 2:
                    continue

                # bin index for each timepoint
                bin_idx = (time // bin_size_min).astype(int)

                for i in range(bin_idx.shape[0] - 1):
                    b = int(bin_idx[i])
                    if 0 <= b < n_bins:
                        incr = float(length[i + 1] - length[i])
                        if not np.isfinite(incr):
                            continue
                        if MAX_INCR is not None and abs(incr) > float(MAX_INCR):
                            continue
                        if np.isfinite(length[i]) and length[i] < float(MIN_LEN):
                            continue
                        bins[cls][acq][b].append(incr)
    return bins

def per_acquisition_stat_matrix(bins_cls_acq, stat="std", n_bins=N_BINS, min_n=MIN_INCR_PER_BIN):
    """
    stat in {"std","mean"} computed per (acq,bin), without pooling across acquisitions.
    Returns:
      acq_names (sorted)
      Y: shape (n_acq, n_bins)
      N: shape (n_acq, n_bins) counts
    """
    acq_names = sorted(bins_cls_acq.keys())
    Y = np.full((len(acq_names), n_bins), np.nan, float)
    N = np.zeros((len(acq_names), n_bins), int)

    for i, acq in enumerate(acq_names):
        for b in range(n_bins):
            arr = np.asarray(bins_cls_acq[acq].get(b, []), float)
            arr = arr[np.isfinite(arr)]
            N[i, b] = arr.size
            if arr.size >= max(1 if stat == "mean" else 2, min_n):
                if stat == "std":
                    Y[i, b] = np.std(arr, ddof=1)
                elif stat == "mean":
                    Y[i, b] = np.mean(arr)
                else:
                    raise ValueError("stat must be 'std' or 'mean'")
    return acq_names, Y, N

def msd_from_trajs(trajs, tau_max):
    """
    trajs: list of 1D arrays x(t) sampled at uniform dt=1 step
    Returns:
      tau, msd(tau) averaged over all trajs 
    """
    tau = np.arange(1, tau_max + 1, dtype=int)
    ss = np.zeros(tau_max, float)
    cc = np.zeros(tau_max, int)

    for x in trajs:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < 2:
            continue
        local_max = min(tau_max, n - 1)
        for k in range(1, local_max + 1):
            d = x[k:] - x[:-k]
            ss[k - 1] += np.sum(d * d)
            cc[k - 1] += d.size

    msd = np.full(tau_max, np.nan, float)
    ok = cc > 0
    msd[ok] = ss[ok] / cc[ok]
    return tau, msd

def alpha_from_msd(tau, msd, tau_min_fit=TAU_MIN_FIT, tau_max_fit=TAU_MAX_FIT, min_points=MIN_POINTS_FIT):
    tau = np.asarray(tau, float)
    msd = np.asarray(msd, float)

    mask = np.isfinite(msd) & (msd > 0) & np.isfinite(tau) & (tau > 0)
    mask &= (tau >= float(tau_min_fit))
    if tau_max_fit is not None:
        mask &= (tau <= float(tau_max_fit))

    if np.sum(mask) < min_points:
        return np.nan, np.nan, np.nan

    x = np.log(tau[mask])
    y = np.log(msd[mask])
    lr = stats.linregress(x, y)
    alpha = lr.slope
    intercept = lr.intercept
    r2 = lr.rvalue**2
    return alpha, intercept, r2

def gather_trajs_truncated(dataset_cls_acq, alpha_max_time_min=ALPHA_MAX_TIME_MIN):
    """
    Collect branch trajectories (length arrays) for a given acquisition, excluding contact branches,
    and truncating to time <= alpha_max_time_min.
    Returns list of 1D arrays.
    """
    trajs = []
    for branch in dataset_cls_acq.keys():
        is_contact = np.asarray(dataset_cls_acq[branch]["is_contact"])
        if np.any(is_contact):
            continue
        time = np.asarray(dataset_cls_acq[branch]["time"], float)
        length = np.asarray(dataset_cls_acq[branch]["length"], float)

        m = np.isfinite(time) & np.isfinite(length) & (time <= float(alpha_max_time_min))
        time = time[m]
        length = length[m]
        if length.size >= 3:
            trajs.append(length)
    return trajs

def main():

    # Load + filter tracks
    if os.path.exists(DATASET_PATH):
        dataset_source_path = DATASET_PATH
    elif os.path.exists(CACHED_DATASET_PATH):
        dataset_source_path = CACHED_DATASET_PATH
    else:
        raise FileNotFoundError(
            f"No dataset found at '{DATASET_PATH}' and no cache found at '{CACHED_DATASET_PATH}'."
        )

    dataset_raw = load_tracks.load_dataset_from_pickle(dataset_source_path)
    dataset = load_tracks.filter_tracks(dataset_raw, MAX_INCR, MIN_LEN, None)

    if os.path.exists(DATASET_PATH):
        try:
            shutil.copy2(DATASET_PATH, CACHED_DATASET_PATH)
        except Exception:
            pass

    present = [c for c in PREFERRED_ORDER if c in dataset.keys()]
    for c in dataset.keys():
        if c not in present:
            present.append(c)

    #increments per acquisition
    bins = build_bins_per_acquisition(dataset, BIN_SIZE_MIN, N_BINS)
    time_axis = bin_centers_hours(START_HOUR_AEL, BIN_SIZE_MIN, N_BINS)
    level_anchor_hour = float(time_axis[0])

    per_class = {}
    for cls in present:
        acq_names, Y_std, N_counts = per_acquisition_stat_matrix(bins[cls], stat="std", n_bins=N_BINS, min_n=MIN_INCR_PER_BIN)
        _,        Y_mean, _      = per_acquisition_stat_matrix(bins[cls], stat="mean", n_bins=N_BINS, min_n=MIN_INCR_PER_BIN)

        per_class[cls] = dict(acq_names=acq_names, Y_std=Y_std, Y_mean=Y_mean, N_counts=N_counts)


    #sigma
    for share_axis in ["all", "none"]:
        fig, axs = plt.subplots(2, 3, figsize=(3 * basic_figsize[0], 2 * basic_figsize[1]),
                                sharex=False, sharey=False)
        col_defs = [["class_I", "class_IV"], ["class_IV", "class_IV_MT_inh"], present]

        for col, classes_in_col in enumerate(col_defs):
            classes_in_col = [c for c in classes_in_col if c in per_class]
            for cls in classes_in_col:
                Y = per_class[cls]["Y_std"]
                mean = np.nanmean(Y, axis=0)
                std = np.nanstd(Y, axis=0)

                for row in range(2):
                    ax = axs[row, col]
                    ax.plot(time_axis, mean, color=CLASS_COLOR[cls], label=CLASS_LABEL[cls])
                    ax.fill_between(time_axis, np.maximum(mean - std, 0.0), mean + std, color=CLASS_COLOR[cls], alpha=0.3, linewidth=0)
                    if row == 1:
                        for k in range(Y.shape[0]):
                            ax.plot(time_axis, Y[k, :], color=CLASS_COLOR[cls], linestyle=":", alpha=0.6)

        for col in range(3):
            axs[0, col].legend()
            for row in range(2):
                ax = axs[row, col]
                ax.set_xlabel("time (hours AEL)")
                format_hours_axis(ax, time_axis)
                ax.set_ylabel("1-min increments $\\sigma$ ($\\mu m$)")
                ax.set_ylim(bottom=0.0)
            
            if col == 0:
                c1, c2 = "class_I", "class_IV"
            elif col == 1:
                c1, c2 = "class_IV", "class_IV_MT_inh"
            else:
                c1 = c2 = None
            if c1 in per_class and c2 in per_class:
                try:
                    _, _, stats_cmp = compare_two_series(
                        per_class[c1]["Y_std"],
                        per_class[c2]["Y_std"],
                        time_axis,
                        center_at=level_anchor_hour,
                        return_per_time=False,
                    )
                except Exception:
                    stats_cmp = {}
                p_level = float(stats_cmp.get("p_level_diff", np.nan))
                p_slope = float(stats_cmp.get("p_slope_diff", np.nan))
                if np.isfinite(p_level) or np.isfinite(p_slope):
                    txt = f"Cluster-robust OLS\nlevel p={p_level:.3g}\nslope p={p_slope:.3g}"
                    axs[0, col].text(
                        0.98, 0.02, txt, transform=axs[0, col].transAxes,
                        ha="right", va="bottom", #fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.8)
                    )

        fig.tight_layout()
        save_all_ext(fig, os.path.join(PLOTS_DIR, f"increment-std-per-acq-mean-std-{share_axis}"))
        plt.close(fig)

    #sigma(t) + D(t)=sigma^2/2 for class I and class IV
    if ("class_I" in per_class) and ("class_IV" in per_class):
        fig, ax_sigma = plt.subplots(1, 1, figsize=(project_style.mm_to_in(70), project_style.mm_to_in(70)))
        ax_sigma.set_box_aspect(1)
        ax_D = ax_sigma.twinx()
        legend_title = None

        try:
            _, _, stats_ci_civ = compare_two_series(
                per_class["class_I"]["Y_std"],
                per_class["class_IV"]["Y_std"],
                time_axis,
                center_at=level_anchor_hour,
                return_per_time=False,
            )
        except Exception:
            stats_ci_civ = {}
        p_level_ci_civ = float(stats_ci_civ.get("p_level_diff", np.nan))
        p_slope_ci_civ = float(stats_ci_civ.get("p_slope_diff", np.nan))
        if np.isfinite(p_level_ci_civ) or np.isfinite(p_slope_ci_civ):
            plevel_txt = f"{p_level_ci_civ:.3g}" if np.isfinite(p_level_ci_civ) else "nan"
            pslope_txt = f"{p_slope_ci_civ:.3g}" if np.isfinite(p_slope_ci_civ) else "nan"
            legend_title = f"Cluster-robust OLS cI vs cIV\nlevel p={plevel_txt}, slope p={pslope_txt}"

        for cls in ["class_I", "class_IV"]:
            Y = per_class[cls]["Y_std"]
            sigma_mean = np.nanmean(Y, axis=0)
            sigma_std = np.nanstd(Y, axis=0)
            D_mean = 0.5 * (sigma_mean ** 2)

            ax_sigma.plot(
                time_axis,
                sigma_mean,
                color=CLASS_COLOR[cls],
                linestyle="-",
                label=f"{CLASS_LABEL[cls]} ",# + r"$\sigma$",
            )
            ax_sigma.fill_between(
                time_axis,
                np.maximum(sigma_mean - sigma_std, 0.0),
                sigma_mean + sigma_std,
                color=CLASS_COLOR[cls],
                alpha=0.18,
                linewidth=0,
            )
            ax_D.plot(
                time_axis,
                D_mean,
                color=CLASS_COLOR[cls],
                linestyle="--",
                # label=f"{CLASS_LABEL[cls]} " + r"$D=\sigma^2/2$",
            )

        ax_sigma.set_xlabel("time (hours AEL)")
        format_hours_axis(ax_sigma, time_axis)
        #ax_sigma.set_ylabel(r"1-min increments $\sigma$ ($\mu m$)")
        ax_sigma.set_ylabel(r"$\sigma_{1min}$ ($\mu m$)")
        ax_sigma.set_ylim(bottom=0.0)
        ax_D.set_ylabel(r"$D=\sigma^2/(2\Delta t)$ ($\mu m^2$/min)")
        ax_D.set_ylim(bottom=0.0)

        h1, l1 = ax_sigma.get_legend_handles_labels()
        h2, l2 = ax_D.get_legend_handles_labels()
        ax_sigma.legend(h1 + h2, l1 + l2, title=legend_title)

        fig.tight_layout()
        save_all_ext(fig, os.path.join(PLOTS_DIR, "increment-std-and-diffusion_classI_classIV"))
        plt.close(fig)

    #distribution of increments used for std per bin:
    for cls in present:
        here_basic_figsize = (project_style.mm_to_in(50), project_style.mm_to_in(50))
        fig, axs = plt.subplots(2, 3, figsize=(3 * here_basic_figsize[0], 2 * here_basic_figsize[1]),
                                sharex=True, sharey=True)
        axs = np.array(axs)
        for b in range(N_BINS):
            ax = axs[b // 3, b % 3]
            pooled = []
            acq_hists = []
            # fixed bins for comparable histograms
            all_vals = []
            for acq in bins[cls].keys():
                arr = np.asarray(bins[cls][acq][b], float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                all_vals.append(arr)
            if all_vals:
                all_concat = np.concatenate(all_vals)
                edges = np.linspace(np.nanmin(all_concat), np.nanmax(all_concat), 31)
                centers = 0.5 * (edges[:-1] + edges[1:])
                for arr in all_vals:
                    hist, _ = np.histogram(arr, bins=edges, density=True)
                    acq_hists.append(hist)

                ax.hist(all_concat, bins=edges, density=True, color=CLASS_COLOR[cls], alpha=0.35)

                H = np.vstack(acq_hists)


                if (cls == "class_IV") and (b == 3):
                    mu = float(np.mean(all_concat))
                    sd = float(np.std(all_concat, ddof=1))
                    y = ax.get_ylim()[1] * 0.85
                    ax.annotate("", xy=(mu - sd, y), xytext=(mu + sd, y),
                                arrowprops=dict(arrowstyle="<->", color="red", lw=1.8))
                    ax.text(mu, y + 0.02 * ax.get_ylim()[1], r"$\sigma$",
                            color="red", ha="center", va="bottom")

            ax.set_title(f"{CLASS_LABEL[cls]} bin {b+1} ({hour_label_from_float(time_axis[b])})")
            ax.set_xlabel("increment ($\\mu m$)")
            ax.set_ylabel("density")
            ax.tick_params(labelbottom=True, labelleft=True)

        fig.tight_layout()
        save_all_ext(fig, os.path.join(PLOTS_DIR, f"increment-distribution-pooled-bin-{cls}"))
        plt.close(fig)

    #difrt mu = mean increment per bin (per acquisition), same style
    for share_axis in ["all", "none"]:
        fig, axs = plt.subplots(2, 3, figsize=(3 * basic_figsize[0], 2 * basic_figsize[1]),
                                sharex=False, sharey=False)
        col_defs = [["class_I", "class_IV"], ["class_IV", "class_IV_MT_inh"], present]

        for col, classes_in_col in enumerate(col_defs):
            classes_in_col = [c for c in classes_in_col if c in per_class]
            for cls in classes_in_col:
                Y = per_class[cls]["Y_mean"]
                mean = np.nanmean(Y, axis=0)
                std = np.nanstd(Y, axis=0)

                for row in range(2):
                    ax = axs[row, col]
                    ax.plot(time_axis, mean, color=CLASS_COLOR[cls], label=CLASS_LABEL[cls])
                    ax.fill_between(time_axis, mean - std, mean + std, color=CLASS_COLOR[cls], alpha=0.3, linewidth=0)
                    if row == 1:
                        for k in range(Y.shape[0]):
                            ax.plot(time_axis, Y[k, :], color=CLASS_COLOR[cls], linestyle=":", alpha=0.6)

        for col in range(3):
            axs[0, col].legend()
            for row in range(2):
                ax = axs[row, col]
                ax.set_xlabel("Time (hours AEL)")
                format_hours_axis(ax, time_axis)
                ax.set_ylabel("drift = mean 1-min increment ($\\mu m$/min)")
                ax.axhline(0.0, lw=1, ls=":", color="k", alpha=0.6)

        fig.tight_layout()
        save_all_ext(fig, os.path.join(PLOTS_DIR, f"increment-drift-per-acq-mean-std-{share_axis}"))
        plt.close(fig)

    #anomalous diffusion exponent alpha per acquisition, with truncation at 300 min
    alpha_dict = {cls: {} for cls in present}
    alpha_trajs = {cls: {} for cls in present}

    for cls in present:
        for acq in dataset[cls].keys():
            trajs = gather_trajs_truncated(dataset[cls][acq], alpha_max_time_min=ALPHA_MAX_TIME_MIN)
            if MIN_LEN_MSD_FIT > 0:
                trajs = [t for t in trajs if len(t) >= MIN_LEN_MSD_FIT]
            if len(trajs) == 0:
                alpha_dict[cls][acq] = dict(alpha=np.nan, intercept=np.nan, r2=np.nan, tau_max=np.nan, tau_max_fit=np.nan)
                alpha_trajs[cls][acq] = []
                continue

            max_len = max(t.size for t in trajs)
            tau_max = min(TAU_MAX_HARD, max_len - 1)
            if tau_max < 2:
                alpha_dict[cls][acq] = dict(alpha=np.nan, intercept=np.nan, r2=np.nan, tau_max=int(tau_max), tau_max_fit=np.nan)
                continue

            tau, msd = msd_from_trajs(trajs, tau_max=tau_max)

            if TAU_MAX_FIT is None:
                tau_max_fit = int(np.clip(max_len // 3, 10, tau_max))
            else:
                tau_max_fit = min(int(TAU_MAX_FIT), int(tau_max))

            alpha, intercept, r2 = alpha_from_msd(tau, msd, tau_min_fit=TAU_MIN_FIT, tau_max_fit=tau_max_fit, min_points=MIN_POINTS_FIT)
            alpha_dict[cls][acq] = dict(alpha=float(alpha), intercept=float(intercept), r2=float(r2), tau_max=int(tau_max), tau_max_fit=int(tau_max_fit))
            alpha_trajs[cls][acq] = trajs

    summary_alpha = {}
    for cls in present:
        vals = np.array([alpha_dict[cls][acq]["alpha"] for acq in alpha_dict[cls].keys()], float)
        vals = vals[np.isfinite(vals)]
        summary_alpha[cls] = dict(
            n=int(vals.size),
            mean=float(np.mean(vals)) if vals.size else np.nan,
            std=float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
        )


    ALPHA_GLOBAL_FIGSIZE = (project_style.mm_to_in(60), project_style.mm_to_in(50))

    def plot_alpha_global(classes, out_name, add_mt_bracket=False, figsize=None):
        fig, ax = plt.subplots(1, 1, figsize=figsize or basic_figsize)
        xs = np.arange(len(classes))
        means = [summary_alpha[c]["mean"] for c in classes]
        stds = [summary_alpha[c]["std"] for c in classes]
        for i, cls in enumerate(classes):
            ax.errorbar(xs[i], means[i], yerr=stds[i], fmt="o", capsize=6,
                        color=CLASS_COLOR[cls], ecolor=CLASS_COLOR[cls])
            vals = np.array([alpha_dict[cls][acq]["alpha"] for acq in alpha_dict[cls].keys()], float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                jitter = (np.random.rand(vals.size) - 0.5) * 0.15
                ax.scatter(np.full(vals.size, xs[i]) + jitter, vals, alpha=0.7,
                           color=CLASS_COLOR[cls])
        ax.set_xticks(xs)
        ax.set_xticklabels([CLASS_LABEL[c] for c in classes], rotation=0)
        ax.set_ylabel(r"Anomalous diffusion exponent $\alpha$")
        ax.axhline(1.0, lw=1, ls=":", color="k", alpha=0.6)


        all_vals = []
        for cls in classes:
            vals = np.array([alpha_dict[cls][acq]["alpha"] for acq in alpha_dict[cls].keys()], float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                all_vals.append(np.nanmax(vals))
        max_y = np.nanmax(all_vals + [v for v in means if np.isfinite(v)] + [1.0])
        y = max_y + 0.08
        for i in range(len(classes) - 1):
            c1 = classes[i]
            c2 = classes[i + 1]
            x = np.array([alpha_dict[c1][acq]["alpha"] for acq in alpha_dict[c1].keys()], float)
            yv = np.array([alpha_dict[c2][acq]["alpha"] for acq in alpha_dict[c2].keys()], float)
            x = x[np.isfinite(x)]; yv = yv[np.isfinite(yv)]
            if x.size >= 2 and yv.size >= 2:
                _, p_t = stats.ttest_ind(yv, x, equal_var=False)
                add_bracket_star_data(ax, i, i + 1, y, p_t)
                y += 0.05
        if add_mt_bracket and ("class_I" in classes) and ("class_IV_MT_inh" in classes):
            i = classes.index("class_I")
            j = classes.index("class_IV_MT_inh")
            x = np.array([alpha_dict["class_I"][acq]["alpha"] for acq in alpha_dict["class_I"].keys()], float)
            yv = np.array([alpha_dict["class_IV_MT_inh"][acq]["alpha"] for acq in alpha_dict["class_IV_MT_inh"].keys()], float)
            x = x[np.isfinite(x)]; yv = yv[np.isfinite(yv)]
            if x.size >= 2 and yv.size >= 2:
                _, p_t = stats.ttest_ind(yv, x, equal_var=False)
                y_mt = y + 0.05
                add_bracket_star_data(ax, i, j, y_mt, p_t)
                y = y_mt + 0.05
        ax.set_ylim(top=y + 0.08)

        # p-values for alpha != 1 
        lines = []
        for cls in classes:
            vals = np.array([alpha_dict[cls][acq]["alpha"] for acq in alpha_dict[cls].keys()], float)
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2:
                _, p_t = stats.ttest_1samp(vals, 1.0)
                lines.append(f"{CLASS_LABEL[cls]}: p={p_t:.3g}")
            elif vals.size == 1:
                lines.append(f"{CLASS_LABEL[cls]}: p=NA (n=1)")
            else:
                lines.append(f"{CLASS_LABEL[cls]}: p=NA")
        if lines:
            ax.text(
                0.98, 0.02, "\n".join(lines),
                transform=ax.transAxes, ha="right", va="bottom", #fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.8)
            )
        fig.tight_layout()
        save_all_ext(fig, os.path.join(PLOTS_DIR, out_name))
        plt.close(fig)

    plot_alpha_global(present, "alpha-global-per-acq-mean-std", add_mt_bracket=True, figsize=ALPHA_GLOBAL_FIGSIZE)
    if ("class_I" in present) and ("class_IV" in present):
        plot_alpha_global(["class_I", "class_IV"], "alpha-global-per-acq-mean-std_cI_cIV", add_mt_bracket=False)

    #MSD fit per acquisition (alpha): pairwise and combined class plots
    def plot_msd_fit_classes(classes, out_name):
        fig, ax = plt.subplots(1, 1, figsize=(basic_figsize[0], basic_figsize[1]))
        cls_slopes = {}
        for cls in classes:
            msd_list = []
            for acq in dataset[cls].keys():
                trajs = alpha_trajs.get(cls, {}).get(acq, [])
                if len(trajs) == 0:
                    continue
                max_len = max(t.size for t in trajs)
                tau_max = min(TAU_MAX_HARD, max_len - 1)
                if tau_max < 2:
                    continue
                tau, msd = msd_from_trajs(trajs, tau_max=tau_max)
                msd_list.append(msd)
            if len(msd_list) == 0:
                continue
            max_tau = max(len(m) for m in msd_list)
            M = np.full((len(msd_list), max_tau), np.nan, float)
            for i, m in enumerate(msd_list):
                M[i, : len(m)] = m
            tau = np.arange(1, max_tau + 1, dtype=float)
            mean = np.nanmean(M, axis=0)
            mn = np.nanmin(M, axis=0)
            mx = np.nanmax(M, axis=0)
            m = np.isfinite(mean) & (mean > 0)
            ax.loglog(tau[m], mean[m], color=CLASS_COLOR[cls])
            ax.fill_between(tau[m], np.maximum(mn[m], 1e-12), mx[m],
                            color=CLASS_COLOR[cls], alpha=0.25)

            # single regression on mean MSD
            tau_max_fit = TAU_MAX_FIT if TAU_MAX_FIT is not None else np.nanmax(tau[m])
            tmask = m & (tau >= float(TAU_MIN_FIT)) & (tau <= float(tau_max_fit))
            if np.sum(tmask) >= MIN_POINTS_FIT:
                x = np.log(tau[tmask])
                y = np.log(mean[tmask])
                lr = stats.linregress(x, y)
                alpha = lr.slope
                intercept = lr.intercept
                t_fit = np.linspace(max(float(TAU_MIN_FIT), 1.0), float(tau_max_fit), 50)
                msd_fit = np.exp(intercept) * (t_fit ** alpha)
                ax.loglog(t_fit, msd_fit, color=CLASS_COLOR[cls], ls="--")
                cls_slopes[cls] = alpha

        ax.set_xlabel("lag $\\tau$ (min)")
        ax.set_ylabel("MSD$(\\tau)$")
        handles = []
        for c in classes:
            if 'cls_slopes' in locals() and c in cls_slopes:
                mu = summary_alpha.get(c, {}).get("mean", np.nan)
                sd = summary_alpha.get(c, {}).get("std", np.nan)
                if np.isfinite(mu) and np.isfinite(sd):
                    label = f"{CLASS_LABEL[c]} $\\alpha$={mu:.2f}±{sd:.2f}"
                elif np.isfinite(mu):
                    label = f"{CLASS_LABEL[c]} $\\alpha$={mu:.2f}"
                else:
                    label = CLASS_LABEL[c]
            else:
                label = CLASS_LABEL[c]
            handles.append(plt.Line2D([0],[0], color=CLASS_COLOR[c], label=label))
        ax.legend(handles=handles, frameon=False)
        fig.tight_layout()
        save_all_ext(fig, os.path.join(PLOTS_DIR, out_name))
        plt.close(fig)

    if ("class_I" in present) and ("class_IV" in present):
        plot_msd_fit_classes(["class_I", "class_IV"], "alpha-global-msd-fit-per-acq_cI_cIV")
    if ("class_IV" in present) and ("class_IV_MT_inh" in present):
        plot_msd_fit_classes(["class_IV", "class_IV_MT_inh"], "alpha-global-msd-fit-per-acq_cIV_cIVMTinh")
    if ("class_I" in present) and ("class_IV_MT_inh" in present):
        plot_msd_fit_classes(["class_I", "class_IV_MT_inh"], "alpha-global-msd-fit-per-acq_cI_cIVMTinh")
    if all(cls in present for cls in ["class_I", "class_IV", "class_IV_MT_inh"]):
        plot_msd_fit_classes(
            ["class_I", "class_IV", "class_IV_MT_inh"],
            "alpha-global-msd-fit-per-acq_cI_cIV_cIVMTinh",
        )

    #trajectories used for alpha computation (aligned to t=0, x=0)
    def plot_alpha_trajs(classes, out_name, max_traj_per_class=1000, tmax=60):
        fig, ax = plt.subplots(1, 1, figsize=(basic_figsize[0], basic_figsize[1]))
        for cls in classes:
            n_plotted = 0
            for acq in dataset[cls].keys():
                trajs = alpha_trajs.get(cls, {}).get(acq, [])
                for t in trajs:
                    t = np.asarray(t, float)
                    if t.size == 0:
                        continue
                    #split into non-overlapping segments of length tmax 
                    seg_len = int(tmax) + 1
                    nseg = t.size // seg_len
                    if nseg < 1:
                        continue
                    z = 3 if cls == "class_I" else 2
                    for s in range(nseg):
                        seg = t[s * seg_len:(s + 1) * seg_len]
                        if seg.size < seg_len:
                            continue
                        x = seg - seg[0]
                        tt = np.arange(x.size, dtype=float)
                        ax.plot(tt, x, color=CLASS_COLOR[cls], alpha=0.2, zorder=z)
                        n_plotted += 1
                        if n_plotted >= max_traj_per_class:
                            break
                    if n_plotted >= max_traj_per_class:
                        break
                if n_plotted >= max_traj_per_class:
                    break
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Displacement ($\\mu m$)")
        ax.set_xlim(0, tmax)
        handles = [plt.Line2D([0],[0], color=CLASS_COLOR[c], label=CLASS_LABEL[c]) for c in classes]
        ax.legend(handles=handles, frameon=False)
        fig.tight_layout()
        save_all_ext(fig, os.path.join(PLOTS_DIR, out_name))
        plt.close(fig)

    if ("class_I" in present) and ("class_IV" in present):
        plot_alpha_trajs(["class_I", "class_IV"], "alpha-trajs-used_cI_cIV")
    if ("class_IV" in present) and ("class_IV_MT_inh" in present):
        plot_alpha_trajs(["class_IV", "class_IV_MT_inh"], "alpha-trajs-used_cIV_cIVMTinh")

    #pairwise alpha statistical tests 
    for i in range(len(present)):
        for j in range(i):
            c1 = present[i]
            c2 = present[j]
            x = np.array([alpha_dict[c1][acq]["alpha"] for acq in alpha_dict[c1].keys()], float)
            y = np.array([alpha_dict[c2][acq]["alpha"] for acq in alpha_dict[c2].keys()], float)
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if x.size < 2 or y.size < 2:
                continue
            _, p_t = stats.ttest_ind(y, x, equal_var=False)
            try:
                _, p_u = stats.mannwhitneyu(y, x, alternative="two-sided")
            except Exception:
                p_u = np.nan
    #save tables
    rows_alpha_global = []
    for cls in present:
        vals = np.array([alpha_dict[cls][acq]["alpha"] for acq in alpha_dict[cls].keys()], float)
        vals = vals[np.isfinite(vals)]
        n_vals = int(vals.size)
        mean_alpha = float(np.mean(vals)) if n_vals else np.nan
        std_alpha = float(np.std(vals, ddof=1)) if n_vals > 1 else np.nan
        se_alpha = float(std_alpha / np.sqrt(n_vals)) if n_vals > 1 and np.isfinite(std_alpha) else np.nan
        df_alpha = float(n_vals - 1) if n_vals > 1 else np.nan
        if vals.size >= 2:
            tt = stats.ttest_1samp(vals, 1.0)
            t_alpha_ne1 = float(tt.statistic)
            p_alpha_ne1 = float(tt.pvalue)
            tcrit = float(stats.t.ppf(1 - 0.05 / 2.0, df_alpha)) if np.isfinite(df_alpha) and df_alpha > 0 else np.nan
            ci95_lo = float(mean_alpha - tcrit * se_alpha) if np.isfinite(tcrit) and np.isfinite(se_alpha) else np.nan
            ci95_hi = float(mean_alpha + tcrit * se_alpha) if np.isfinite(tcrit) and np.isfinite(se_alpha) else np.nan
        else:
            t_alpha_ne1 = np.nan
            p_alpha_ne1 = np.nan
            ci95_lo = np.nan
            ci95_hi = np.nan
        rows_alpha_global.append(
            dict(
                cls=cls,
                class_label=CLASS_LABEL[cls],
                n_alpha_global=n_vals,
                alpha_global_mean=mean_alpha,
                alpha_global_std=std_alpha,
                alpha_global_ci95_low=ci95_lo,
                alpha_global_ci95_high=ci95_hi,
                alpha_test_name="One-sample t-test against 1",
                alpha_test_stat_t=t_alpha_ne1,
                alpha_test_df=df_alpha,
                alpha_test_h0="H0: mean alpha = 1",
                alpha_test_sidedness="two-sided",
                p_alpha_global_ne_1=p_alpha_ne1,
            )
        )
    df_alpha_global = pd.DataFrame(rows_alpha_global)
    path_alpha_md = os.path.join(RESULT_DIR, "table_alpha_global_summary.md")
    path_alpha_csv = os.path.join(RESULT_DIR, "table_alpha_global_summary.csv")
    df_alpha_global_round = df_alpha_global.round(6)
    _write_df_markdown(df_alpha_global_round, path_alpha_md)
    _write_df_csv(df_alpha_global, path_alpha_csv)

    #tables
    rows_sigma_decrease = []
    for cls in present:
        Y = per_class[cls]["Y_std"]
        n_exp, n_t = Y.shape
        a = np.tile(time_axis, n_exp)
        exp = np.repeat(np.arange(n_exp), n_t)
        df = pd.DataFrame({"y": Y.ravel(), "a": a, "exp": exp})
        df = df[np.isfinite(df["y"]) & np.isfinite(df["a"])].copy()
        if df.empty:
            rows_sigma_decrease.append(
                dict(
                    cls=cls,
                    class_label=CLASS_LABEL[cls],
                    n_experiments=n_exp,
                    n_timepoints=n_t,
                    slope_per_hour=np.nan,
                    slope_se=np.nan,
                    slope_ci95_low=np.nan,
                    slope_ci95_high=np.nan,
                    p_two_sided=np.nan,
                    p_decrease_one_sided=np.nan,
                    decrease_significant_0p05=False,
                    test_model="cluster_robust_FE",
                )
            )
            continue
        try:
            models, _ = slope_intercept_with_robust_and_cluster(
                df, y="y", a="a", exp="exp", alpha=0.05
            )
            model_for_test = models.get("fe_cluster", models.get("hc3", models["base"]))
            slope, slope_se, slope_ci95_low, slope_ci95_high, p_two_sided, p_decrease = extract_slope_stats_from_model(model_for_test, term="a")
        except Exception as e:
            slope, slope_se, slope_ci95_low, slope_ci95_high, p_two_sided, p_decrease = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        rows_sigma_decrease.append(
            dict(
                cls=cls,
                class_label=CLASS_LABEL[cls],
                n_experiments=n_exp,
                n_timepoints=n_t,
                slope_per_hour=slope,
                slope_se=slope_se,
                slope_ci95_low=slope_ci95_low,
                slope_ci95_high=slope_ci95_high,
                p_two_sided=p_two_sided,
                p_decrease_one_sided=p_decrease,
                decrease_significant_0p05=bool(np.isfinite(p_decrease) and (p_decrease < 0.05)),
                test_model="cluster_robust_FE",
            )
        )
    df_sigma_decrease = pd.DataFrame(rows_sigma_decrease)

    rows_sigma_pair = []
    slope_by_class = {}
    for _, r in df_sigma_decrease.iterrows():
        slope_by_class[str(r["cls"])] = float(r["slope_per_hour"]) if np.isfinite(r["slope_per_hour"]) else np.nan
    pair_classes = [c for c in PREFERRED_ORDER if c in per_class]
    for i in range(len(pair_classes)):
        for j in range(i):
            c1 = pair_classes[j]
            c2 = pair_classes[i]
            try:
                _, m_pair, stats_pair = compare_two_series(
                    per_class[c1]["Y_std"],
                    per_class[c2]["Y_std"],
                    time_axis,
                    center_at=level_anchor_hour,
                    return_per_time=False,
                )
                p_level = float(stats_pair.get("p_level_diff", np.nan))
                p_slope = float(stats_pair.get("p_slope_diff", np.nan))
                eff = extract_level_slope_effects_from_model(m_pair)
            except Exception as e:
                p_level, p_slope = (np.nan, np.nan)
                eff = {
                    "level_diff": np.nan,
                    "level_ci95_low": np.nan,
                    "level_ci95_high": np.nan,
                    "slope_diff": np.nan,
                    "slope_ci95_low": np.nan,
                    "slope_ci95_high": np.nan,
                }
            level_a = float(np.nanmean(per_class[c1]["Y_std"][:, 0])) if per_class[c1]["Y_std"].size else np.nan
            level_b = float(np.nanmean(per_class[c2]["Y_std"][:, 0])) if per_class[c2]["Y_std"].size else np.nan
            rows_sigma_pair.append(
                dict(
                    group_a=CLASS_LABEL[c1],
                    group_b=CLASS_LABEL[c2],
                    n_a=int(per_class[c1]["Y_std"].shape[0]),
                    n_b=int(per_class[c2]["Y_std"].shape[0]),
                    level_a_anchor=float(level_a),
                    level_b_anchor=float(level_b),
                    level_diff_a_minus_b=eff["level_diff"],
                    level_diff_ci95=f"[{eff['level_ci95_low']:.6g}, {eff['level_ci95_high']:.6g}]"
                    if (np.isfinite(eff["level_ci95_low"]) and np.isfinite(eff["level_ci95_high"])) else "NA",
                    p_level_diff=p_level,
                    slope_a=slope_by_class.get(c1, np.nan),
                    slope_b=slope_by_class.get(c2, np.nan),
                    slope_diff_a_minus_b=eff["slope_diff"],
                    slope_diff_ci95=f"[{eff['slope_ci95_low']:.6g}, {eff['slope_ci95_high']:.6g}]"
                    if (np.isfinite(eff["slope_ci95_low"]) and np.isfinite(eff["slope_ci95_high"])) else "NA",
                    p_slope_diff=p_slope,
                    test_model="compare_two_series_cluster_robust",
                )
            )
    df_sigma_pair = pd.DataFrame(rows_sigma_pair)

    path_sigma_dec_md = os.path.join(RESULT_DIR, "table_sigma_1min_decrease_cluster_ols.md")
    path_sigma_dec_csv = os.path.join(RESULT_DIR, "table_sigma_1min_decrease_cluster_ols.csv")
    _write_df_markdown(df_sigma_decrease.round(6), path_sigma_dec_md)
    _write_df_csv(df_sigma_decrease, path_sigma_dec_csv)

    path_sigma_pair_md = os.path.join(RESULT_DIR, "table_sigma_1min_pairwise_all_cluster_ols.md")
    path_sigma_pair_csv = os.path.join(RESULT_DIR, "table_sigma_1min_pairwise_all_cluster_ols.csv")
    _write_df_markdown(df_sigma_pair.round(6), path_sigma_pair_md)
    _write_df_csv(df_sigma_pair, path_sigma_pair_csv)

    # 9) Raw data exports
    # Raw branch-level records JSON (same list-of-dicts idea as other scripts).
    with open(RAW_JSON_PATH, "w") as f:
        json.dump(_dataset_to_branch_records(dataset_raw), f, indent=2)

    sigma_wide = {"time (h) AEL": [hour_label_from_float(t) for t in time_axis]}
    mu_wide = {"time (h) AEL": [hour_label_from_float(t) for t in time_axis]}
    rows_alpha = []
    for cls in present:
        acq_names = per_class[cls]["acq_names"]
        Y_sigma = per_class[cls]["Y_std"]
        Y_mu = per_class[cls]["Y_mean"]
        acq_to_alpha = alpha_dict.get(cls, {})

        for i, acq in enumerate(acq_names):
            col = f"{cls}_{acq}"
            sigma_wide[col] = Y_sigma[i, :].tolist()
            mu_wide[col] = Y_mu[i, :].tolist()
            rows_alpha.append(
                {
                    "class": cls,
                    "acquisition": acq,
                    "alpha_global": acq_to_alpha.get(acq, {}).get("alpha", np.nan),
                }
            )

    _write_df_csv(pd.DataFrame(sigma_wide), os.path.join(RAW_DATA_DIR, "sigma_per_acquisition_timebin.csv"))
    _write_df_csv(pd.DataFrame(mu_wide), os.path.join(RAW_DATA_DIR, "mu_per_acquisition_timebin.csv"))
    _write_df_csv(pd.DataFrame(rows_alpha), os.path.join(RAW_DATA_DIR, "alpha_per_acquisition.csv"))

if __name__ == "__main__":
    main()
