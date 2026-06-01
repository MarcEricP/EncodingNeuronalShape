import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress, ttest_ind
from skimage import io, morphology
import tqdm

from dendrogenesis import project_style
import dendrotree


project_style.set_style()
plt.rcParams.update({"font.size": 20})


BASE_DIR_CANDIDATES = [
    "/mnt/c/Users/menir/Documents/0000NeuronsData/tubulin_test_cache",
]
WORKING_DIR = os.getcwd()
RESULT_DIR = os.path.join(WORKING_DIR, "DataAnalysis", "Result", "8_tubulin_mt_only_invasion")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
RAW_DATA_DIR = os.path.join(RESULT_DIR, "raw_data")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

SOURCE_DIRS = {
    "cI": "tubulin_class_I",
    "cIV": "tubulin_class_IV",
}
CLASS_LABEL = {
    "cI": "WT cI",
    "cIV": "WT cIV",
}
REQUESTED_COLOR = {
    "cI": "tab:green",
    "cIV": "tab:pink",
}
DEFAULT_TUBULIN_LABEL = {
    "cI": 1,
    "cIV": 1,
}
PIXELS_PER_MICRON = 16.0
EXCLUDED_DIRS = {"flow_track", "flow_track_opened_skel"}

# These acquisitions were explicitly skipped in the original analysis.
SKIP_ACQUISITIONS = {
    "25.06.11 RluvCM JF649 1 min",
    "25.07.02 RluvCM JF646 1min",
    "C2-MAX_25-09-17-ppkIII_TubmNG_Halo_1min_6h_concatenated-contrast_vaguely_ok_croped-1",
    "C2-MAX_25-09-11-ppkIII_TubmNG_Halo_1min_4h_w1CSU 491_t121_cropped-1",
    "C2-MAX_25-07-21-ppkIII_TubmNG_Halo_e2_2min_3h_w1CSU 491_t1-1",
}
ACQUISITION_OVERRIDES = {
    "1_rescaled_25.06.11 RluvCM JF649 1 min movie-1": {"offset_16AEL_min": 0, "tubulin_label": 1},
    "2_rescaled_25.06.11 RluvCM JF649 1 min movie-1": {"offset_16AEL_min": 0, "tubulin_label": 9},
    "rescaled_20241001_e1_ev1min_8h_Tub+CAAX_Merged_Crop1-1": {"offset_16AEL_min": -110, "tubulin_label": 9},
    "Elise-Movie_every2min_12h10_proj-1-2": {"dt_min": 2},
    "Elise-e1-ev2min_8h_stk-1-1-2": {"dt_min": 2},
}


def _find_base_dir():
    for candidate in BASE_DIR_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"No tubulin base directory found in: {BASE_DIR_CANDIDATES}")


def _save_all_ext_no_square_patch(fig, path_wo_ext, dpi=300):
    """
    Save using the original Figure.savefig to avoid the global square-axes patch
    from project_style on figures that intentionally use non-square axes.
    """
    orig_savefig = getattr(project_style, "_ORIG_FIGURE_SAVEFIG", None)
    for ext in ("png", "pdf", "svg"):
        out = f"{path_wo_ext}.{ext}"
        if callable(orig_savefig):
            orig_savefig(fig, out, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out, dpi=dpi, bbox_inches="tight")


def _write_df_markdown(df, path):
    with open(path, "w") as f:
        f.write("| " + " | ".join(str(c) for c in df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
        for row in df.to_numpy():
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def _fmt_mean_se(mean, se, digits_mean=4, digits_se=6):
    if not (np.isfinite(mean) and np.isfinite(se)):
        return "NA"
    return f"{mean:.{digits_mean}f} ± {se:.{digits_se}f}"


def _fmt_ci(low, high, digits=4):
    if not (np.isfinite(low) and np.isfinite(high)):
        return "NA"
    return f"[{low:.{digits}f},{high:.{digits}f}]"


def _fmt_p(p):
    if not np.isfinite(p):
        return "p = NA"
    if p < 1e-3:
        return "p < 0.001"
    return f"p = {p:.3f}".rstrip("0").rstrip(".")


def _iter_skeleton_paths(root_dir):
    for root, dirs, _files in os.walk(root_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        skel_path = os.path.join(root, "claire_annotation", "skel_stack.tif")
        if os.path.exists(skel_path):
            yield root, skel_path


def _sanitize_for_filename(name):
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "acquisition"


def _cache_csv_path(class_key, acquisition_name):
    fname = f"class_{class_key}__{_sanitize_for_filename(acquisition_name)}.csv"
    return os.path.join(RAW_DATA_DIR, fname)


def _get_acquisition_config(class_key, acquisition_name):
    cfg = {
        "dt_min": 1,
        "offset_16AEL_min": 0,
        "tubulin_label": DEFAULT_TUBULIN_LABEL[class_key],
    }
    cfg.update(ACQUISITION_OVERRIDES.get(acquisition_name, {}))
    return cfg


def _sample_frames(class_key, stack_shape_0, dt_min):
    if class_key == "cI":
        return None
    step = max(1, int(round(10 / dt_min)))
    return np.arange(0, stack_shape_0, step, dtype=int)


def _safe_linregress(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None
    return linregress(x, y)


def _compute_acquisition_metrics(class_key, acquisition_name, skel_path):
    cfg = _get_acquisition_config(class_key, acquisition_name)
    stack = io.imread(skel_path)
    if acquisition_name in SKIP_ACQUISITIONS:
        return None

    if class_key == "cI":
        frame_idx = np.sort(np.unique(np.nonzero(stack)[0]))
    else:
        frame_idx = _sample_frames(class_key, stack.shape[0], cfg["dt_min"])

    if frame_idx is None or len(frame_idx) == 0:
        return None

    tubulin_lengths = []
    n_tubulin_tips = []
    time_min = []

    for t in tqdm.tqdm(frame_idx, desc=f"{CLASS_LABEL[class_key]} {acquisition_name}", leave=False):
        tubulin_mask = morphology.skeletonize(stack[t] == cfg["tubulin_label"])
        try:
            graph_tub = dendrotree.nx_graph.nx_graph.mask2nx(tubulin_mask)
            n_tips = len(dendrotree.nx_graph.nx_graph.get_tips_and_junctions(graph_tub)[0])
            tub_len = dendrotree.nx_graph.metrics.total_length(graph_tub) / PIXELS_PER_MICRON
        except Exception:
            n_tips = np.nan
            tub_len = np.nan

        n_tubulin_tips.append(n_tips)
        tubulin_lengths.append(tub_len)
        time_min.append((float(t) + cfg["offset_16AEL_min"]) * cfg["dt_min"])

    time_min = np.asarray(time_min, dtype=float)
    if class_key == "cI" and time_min.size:
        # Ensure every class-I trajectory starts at 0 min.
        time_min = time_min - np.nanmin(time_min)
    tubulin_lengths = np.asarray(tubulin_lengths, dtype=float)
    n_tubulin_tips = np.asarray(n_tubulin_tips, dtype=float)

    tubulin_reg = _safe_linregress(time_min, tubulin_lengths)
    mean_tips = float(np.nanmean(n_tubulin_tips)) if np.isfinite(n_tubulin_tips).any() else np.nan
    speed_per_tip = np.nan
    if tubulin_reg is not None and np.isfinite(mean_tips) and mean_tips > 0:
        speed_per_tip = float(tubulin_reg.slope / mean_tips)

    timeseries_rows = []
    for i, t in enumerate(frame_idx):
        timeseries_rows.append(
            {
                "class": CLASS_LABEL[class_key],
                "class_key": class_key,
                "acquisition": acquisition_name,
                "frame_index": int(t),
                "time_min": float(time_min[i]),
                "tubulin_length_um": float(tubulin_lengths[i]) if np.isfinite(tubulin_lengths[i]) else np.nan,
                "n_tubulin_tips": float(n_tubulin_tips[i]) if np.isfinite(n_tubulin_tips[i]) else np.nan,
            }
        )

    summary_row = {
        "class": CLASS_LABEL[class_key],
        "class_key": class_key,
        "acquisition": acquisition_name,
        "dt_min": cfg["dt_min"],
        "offset_16AEL_min": cfg["offset_16AEL_min"],
        "tubulin_label": cfg["tubulin_label"],
        "n_timepoints": int(len(time_min)),
        "mean_n_tubulin_tips": mean_tips,
        "slope_tubulin_length_um_per_min": tubulin_reg.slope if tubulin_reg is not None else np.nan,
        "p_tubulin_length": tubulin_reg.pvalue if tubulin_reg is not None else np.nan,
        "r_tubulin_length": tubulin_reg.rvalue if tubulin_reg is not None else np.nan,
        "speed_tubulin_per_tip_um_per_min": speed_per_tip,
    }

    plot_payload = {
        "time_min": time_min,
        "tubulin_length_um": tubulin_lengths,
        "tubulin_reg": tubulin_reg,
    }
    return timeseries_rows, summary_row, plot_payload


def _build_from_timeseries_df(timeseries_df, class_key, acquisition_name):
    cfg = _get_acquisition_config(class_key, acquisition_name)
    if "frame_index" in timeseries_df.columns:
        adf = timeseries_df.sort_values("frame_index").reset_index(drop=True)
    else:
        adf = timeseries_df.sort_values("time_min").reset_index(drop=True)
    # Normalize cache-loaded rows so downstream pivots/groupbys always have these keys.
    adf["class"] = CLASS_LABEL[class_key]
    adf["class_key"] = class_key
    adf["acquisition"] = acquisition_name

    time_min = adf["time_min"].to_numpy(float)
    tubulin_lengths = adf["tubulin_length_um"].to_numpy(float)
    n_tubulin_tips = adf["n_tubulin_tips"].to_numpy(float)

    tubulin_reg = _safe_linregress(time_min, tubulin_lengths)
    mean_tips = float(np.nanmean(n_tubulin_tips)) if np.isfinite(n_tubulin_tips).any() else np.nan
    speed_per_tip = np.nan
    if tubulin_reg is not None and np.isfinite(mean_tips) and mean_tips > 0:
        speed_per_tip = float(tubulin_reg.slope / mean_tips)

    summary_row = {
        "class": CLASS_LABEL[class_key],
        "class_key": class_key,
        "acquisition": acquisition_name,
        "dt_min": cfg["dt_min"],
        "offset_16AEL_min": cfg["offset_16AEL_min"],
        "tubulin_label": cfg["tubulin_label"],
        "n_timepoints": int(len(time_min)),
        "mean_n_tubulin_tips": mean_tips,
        "slope_tubulin_length_um_per_min": tubulin_reg.slope if tubulin_reg is not None else np.nan,
        "p_tubulin_length": tubulin_reg.pvalue if tubulin_reg is not None else np.nan,
        "r_tubulin_length": tubulin_reg.rvalue if tubulin_reg is not None else np.nan,
        "speed_tubulin_per_tip_um_per_min": speed_per_tip,
    }
    plot_payload = {
        "time_min": time_min,
        "tubulin_length_um": tubulin_lengths,
        "tubulin_reg": tubulin_reg,
    }
    return adf.to_dict("records"), summary_row, plot_payload


def _build_speed_stats(summary_df):
    ci = summary_df.loc[summary_df["class_key"] == "cI", "speed_tubulin_per_tip_um_per_min"].to_numpy(dtype=float)
    civ = summary_df.loc[summary_df["class_key"] == "cIV", "speed_tubulin_per_tip_um_per_min"].to_numpy(dtype=float)
    ci = ci[np.isfinite(ci)]
    civ = civ[np.isfinite(civ)]
    mean_ci = float(np.mean(ci)) if ci.size else np.nan
    mean_civ = float(np.mean(civ)) if civ.size else np.nan
    mean_diff = float(mean_ci - mean_civ) if (ci.size and civ.size) else np.nan
    welch_df = np.nan
    ci95_low = np.nan
    ci95_high = np.nan
    if ci.size >= 2 and civ.size >= 2:
        test = ttest_ind(ci, civ, equal_var=False)
        t_stat = float(test.statistic)
        p_value = float(test.pvalue)
        s2_ci = float(np.var(ci, ddof=1))
        s2_civ = float(np.var(civ, ddof=1))
        v1 = s2_ci / ci.size
        v2 = s2_civ / civ.size
        se_diff = float(np.sqrt(v1 + v2))
        denom = (v1 ** 2) / (ci.size - 1) + (v2 ** 2) / (civ.size - 1)
        welch_df = float(((v1 + v2) ** 2) / denom) if denom > 0 else np.nan
        if np.isfinite(welch_df) and np.isfinite(se_diff) and se_diff > 0:
            tcrit = float(stats.t.ppf(1 - 0.05 / 2.0, welch_df))
            ci95_low = float(mean_diff - tcrit * se_diff)
            ci95_high = float(mean_diff + tcrit * se_diff)
    else:
        t_stat = np.nan
        p_value = np.nan
    return pd.DataFrame(
        [
            {
                "metric": "speed_tubulin_per_tip_um_per_min",
                "group_a": "WT cI",
                "group_b": "WT cIV",
                "n_neurons_cI": int(ci.size),
                "n_neurons_cIV": int(civ.size),
                "mean_cI": mean_ci,
                "mean_cIV": mean_civ,
                "sd_cI": float(np.std(ci, ddof=1)) if ci.size > 1 else np.nan,
                "sd_cIV": float(np.std(civ, ddof=1)) if civ.size > 1 else np.nan,
                "mean_diff_cI_minus_cIV": mean_diff,
                "mean_diff_ci95_low": ci95_low,
                "mean_diff_ci95_high": ci95_high,
                "test": "Welch_t_test",
                "t_stat": t_stat,
                "welch_df": welch_df,
                "sidedness": "two-sided",
                "p_value": p_value,
            }
        ]
    )


def _build_requested_table(stats_df):
    row = stats_df.iloc[0]
    se_ci = row["sd_cI"] / np.sqrt(row["n_neurons_cI"]) if row["n_neurons_cI"] > 1 and np.isfinite(row["sd_cI"]) else np.nan
    se_civ = row["sd_cIV"] / np.sqrt(row["n_neurons_cIV"]) if row["n_neurons_cIV"] > 1 and np.isfinite(row["sd_cIV"]) else np.nan
    return pd.DataFrame(
        [
            {
                "Metric tested": "Microtubule expansion speed per branch tip",
                "Group A": f"Wild type class I (n={int(row['n_neurons_cI'])})",
                "Group B": f"Wild type class IV (n={int(row['n_neurons_cIV'])})",
                "Mean in class I ± SE (µm/min)": _fmt_mean_se(row["mean_cI"], se_ci),
                "Mean in class IV ± SE (µm/min)": _fmt_mean_se(row["mean_cIV"], se_civ),
                "Mean difference: class I minus class IV": f"{row['mean_diff_cI_minus_cIV']:.4f}" if np.isfinite(row["mean_diff_cI_minus_cIV"]) else "NA",
                "Mean difference 95% CI": _fmt_ci(row["mean_diff_ci95_low"], row["mean_diff_ci95_high"]),
                "Two-sided Welch t-test (H0: mean in class I = mean in class IV)": _fmt_p(row["p_value"]),
                "Test statistic": f"{row['t_stat']:.2f}" if np.isfinite(row["t_stat"]) else "NA",
                "Test Degrees of freedom": f"{row['welch_df']:.2f}" if np.isfinite(row["welch_df"]) else "NA",
            }
        ]
    )


def _add_bracket_star_data(ax, x1, x2, y, p):
    if np.isfinite(p):
        label = f"p={p:.3g}"
    else:
        label = "p=NA"
    dy = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], color="k", lw=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + dy, label, ha="center", va="bottom")


def _plot_requested_three_panel(timeseries_df, summary_df, stats_df):
    width_ratios = [1.0, 1.0, 0.5]
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(project_style.mm_to_in(190), project_style.mm_to_in(70)),
        gridspec_kw={"width_ratios": width_ratios},
    )

    # Panel 1: MT network length over time (all trajectories)
    ax = axs[0]
    for class_key in ["cI", "cIV"]:
        sdf = timeseries_df[timeseries_df["class_key"] == class_key]
        for acq, adf in sdf.groupby("acquisition", sort=True):
            adf = adf.sort_values("time_min")
            ax.plot(
                adf["time_min"].to_numpy(float),
                adf["tubulin_length_um"].to_numpy(float),
                color=REQUESTED_COLOR[class_key],
                alpha=0.7,
                lw=1.5,
                linestyle="-",
                label=CLASS_LABEL[class_key] if acq == sorted(sdf["acquisition"].unique())[0] else None,
            )
    # ax.set_title("MT network over time")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"MT network length (µm)")
    ax.set_box_aspect(1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen and l is not None:
                seen.add(l)
                uniq_h.append(h)
                uniq_l.append(l)
        ax.legend(uniq_h, uniq_l, frameon=False, loc="best")

    # Panel 2: Number of MT tips over time (all trajectories)
    ax = axs[1]
    for class_key in ["cI", "cIV"]:
        sdf = timeseries_df[timeseries_df["class_key"] == class_key]
        for acq, adf in sdf.groupby("acquisition", sort=True):
            adf = adf.sort_values("time_min")
            ax.plot(
                adf["time_min"].to_numpy(float),
                adf["n_tubulin_tips"].to_numpy(float),
                color=REQUESTED_COLOR[class_key],
                alpha=0.7,
                lw=1.5,
                linestyle="-",
                label=CLASS_LABEL[class_key] if acq == sorted(sdf["acquisition"].unique())[0] else None,
            )
    # ax.set_title("MT tips over time")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Number of MT tips")
    ax.set_box_aspect(1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen and l is not None:
                seen.add(l)
                uniq_h.append(h)
                uniq_l.append(l)
        ax.legend(uniq_h, uniq_l, frameon=False, loc="best")

    # Panel 3: Average MT invasion velocity per tip + significance bracket
    ax = axs[2]
    ci = summary_df.loc[summary_df["class_key"] == "cI", "speed_tubulin_per_tip_um_per_min"].to_numpy(float)
    civ = summary_df.loc[summary_df["class_key"] == "cIV", "speed_tubulin_per_tip_um_per_min"].to_numpy(float)
    ci = ci[np.isfinite(ci)]
    civ = civ[np.isfinite(civ)]
    groups = [ci, civ]
    class_order = ["cI", "cIV"]
    xs = np.arange(2)
    rng = np.random.default_rng(0)
    for i, (vals, class_key) in enumerate(zip(groups, class_order)):
        if vals.size:
            jitter = rng.uniform(-0.08, 0.08, size=vals.size)
            ax.scatter(np.full(vals.size, xs[i]) + jitter, vals, color=REQUESTED_COLOR[class_key], alpha=0.8, s=40)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            ax.errorbar(xs[i], mean, yerr=sd, fmt="o", color=REQUESTED_COLOR[class_key], ecolor=REQUESTED_COLOR[class_key], capsize=6, lw=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([CLASS_LABEL[c] for c in class_order], rotation=0)
    ax.set_ylabel(r"MT invasion velocity per tip (µm/min)")
    # ax.set_title("MT invasion velocity per tip")
    ax.set_aspect("auto", adjustable="box")
    # Match panel-1/2 height while keeping a narrower panel-3 width.
    ax.set_box_aspect(width_ratios[0] / width_ratios[2])
    p_value = float(stats_df["p_value"].iloc[0]) if ("p_value" in stats_df.columns and len(stats_df) > 0) else np.nan
    y_max = np.nanmax(np.concatenate([ci, civ])) if (ci.size or civ.size) else 1.0
    y_min = np.nanmin(np.concatenate([ci, civ])) if (ci.size or civ.size) else 0.0
    pad = 0.15 * max(1e-9, (y_max - y_min))
    y_bracket = y_max + pad
    ax.set_ylim(bottom=-0.001, top=y_bracket + pad)
    _add_bracket_star_data(ax, xs[0], xs[1], y_bracket, p_value)

    fig.tight_layout()
    return fig


def main():
    timeseries_rows = []
    summary_rows = []
    plot_data = {"cI": {}, "cIV": {}}
    base_dir = None
    try:
        base_dir = _find_base_dir()
    except FileNotFoundError:
        base_dir = None

    if base_dir is not None:
        for class_key, rel_dir in SOURCE_DIRS.items():
            class_root = os.path.join(base_dir, rel_dir)
            for root, skel_path in _iter_skeleton_paths(class_root):
                acquisition_name = os.path.basename(root)
                cache_path = _cache_csv_path(class_key, acquisition_name)
                if os.path.exists(cache_path):
                    adf = pd.read_csv(cache_path)
                    result = _build_from_timeseries_df(adf, class_key, acquisition_name)
                else:
                    result = _compute_acquisition_metrics(class_key, acquisition_name, skel_path)
                    if result is not None:
                        acq_timeseries_cache, _, _ = result
                        pd.DataFrame(acq_timeseries_cache)[
                            ["time_min", "frame_index", "tubulin_length_um", "n_tubulin_tips"]
                        ].to_csv(cache_path, index=False)
                if result is None:
                    continue
                acq_timeseries, acq_summary, acq_plot = result
                timeseries_rows.extend(acq_timeseries)
                summary_rows.append(acq_summary)
                plot_data[class_key][acquisition_name] = acq_plot
    else:
        cache_files = [
            p for p in os.listdir(RAW_DATA_DIR)
            if p.endswith(".csv") and p.startswith("class_") and "__" in p
        ]
        for fname in sorted(cache_files):
            fpath = os.path.join(RAW_DATA_DIR, fname)
            class_part, acq_part = fname[:-4].split("__", 1)
            class_key = class_part.replace("class_", "")
            if class_key not in {"cI", "cIV"}:
                continue
            adf = pd.read_csv(fpath)
            if adf.empty:
                continue
            acquisition_name = str(adf["acquisition"].iloc[0]) if "acquisition" in adf.columns else acq_part
            result = _build_from_timeseries_df(adf, class_key, acquisition_name)
            acq_timeseries, acq_summary, acq_plot = result
            timeseries_rows.extend(acq_timeseries)
            summary_rows.append(acq_summary)
            plot_data[class_key][acquisition_name] = acq_plot

    if not summary_rows:
        raise RuntimeError("No Claire tubulin acquisitions were found (dataset and cache both unavailable).")

    timeseries_df = pd.DataFrame(timeseries_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(["class_key", "acquisition"]).reset_index(drop=True)
    stats_df = _build_speed_stats(summary_df)

    requested_table = _build_requested_table(stats_df)

    summary_df.to_csv(os.path.join(RAW_DATA_DIR, "tubulin_mt_only_claire_cI_cIV_metrics.csv"), index=False)
    wide_tubulin = (
        timeseries_df.pivot_table(index="time_min", columns="acquisition", values="tubulin_length_um", aggfunc="first")
        .sort_index()
    )
    wide_tubulin.index.name = "time (min)"
    wide_tubulin.to_csv(os.path.join(RAW_DATA_DIR, "tubulin_mt_only_claire_cI_cIV_timeseries_tubulin_length.csv"))

    wide_tips = (
        timeseries_df.pivot_table(index="time_min", columns="acquisition", values="n_tubulin_tips", aggfunc="first")
        .sort_index()
    )
    wide_tips.index.name = "time (min)"
    wide_tips.to_csv(os.path.join(RAW_DATA_DIR, "tubulin_mt_only_claire_cI_cIV_timeseries_n_tubulin_tips.csv"))
    requested_table.to_csv(os.path.join(RESULT_DIR, "tubulin_mt_only_claire_cI_cIV_speed_per_tip_stats.csv"), index=False)
    _write_df_markdown(requested_table, os.path.join(RESULT_DIR, "tubulin_mt_only_claire_cI_cIV_speed_per_tip_stats.md"))

    fig_three = _plot_requested_three_panel(timeseries_df, summary_df, stats_df)
    _save_all_ext_no_square_patch(fig_three, os.path.join(PLOTS_DIR, "tubulin_mt_only_claire_cI_cIV_three_panel"))
    plt.close(fig_three)

    print(f"outputs saved in {RESULT_DIR}")


if __name__ == "__main__":
    main()
