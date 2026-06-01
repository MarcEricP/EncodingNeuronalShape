import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


from dendrogenesis import project_style
from dendrogenesis import useful_plt
from dendrogenesis.useful_stats import compare_two_series, slope_intercept_with_robust_and_cluster

project_style.set_style()


INPUT_BASE_DIR_CANDIDATES = [
    "/mnt/c/Users/menir/Documents/0000NeuronsData/clean_movies_1_min",
]
PIXELS_PER_MICRON = 16
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = os.path.join(REPO_ROOT, "DataAnalysis", "Result", "2_branching_rate")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
RAW_DATA_DIR = os.path.join(RESULT_DIR, "raw_data")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
SMOOTHING_MAX_WINDOW = 51
SMOOTHING_POLYORDER = 1
MAX_FRAME = 300 - 1
FRAME_START = 25


def _write_markdown_table(path, headers, rows):
    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def _write_df_markdown(df, path):
    _write_markdown_table(path, [str(c) for c in df.columns], df.to_numpy())


def _write_df_csv(df, path):
    df.to_csv(path, index=False)


def _format_p_value(p):
    if np.isfinite(p):
        return f"{p:.3g}"
    return "nan"


def _extract_pairwise_effects(model):
    names = list(model.model.exog_names)
    idx = {nm: i for i, nm in enumerate(names)}
    level_term = "C(series)[T.S2]"
    cand1 = "a:C(series)[T.S2]"
    cand2 = "C(series)[T.S2]:a"
    slope_term = cand1 if cand1 in idx else (cand2 if cand2 in idx else None)

    out = {
        "level_diff": np.nan,
        "level_diff_ci95_low": np.nan,
        "level_diff_ci95_high": np.nan,
        "slope_diff": np.nan,
        "slope_diff_ci95_low": np.nan,
        "slope_diff_ci95_high": np.nan,
    }
    if level_term in idx:
        tt = model.t_test(level_term)
        ci = np.asarray(tt.conf_int(alpha=0.05), dtype=float).ravel()
        out["level_diff"] = float(np.asarray(tt.effect).item())
        if ci.size >= 2:
            out["level_diff_ci95_low"] = float(ci[0])
            out["level_diff_ci95_high"] = float(ci[1])
    if slope_term and slope_term in idx:
        tt = model.t_test(slope_term)
        ci = np.asarray(tt.conf_int(alpha=0.05), dtype=float).ravel()
        out["slope_diff"] = float(np.asarray(tt.effect).item())
        if ci.size >= 2:
            out["slope_diff_ci95_low"] = float(ci[0])
            out["slope_diff_ci95_high"] = float(ci[1])
    return out


def _find_base_dir():
    for p in INPUT_BASE_DIR_CANDIDATES:
        if os.path.isdir(p):
            return p
    return None


def _savgol_window(n, max_window=SMOOTHING_MAX_WINDOW):
    if n < 3:
        return None
    return min(n, max_window)


def _read_all_series(source_dirs):
    dfs = {}
    for c, temp_root in source_dirs.items():
        dfs[c] = {}
        exclude = ["flow_track", "flow_track_opened_skel"]
        for root, dirs, files in os.walk(temp_root, topdown=True):
            dirs[:] = [d for d in dirs if d not in exclude]
            fname = "new_branches_ad.csv"
            if fname not in files:
                continue
            name = os.path.split(os.path.split(root)[0])[1]
            dfs[c][name] = {}
            temp_csv = pd.read_csv(os.path.join(root, fname))

            temp_frame_numbers = np.array(temp_csv["frame"])
            temp_total_length = np.array(temp_csv["neuron_total_length"]) / PIXELS_PER_MICRON

            n = int(temp_frame_numbers.max() + 1)
            dfs[c][name]["num_new_branch"] = np.zeros((n,))
            dfs[c][name]["total_length"] = np.zeros((n,))

            for i in range(temp_frame_numbers.shape[0]):
                frame_i = int(temp_frame_numbers[i])
                dfs[c][name]["num_new_branch"][frame_i] += 1
                dfs[c][name]["total_length"][frame_i] = temp_total_length[i]

            dfs[c][name]["num_new_branch"] = dfs[c][name]["num_new_branch"][1:]
            dfs[c][name]["total_length"] = dfs[c][name]["total_length"][1:]
            dfs[c][name]["branching_rate"] = dfs[c][name]["num_new_branch"] / np.maximum(
                dfs[c][name]["total_length"], 1e-7
            )

            wn = _savgol_window(dfs[c][name]["num_new_branch"].shape[0])
            wb = _savgol_window(dfs[c][name]["branching_rate"].shape[0])
            if wn is None:
                dfs[c][name]["num_new_branch_running_mean"] = dfs[c][name]["num_new_branch"].copy()
            else:
                dfs[c][name]["num_new_branch_running_mean"] = savgol_filter(
                    dfs[c][name]["num_new_branch"], wn, SMOOTHING_POLYORDER
                )
            if wb is None:
                dfs[c][name]["branching_rate_running_mean"] = dfs[c][name]["branching_rate"].copy()
            else:
                dfs[c][name]["branching_rate_running_mean"] = savgol_filter(
                    dfs[c][name]["branching_rate"], wb, SMOOTHING_POLYORDER
                )
    return dfs


def _read_all_series_from_raw_csv(raw_dir):
    dfs = {"WT cI": {}, "WT cIV": {}, "MT inh cIV": {}}
    if not os.path.isdir(raw_dir):
        return dfs

    class_prefix_to_key = {
        "class_I_": "WT cI",
        "class_IV_": "WT cIV",
        "class_IV_MT_inh_": "MT inh cIV",
    }

    for fname in os.listdir(raw_dir):
        if not fname.endswith(".csv"):
            continue
        cls_key = None
        acq_name = None
        for prefix, mapped in class_prefix_to_key.items():
            if fname.startswith(prefix):
                cls_key = mapped
                acq_name = fname[len(prefix): -4]
                break
        if cls_key is None or acq_name is None:
            continue

        fpath = os.path.join(raw_dir, fname)
        df = pd.read_csv(fpath)
        req_cols = [
            "number of new branches (#/min)",
            "total length (µm)",
            "lineic branching rate lambda (µm^-1.min^-1)",
        ]
        if any(c not in df.columns for c in req_cols):
            continue

        num_new_branch = np.asarray(df[req_cols[0]], dtype=float)
        total_length = np.asarray(df[req_cols[1]], dtype=float)
        branching_rate = np.asarray(df[req_cols[2]], dtype=float)
        n = min(num_new_branch.shape[0], total_length.shape[0], branching_rate.shape[0], MAX_FRAME)
        if n <= 0:
            continue
        num_new_branch = num_new_branch[:n]
        total_length = total_length[:n]
        branching_rate = branching_rate[:n]

        wn = _savgol_window(num_new_branch.shape[0])
        wb = _savgol_window(branching_rate.shape[0])
        if wn is None:
            num_new_branch_running_mean = num_new_branch.copy()
        else:
            num_new_branch_running_mean = savgol_filter(num_new_branch, wn, SMOOTHING_POLYORDER)
        if wb is None:
            branching_rate_running_mean = branching_rate.copy()
        else:
            branching_rate_running_mean = savgol_filter(branching_rate, wb, SMOOTHING_POLYORDER)

        dfs[cls_key][acq_name] = {
            "num_new_branch": num_new_branch,
            "total_length": total_length,
            "branching_rate": branching_rate,
            "num_new_branch_running_mean": num_new_branch_running_mean,
            "branching_rate_running_mean": branching_rate_running_mean,
        }
    return dfs


def _to_matrix(dfs, series_key):
    out = {}
    for c in dfs:
        rows = [np.array(dfs[c][name][series_key])[:MAX_FRAME] for name in dfs[c]]
        out[c] = np.array(rows) if rows else np.empty((0, MAX_FRAME))
    return out


def _save_figures(dfs):
    nrows = 2
    ncols = 3
    basic_figsize = (project_style.mm_to_in(55), project_style.mm_to_in(50))
    time_axis = np.arange(MAX_FRAME)

    for share_axis in ["all", "none"]:
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(basic_figsize[0] * ncols, basic_figsize[1] * nrows), sharex=share_axis, sharey=share_axis
        )
        dict_all_class_lineic = {}
        dict_all_class_lineic_unsmoothed = {}
        for c, color, cols in zip(
            ["WT cI", "WT cIV", "MT inh cIV"],
            ["tab:green", "tab:pink", "#353535"],
            [[0, 1], [0, 1, 2], [0, 2]],
        ):
            list_single_trajs = [np.array(dfs[c][name]["branching_rate_running_mean"])[:MAX_FRAME] for name in dfs[c]]
            if not list_single_trajs:
                continue
            list_single_trajs = np.array(list_single_trajs)
            dict_all_class_lineic[c] = list_single_trajs
            list_single_trajs_unsmoothed = [np.array(dfs[c][name]["branching_rate"])[:MAX_FRAME] for name in dfs[c]]
            dict_all_class_lineic_unsmoothed[c] = np.array(list_single_trajs_unsmoothed)
            mean = np.mean(list_single_trajs, axis=0)
            std = np.std(list_single_trajs, axis=0)
            for i in [0, 1]:
                for j in cols:
                    axs[i, j].plot(time_axis[FRAME_START:], mean[FRAME_START:], c=color, label=c)
                    axs[i, j].fill_between(
                        time_axis[FRAME_START:],
                        (mean - std)[FRAME_START:],
                        (mean + std)[FRAME_START:],
                        color=color,
                        alpha=0.3,
                    )
                    if i == 1:
                        for n in range(list_single_trajs.shape[0]):
                            axs[i, j].plot(
                                time_axis[FRAME_START:],
                                list_single_trajs[n][FRAME_START:],
                                color=color,
                                linestyle=":",
                                alpha=0.7,
                            )

        legend_title_by_col = {}
        pair_by_col = {1: ("WT cI", "WT cIV"), 2: ("WT cI", "MT inh cIV")}
        for col, (c1, c2) in pair_by_col.items():
            if c1 not in dict_all_class_lineic_unsmoothed or c2 not in dict_all_class_lineic_unsmoothed:
                continue
            _, _, stats = compare_two_series(
                dict_all_class_lineic_unsmoothed[c1],
                dict_all_class_lineic_unsmoothed[c2],
                time_axis,
                return_per_time=False,
            )
            p_level = float(stats.get("p_level_diff", np.nan))
            p_slope = float(stats.get("p_slope_diff", np.nan))
            if np.isfinite(p_level) or np.isfinite(p_slope):
                legend_title_by_col[col] = (
                    "Cluster-robust OLS\n"
                    f"level p={_format_p_value(p_level)}\n"
                    f"slope p={_format_p_value(p_slope)}"
                )

        for j in range(ncols):
            axs[1, j].set_xlabel("Time (hours AEL)")
            axs[1, j] = useful_plt.format_hr_x_axis(axs[1, j], 16)
            if share_axis == "none":
                axs[0, j].set_xlabel("Time (hours AEL)")
                axs[0, j] = useful_plt.format_hr_x_axis(axs[0, j], 16)
            axs[0, j].legend(title=legend_title_by_col.get(j, None))

        for i in range(nrows):
            axs[i, 0].set_ylabel("$\\lambda$ ($µm^{-1}$.$min^{-1}$)")
            if share_axis == "none":
                for j in range(ncols):
                    axs[i, j].set_ylabel("$\\lambda$ ($µm^{-1}$.$min^{-1}$)")
                    axs[i, j].set_ylim(bottom=0)
        plt.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(PLOTS_DIR, f"branching-rate-{share_axis}.{ext}"))
        plt.close()


def _extract_slope_stats(model):
    tt = model.t_test("a = 0")
    slope = float(np.asarray(tt.effect).item())
    slope_se = float(np.asarray(tt.sd).item())
    p_two_sided = float(np.asarray(tt.pvalue).item())
    ci95 = np.asarray(tt.conf_int(alpha=0.05), dtype=float).ravel()
    slope_ci95_low = float(ci95[0]) if ci95.size >= 2 else np.nan
    slope_ci95_high = float(ci95[1]) if ci95.size >= 2 else np.nan
    if slope < 0:
        p_decrease = p_two_sided / 2.0
    else:
        p_decrease = 1.0 - p_two_sided / 2.0
    return slope, slope_se, slope_ci95_low, slope_ci95_high, p_two_sided, p_decrease


def _extract_level_stats(model):
    tt = model.t_test("Intercept = 0")
    level = float(np.asarray(tt.effect).item())
    level_se = float(np.asarray(tt.sd).item())
    ci95 = np.asarray(tt.conf_int(alpha=0.05), dtype=float).ravel()
    level_ci95_low = float(ci95[0]) if ci95.size >= 2 else np.nan
    level_ci95_high = float(ci95[1]) if ci95.size >= 2 else np.nan
    p_two_sided = float(np.asarray(tt.pvalue).item())
    return level, level_se, level_ci95_low, level_ci95_high, p_two_sided


def _fmt_pm(v, se, ff="%.3g"):
    if np.isfinite(v) and np.isfinite(se):
        return f"{ff % v} ± {ff % se}"
    if np.isfinite(v):
        return ff % v
    return "NA"


def _fmt_ci(lo, hi, ff="%.3g"):
    if np.isfinite(lo) and np.isfinite(hi):
        return f"[{ff % lo}, {ff % hi}]"
    return "NA"


def _fmt_p_one_sided(p):
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return "p < 0.001"
    return f"p = {p:.4g}"


def _safe_name(s):
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(s)).strip("_")


def _export_raw_data_csvs(dfs):
    class_prefix = {
        "WT cI": "class_I",
        "WT cIV": "class_IV",
        "MT inh cIV": "class_IV_MT_inh",
    }
    for cls, acqs in dfs.items():
        if cls not in class_prefix:
            continue
        for acq, d in acqs.items():
            n = min(len(d["num_new_branch"]), len(d["total_length"]), len(d["branching_rate"]))
            if n <= 0:
                continue
            df = pd.DataFrame({
                "time after 16 h AEL (min)": np.arange(n, dtype=int),
                "number of new branches (#/min)": np.asarray(d["num_new_branch"], dtype=float)[:n],
                "total length (µm)": np.asarray(d["total_length"], dtype=float)[:n],
                "lineic branching rate lambda (µm^-1.min^-1)": np.asarray(d["branching_rate"], dtype=float)[:n],
            })
            out_name = f"{class_prefix[cls]}_{_safe_name(acq)}.csv"
            df.to_csv(os.path.join(RAW_DATA_DIR, out_name), index=False)


def _export_stats(dfs):
    time_axis = np.arange(MAX_FRAME)
    unsmoothed = _to_matrix(dfs, "branching_rate")

    single_rows = []
    pair_rows = []

    for key in unsmoothed:
        Y = unsmoothed[key]
        if Y.shape[0] == 0:
            continue
        n_exp, n_t = Y.shape
        df = pd.DataFrame({"y": Y.ravel(), "a": np.tile(time_axis[:n_t], n_exp), "exp": np.repeat(np.arange(n_exp), n_t)})
        models, _ = slope_intercept_with_robust_and_cluster(df, y="y", a="a", exp="exp", alpha=0.05)
        model_for_test = models.get("fe_cluster", models.get("hc3", models["base"]))
        slope, slope_se, slope_ci95_low, slope_ci95_high, p_two_sided, p_decrease = _extract_slope_stats(model_for_test)
        level, level_se, level_ci95_low, level_ci95_high, p_level_two_sided = _extract_level_stats(model_for_test)
        single_rows.append(
            {
                "class": key,
                "n_experiments": n_exp,
                "n_timepoints": n_t,
                "level_at_16h": level,
                "level_se": level_se,
                "level_ci95_low": level_ci95_low,
                "level_ci95_high": level_ci95_high,
                "p_level_two_sided": p_level_two_sided,
                "slope_per_min": slope,
                "slope_se": slope_se,
                "slope_ci95_low": slope_ci95_low,
                "slope_ci95_high": slope_ci95_high,
                "p_two_sided": p_two_sided,
                "p_decrease_one_sided": p_decrease,
                "decrease_significant_0p05": bool(p_decrease < 0.05),
                "model_formula": "y_it = beta0 + beta1*time_min_it + gamma_acquisition_i + eps_it",
                "cluster_variable": "acquisition",
                "n_clusters": int(n_exp),
                "decrease_test_name": "One-sided Wald t-test on slope (H0: beta1 >= 0; tested against decrease) from cluster-robust FE OLS",
                "test_model": "cluster_robust_FE",
            }
        )

    classes = [k for k in unsmoothed.keys() if unsmoothed[k].shape[0] > 0]
    for i in range(len(classes)):
        for j in range(i):
            c1, c2 = classes[i], classes[j]
            _, model_overall, stats = compare_two_series(
                unsmoothed[c1],
                unsmoothed[c2],
                time_axis,
                return_per_time=False,
            )
            p_level = float(stats.get("p_level_diff", np.nan))
            p_slope = float(stats.get("p_slope_diff", np.nan))
            effects = _extract_pairwise_effects(model_overall)
            n_clusters_a = int(unsmoothed[c2].shape[0])
            n_clusters_b = int(unsmoothed[c1].shape[0])
            pair_rows.append(
                {
                    "group_a": c2,
                    "group_b": c1,
                    "model_formula": "y ~ a + C(series) + a:C(series) + C(exp):C(series)",
                    "cluster_variable": "exp_series (acquisition within group)",
                    "n_clusters_group_a": n_clusters_a,
                    "n_clusters_group_b": n_clusters_b,
                    "test_name": "Two-sided Wald t-tests on model contrasts (cluster-robust FE OLS)",
                    "h0_level": "H0: level difference = 0",
                    "h0_slope": "H0: slope difference = 0",
                    "sidedness": "two-sided",
                    "level_diff": effects["level_diff"],
                    "level_diff_ci95_low": effects["level_diff_ci95_low"],
                    "level_diff_ci95_high": effects["level_diff_ci95_high"],
                    "p_level_diff": p_level,
                    "slope_diff": effects["slope_diff"],
                    "slope_diff_ci95_low": effects["slope_diff_ci95_low"],
                    "slope_diff_ci95_high": effects["slope_diff_ci95_high"],
                    "p_slope_diff": p_slope,
                }
            )


    single_df = pd.DataFrame(single_rows)
    pair_df = pd.DataFrame(pair_rows)
    class_display = {
        "WT cI": "Wild type class I",
        "WT cIV": "Wild type class IV",
        "MT inh cIV": "Microtubule inhibited class IV",
    }
    single_md = single_df.copy()
    single_md["Neuron class"] = single_md["class"].map(class_display).fillna(single_md["class"])
    single_md["Number of neurons/acquisitions"] = single_md["n_experiments"].astype(int)
    single_md["Time range"] = "16 h AEL - 21 h AEL"
    single_md["Number of time points"] = single_md["n_timepoints"].astype(int)
    single_md["Slope ± SE (1/µm/min^2)"] = [
        _fmt_pm(v, se, "%.3g") for v, se in zip(single_md["slope_per_min"], single_md["slope_se"])
    ]
    single_md["Slope 95% CI"] = [
        _fmt_ci(lo, hi, "%.3g") for lo, hi in zip(single_md["slope_ci95_low"], single_md["slope_ci95_high"])
    ]
    single_md["One-sided Wald t-test on slope  (H0: slope >= 0) p-value"] = single_md["p_decrease_one_sided"].map(_fmt_p_one_sided)
    single_md = single_md[[
        "Neuron class",
        "Number of neurons/acquisitions",
        "Time range",
        "Number of time points",
        "Slope ± SE (1/µm/min^2)",
        "Slope 95% CI",
        "One-sided Wald t-test on slope  (H0: slope >= 0) p-value",
    ]]

    single_lookup = single_df.set_index("class")
    pair_md = pair_df.copy()
    pair_md["Group A"] = pair_md["group_a"].map(class_display).fillna(pair_md["group_a"]) + " (n-" + pair_md["n_clusters_group_a"].astype(int).astype(str) + ")"
    pair_md["Group B"] = pair_md["group_b"].map(class_display).fillna(pair_md["group_b"]) + " (n-" + pair_md["n_clusters_group_b"].astype(int).astype(str) + ")"
    pair_md["Group A level at 16h AEL (1/µm/min)"] = pair_md["group_a"].map(single_lookup["level_at_16h"])
    pair_md["Group B level at 16h AEL(1/µm/min)"] = pair_md["group_b"].map(single_lookup["level_at_16h"])
    pair_md["Level difference A - B at 16 h AEL (1/µm/min)"] = pair_md["level_diff"]
    pair_md["Level difference 95% CI"] = [
        _fmt_ci(lo, hi, "%.3g") for lo, hi in zip(pair_md["level_diff_ci95_low"], pair_md["level_diff_ci95_high"])
    ]
    pair_md["Two-sided Wald t-test on level difference (H0: difference = 0) p-value"] = pair_md["p_level_diff"].map(_fmt_p_one_sided)
    pair_md["Group A slope (1/µm/min^2)"] = pair_md["group_a"].map(single_lookup["slope_per_min"])
    pair_md["Group B slope (1/µm/min^2)"] = pair_md["group_b"].map(single_lookup["slope_per_min"])
    pair_md["Slope difference A - B (1/µm/min^2)"] = pair_md["slope_diff"]
    pair_md["Slope difference 95% CI"] = [
        _fmt_ci(lo, hi, "%.3g") for lo, hi in zip(pair_md["slope_diff_ci95_low"], pair_md["slope_diff_ci95_high"])
    ]
    pair_md["Two-sided Wald t-test on slope difference (H0: difference = 0) p-value"] = pair_md["p_slope_diff"].map(_fmt_p_one_sided)
    pair_md = pair_md[[
        "Group A",
        "Group B",
        "Group A level at 16h AEL (1/µm/min)",
        "Group B level at 16h AEL(1/µm/min)",
        "Level difference A - B at 16 h AEL (1/µm/min)",
        "Level difference 95% CI",
        "Two-sided Wald t-test on level difference (H0: difference = 0) p-value",
        "Group A slope (1/µm/min^2)",
        "Group B slope (1/µm/min^2)",
        "Slope difference A - B (1/µm/min^2)",
        "Slope difference 95% CI",
        "Two-sided Wald t-test on slope difference (H0: difference = 0) p-value",
    ]]

    _write_df_markdown(single_md, os.path.join(RESULT_DIR, "branching_rate_decrease_tests.md"))
    _write_df_csv(single_md, os.path.join(RESULT_DIR, "branching_rate_decrease_tests.csv"))
    _write_df_markdown(pair_md, os.path.join(RESULT_DIR, "branching_rate_pairwise_series_tests.md"))
    _write_df_csv(pair_md, os.path.join(RESULT_DIR, "branching_rate_pairwise_series_tests.csv"))


def main():
    base_dir = _find_base_dir()
    source_dirs = {}
    if base_dir is not None:
        source_dirs = {
            "WT cI": os.path.join(base_dir, "class_I"),
            "WT cIV": os.path.join(base_dir, "class_IV"),
            "MT inh cIV": os.path.join(base_dir, "class_IV_MT_inh"),
        }
        dfs = _read_all_series(source_dirs)
    else:
        dfs = {"WT cI": {}, "WT cIV": {}, "MT inh cIV": {}}

    if not any(len(v) for v in dfs.values()):
        dfs = _read_all_series_from_raw_csv(RAW_DATA_DIR)
    if not any(len(v) for v in dfs.values()):
        raise RuntimeError(
            "No branching-rate input found. Provide raw dataset in one of:\n"
            + "\n".join(INPUT_BASE_DIR_CANDIDATES)
            + f"\nOR provide cached per-acquisition CSV files in:\n{RAW_DATA_DIR}"
        )

    _save_figures(dfs)
    if base_dir is not None:
        _export_raw_data_csvs(dfs)
    _export_stats(dfs)


if __name__ == "__main__":
    main()
