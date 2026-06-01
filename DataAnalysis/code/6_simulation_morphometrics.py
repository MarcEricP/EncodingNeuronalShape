import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

from skimage import io
import networkx as nx

from dendrotree.nx_graph import nx_graph, metrics
from dendrotree.tree_structure import swc_rx
from dendrogenesis import useful_plt

import sys
from pathlib import Path


from dendrogenesis import project_style
project_style.set_style()

BASE_DIR_DATA = "/mnt/c/Users/menir/Documents/0000NeuronsData/clean_movies_1_min_test_fail"
SIMU_BASE =  os.path.join(os.getcwd(), "SimulationTree","simulation_result_3_models")
RESULT_DIR = os.path.join(os.getcwd(), "DataAnalysis", "Result", "6_simulation_morphometrics")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
RAW_DATA_DIR = os.path.join(RESULT_DIR, "raw_data")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

classes = ["class_I", "class_IV", "class_IV_MT_inh"]
simu_names = [
    "PALAVALLI",
    "PALAVALLI_zero_drift",
    "GENNORM_IID",
    "GENNORM_IID_zero_drift",
    "ARFIMA",
    "ARFIMA_zero_drift",
]

metrics_to_use = [
    "total_length",
    "semi-minor_axis",
    "number_of_branches",
    "density",
]
metric_labels = {
    "total_length": r"Total length $\mu m$",
    "semi-minor_axis": r"Semi-minor axis $\mu m$",
    "number_of_branches": "Branching points #",
    "density": r"Density $\mu m^{-1}$",
}

basic_figsize = (project_style.mm_to_in(50), project_style.mm_to_in(50))
max_simu_time = 300
pixels_per_microns = 16
save_metric_name = "metrics.json"
start_hour = 16
CACHE_VERSION = 3


def _sanitize_name(name):
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(name)).strip("_")


def _compute_metrics_from_mask_movie(n_class, movie_dir, movie_tif_path, save_metric_path):
    if os.path.exists(save_metric_path):
        with open(save_metric_path, "r") as f:
            return n_class, movie_dir, json.load(f)

    movie = io.imread(movie_tif_path)
    out = []
    for t in tqdm.trange(movie.shape[0], desc=f"REAL {os.path.basename(movie_tif_path)}"):
        g = nx_graph.mask2nx(movie[t], has_contact=True, has_root=True)
        g = nx_graph.refine_graph(g, max_l=8)
        tl = metrics.total_length(g)
        res_ellipse = metrics.tree_inertia_and_equivalent_ellipse(g)
        area = float(np.pi * res_ellipse["semi_minor"] * res_ellipse["semi_major"])

        g_loop = nx_graph.mask2nx(movie[t] > 0, has_contact=False, has_root=False)
        n_br = int(metrics.n_branch_points(g_loop)) - len(nx.cycle_basis(g_loop))
        out.append({
            "time": int(t),
            "total_length": float(tl / pixels_per_microns),
            "semi-minor_axis": float(res_ellipse["semi_minor"] / pixels_per_microns),
            "semi-major_axis": float(res_ellipse["semi_major"] / pixels_per_microns),
            "angle_ellipse": float(res_ellipse["angle_rad"]),
            "centroid_ellipse": [float(a / pixels_per_microns) for a in res_ellipse["centroid"]],
            "number_of_branches": n_br,
            "area": float(area / (pixels_per_microns ** 2)),
            "density": float((tl / area) * pixels_per_microns) if area > 0 else float("nan"),
        })

    with open(save_metric_path, "w") as f:
        json.dump(out, f)
    return n_class, movie_dir, out


def _compute_metrics_from_swc_folder(n_class, movie_dir, temp_dir, swc_dir, save_metric_path):
    if os.path.exists(save_metric_path):
        with open(save_metric_path, "r") as f:
            return n_class, movie_dir, temp_dir, json.load(f)

    out = []
    for t in tqdm.trange(max_simu_time + 1, desc=f"SIM {swc_dir}"):
        swc_path = os.path.join(swc_dir, f"{t}.swc")
        if not os.path.exists(swc_path):
            continue
        with open(swc_path, "r") as f:
            swc_str = f.read()
        g_rx = swc_rx.swc2rx(swc_str)
        g = nx_graph.rx2nx(g_rx)
        tl = metrics.total_length(g)
        res_ellipse = metrics.tree_inertia_and_equivalent_ellipse(g)
        area = float(np.pi * res_ellipse["semi_minor"] * res_ellipse["semi_major"])
        out.append({
            "time": int(t),
            "total_length": float(tl),
            "semi-minor_axis": float(res_ellipse["semi_minor"]),
            "semi-major_axis": float(res_ellipse["semi_major"]),
            "angle_ellipse": float(res_ellipse["angle_rad"]),
            "centroid_ellipse": [float(a) for a in res_ellipse["centroid"]],
            "number_of_branches": int(metrics.n_branch_points(g)),
            "area": float(area),
            "density": float(tl / area) if area > 0 else float("nan"),
        })

    with open(save_metric_path, "w") as f:
        json.dump(out, f)
    return n_class, movie_dir, temp_dir, out


def _standardize_arrays(dict_data):
    std_dict = {}
    for n_class in classes:
        std_dict[n_class] = {}
        class_block = dict_data.get(n_class, {"real": {}, "simu": {}})
        for datatype, obs_dict in class_block.items():
            if datatype.startswith("_"):
                continue
            std_dict[n_class][datatype] = {}
            for metric in metrics_to_use:
                res_array = []
                for obs in obs_dict.keys():
                    seq = [a[metric] for a in obs_dict[obs]]
                    seq = seq[:max_simu_time]
                    if len(seq) < max_simu_time:
                        seq = seq + [np.nan] * (max_simu_time - len(seq))
                    res_array.append(seq)
                std_dict[n_class][datatype][metric] = np.array(res_array) if res_array else np.empty((0, max_simu_time))
    return std_dict


def _mean_std(series_2d):
    if series_2d.size == 0:
        return None, None
    return np.nanmean(series_2d, axis=0), np.nanstd(series_2d, axis=0)


def load_real():
    cache_path = os.path.join(RAW_DATA_DIR, "save_compute_REAL.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        # basic validity: version + all classes present with "real" dicts (can be empty)
        if cached.get("_cache_version") == CACHE_VERSION and all(cls in cached and "real" in cached[cls] for cls in classes):
            return cached
        else:
            print("[cache note] REAL cache incomplete; recomputing.")

    dict_real_only = {c: {"real": {}} for c in classes}
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as ex:
        futures = []
        for n_class in classes:
            class_real_dir = os.path.join(BASE_DIR_DATA, n_class)
            if not os.path.isdir(class_real_dir):
                continue
            for movie_dir in os.listdir(class_real_dir):
                structure_path = os.path.join(class_real_dir, movie_dir, "ij_sift", "neuron_structure.tif")
                if os.path.exists(structure_path):
                    metric_path = os.path.join(class_real_dir, movie_dir, "ij_sift", save_metric_name)
                    futures.append(ex.submit(_compute_metrics_from_mask_movie, n_class, movie_dir, structure_path, metric_path))
        pbar = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Compute REAL metrics")
        for fut in pbar:
            n_class, movie_dir, res = fut.result()
            dict_real_only[n_class]["real"][movie_dir] = res

    dict_real_only["_cache_version"] = CACHE_VERSION
    with open(cache_path, "w") as f:
        json.dump(dict_real_only, f)
    return dict_real_only


def load_simu(simu_name, dict_real_only):
    cache_dir = os.path.join(RAW_DATA_DIR, simu_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "save_compute.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        valid = cached.get("_cache_version") == CACHE_VERSION
        for cls in classes:
            if not valid:
                break
            if cls not in cached or "simu" not in cached[cls]:
                valid = False
                break
        if valid:
            return cached
        else:
            print(f"[cache note] SIMU cache for {simu_name} incomplete; recomputing.")

    dict_data = {
        cls: {"real": dict_real_only.get(cls, {}).get("real", {}), "simu": {}}
        for cls in classes
    }

    base_dir_simu = os.path.join(SIMU_BASE, simu_name)
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as ex:
        for n_class in classes:
            class_dir = os.path.join(base_dir_simu, n_class)
            if not os.path.isdir(class_dir):
                continue
            for movie_dir in os.listdir(class_dir):
                start_path = os.path.join(class_dir, movie_dir)
                if not os.path.isdir(start_path):
                    continue
                for temp_dir in os.listdir(start_path):
                    swc_dir = os.path.join(start_path, temp_dir, "swc")
                    if os.path.exists(swc_dir):
                        metric_path = os.path.join(start_path, temp_dir, save_metric_name)
                        futures.append(
                            ex.submit(
                                _compute_metrics_from_swc_folder,
                                n_class,
                                movie_dir,
                                temp_dir,
                                swc_dir,
                                metric_path,
                            )
                        )

        pbar = tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"Simu {simu_name}")
        for fut in pbar:
            n_class, movie_dir, temp_dir, res = fut.result()
            run_key = f"{movie_dir}__{temp_dir}"
            dict_data.setdefault(n_class, {"real": dict_real_only.get(n_class, {}).get("real", {}), "simu": {}})
            dict_data[n_class]["simu"][run_key] = res

    dict_data["_cache_version"] = CACHE_VERSION
    with open(cache_path, "w") as f:
        json.dump(dict_data, f)
    return dict_data


def _print_loaded_counts(dict_data, label):
    print(f"[counts] {label}")
    for n_class in classes:
        keys = list(dict_data.get(n_class, {}).get("simu", {}).keys())
        movies = {k.rsplit("__", 1)[0] for k in keys if "__" in k}
        print(f"  {n_class}: runs={len(keys)}, movies={len(movies)}")


def _expected_run_counts_for_model(simu_name):
    expected = {c: 0 for c in classes}
    base_dir_simu = os.path.join(SIMU_BASE, simu_name)
    for n_class in classes:
        class_dir = os.path.join(base_dir_simu, n_class)
        if not os.path.isdir(class_dir):
            continue
        for movie_dir in os.listdir(class_dir):
            movie_path = os.path.join(class_dir, movie_dir)
            if not os.path.isdir(movie_path):
                continue
            for run_dir in os.listdir(movie_path):
                swc_dir = os.path.join(movie_path, run_dir, "swc")
                if os.path.isdir(swc_dir):
                    expected[n_class] += 1
    return expected


def _export_simu_raw_excels(simu_name, dict_data):
    base_out = os.path.join(RAW_DATA_DIR, simu_name)
    for n_class in classes:
        class_runs = dict_data.get(n_class, {}).get("simu", {})
        if not class_runs:
            continue
        class_out = os.path.join(base_out, n_class)
        os.makedirs(class_out, exist_ok=True)

        by_start = {}
        for run_key, seq in class_runs.items():
            if "__" in run_key:
                start_name, real_name = run_key.rsplit("__", 1)
            else:
                start_name, real_name = run_key, "run"
            by_start.setdefault(start_name, {})[real_name] = seq

        for start_name, runs_dict in by_start.items():
            xlsx_name = f"{_sanitize_name(start_name)}.xlsx"
            xlsx_path = os.path.join(class_out, xlsx_name)
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                used_sheet_names = set()
                for metric in metrics_to_use:
                    cols = {}
                    for run_name, seq in sorted(runs_dict.items()):
                        vals = [a.get(metric, np.nan) for a in seq]
                        vals = vals[:max_simu_time]
                        if len(vals) < max_simu_time:
                            vals = vals + [np.nan] * (max_simu_time - len(vals))
                        cols[_sanitize_name(run_name)] = vals
                    if not cols:
                        continue
                    df_metric = pd.DataFrame(cols)
                    df_metric.insert(0, "time", np.arange(max_simu_time, dtype=int))
                    base_sheet = (_sanitize_name(metric) or "metric")[:31]
                    sheet = base_sheet
                    k = 1
                    while sheet in used_sheet_names:
                        suffix = f"_{k}"
                        sheet = f"{base_sheet[:31 - len(suffix)]}{suffix}"
                        k += 1
                    used_sheet_names.add(sheet)
                    df_metric.to_excel(writer, index=False, sheet_name=sheet)


def plot_special_panel(std_real, per_simu_std):
    row_classes = [c for c in ["class_I", "class_IV", "class_IV_MT_inh"] if c in std_real]
    n_rows = len(row_classes)
    if n_rows == 0:
        return
    fig, axs = plt.subplots(n_rows, 4, figsize=(basic_figsize[0] * 4, basic_figsize[1] * n_rows))
    axs = np.atleast_2d(axs)

    col_titles = ["Experiment", "2-state model", "Correct scale diffusive model", "Subdiffusive model"]

    color_map = {
        "Experiment": "black",
        "2-state model": "tab:orange",
        "Correct scale diffusive model": "tab:blue",
        "Subdiffusive model": "tab:red",
    }

    def _pick_sim(class_key, label):
        if label == "2-state model":
            if class_key == "class_IV_MT_inh":
                return None
            return "PALAVALLI_zero_drift"
        if label == "Correct scale diffusive model":
            return "GENNORM_IID" if class_key == "class_I" else "GENNORM_IID_zero_drift"
        if label == "Subdiffusive model":
            return "ARFIMA_zero_drift"
        return None

    for c_idx, metric in enumerate(["total_length", "semi-minor_axis", "number_of_branches", "density"]):
        for r_idx, class_key in enumerate(row_classes):
            ax = axs[r_idx, c_idx]
            ylim_candidates = []

            # Experiment
            mR, sR = _mean_std(std_real[class_key]["real"][metric])
            if mR is not None:
                x = np.arange(len(mR))
                ax.plot(x, mR, c=color_map["Experiment"], label="Experiment")
                ax.fill_between(x, mR - sR, mR + sR, color=color_map["Experiment"], alpha=0.15)
                ylim_candidates.append(mR - sR)
                ylim_candidates.append(mR + sR)

            # models
            for label in col_titles[1:]:
                sim_key = _pick_sim(class_key, label)
                if sim_key is None:
                    continue
                container = "simu"
                series = per_simu_std.get(sim_key, {}).get(class_key, {}).get(container, {}).get(metric, np.empty((0, max_simu_time)))
                mS, sS = _mean_std(series)
                if mS is None:
                    continue
                x = np.arange(len(mS))
                ax.plot(x, mS, c=color_map[label], label=label)
                ax.fill_between(x, mS - sS, mS + sS, color=color_map[label], alpha=0.15)
                # Keep the 2-state trace visible but do not let it set the y-scale.
                if label != "2-state model":
                    ylim_candidates.append(mS - sS)
                    ylim_candidates.append(mS + sS)

            ax.set_ylabel(metric_labels[metric])
            ax = useful_plt.format_hr_x_axis(ax, start_hour)
            ax.set_xlabel("Time hours AEL")
            if ylim_candidates:
                y_all = np.concatenate([np.ravel(a) for a in ylim_candidates])
                y_all = y_all[np.isfinite(y_all)]
                if y_all.size > 0:
                    y_min = float(np.min(y_all))
                    y_max = float(np.max(y_all))
                    if y_min == y_max:
                        pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
                    else:
                        pad = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_box_aspect(1)

    # Row labels
    y_positions = np.linspace(0.85, 0.15, n_rows)
    row_text = {
        "class_I": "Class I",
        "class_IV": "Class IV",
        "class_IV_MT_inh": "Class IV MT inh",
    }
    for y, c in zip(y_positions, row_classes):
        fig.text(0.005, float(y), row_text.get(c, c), rotation=90, va="center", fontsize=9)

    # legend inside bottom-right subplot from line labels
    axs[-1, -1].legend()
    fig.tight_layout(rect=[0.01, 0.0, 1.0, 1.0])
    base = "special_4col_2row_I_IV_separate_script"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{base}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_special_panel_markov_only(std_real, per_simu_std):
    """2x4 panel: class I / class IV, Experiment + 2-state model only."""
    row_classes = [c for c in ["class_I", "class_IV"] if c in std_real]
    n_rows = len(row_classes)
    if n_rows == 0:
        return

    fig, axs = plt.subplots(n_rows, 4, figsize=(basic_figsize[0] * 4, basic_figsize[1] * n_rows))
    axs = np.atleast_2d(axs)
    color_map = {"Experiment": "black", "2-state model": "tab:orange"}

    for c_idx, metric in enumerate(["total_length", "semi-minor_axis", "number_of_branches", "density"]):
        for r_idx, class_key in enumerate(row_classes):
            ax = axs[r_idx, c_idx]
            ylim_candidates = []

            mR, sR = _mean_std(std_real[class_key]["real"][metric])
            if mR is not None:
                x = np.arange(len(mR))
                ax.plot(x, mR, c=color_map["Experiment"], label="Experiment")
                ax.fill_between(x, mR - sR, mR + sR, color=color_map["Experiment"], alpha=0.15)
                ylim_candidates.append(mR - sR)
                ylim_candidates.append(mR + sR)

            sim_key = "PALAVALLI_zero_drift"
            series = per_simu_std.get(sim_key, {}).get(class_key, {}).get("simu", {}).get(metric, np.empty((0, max_simu_time)))
            mS, sS = _mean_std(series)
            if mS is not None:
                x = np.arange(len(mS))
                ax.plot(x, mS, c=color_map["2-state model"], label="2-state model")
                ax.fill_between(x, mS - sS, mS + sS, color=color_map["2-state model"], alpha=0.15)
                ylim_candidates.append(mS - sS)
                ylim_candidates.append(mS + sS)

            ax.set_ylabel(metric_labels[metric])
            ax = useful_plt.format_hr_x_axis(ax, start_hour)
            ax.set_xlabel("Time hours AEL")
            if ylim_candidates:
                y_all = np.concatenate([np.ravel(a) for a in ylim_candidates])
                y_all = y_all[np.isfinite(y_all)]
                if y_all.size > 0:
                    y_min = float(np.min(y_all))
                    y_max = float(np.max(y_all))
                    if y_min == y_max:
                        pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
                    else:
                        pad = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_box_aspect(1)

    y_positions = np.linspace(0.85, 0.15, n_rows)
    row_text = {"class_I": "Class I", "class_IV": "Class IV"}
    for y, c in zip(y_positions, row_classes):
        fig.text(0.005, float(y), row_text.get(c, c), rotation=90, va="center", fontsize=9)

    axs[-1, -1].legend()
    fig.tight_layout(rect=[0.01, 0.0, 1.0, 1.0])
    base = "special_4col_2row_I_IV_markov_only"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{base}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_special_panel_all_simus(std_real, per_simu_std):
    row_classes = [c for c in ["class_I", "class_IV", "class_IV_MT_inh"] if c in std_real]
    n_rows = len(row_classes)
    fig, axs = plt.subplots(n_rows, 4, figsize=(basic_figsize[0] * 4, basic_figsize[1] * n_rows))
    axs = np.atleast_2d(axs)

    sim_labels = {
        "PALAVALLI": "palavalli",
        "PALAVALLI_zero_drift": "palavalli zero drift",
        "GENNORM_IID": "gennorm",
        "GENNORM_IID_zero_drift": "gennorm zero drift",
        "ARFIMA": "arfima",
        "ARFIMA_zero_drift": "arfima zero drift",
    }
    sim_colors = {
        "PALAVALLI": "tab:orange",
        "PALAVALLI_zero_drift": "tab:brown",
        "GENNORM_IID": "tab:blue",
        "GENNORM_IID_zero_drift": "tab:cyan",
        "ARFIMA": "tab:red",
        "ARFIMA_zero_drift": "tab:pink",
    }

    for c_idx, metric in enumerate(["total_length", "semi-minor_axis", "number_of_branches", "density"]):
        for r_idx, class_key in enumerate(row_classes):
            ax = axs[r_idx, c_idx]
            ylim_candidates = []

            # Experiment
            mR, sR = _mean_std(std_real[class_key]["real"][metric])
            if mR is not None:
                x = np.arange(len(mR))
                ax.plot(x, mR, c="black", label="Experiment")
                ax.fill_between(x, mR - sR, mR + sR, color="black", alpha=0.15)
                ylim_candidates.append(mR - sR)
                ylim_candidates.append(mR + sR)

            # all simus
            for sim_key in simu_names:
                series = per_simu_std.get(sim_key, {}).get(class_key, {}).get("simu", {}).get(
                    metric, np.empty((0, max_simu_time))
                )
                mS, sS = _mean_std(series)
                if mS is None:
                    continue
                x = np.arange(len(mS))
                ax.plot(x, mS, c=sim_colors.get(sim_key, "0.5"), label=sim_labels.get(sim_key, sim_key))
                ax.fill_between(x, mS - sS, mS + sS, color=sim_colors.get(sim_key, "0.5"), alpha=0.15)
                if sim_key not in ("PALAVALLI", "PALAVALLI_zero_drift", "GENNORM_IID_zero_drift"):
                    ylim_candidates.append(mS - sS)
                    ylim_candidates.append(mS + sS)

            ax.set_ylabel(metric_labels[metric])
            ax = useful_plt.format_hr_x_axis(ax, start_hour)
            ax.set_xlabel("Time hours AEL")
            if ylim_candidates:
                y_all = np.concatenate([np.ravel(a) for a in ylim_candidates])
                y_all = y_all[np.isfinite(y_all)]
                if y_all.size > 0:
                    y_min = float(np.min(y_all))
                    y_max = float(np.max(y_all))
                    if y_min == y_max:
                        pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
                    else:
                        pad = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_box_aspect(1)

    y_positions = np.linspace(0.85, 0.15, n_rows)
    row_text = {
        "class_I": "Class I",
        "class_IV": "Class IV",
        "class_IV_MT_inh": "Class IV MT inh",
    }
    for y, c in zip(y_positions, row_classes):
        fig.text(0.005, float(y), row_text.get(c, c), rotation=90, va="center", fontsize=9)

    axs[-1, -1].legend(frameon=False)#, fontsize=8)
    fig.tight_layout(rect=[0.01, 0.0, 1.0, 1.0])
    base = "special_4col_2row_I_IV_all_simus"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{base}.{ext}"), bbox_inches="tight")
    plt.close(fig)

def plot_special_panel_experiment_plus_one_model(std_real, per_simu_std, sim_key, model_label, base_name):
    """
    2x4 panel: classI/classIV with Experiment + one simulation model.
    """
    row_classes = [c for c in ["class_I", "class_IV"] if c in std_real]
    n_rows = len(row_classes)
    if n_rows == 0:
        return

    fig, axs = plt.subplots(n_rows, 4, figsize=(basic_figsize[0] * 4, basic_figsize[1] * n_rows))
    axs = np.atleast_2d(axs)
    model_color = "tab:blue" if "gennorm" in sim_key.lower() else ("tab:orange" if "palavalli" in sim_key.lower() else "tab:red")

    for c_idx, metric in enumerate(["total_length", "semi-minor_axis", "number_of_branches", "density"]):
        for r_idx, class_key in enumerate(row_classes):
            ax = axs[r_idx, c_idx]
            ylim_candidates = []

            mR, sR = _mean_std(std_real[class_key]["real"][metric])
            if mR is not None:
                x = np.arange(len(mR))
                ax.plot(x, mR, c="black", label="Experiment")
                ax.fill_between(x, mR - sR, mR + sR, color="black", alpha=0.15)
                ylim_candidates.append(mR - sR)
                ylim_candidates.append(mR + sR)

            series = per_simu_std.get(sim_key, {}).get(class_key, {}).get("simu", {}).get(metric, np.empty((0, max_simu_time)))
            mS, sS = _mean_std(series)
            if mS is not None:
                x = np.arange(len(mS))
                ax.plot(x, mS, c=model_color, label=model_label)
                ax.fill_between(x, mS - sS, mS + sS, color=model_color, alpha=0.15)
                ylim_candidates.append(mS - sS)
                ylim_candidates.append(mS + sS)

            ax.set_ylabel(metric_labels[metric])
            ax = useful_plt.format_hr_x_axis(ax, start_hour)
            ax.set_xlabel("Time hours AEL")
            if ylim_candidates:
                y_all = np.concatenate([np.ravel(a) for a in ylim_candidates])
                y_all = y_all[np.isfinite(y_all)]
                if y_all.size > 0:
                    y_min = float(np.min(y_all))
                    y_max = float(np.max(y_all))
                    if y_min == y_max:
                        pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
                    else:
                        pad = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_box_aspect(1)

    y_positions = np.linspace(0.85, 0.15, n_rows)
    row_text = {"class_I": "Class I", "class_IV": "Class IV"}
    for y, c in zip(y_positions, row_classes):
        fig.text(0.005, float(y), row_text.get(c, c), rotation=90, va="center", fontsize=9)

    axs[-1, -1].legend(frameon=False)#, fontsize=8)
    fig.tight_layout(rect=[0.01, 0.0, 1.0, 1.0])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{base_name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_special_panel_experiment_plus_correct_scale_with_markov_on_classI(std_real, per_simu_std):
    """
    - Base traces: Experiment + Correct scale diffusive model (GENNORM_IID_zero_drift)
    - Extra trace only on class I panels: 2-state Markov model (PALAVALLI_zero_drift)
    """
    row_classes = [c for c in ["class_I", "class_IV"] if c in std_real]
    n_rows = len(row_classes)
    if n_rows == 0:
        return

    fig, axs = plt.subplots(n_rows, 4, figsize=(basic_figsize[0] * 4, basic_figsize[1] * n_rows))
    axs = np.atleast_2d(axs)

    base_sim_key = "GENNORM_IID_zero_drift"
    base_model_label = "Correct scale diffusive model"
    extra_sim_key = "PALAVALLI_zero_drift"
    extra_label = "2-state Markov model"

    for c_idx, metric in enumerate(["total_length", "semi-minor_axis", "number_of_branches", "density"]):
        for r_idx, class_key in enumerate(row_classes):
            ax = axs[r_idx, c_idx]
            ylim_candidates = []

            mR, sR = _mean_std(std_real[class_key]["real"][metric])
            if mR is not None:
                x = np.arange(len(mR))
                ax.plot(x, mR, c="black", label="Experiment")
                ax.fill_between(x, mR - sR, mR + sR, color="black", alpha=0.15)
                ylim_candidates.append(mR - sR)
                ylim_candidates.append(mR + sR)

            base_series = per_simu_std.get(base_sim_key, {}).get(class_key, {}).get(
                "simu", {}
            ).get(metric, np.empty((0, max_simu_time)))
            mB, sB = _mean_std(base_series)
            if mB is not None:
                x = np.arange(len(mB))
                ax.plot(x, mB, c="tab:blue", label=base_model_label)
                ax.fill_between(x, mB - sB, mB + sB, color="tab:blue", alpha=0.15)
                ylim_candidates.append(mB - sB)
                ylim_candidates.append(mB + sB)

            # Overlay Markov only for class I, without affecting y-limits.
            if class_key == "class_I":
                extra_series = per_simu_std.get(extra_sim_key, {}).get(class_key, {}).get(
                    "simu", {}
                ).get(metric, np.empty((0, max_simu_time)))
                mE, sE = _mean_std(extra_series)
                if mE is not None:
                    x = np.arange(len(mE))
                    ax.plot(x, mE, c="tab:orange", linestyle="--", label=extra_label)
                    ax.fill_between(x, mE - sE, mE + sE, color="tab:orange", alpha=0.12)

            ax.set_ylabel(metric_labels[metric])
            ax = useful_plt.format_hr_x_axis(ax, start_hour)
            ax.set_xlabel("Time hours AEL")
            if ylim_candidates:
                y_all = np.concatenate([np.ravel(a) for a in ylim_candidates])
                y_all = y_all[np.isfinite(y_all)]
                if y_all.size > 0:
                    y_min = float(np.min(y_all))
                    y_max = float(np.max(y_all))
                    if y_min == y_max:
                        pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
                    else:
                        pad = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_box_aspect(1)

    y_positions = np.linspace(0.85, 0.15, n_rows)
    row_text = {"class_I": "Class I", "class_IV": "Class IV"}
    for y, c in zip(y_positions, row_classes):
        fig.text(0.005, float(y), row_text.get(c, c), rotation=90, va="center", fontsize=9)

    # Build legend from all axes so class-I-only overlays are included.
    legend_handles = {}
    for ax in np.ravel(axs):
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l and l not in legend_handles:
                legend_handles[l] = h
    axs[-1, -1].legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        frameon=False,
    )
    fig.tight_layout(rect=[0.01, 0.0, 1.0, 1.0])
    base_name = "special_4col_2row_I_IV_experiment_plus_correct_scale_diffusive_with_markov_on_classI"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{base_name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_special_panel_experiment_plus_diffusive_pair(std_real, per_simu_std):
    """
    2x4 panel: class I / class IV with
    Experiment + GENNORM_IID + GENNORM_IID_zero_drift.
    """
    row_classes = [c for c in ["class_I", "class_IV"] if c in std_real]
    n_rows = len(row_classes)
    if n_rows == 0:
        return

    fig, axs = plt.subplots(n_rows, 4, figsize=(basic_figsize[0] * 4, basic_figsize[1] * n_rows))
    axs = np.atleast_2d(axs)

    model_defs = [
        ("GENNORM_IID", "Diffusive with drift", "tab:cyan"),
        ("GENNORM_IID_zero_drift", "Diffusive", "tab:blue"),
    ]

    for c_idx, metric in enumerate(["total_length", "semi-minor_axis", "number_of_branches", "density"]):
        for r_idx, class_key in enumerate(row_classes):
            ax = axs[r_idx, c_idx]
            ylim_candidates = []

            mR, sR = _mean_std(std_real[class_key]["real"][metric])
            if mR is not None:
                x = np.arange(len(mR))
                ax.plot(x, mR, c="black", label="Experiment")
                ax.fill_between(x, mR - sR, mR + sR, color="black", alpha=0.15)
                ylim_candidates.append(mR - sR)
                ylim_candidates.append(mR + sR)

            for sim_key, label, color in model_defs:
                series = per_simu_std.get(sim_key, {}).get(class_key, {}).get("simu", {}).get(
                    metric, np.empty((0, max_simu_time))
                )
                mS, sS = _mean_std(series)
                if mS is None:
                    continue
                x = np.arange(len(mS))
                ax.plot(x, mS, c=color, label=label)
                ax.fill_between(x, mS - sS, mS + sS, color=color, alpha=0.15)
                ylim_candidates.append(mS - sS)
                ylim_candidates.append(mS + sS)

            ax.set_ylabel(metric_labels[metric])
            ax = useful_plt.format_hr_x_axis(ax, start_hour)
            ax.set_xlabel("Time hours AEL")
            if ylim_candidates:
                y_all = np.concatenate([np.ravel(a) for a in ylim_candidates])
                y_all = y_all[np.isfinite(y_all)]
                if y_all.size > 0:
                    y_min = float(np.min(y_all))
                    y_max = float(np.max(y_all))
                    if y_min == y_max:
                        pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
                    else:
                        pad = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_box_aspect(1)

    y_positions = np.linspace(0.85, 0.15, n_rows)
    row_text = {"class_I": "Class I", "class_IV": "Class IV"}
    for y, c in zip(y_positions, row_classes):
        fig.text(0.005, float(y), row_text.get(c, c), rotation=90, va="center", fontsize=9)

    axs[-1, -1].legend(frameon=False)
    fig.tight_layout(rect=[0.01, 0.0, 1.0, 1.0])
    base_name = "special_4col_2row_I_IV_experiment_plus_diffusive_pair"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{base_name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    std_real = {}

    # Load REAL
    dict_real_only = load_real()
    std_real = {}
    for n_class in classes:
        std_real[n_class] = {"real": {}}
        for metric in metrics_to_use:
            series = []
            for obs in dict_real_only[n_class]["real"].values():
                seq = [a[metric] for a in obs]
                seq = seq[:max_simu_time]
                if len(seq) < max_simu_time:
                    seq = seq + [np.nan] * (max_simu_time - len(seq))
                series.append(seq)
            std_real[n_class]["real"][metric] = np.array(series) if series else np.empty((0, max_simu_time))

    # Load SIMU
    per_simu_std = {}
    for simu in simu_names:
        dict_data = load_simu(simu, dict_real_only)
        _export_simu_raw_excels(simu, dict_data)
        _print_loaded_counts(dict_data, simu)
        expected_counts = _expected_run_counts_for_model(simu)
        for n_class in classes:
            loaded = len(dict_data.get(n_class, {}).get("simu", {}))
            expected = expected_counts.get(n_class, 0)
            if loaded != expected:
                print(f"{simu} {n_class}: loaded={loaded} expected={expected}")
        std_dict = _standardize_arrays(dict_data)
        per_simu_std[simu] = std_dict

    plot_special_panel(std_real, per_simu_std)
    plot_special_panel_markov_only(std_real, per_simu_std)
    plot_special_panel_all_simus(std_real, per_simu_std)
    plot_special_panel_experiment_plus_one_model(
        std_real, per_simu_std,
        sim_key="PALAVALLI_zero_drift",
        model_label="2-state Markov model",
        base_name="special_4col_2row_I_IV_experiment_plus_2state_markov",
    )
    plot_special_panel_experiment_plus_one_model(
        std_real, per_simu_std,
        sim_key="GENNORM_IID_zero_drift",
        model_label="Correct scale diffusive model",
        base_name="special_4col_2row_I_IV_experiment_plus_correct_scale_diffusive",
    )
    plot_special_panel_experiment_plus_one_model(
        std_real, per_simu_std,
        sim_key="ARFIMA_zero_drift",
        model_label="Subdiffusive",
        base_name="special_4col_2row_I_IV_experiment_plus_arfima_zero_drift",
    )
    plot_special_panel_experiment_plus_correct_scale_with_markov_on_classI(std_real, per_simu_std)
    plot_special_panel_experiment_plus_diffusive_pair(std_real, per_simu_std)
