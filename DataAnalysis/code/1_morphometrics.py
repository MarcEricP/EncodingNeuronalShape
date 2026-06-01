import os
import json

import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
from pathlib import Path


from dendrogenesis import project_style

from dendrogenesis import useful as uf
from dendrogenesis import useful_plt
from skimage import io
import tqdm
from dendrotree.nx_graph import nx_graph, metrics
import networkx as nx
from scipy.stats import linregress, norm
import piecewise_regression

project_style.set_style()

PIXELS_PER_MICRON = 16  


def _compute_metrics_from_mask_movie(movie_tif_path, pixels_per_microns, save_metric_path):
    """One process handles the whole real movie (all frames) and writes metrics.json in the same folder.
    Returns (movie_parent_dir_name, list_of_dicts)."""
    movie_dir = os.path.dirname(movie_tif_path)  # .../ij_sift
    run_name = os.path.basename(os.path.dirname(movie_dir))  # movie folder name

    # If metrics.json already exists, just load it
    if os.path.exists(save_metric_path):
        with open(save_metric_path, "r") as f:
            res = json.load(f)
        return run_name, res

    movie = io.imread(movie_tif_path)
    out = []
    for t in tqdm.trange(movie.shape[0], desc=f"Computing metrics for {run_name}"):
        # Skeleton graph for length + ellipse
        g = nx_graph.mask2nx(movie[t], has_contact=True, has_root=True)
        g = nx_graph.refine_graph(g, max_l=8)
        tl = metrics.total_length(g)
        res_ellipse = metrics.tree_inertia_and_equivalent_ellipse(g)
        area = float(np.pi * res_ellipse["semi_minor"] * res_ellipse["semi_major"])

        # Graph from full mask for branches + loops
        g_loop = nx_graph.mask2nx(movie[t] > 0, has_contact=False, has_root=False)
        number_of_branches = int(metrics.n_branch_points(g_loop))
        number_of_loops = len(nx.cycle_basis(g_loop))
        number_of_branches -= number_of_loops

        out.append({
            "time": int(t),
            "total_length": float(tl / pixels_per_microns),
            "semi-minor_axis": float(res_ellipse["semi_minor"] / pixels_per_microns),
            "semi-major_axis": float(res_ellipse["semi_major"] / pixels_per_microns),
            "angle_ellipse": float(res_ellipse["angle_rad"]),
            "centroid_ellipse": [
                float(a / pixels_per_microns) for a in res_ellipse["centroid"]
            ],
            "number_of_branches": number_of_branches,
            "area": float(area / (pixels_per_microns ** 2)),
            "density": float((tl / area) * pixels_per_microns) if area > 0 else float("nan"),
        })

    with open(save_metric_path, "w") as f:
        json.dump(out, f)
    return run_name, out


if __name__ == "__main__":
    base_dir = r"/mnt/c/Users/menir/Documents/0000NeuronsData/clean_movies_1_min"
    # base_dir must contain class_I_very_young, class_I, class_IV, etc.

    repo_root = Path(__file__).resolve().parents[2]
    result_dir = os.path.join(repo_root, "DataAnalysis", "Result", "1_morphometrics")
    plots_dir = os.path.join(result_dir, "plots")
    raw_data_dir = os.path.join(result_dir, "raw_data")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)

    t_range = (0, 300)  # minutes (5 hours)
    save_compute = os.path.join(result_dir, "results.json")
    GENERATE_PER_NEURON_FIT_TABLES = False

    metric_colors = {
        "total_length": "tab:orange",
        "semi-minor_axis": "tab:blue",
        "semi-major_axis": "tab:purple",
        "ellipse_aspect_ratio": "tab:blue",
        "number_of_branches": "black",
        "density" : "tab:red",
    }

    metric_labels = {
        "total_length": "Total length (µm)",
        "semi-minor_axis": "Semi-minor axis (µm)",
        "semi-major_axis": "Semi-major axis (µm)",
        "ellipse_aspect_ratio": "Aspect ratio",
        "number_of_branches": "Branching points #",
        "density" : r"Density ($µm^{-1}$)",
    }

    basic_figsize = (project_style.mm_to_in(60), project_style.mm_to_in(50))

    def compute_results(base_dir, save_path, pixels_per_microns):
        """
        Compute dict_res using ij_sift/metrics.json for each neuron.
        If ij_sift/metrics.json does not exist, compute it from ij_sift/skel_stack.tif.
        """
        exclude = ["flow_track", "flow_track_opened_skel"]

        # Find all skel_stack.tif (one per movie/neuron)
        list_paths = uf.find_files(
            [base_dir],
            [["ij_sift/skel_stack.tif"]],
            exclude=exclude,
        )

        dict_res = {}

        for elem in list_paths:
            neuron_dir = elem[0]          # .../class_X/neuron_name
            movie_tif_path = elem[1]      # .../class_X/neuron_name/ij_sift/skel_stack.tif
            ij_sift_dir = os.path.dirname(movie_tif_path)
            metrics_path = os.path.join(ij_sift_dir, "metrics.json")

            # Compute metrics.json if needed (or load if already exists)
            _, metrics_list = _compute_metrics_from_mask_movie(
                movie_tif_path,
                pixels_per_microns,
                metrics_path,
            )

            name = os.path.basename(neuron_dir)
            neuron_class = os.path.basename(os.path.dirname(neuron_dir))
            if neuron_class not in dict_res:
                dict_res[neuron_class] = {}
            dict_res[neuron_class][name] = {}

            metrics_df = pandas.DataFrame(metrics_list)

            # Basic morphometrics (direct from JSON)
            for key in ["total_length", "semi-major_axis", "semi-minor_axis", "number_of_branches"]:
                if key in metrics_df.columns:
                    dict_res[neuron_class][name][key] = metrics_df[key].to_numpy()
                else:
                    n_frames = len(metrics_df)
                    dict_res[neuron_class][name][key] = np.full(n_frames, np.nan)

            # Derived quantities from semi-axes
            s_min = dict_res[neuron_class][name]["semi-minor_axis"]
            s_maj = dict_res[neuron_class][name]["semi-major_axis"]
            dict_res[neuron_class][name]["ellipse_area"] = np.pi * s_min * s_maj
            dict_res[neuron_class][name]["ellipse_aspect_ratio"] = s_maj / s_min
            dict_res[neuron_class][name]["density"] = dict_res[neuron_class][name]["total_length"]/dict_res[neuron_class][name]["ellipse_area"]

        with open(save_path, "w") as f:
            json.dump(dict_res, f, cls=uf.NumpyEncoder)

        return dict_res

    def load_or_compute_results(base_dir, save_path, pixels_per_microns):
        """
        Load dict_res from JSON, or recompute if file is missing or
        if some required metrics are missing.
        """
        if not os.path.exists(save_path):
            dict_res_ = compute_results(base_dir, save_path, pixels_per_microns)
        else:
            with open(save_path, "r") as f:
                dict_res_ = json.load(f)

            required_keys = ["ellipse_aspect_ratio", "ellipse_area", "number_of_branches"]
            needs_recompute = False

            for cl in dict_res_.values():
                for neuron_data in cl.values():
                    if any(k not in neuron_data for k in required_keys):
                        needs_recompute = True
                        break
                if needs_recompute:
                    break

            if needs_recompute:
                dict_res_ = compute_results(base_dir, save_path, pixels_per_microns)

        return dict_res_

    dict_res = load_or_compute_results(
        base_dir,
        save_compute,
        PIXELS_PER_MICRON,
    )

    def _safe_name(s):
        return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(s)).strip("_")

    def export_raw_metric_csvs(dict_res, out_dir):
        export_metric_order = [
            "total_length",
            "semi-major_axis",
            "semi-minor_axis",
            "ellipse_area",
            "ellipse_aspect_ratio",
            "number_of_branches",
            "density",
        ]
        metric_colnames = {
            "time_after_16": "time after 16 h AEL (min)",
            "time_after_13": "time after 13 h AEL (min)",
            "total_length": "total_length (µm)",
            "semi-major_axis": "semi-major_axis (µm)",
            "semi-minor_axis": "semi-minor_axis (µm)",
            "ellipse_area": "ellipse_area (µm^2)",
            "ellipse_aspect_ratio": "ellipse_aspect_ratio (-)",
            "number_of_branches": "number_of_branches (#)",
            "density": "density (µm^-1)",
        }
        allowed = {
            "class_I": "class_I",
            "class_I_early_5_min": "class_I_early_5_min",
            "class_IV": "class_IV",
            "class_IV_MT_inh": "class_IV_MT_inh",
            "class_IV_mt_inh": "class_IV_MT_inh",
        }
        for class_key, neurons in dict_res.items():
            if class_key not in allowed:
                continue
            class_prefix = allowed[class_key]
            for acquisition_name, metric_dict in neurons.items():
                metric_keys = [
                    k for k in export_metric_order
                    if (k in metric_dict and isinstance(metric_dict[k], (list, tuple, np.ndarray)))
                ]
                if not metric_keys:
                    continue
                lengths = [len(np.asarray(metric_dict[k])) for k in metric_keys]
                n_t = max(lengths) if lengths else 0
                if n_t == 0:
                    continue
                if class_key == "class_I_early_5_min":
                    # Young class I is sampled every 5 min from 13h to 16h AEL.
                    time_axis = 5 * np.arange(n_t, dtype=int)
                    time_col = metric_colnames["time_after_13"]
                else:
                    # Other classes use 1-min sampling from 16h AEL onward.
                    time_axis = np.arange(n_t, dtype=int)
                    time_col = metric_colnames["time_after_16"]
                data = {time_col: time_axis}
                for k in metric_keys:
                    try:
                        arr = np.asarray(metric_dict[k], dtype=float)
                        col = np.full(n_t, np.nan, dtype=float)
                        col[:arr.shape[0]] = arr
                    except Exception:
                        arr = np.asarray(metric_dict[k], dtype=object)
                        col = np.full(n_t, np.nan, dtype=object)
                        col[:arr.shape[0]] = arr
                    data[metric_colnames.get(k, k)] = col
                df = pandas.DataFrame(data)
                out_name = f"{class_prefix}_{_safe_name(acquisition_name)}.csv"
                df.to_csv(os.path.join(out_dir, out_name), index=False)

    export_raw_metric_csvs(dict_res, raw_data_dir)

    COLORS = {
        "class_I": "tab:green",
        "class_IV": "tab:pink",
        "class_IV_MT_inh": "#353535",
    }

    # 1 row x 4 cols: class IV vs class IV MT inh across 4 metrics.
    panel_metrics = [
        ("total_length", r"Total length ($\mu$m)"),
        ("semi-minor_axis", r"Semi-minor axis ($\mu$m)"),
        ("number_of_branches", "Branching points #"),
        ("density", r"Density ($\mu$m$^{-1}$)"),
    ]
    basic_figsize = (project_style.mm_to_in(53), project_style.mm_to_in(50))
    fig_panel, axs_panel = plt.subplots(1, 4, figsize=(basic_figsize[0] * 4, basic_figsize[1]), sharex=False, sharey=False)
    for ax, (metric_key, y_label) in zip(axs_panel, panel_metrics):
        for class_key, legend_label in [("class_IV", "WT cIV"), ("class_IV_MT_inh", "MT inh cIV")]:
            if class_key not in dict_res:
                continue
            temp_tab = []
            for movie_vals in dict_res[class_key].values():
                if metric_key not in movie_vals:
                    continue
                values = np.asarray(movie_vals[metric_key], dtype=float)
                temp_fill = np.nan * np.ones(t_range[1])
                n_keep = min(t_range[1], values.shape[0])
                temp_fill[:n_keep] = values[:n_keep]
                temp_tab.append(temp_fill)
            if not temp_tab:
                continue
            metric_mat = np.asarray(temp_tab, dtype=float)
            time_array = np.arange(t_range[1])
            mean = np.nanmean(metric_mat, axis=0)
            # mean = mean - mean[0]
            std = np.nanstd(metric_mat, axis=0)
            ax.plot(time_array, mean, color=COLORS[class_key], label=legend_label)
            ax.fill_between(time_array, mean - std, mean + std, color=COLORS[class_key], alpha=0.3)

        ax.set_ylabel(y_label)
        ax.set_xlabel("Time (hours AEL)")
        ax = useful_plt.format_hr_x_axis(ax, 16)
        ax.set_box_aspect(1)
    axs_panel[0].legend(frameon=False)
    plt.tight_layout()
    for extension in ["png", "pdf", "svg"]:
        plt.savefig(os.path.join(plots_dir, f"main-morphometrics-cIV-vs-MTinh-1row4col.{extension}"))
    plt.close(fig_panel)

    def build_ci_trajectories(dict_res, metrics, t_range):
        """
        Build time array and trajectories for WT class I (13–21 h AEL)
        from:
          - class_I_very_young (every 5 min, 13–16 h AEL),
          - class_I (after 16 h AEL, 16–21 h).
        """
        cI_young = dict_res["class_I_early_5_min"]
        cI_late = dict_res["class_I"]

        # minimum number of frames among very young movies
        minimax_frame_young = min(
            [len(list(a.values())[0]) for a in cI_young.values()]
        )

        # Time in minutes (0 at 13h AEL, end at 21h AEL)
        time_array = np.concatenate(
            [
                5 * np.arange(minimax_frame_young),          # young (13–16h AEL), 5 min step
                5 * minimax_frame_young + np.arange(t_range[1]),  # late (16–21h AEL), 1 min step
            ]
        )

        all_class = {}
        for key in metrics:
            temp_tab = []

            # young part (class_I_very_young, every 5 min)
            for name in cI_young.keys():
                temp_fill = np.nan * np.ones_like(time_array, dtype=float)
                vals = np.array(cI_young[name][key])[:minimax_frame_young]
                temp_fill[:minimax_frame_young] = vals
                temp_tab.append(temp_fill)

            # late part (class_I, after 16h AEL)
            for name in cI_late.keys():
                temp_fill = np.nan * np.ones_like(time_array, dtype=float)
                vals = np.array(cI_late[name][key])[: t_range[1]]
                temp_fill[minimax_frame_young:] = vals
                temp_tab.append(temp_fill)

            all_class[key] = np.array(temp_tab)

        return time_array, all_class, minimax_frame_young

    def build_civ_trajectories(dict_res, metrics, t_range, class_key="class_IV"):
        """
        Build time array and trajectories for WT class IV (16–21 h AEL).
        """
        temp_dict = dict_res[class_key]
        time_array = np.arange(t_range[1])  # minutes (0–300 => 16–21 h AEL)

        all_class = {}
        for key in metrics:
            temp_tab = []
            for name in temp_dict.keys():
                vals = np.array(temp_dict[name][key])[: t_range[1]]
                temp_tab.append(vals)
            all_class[key] = np.array(temp_tab)

        return time_array, all_class

    metrics_to_plot = [
        "total_length",
        "semi-minor_axis",
        "semi-major_axis",
        "ellipse_aspect_ratio",
        "number_of_branches",
        "density",
    ]

    time_ci, all_ci, minimax_frame_young = build_ci_trajectories(
        dict_res,
        metrics_to_plot,
        t_range,
    )
    time_civ, all_civ = build_civ_trajectories(
        dict_res,
        metrics_to_plot,
        t_range,
        class_key="class_IV",
    )


    def _plot_mean_std(
        ax,
        time_array,
        trajs,
        color,
        label,
        bottom_zero=True,
        label_color=None,
    ):
        trajs = np.array(trajs, dtype=float)
        mean = np.nanmean(trajs, axis=0)
        std = np.nanstd(trajs, axis=0)

        ax.plot(time_array, mean, color=color)
        ax.fill_between(time_array, mean - std, mean + std, color=color, alpha=0.3)
        if label_color is None:
            label_color = color
        ax.set_ylabel(label, color=label_color)
        ax.tick_params(axis="y", colors=label_color)
        if bottom_zero:
            ax.set_ylim(bottom=0)

    def _plot_two_metrics_same_axis(
        ax,
        time_array,
        trajs_a,
        trajs_b,
        color_a,
        color_b,
        label_a,
        label_b,
        bottom_zero=True,
    ):
        """
        Overlay two metrics on a single y-axis (no twinx), keeping distinct colors.
        """
        trajs_a = np.array(trajs_a, dtype=float)
        mean_a = np.nanmean(trajs_a, axis=0)
        std_a = np.nanstd(trajs_a, axis=0)
        ax.plot(time_array, mean_a, color=color_a, label=label_a)
        ax.fill_between(time_array, mean_a - std_a, mean_a + std_a, color=color_a, alpha=0.3)

        trajs_b = np.array(trajs_b, dtype=float)
        mean_b = np.nanmean(trajs_b, axis=0)
        std_b = np.nanstd(trajs_b, axis=0)
        ax.plot(time_array, mean_b, color=color_b, label=label_b)
        ax.fill_between(time_array, mean_b - std_b, mean_b + std_b, color=color_b, alpha=0.3)

        ax.set_ylabel("Semi-axes (µm)", color="black")
        ax.tick_params(axis="y", colors="black")
        if bottom_zero:
            ax.set_ylim(bottom=0)

    def plot_morphometrics_panel(
        time_ci,
        all_ci,
        time_civ,
        all_civ,
        metric_colors,
        metric_labels,
        start_hour_ci,
        start_hour_civ,
        out_prefix,
        force_zero=False,
        ci_breakpoints=None,
    ):
        fig, axes = plt.subplots(2, 4, figsize=(basic_figsize[0] * 4, basic_figsize[1] * 2), sharex=False)

        # Row 0: class IV
        _plot_mean_std(
            axes[0, 0],
            time_civ,
            all_civ["total_length"],
            metric_colors["total_length"],
            metric_labels["total_length"],
            label_color="black",
            bottom_zero=force_zero,
        )

        _plot_two_metrics_same_axis(
            axes[0, 1],
            time_civ,
            all_civ["semi-minor_axis"],
            all_civ["semi-major_axis"],
            metric_colors["semi-minor_axis"],
            metric_colors["semi-major_axis"],
            metric_labels["semi-minor_axis"],
            metric_labels["semi-major_axis"],
            bottom_zero=force_zero,
        )

        _plot_mean_std(
            axes[0, 2],
            time_civ,
            all_civ["number_of_branches"],
            metric_colors["number_of_branches"],
            metric_labels["number_of_branches"],
            label_color="black",
            bottom_zero=force_zero,
        )
        _plot_mean_std(
            axes[0, 3],
            time_civ,
            all_civ["density"],
            metric_colors["density"],
            metric_labels["density"],
            bottom_zero=force_zero,
            label_color="black",
        )

        # Row 1: class I
        _plot_mean_std(
            axes[1, 0],
            time_ci,
            all_ci["total_length"],
            metric_colors["total_length"],
            metric_labels["total_length"],
            label_color="black",
            bottom_zero=force_zero,
        )

        _plot_mean_std(
            axes[1, 1],
            time_ci,
            all_ci["semi-minor_axis"],
            metric_colors["semi-minor_axis"],
            metric_labels["semi-minor_axis"],
            label_color="black",
            bottom_zero=force_zero,
        )

        _plot_mean_std(axes[1, 2],
            time_ci,
            all_ci["number_of_branches"],
            metric_colors["number_of_branches"],
            metric_labels["number_of_branches"],
            label_color="black",
            bottom_zero=force_zero,
        )
        _plot_mean_std(
            axes[1, 3],
            time_ci,
            all_ci["density"],
            metric_colors["density"],
            metric_labels["density"],
            bottom_zero=force_zero,
            label_color="black",
        )

        # X axis formatting
        for col in range(4):
            useful_plt.format_hr_x_axis(axes[0, col], start_hour_civ)
            useful_plt.format_hr_x_axis(axes[1, col], start_hour_ci)
            axes[1, col].xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
            axes[0, col].set_xlabel("Time (hours AEL)")
            axes[1, col].set_xlabel("Time (hours AEL)")
            axes[0, col].set_box_aspect(1)
            axes[1, col].set_box_aspect(1)
            if ci_breakpoints:
                for bp in ci_breakpoints:
                    axes[1, col].axvline(bp, color="0.5", linestyle="--")

        # fig.subplots_adjust(
        #     left=0.06, right=0.98,
        #     bottom=0.08, top=0.95,
        #     wspace=0.5, hspace=0.2  # smaller gaps
        # )
        fig.tight_layout()

        for ext in ["png", "pdf", "svg"]:
            fig.savefig(
                os.path.join(plots_dir, f"{out_prefix}.{ext}"),
                bbox_inches="tight",
            )
        plt.close(fig)

    def plot_supplementary_all_metrics_three_classes(
        dict_res,
        metrics_to_plot,
        metric_colors,
        metric_labels,
        t_range,
        out_prefix,
    ):
        class_labels = {
            "class_I": "WT cI",
            "class_IV": "WT cIV",
            "class_IV_MT_inh": "MT inh cIV",
            "class_IV_mt_inh": "MT inh cIV",
        }
        class_plot_colors = {
            "class_I": "tab:green",
            "class_IV": "tab:pink",
            "class_IV_MT_inh": "#353535",
            "class_IV_mt_inh": "#353535",
        }
        candidate_classes = ["class_I", "class_IV", "class_IV_MT_inh", "class_IV_mt_inh"]
        classes_present = [c for c in candidate_classes if c in dict_res]

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(basic_figsize[0] * 3, basic_figsize[1] * 2),
            sharex=False,
            sharey=False,
        )
        axes = axes.ravel()

        for ax, metric_key in zip(axes, metrics_to_plot):
            for class_key in classes_present:
                temp_tab = []
                for movie_vals in dict_res[class_key].values():
                    if metric_key not in movie_vals:
                        continue
                    values = np.asarray(movie_vals[metric_key], dtype=float)
                    temp_fill = np.nan * np.ones(t_range[1], dtype=float)
                    n_keep = min(t_range[1], values.shape[0])
                    temp_fill[:n_keep] = values[:n_keep]
                    temp_tab.append(temp_fill)

                if not temp_tab:
                    continue

                metric_mat = np.asarray(temp_tab, dtype=float)
                time_array = np.arange(t_range[1])
                mean = np.nanmean(metric_mat, axis=0)
                std = np.nanstd(metric_mat, axis=0)

                ax.plot(
                    time_array,
                    mean,
                    color=class_plot_colors[class_key],
                    label=class_labels[class_key],
                )
                ax.fill_between(
                    time_array,
                    mean - std,
                    mean + std,
                    color=class_plot_colors[class_key],
                    alpha=0.2,
                )

            ax.set_ylabel(metric_labels.get(metric_key, metric_key))
            ax.tick_params(axis="y")
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Time (hours AEL)")
            useful_plt.format_hr_x_axis(ax, 16)
            ax.set_box_aspect(1)

        if classes_present:
            axes[0].legend(frameon=False)

        fig.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(
                os.path.join(plots_dir, f"{out_prefix}.{ext}"),
                bbox_inches="tight",
            )
        plt.close(fig)

    def plot_supplementary_all_metrics_single_class(
        dict_res,
        class_key,
        metrics_to_plot,
        metric_labels,
        t_range,
        out_prefix,
        line_color="black",
    ):
        if class_key not in dict_res:
            return

        class_labels = {
            "class_I": "WT cI",
            "class_IV": "WT cIV",
            "class_IV_MT_inh": "MT inh cIV",
            "class_IV_mt_inh": "MT inh cIV",
        }
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(basic_figsize[0] * 3, basic_figsize[1] * 2),
            sharex=False,
            sharey=False,
        )
        axes = axes.ravel()

        ci_minimax_frame_young = None
        ci_start_hour = 16.0
        if class_key == "class_I" and "class_I_early_5_min" in dict_res and len(dict_res["class_I_early_5_min"]) > 0:
            ci_minimax_frame_young = min(
                [len(list(a.values())[0]) for a in dict_res["class_I_early_5_min"].values()]
            )
            ci_start_hour = 16 - ci_minimax_frame_young * 5 / 60.0

        for ax, metric_key in zip(axes, metrics_to_plot):
            temp_tab = []
            if class_key == "class_I" and ci_minimax_frame_young is not None:
                time_array = np.concatenate(
                    [
                        5 * np.arange(ci_minimax_frame_young, dtype=float),
                        5 * ci_minimax_frame_young + np.arange(t_range[1], dtype=float),
                    ]
                )
                total_len = time_array.shape[0]

                for movie_vals in dict_res["class_I_early_5_min"].values():
                    if metric_key not in movie_vals:
                        continue
                    values = np.asarray(movie_vals[metric_key], dtype=float)
                    temp_fill = np.nan * np.ones(total_len, dtype=float)
                    n_keep = min(ci_minimax_frame_young, values.shape[0])
                    temp_fill[:n_keep] = values[:n_keep]
                    temp_tab.append(temp_fill)
                    ax.plot(time_array, temp_fill, linestyle="--", color=line_color, alpha=0.25)

                for movie_vals in dict_res["class_I"].values():
                    if metric_key not in movie_vals:
                        continue
                    values = np.asarray(movie_vals[metric_key], dtype=float)
                    temp_fill = np.nan * np.ones(total_len, dtype=float)
                    n_keep = min(t_range[1], values.shape[0])
                    temp_fill[ci_minimax_frame_young: ci_minimax_frame_young + n_keep] = values[:n_keep]
                    temp_tab.append(temp_fill)
                    ax.plot(time_array, temp_fill, linestyle="--", color=line_color, alpha=0.25)
            else:
                time_array = np.arange(t_range[1], dtype=float)
                for movie_vals in dict_res[class_key].values():
                    if metric_key not in movie_vals:
                        continue
                    values = np.asarray(movie_vals[metric_key], dtype=float)
                    temp_fill = np.nan * np.ones(t_range[1], dtype=float)
                    n_keep = min(t_range[1], values.shape[0])
                    temp_fill[:n_keep] = values[:n_keep]
                    temp_tab.append(temp_fill)
                    ax.plot(time_array, temp_fill, linestyle="--", color=line_color, alpha=0.25)

            if temp_tab:
                metric_mat = np.asarray(temp_tab, dtype=float)
                mean = np.nanmean(metric_mat, axis=0)
                std = np.nanstd(metric_mat, axis=0)
                ax.plot(time_array, mean, color=line_color)
                ax.fill_between(time_array, mean - std, mean + std, color=line_color, alpha=0.2)

            ax.set_title(class_labels.get(class_key, class_key))
            ax.set_ylabel(metric_labels.get(metric_key, metric_key))
            ax.tick_params(axis="y")
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Time (hours AEL)")
            useful_plt.format_hr_x_axis(ax, ci_start_hour if class_key == "class_I" else 16)
            ax.set_box_aspect(1)

        fig.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(
                os.path.join(plots_dir, f"{out_prefix}.{ext}"),
                bbox_inches="tight",
            )
        plt.close(fig)

    def _fmt_num(x, ff="%.2g"):
        return (ff % x) if np.isfinite(x) else "NA"

    def _fmt_pm(val, se, ff="%.2g"):
        if np.isfinite(val) and np.isfinite(se):
            return f"{ff % val} ± {ff % se}"
        if np.isfinite(val):
            return ff % val
        return "NA"

    def compute_or_load_pooled_fit_results(
        metrics_to_plot,
        time_ci,
        all_ci,
        time_civ,
        all_civ,
        dict_res,
        t_range,
        cache_path,
        force_recompute=False,
    ):
        civ_mtinh_key = "class_IV_MT_inh" if "class_IV_MT_inh" in dict_res else "class_IV_mt_inh"
        class_payload = {
            "class_I": {"time": time_ci, "all_class": all_ci, "fit_type": "piecewise"},
            "class_IV": {"time": time_civ, "all_class": all_civ, "fit_type": "linear"},
        }
        if civ_mtinh_key in dict_res:
            time_mt, all_mt = build_civ_trajectories(dict_res, metrics_to_plot, t_range, class_key=civ_mtinh_key)
            class_payload[civ_mtinh_key] = {"time": time_mt, "all_class": all_mt, "fit_type": "linear"}

        class_display = {
            "class_I": "cI",
            "class_IV": "cIV",
            "class_IV_MT_inh": "cIV MT inh",
            "class_IV_mt_inh": "cIV MT inh",
        }
        class_neuron_counts = {
            "class_I": len(dict_res.get("class_I", {})),
            "class_IV": len(dict_res.get("class_IV", {})),
            "class_IV_MT_inh": len(dict_res.get("class_IV_MT_inh", {})),
            "class_IV_mt_inh": len(dict_res.get("class_IV_mt_inh", {})),
        }
        breakpoints_ci = {
            "total_length": 1,
            "semi-minor_axis": 2,
            "semi-major_axis": 1,
            "ellipse_aspect_ratio": 2,
            "number_of_branches": 2,
            "density": 1,
        }

        cache_meta = {
            "metrics_to_plot": list(metrics_to_plot),
            "t_range": list(t_range),
            "class_counts": class_neuron_counts,
            "cache_version": 1,
        }

        if (not force_recompute) and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)
                if cache_data.get("meta") == cache_meta and "results" in cache_data:
                    return cache_data["results"]
            except Exception:
                pass

        pooled_results = []
        for class_key, payload in class_payload.items():
            time_array = payload["time"]
            all_class = payload["all_class"]
            fit_type = payload["fit_type"]

            for metric_key in metrics_to_plot:
                if metric_key not in all_class:
                    continue
                t = np.concatenate([time_array] * all_class[metric_key].shape[0]).astype(float)
                y = np.concatenate(all_class[metric_key]).astype(float)
                keep = np.isfinite(t) & np.isfinite(y)
                t = t[keep]
                y = y[keep]

                entry = {
                    "class_key": class_key,
                    "class_label": class_display.get(class_key, class_key),
                    "metric_key": metric_key,
                    "fit_type": fit_type,
                    "n_neurons": int(class_neuron_counts.get(class_key, all_class[metric_key].shape[0])),
                    "status": "ok",
                    "linear": None,
                    "piecewise": None,
                }

                if y.size < 3:
                    entry["status"] = "insufficient_points"
                    pooled_results.append(entry)
                    continue

                if fit_type == "linear":
                    try:
                        lr = linregress(t, y)
                        entry["linear"] = {
                            "slope": float(lr.slope),
                            "slope_se": float(getattr(lr, "stderr", np.nan)),
                            "pvalue": float(lr.pvalue),
                            "r2": float(lr.rvalue ** 2),
                        }
                    except Exception:
                        entry["status"] = "fit_error"
                else:
                    try:
                        n_bp = breakpoints_ci.get(metric_key, 1)
                        pw_fit = piecewise_regression.Fit(t, y, n_breakpoints=n_bp)
                        res = pw_fit.get_results()
                        est = res["estimates"]
                        alphas = []
                        breakpoints = []
                        for i in range(1, n_bp + 2):
                            key = f"alpha{i}"
                            if key in est:
                                alphas.append({
                                    "estimate": float(est[key]["estimate"]),
                                    "se": float(est[key]["se"]),
                                })
                        for i in range(1, n_bp + 1):
                            key = f"breakpoint{i}"
                            if key in est:
                                breakpoints.append({
                                    "estimate": float(est[key]["estimate"]),
                                    "se": float(est[key]["se"]),
                                })
                        entry["piecewise"] = {
                            "n_breakpoints": n_bp,
                            "alphas": alphas,
                            "breakpoints": breakpoints,
                        }
                    except Exception:
                        entry["status"] = "fit_error"

                pooled_results.append(entry)

        with open(cache_path, "w") as f:
            json.dump({"meta": cache_meta, "results": pooled_results}, f, indent=2)
        return pooled_results

    def build_fit_summary_table(
        metrics_to_plot,
        time_ci,
        all_ci,
        time_civ,
        all_civ,
        dict_res,
        t_range,
        out_prefix,
        generate_per_neuron_tables=False,
        pooled_fit_results=None,
    ):
        civ_mtinh_key = "class_IV_MT_inh" if "class_IV_MT_inh" in dict_res else "class_IV_mt_inh"
        class_payload = {
            "class_I": {"time": time_ci, "all_class": all_ci, "fit_type": "piecewise"},
            "class_IV": {"time": time_civ, "all_class": all_civ, "fit_type": "linear"},
        }
        if civ_mtinh_key in dict_res:
            time_mt, all_mt = build_civ_trajectories(dict_res, metrics_to_plot, t_range, class_key=civ_mtinh_key)
            class_payload[civ_mtinh_key] = {"time": time_mt, "all_class": all_mt, "fit_type": "linear"}

        class_display = {
            "class_I": "cI",
            "class_IV": "cIV",
            "class_IV_MT_inh": "cIV MT inh",
            "class_IV_mt_inh": "cIV MT inh",
        }
        class_neuron_counts = {
            "class_I": len(dict_res.get("class_I", {})),
            "class_IV": len(dict_res.get("class_IV", {})),
            "class_IV_MT_inh": len(dict_res.get("class_IV_MT_inh", {})),
            "class_IV_mt_inh": len(dict_res.get("class_IV_mt_inh", {})),
        }
        class_i_early_n = len(dict_res.get("class_I_early_5_min", {}))
        class_i_late_n = len(dict_res.get("class_I", {}))
        class_i_neuron_label = f"from 13 AEL {class_i_early_n} + from 16 AEL {class_i_late_n}"
        breakpoints_ci = {
            "total_length": 1,
            "semi-minor_axis": 2,
            "semi-major_axis": 1,
            "ellipse_aspect_ratio": 2,
            "number_of_branches": 2,
            "density": 1,
        }
        linear_pvalue_test = "Wald t-test on slope (SciPy linregress)"
        linear_pvalue_model = "OLS linear model: y = intercept + slope*time"
        linear_h0 = "H0: slope = 0"
        linear_h1 = "H1: slope != 0"
        linear_sidedness = "two-sided"
        piecewise_slope_test = "Wald test on segment slope (z=estimate/SE)"
        piecewise_model = "Piecewise linear regression"
        piecewise_h0 = "H0: slope_phase_k = 0"
        piecewise_h1 = "H1: slope_phase_k != 0"
        piecewise_sidedness = "two-sided"

        if pooled_fit_results is None:
            pooled_fit_results = compute_or_load_pooled_fit_results(
                metrics_to_plot=metrics_to_plot,
                time_ci=time_ci,
                all_ci=all_ci,
                time_civ=time_civ,
                all_civ=all_civ,
                dict_res=dict_res,
                t_range=t_range,
                cache_path=os.path.join(result_dir, "supplementary_morphometrics_pooled_fit_cache.json"),
                force_recompute=False,
            )

        rows = []
        def _fmt_ci95(est, se):
            if (not np.isfinite(est)) or (not np.isfinite(se)):
                return "NA"
            lo = est - 1.96 * se
            hi = est + 1.96 * se
            return f"[{_fmt_num(lo, '%.3g')}, {_fmt_num(hi, '%.3g')}]"

        def _wald_two_sided_p(est, se):
            if (not np.isfinite(est)) or (not np.isfinite(se)) or se <= 0:
                return np.nan
            z = est / se
            return float(2.0 * norm.sf(abs(z)))

        def _segment_counts_from_breakpoints(entry):
            if entry.get("class_key") != "class_I":
                return []
            metric_key = entry.get("metric_key")
            if metric_key not in all_ci:
                return []
            piece = entry.get("piecewise")
            if not isinstance(piece, dict):
                return []
            bps = [float(bp.get("estimate", np.nan)) for bp in piece.get("breakpoints", [])]
            t = np.concatenate([time_ci] * all_ci[metric_key].shape[0]).astype(float)
            y = np.concatenate(all_ci[metric_key]).astype(float)
            keep = np.isfinite(t) & np.isfinite(y)
            t = t[keep]
            if t.size == 0:
                return []
            counts = []
            low = -np.inf
            for bp in bps:
                counts.append(int(np.sum((t > low) & (t <= bp))))
                low = bp
            counts.append(int(np.sum(t > low)))
            return counts

        for entry in pooled_fit_results:
            row = {
                "Class": entry["class_label"],
                "Metric": metric_labels.get(entry["metric_key"], entry["metric_key"]),
                "Model": entry["fit_type"],
                "n_neurons": int(entry["n_neurons"]),
                "Slope1": "NA",
                "Slope1 95% CI": "NA",
                "Slope1 p (Wald)": "NA",
                "n_phase1": "NA",
                "Slope2": "NA",
                "Slope2 95% CI": "NA",
                "Slope2 p (Wald)": "NA",
                "n_phase2": "NA",
                "Slope3": "NA",
                "Slope3 95% CI": "NA",
                "Slope3 p (Wald)": "NA",
                "n_phase3": "NA",
                "Breakpoint1 (min)": "NA",
                "Breakpoint1 95% CI (min)": "NA",
                "Breakpoint2 (min)": "NA",
                "Breakpoint2 95% CI (min)": "NA",
                "Linear p": "NA",
                "Linear R2": "NA",
                "p-value test": "NA",
                "p-value model": "NA",
                "Hypothesis": "NA",
                "Alternative": "NA",
                "Sidedness": "NA",
            }
            if entry["fit_type"] == "linear" and isinstance(entry.get("linear"), dict):
                lin = entry["linear"]
                row["Slope1"] = _fmt_pm(lin.get("slope", np.nan), lin.get("slope_se", np.nan))
                row["Slope1 95% CI"] = _fmt_ci95(lin.get("slope", np.nan), lin.get("slope_se", np.nan))
                row["Linear p"] = _fmt_num(lin.get("pvalue", np.nan))
                row["Linear R2"] = _fmt_num(lin.get("r2", np.nan))
                row["p-value test"] = linear_pvalue_test
                row["p-value model"] = linear_pvalue_model
                row["Hypothesis"] = linear_h0
                row["Alternative"] = linear_h1
                row["Sidedness"] = linear_sidedness
            if entry["fit_type"] == "piecewise" and isinstance(entry.get("piecewise"), dict):
                piece = entry["piecewise"]
                alphas = piece.get("alphas", [])
                bps = piece.get("breakpoints", [])
                seg_counts = _segment_counts_from_breakpoints(entry)
                if len(alphas) > 0:
                    est = alphas[0].get("estimate", np.nan)
                    se = alphas[0].get("se", np.nan)
                    row["Slope1"] = _fmt_pm(est, se)
                    row["Slope1 95% CI"] = _fmt_ci95(est, se)
                    row["Slope1 p (Wald)"] = _fmt_num(_wald_two_sided_p(est, se))
                    if len(seg_counts) > 0:
                        row["n_phase1"] = str(seg_counts[0])
                if len(alphas) > 1:
                    est = alphas[1].get("estimate", np.nan)
                    se = alphas[1].get("se", np.nan)
                    row["Slope2"] = _fmt_pm(est, se)
                    row["Slope2 95% CI"] = _fmt_ci95(est, se)
                    row["Slope2 p (Wald)"] = _fmt_num(_wald_two_sided_p(est, se))
                    if len(seg_counts) > 1:
                        row["n_phase2"] = str(seg_counts[1])
                if len(alphas) > 2:
                    est = alphas[2].get("estimate", np.nan)
                    se = alphas[2].get("se", np.nan)
                    row["Slope3"] = _fmt_pm(est, se)
                    row["Slope3 95% CI"] = _fmt_ci95(est, se)
                    row["Slope3 p (Wald)"] = _fmt_num(_wald_two_sided_p(est, se))
                    if len(seg_counts) > 2:
                        row["n_phase3"] = str(seg_counts[2])
                if len(bps) > 0:
                    bp1_est = bps[0].get("estimate", np.nan)
                    bp1_se = bps[0].get("se", np.nan)
                    row["Breakpoint1 (min)"] = _fmt_pm(bp1_est, bp1_se)
                    row["Breakpoint1 95% CI (min)"] = _fmt_ci95(bp1_est, bp1_se)
                if len(bps) > 1:
                    bp2_est = bps[1].get("estimate", np.nan)
                    bp2_se = bps[1].get("se", np.nan)
                    row["Breakpoint2 (min)"] = _fmt_pm(bp2_est, bp2_se)
                    row["Breakpoint2 95% CI (min)"] = _fmt_ci95(bp2_est, bp2_se)
                row["Model"] = piecewise_model
                row["p-value test"] = piecewise_slope_test
                row["Hypothesis"] = piecewise_h0
                row["Alternative"] = piecewise_h1
                row["Sidedness"] = piecewise_sidedness
            rows.append(row)

        table_df = pandas.DataFrame(rows, columns=[
            "Class",
            "Metric",
            "Model",
            "n_neurons",
            "Slope1",
            "Slope1 95% CI",
            "Slope1 p (Wald)",
            "n_phase1",
            "Slope2",
            "Slope2 95% CI",
            "Slope2 p (Wald)",
            "n_phase2",
            "Slope3",
            "Slope3 95% CI",
            "Slope3 p (Wald)",
            "n_phase3",
            "Breakpoint1 (min)",
            "Breakpoint1 95% CI (min)",
            "Breakpoint2 (min)",
            "Breakpoint2 95% CI (min)",
            "Linear p",
            "Linear R2",
            "p-value test",
            "p-value model",
            "Hypothesis",
            "Alternative",
            "Sidedness",
        ])

        def _save_table_bundle(df, suffix):
            csv_path = os.path.join(result_dir, f"{out_prefix}_{suffix}.csv")
            md_path = os.path.join(result_dir, f"{out_prefix}_{suffix}.md")

            df.to_csv(csv_path, index=False)

            try:
                md_text = df.to_markdown(index=False)
            except Exception:
                headers = list(df.columns)
                sep = ["---"] * len(headers)
                data_lines = [
                    "| " + " | ".join(str(v) for v in row) + " |"
                    for row in df.values.tolist()
                ]
                md_text = "\n".join([
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(sep) + " |",
                    *data_lines,
                ])
            with open(md_path, "w") as f:
                f.write(md_text + "\n")
        def _fmt_p_compact(x):
            try:
                p = float(x)
            except Exception:
                return "NA"
            if not np.isfinite(p):
                return "NA"
            if p < 1e-3:
                return "p < 0.001"
            return f"p = {p:.3g}"

        table_ci = table_df[table_df["Class"] == "cI"].copy()
        if not table_ci.empty:
            table_ci = table_ci.astype({"n_neurons": "object"})
            table_ci.loc[:, "n_neurons"] = class_i_neuron_label
        table_civ = table_df[table_df["Class"] == "cIV"].copy()
        table_mtinh = table_df[table_df["Class"] == "cIV MT inh"].copy()
        table_ci = table_ci.rename(columns={
            "Metric": "Morphometric metric",
            "n_neurons": "Number of neurons",
            "Slope1": "Slope phase 1 ± SE",
            "Slope1 95% CI": "Slope phase 1 95% CI",
            "Slope1 p (Wald)": "Slope phase 1 Wald t-test two-sided p-value (H0: slope = 0)",
            "Slope2": "Slope phase 2 ± SE",
            "Slope2 95% CI": "Slope phase 2 95% CI",
            "Slope2 p (Wald)": "Slope phase 2 Wald t-test two-sided p-value (H0: slope = 0)",
            "Slope3": "Slope phase 3 ± SE",
            "Slope3 95% CI": "Slope phase 3 95% CI",
            "Slope3 p (Wald)": "Slope phase 3 Wald t-test two-sided p-value (H0: slope = 0)",
            "Breakpoint1 (min)": "Breakpoint 1 ± SE (min)",
            "Breakpoint1 95% CI (min)": "Breakpoint 1 95% CI (min)",
            "Breakpoint2 (min)": "Breakpoint 2 ± SE (min)",
            "Breakpoint2 95% CI (min)": "Breakpoint 2 95% CI (min)",
        })[[
            "Morphometric metric",
            "Number of neurons",
            "Slope phase 1 ± SE",
            "Slope phase 1 95% CI",
            "Slope phase 1 Wald t-test two-sided p-value (H0: slope = 0)",
            "Slope phase 2 ± SE",
            "Slope phase 2 95% CI",
            "Slope phase 2 Wald t-test two-sided p-value (H0: slope = 0)",
            "Slope phase 3 ± SE",
            "Slope phase 3 95% CI",
            "Slope phase 3 Wald t-test two-sided p-value (H0: slope = 0)",
            "Breakpoint 1 ± SE (min)",
            "Breakpoint 1 95% CI (min)",
            "Breakpoint 2 ± SE (min)",
            "Breakpoint 2 95% CI (min)",
        ]]
        _save_table_bundle(table_ci, "classI")

        def _linear_table(df):
            if df.empty:
                return df
            out = df.copy()
            out["Fit model"] = "OLS linear model"
            out["Time Range"] = "16 h AEL - 21 h AEL"
            out["Number of time points"] = str(t_range[1])
            out["Wald t-test two-sided p-value (H0: slope = 0)"] = out["Linear p"].map(_fmt_p_compact)
            out = out.rename(columns={
                "Metric": "Morphometric metric",
                "n_neurons": "Number of neurons",
                "Slope1": "Slope ± SE",
                "Slope1 95% CI": "Slope 95% CI",
                "Linear R2": "Linear fit R squared",
            })[[
                "Morphometric metric",
                "Number of neurons",
                "Fit model",
                "Time Range",
                "Number of time points",
                "Slope ± SE",
                "Slope 95% CI",
                "Wald t-test two-sided p-value (H0: slope = 0)",
                "Linear fit R squared",
            ]]
            return out

        table_civ = _linear_table(table_civ)
        table_mtinh = _linear_table(table_mtinh)
        if not table_civ.empty:
            _save_table_bundle(table_civ, "classIV")
        if not table_mtinh.empty:
            _save_table_bundle(table_mtinh, "classIV_MTinh")

        def _fmt_mean_sd(values, ff="%.2g"):
            vals = np.asarray(values, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return "NA"
            mean_val = float(np.nanmean(vals))
            if vals.size >= 2:
                sd_val = float(np.nanstd(vals, ddof=1))
                return f"{ff % mean_val} ± {ff % sd_val}"
            return ff % mean_val

        if generate_per_neuron_tables:
            neuron_rows = []
            for class_key, payload in class_payload.items():
                time_array = np.asarray(payload["time"], dtype=float)
                all_class = payload["all_class"]
                fit_type = payload["fit_type"]

                for metric_key in metrics_to_plot:
                    if metric_key not in all_class:
                        continue

                    trajs = np.asarray(all_class[metric_key], dtype=float)
                    n_neurons_total = int(trajs.shape[0])

                    slope1_vals = []
                    slope2_vals = []
                    slope3_vals = []
                    bp1_vals = []
                    bp2_vals = []
                    p_vals = []
                    r2_vals = []
                    n_used = 0

                    for i in range(n_neurons_total):
                        y_i = np.asarray(trajs[i], dtype=float)
                        keep_i = np.isfinite(time_array) & np.isfinite(y_i)
                        t_i = time_array[keep_i]
                        y_i = y_i[keep_i]
                        if y_i.size < 3:
                            continue

                        if fit_type == "linear":
                            if np.unique(t_i).size < 2:
                                continue
                            try:
                                lr_i = linregress(t_i, y_i)
                                slope1_vals.append(float(lr_i.slope))
                                p_vals.append(float(lr_i.pvalue))
                                r2_vals.append(float(lr_i.rvalue ** 2))
                                n_used += 1
                            except Exception:
                                continue
                        else:
                            n_bp = breakpoints_ci.get(metric_key, 1)
                            if y_i.size <= (n_bp + 2):
                                continue
                            try:
                                fit_i = piecewise_regression.Fit(t_i, y_i, n_breakpoints=n_bp)
                                res_i = fit_i.get_results()
                                est_i = res_i["estimates"]
                                if "alpha1" in est_i:
                                    slope1_vals.append(float(est_i["alpha1"]["estimate"]))
                                if "alpha2" in est_i:
                                    slope2_vals.append(float(est_i["alpha2"]["estimate"]))
                                if "alpha3" in est_i:
                                    slope3_vals.append(float(est_i["alpha3"]["estimate"]))
                                if "breakpoint1" in est_i:
                                    bp1_vals.append(float(est_i["breakpoint1"]["estimate"]))
                                if "breakpoint2" in est_i:
                                    bp2_vals.append(float(est_i["breakpoint2"]["estimate"]))
                                n_used += 1
                            except Exception:
                                continue

                    neuron_rows.append({
                        "Class": class_display.get(class_key, class_key),
                        "Metric": metric_labels.get(metric_key, metric_key),
                        "Model": f"{fit_type} (per-neuron avg)",
                        "n_neurons": n_neurons_total,
                        "n_neurons_used": n_used,
                        "Slope1": _fmt_mean_sd(slope1_vals),
                        "Slope2": _fmt_mean_sd(slope2_vals),
                        "Slope3": _fmt_mean_sd(slope3_vals),
                        "Breakpoint1 (min)": _fmt_mean_sd(bp1_vals),
                        "Breakpoint2 (min)": _fmt_mean_sd(bp2_vals),
                        "Linear p": _fmt_mean_sd(p_vals) if fit_type == "linear" else "NA",
                        "Linear R2": _fmt_mean_sd(r2_vals) if fit_type == "linear" else "NA",
                    })

            neuron_table_df = pandas.DataFrame(neuron_rows, columns=[
                "Class",
                "Metric",
                "Model",
                "n_neurons",
                "n_neurons_used",
                "Slope1",
                "Slope2",
                "Slope3",
                "Breakpoint1 (min)",
                "Breakpoint2 (min)",
                "Linear p",
                "Linear R2",
            ])
            neuron_table_ci = neuron_table_df[neuron_table_df["Class"] == "cI"].copy()
            if not neuron_table_ci.empty:
                neuron_table_ci = neuron_table_ci.astype({"n_neurons": "object"})
                neuron_table_ci.loc[:, "n_neurons"] = class_i_neuron_label
            neuron_table_civ = neuron_table_df[neuron_table_df["Class"] == "cIV"].copy()
            neuron_table_mtinh = neuron_table_df[neuron_table_df["Class"] == "cIV MT inh"].copy()
            neuron_table_ci = neuron_table_ci[[
                "Metric",
                "n_neurons",
                "n_neurons_used",
                "Slope1",
                "Slope2",
                "Slope3",
                "Breakpoint1 (min)",
                "Breakpoint2 (min)",
            ]]
            neuron_table_civ = neuron_table_civ[[
                "Class",
                "Metric",
                "Model",
                "n_neurons",
                "n_neurons_used",
                "Slope1",
                "Linear p",
                "Linear R2",
            ]]
            neuron_table_mtinh = neuron_table_mtinh[[
                "Class",
                "Metric",
                "Model",
                "n_neurons",
                "n_neurons_used",
                "Slope1",
                "Linear p",
                "Linear R2",
            ]]
            _save_table_bundle(
                neuron_table_ci,
                "classI_per_neuron_avg",
                "Morphometrics fit summary - cI per-neuron averages",
                fig_w=60,
                fontsize=10,
                row_scale=1.35,
            )
            if not neuron_table_civ.empty:
                _save_table_bundle(
                    neuron_table_civ,
                    "classIV_per_neuron_avg",
                    "Morphometrics fit summary - cIV per-neuron averages",
                )
            if not neuron_table_mtinh.empty:
                _save_table_bundle(
                    neuron_table_mtinh,
                    "classIV_MTinh_per_neuron_avg",
                    "Morphometrics fit summary - cIV MT inh per-neuron averages",
                )
        return pooled_fit_results


    # Class IV: from 16h to 21h AEL
    supplementary_metrics = [
        "total_length",
        "semi-minor_axis",
        "semi-major_axis",
        "ellipse_aspect_ratio",
        "number_of_branches",
        "density",
    ]

    plot_supplementary_all_metrics_three_classes(
        dict_res=dict_res,
        metrics_to_plot=supplementary_metrics,
        metric_colors=metric_colors,
        metric_labels=metric_labels,
        t_range=t_range,
        out_prefix="supplementary_morphometrics_cI_cIV_cIVMTinh_all_metrics",
    )
    plot_supplementary_all_metrics_single_class(
        dict_res=dict_res,
        class_key="class_I",
        metrics_to_plot=supplementary_metrics,
        metric_labels=metric_labels,
        t_range=t_range,
        out_prefix="supplementary_morphometrics_cI_all_metrics_single_neurons",
        line_color="tab:green",
    )
    plot_supplementary_all_metrics_single_class(
        dict_res=dict_res,
        class_key="class_IV",
        metrics_to_plot=supplementary_metrics,
        metric_labels=metric_labels,
        t_range=t_range,
        out_prefix="supplementary_morphometrics_cIV_all_metrics_single_neurons",
        line_color="tab:pink",
    )
    mtinh_key = "class_IV_MT_inh" if "class_IV_MT_inh" in dict_res else "class_IV_mt_inh"
    plot_supplementary_all_metrics_single_class(
        dict_res=dict_res,
        class_key=mtinh_key,
        metrics_to_plot=supplementary_metrics,
        metric_labels=metric_labels,
        t_range=t_range,
        out_prefix="supplementary_morphometrics_cIVMTinh_all_metrics_single_neurons",
        line_color="#353535",
    )

    # Class I: from 13h to 21h AEL
    start_hour_ci = 16 - minimax_frame_young * 5 / 60.0  # 13h if young covers 3h at 5 min
    start_hour_civ = 16.0
    pooled_fit_cache_path = os.path.join(result_dir, "supplementary_morphometrics_pooled_fit_cache.json")
    pooled_fit_results = compute_or_load_pooled_fit_results(
        metrics_to_plot=supplementary_metrics,
        time_ci=time_ci,
        all_ci=all_ci,
        time_civ=time_civ,
        all_civ=all_civ,
        dict_res=dict_res,
        t_range=t_range,
        cache_path=pooled_fit_cache_path,
        force_recompute=False,
    )

    ci_breakpoints = None
    for entry in pooled_fit_results:
        if entry.get("class_key") == "class_I" and entry.get("metric_key") == "semi-minor_axis":
            piece = entry.get("piecewise")
            if isinstance(piece, dict) and len(piece.get("breakpoints", [])) >= 2:
                ci_breakpoints = [
                    piece["breakpoints"][0]["estimate"],
                    piece["breakpoints"][1]["estimate"],
                ]
            break
    if ci_breakpoints is None:
        t_ci = np.concatenate([time_ci] * all_ci["semi-minor_axis"].shape[0])
        metric_ci = np.concatenate(all_ci["semi-minor_axis"])
        keep_ci = (~np.isnan(t_ci)) * (~np.isnan(metric_ci))
        t_ci = t_ci[keep_ci]
        metric_ci = metric_ci[keep_ci]
        pw_fit_ci = piecewise_regression.Fit(t_ci, metric_ci, n_breakpoints=2)
        res_ci = pw_fit_ci.get_results()
        ci_breakpoints = [
            res_ci["estimates"]["breakpoint1"]["estimate"],
            res_ci["estimates"]["breakpoint2"]["estimate"],
        ]

    plot_morphometrics_panel(
        time_ci=time_ci,
        all_ci=all_ci,
        time_civ=time_civ,
        all_civ=all_civ,
        metric_colors=metric_colors,
        metric_labels=metric_labels,
        start_hour_ci=start_hour_ci,
        start_hour_civ=start_hour_civ,
        out_prefix="WT_cI_cIV_morphometrics_panels_zero",
        force_zero=True,
        ci_breakpoints=ci_breakpoints,
    )

    plot_morphometrics_panel(
        time_ci=time_ci,
        all_ci=all_ci,
        time_civ=time_civ,
        all_civ=all_civ,
        metric_colors=metric_colors,
        metric_labels=metric_labels,
        start_hour_ci=start_hour_ci,
        start_hour_civ=start_hour_civ,
        out_prefix="WT_cI_cIV_morphometrics_panels_auto",
        force_zero=False,
        ci_breakpoints=ci_breakpoints,
    )

    build_fit_summary_table(
        metrics_to_plot=supplementary_metrics,
        time_ci=time_ci,
        all_ci=all_ci,
        time_civ=time_civ,
        all_civ=all_civ,
        dict_res=dict_res,
        t_range=t_range,
        out_prefix="supplementary_morphometrics_fit_summary",
        generate_per_neuron_tables=GENERATE_PER_NEURON_FIT_TABLES,
        pooled_fit_results=pooled_fit_results,
    )
