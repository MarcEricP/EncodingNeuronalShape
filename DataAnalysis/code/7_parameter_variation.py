import os
import json
import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib

matplotlib.use("Agg")
from dendrogenesis import project_style
import matplotlib.pyplot as plt

from dendrotree.nx_graph import nx_graph, metrics
from dendrotree.tree_structure import swc_rx

project_style.set_style()


def _compute_metrics_from_swc_file(swc_path, t_index):
    with open(swc_path, "r") as f:
        swc_str = f.read()

    g_rx = swc_rx.swc2rx(swc_str)
    g = nx_graph.rx2nx(g_rx)
    tl = metrics.total_length(g)
    res_ellipse = metrics.tree_inertia_and_equivalent_ellipse(g)
    area = float(np.pi * res_ellipse["semi_minor"] * res_ellipse["semi_major"])

    return {
        "time": int(t_index),
        "total_length": float(tl),
        "semi-minor_axis": float(res_ellipse["semi_minor"]),
        "semi-major_axis": float(res_ellipse["semi_major"]),
        "aspect_ratio": float(res_ellipse["semi_major"]) / float(res_ellipse["semi_minor"]),
        "angle_ellipse": float(res_ellipse["angle_rad"]),
        "centroid_ellipse": [float(a) for a in res_ellipse["centroid"]],
        "number_of_branches": int(metrics.n_branch_points(g)),
        "area": float(area),
        "density": float(tl / area) if area > 0 else float("nan"),
    }


def _compute_metrics_from_swc_folder(swc_dir, save_metric_path, neuron_class, param_name, param_value_key):
    identifiers = neuron_class, param_name, param_value_key

    if os.path.exists(save_metric_path):
        with open(save_metric_path, "r") as f:
            res = json.load(f)
        return identifiers, res

    frames = []
    for fname in os.listdir(swc_dir):
        if not fname.endswith(".swc"):
            continue
        stem = os.path.splitext(fname)[0]
        try:
            t_index = int(stem)
        except ValueError:
            continue
        frames.append((t_index, os.path.join(swc_dir, fname)))

    if not frames:
        return identifiers, None

    frames.sort(key=lambda x: x[0])
    res = []
    for t_index, swc_path in frames:
        res.append(_compute_metrics_from_swc_file(swc_path, t_index))

    with open(save_metric_path, "w") as f:
        json.dump(res, f)
    return identifiers, res


def _parse_param_folder(folder_name):
    if folder_name.startswith("lambda_branch_x"):
        param_name = "lambda"
        value_str = folder_name[len("lambda_branch_x"):]
    elif folder_name.startswith("kappa_x"):
        param_name = "kappa"
        value_str = folder_name[len("kappa_x"):]
    elif folder_name.startswith("sigma_x"):
        param_name = "sigma"
        value_str = folder_name[len("sigma_x"):]
    elif folder_name.startswith("alpha_"):
        param_name = "alpha"
        value_str = folder_name[len("alpha_"):]
    else:
        return None, None, None

    value_key = value_str.replace("p", ".")
    try:
        value_float = float(value_key)
    except ValueError:
        return None, None, None
    return param_name, value_key, value_float


def _savefig_all_formats(fig, outdir, base_no_ext):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(outdir, f"{base_no_ext}.{ext}"), bbox_inches="tight")


path_simus = "/home/menir/simulation_tree/simulation_result_parameters_variation_lock_to_mean/ARFIMA_zero_drift"
WORKING_DIR = os.getcwd()
RESULT_DIR = os.path.join(WORKING_DIR, "DataAnalysis", "Result", "7_parameter_variation")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
RAW_DATA_DIR = os.path.join(RESULT_DIR, "raw_data")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
save_compute = os.path.join(RAW_DATA_DIR, "save_compute.json")

basic_figsize = (project_style.mm_to_in(54), project_style.mm_to_in(50))
param_order = ["lambda", "kappa", "sigma", "alpha"]
param_labels = {
    "lambda": "Branching rate $\\mathrm{\\lambda}$\nMultiplicative factor (1 = $\\mathit{in\\ vivo}$)",
    "kappa": "Post-contact retraction $\\mathrm{\\kappa}$\nMultiplicative factor (1 = $\\mathit{in\\ vivo}$)",
    "sigma": "Increments std $\\mathrm{\\sigma}$\nMultiplicative factor (1 = $\\mathit{in\\ vivo}$)",
    "alpha": "Anomalous diffusion exponent $\\mathrm{\\alpha}$",
}
class_labels = {"class_I": "class I", "class_IV": "class IV"}
colors = {"class_I": "tab:green", "class_IV": "tab:pink"}

panel_metrics = ["total_length", "semi-minor_axis", "number_of_branches", "density"]
panel_metric_names = [
    "Total length (µm)",
    "Semi-minor axis (µm)",
    "Branching points #",
    "Density (µm⁻¹)",
]


def _sorted_param_values(param_dict):
    values = []
    for value_key in param_dict.keys():
        try:
            values.append((float(value_key), value_key))
        except ValueError:
            continue
    return [vk for _, vk in sorted(values)]


def _collect_metric_stats(dict_data, neuron_class, param_name, metric, mode):
    param_dict = dict_data.get(neuron_class, {}).get(param_name, {})
    x_vals = []
    mean_vals = []
    std_vals = []
    raw_points = []
    for value_key in _sorted_param_values(param_dict):
        sequences = param_dict.get(value_key, [])
        values = []
        for seq in sequences:
            if not seq:
                continue
            if mode == "last":
                values.append(seq[-1].get(metric, np.nan))
            else:
                first = seq[0]
                last = seq[-1]
                dt = last.get("time", np.nan) - first.get("time", np.nan)
                if not np.isfinite(dt) or dt == 0:
                    values.append(np.nan)
                else:
                    values.append((last.get(metric, np.nan) - first.get(metric, np.nan)) / dt)
        values = [v for v in values if np.isfinite(v)]
        if not values:
            continue
        x_val = float(value_key)
        x_vals.append(x_val)
        mean_vals.append(float(np.mean(values)))
        std_vals.append(float(np.std(values)))
        raw_points.append((x_val, values))
    return x_vals, mean_vals, std_vals, raw_points

if __name__ == "__main__":
    need_recompute = True
    if os.path.exists(save_compute):
        with open(save_compute, "r") as f:
            dict_data = json.load(f)
        need_recompute = False

    if need_recompute:
        dict_data = {}
        futures = []
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as ex:
            for neuron_class in ["class_I", "class_IV"]:
                dict_data[neuron_class] = {}
                class_dir = os.path.join(path_simus, neuron_class)
                if not os.path.isdir(class_dir):
                    continue
                for param_folder in os.listdir(class_dir):
                    param_name, value_key, _ = _parse_param_folder(param_folder)
                    if param_name is None:
                        continue
                    dict_data[neuron_class].setdefault(param_name, {})
                    dict_data[neuron_class][param_name].setdefault(value_key, [])

                    param_dir = os.path.join(class_dir, param_folder)
                    for movie_name in os.listdir(param_dir):
                        movie_dir = os.path.join(param_dir, movie_name)
                        for sim_number in os.listdir(movie_dir):
                            current_simu = os.path.join(movie_dir, sim_number)
                            swc_dir = os.path.join(current_simu, "swc")
                            if not os.path.exists(swc_dir):
                                continue
                            save_metric_path = os.path.join(current_simu, "metrics.json")
                            futures.append(
                                ex.submit(
                                    _compute_metrics_from_swc_folder,
                                    swc_dir,
                                    save_metric_path,
                                    neuron_class,
                                    param_name,
                                    value_key,
                                )
                            )

            pbar = tqdm.tqdm(as_completed(futures), total=len(futures), desc="loading")
            for fut in pbar:
                identifiers, res = fut.result()
                if res is None:
                    continue
                neuron_class, param_name, value_key = identifiers
                dict_data[neuron_class][param_name][value_key].append(res)

        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        with open(save_compute, "w") as f:
            json.dump(dict_data, f)

    panel_rows = len(panel_metrics)
    panel_cols = len(param_order)
    fig, axs = plt.subplots(
        panel_rows,
        panel_cols,
        figsize=(basic_figsize[0] * panel_cols, basic_figsize[1] * panel_rows),
        sharex=False,
        sharey=False,
    )
    axs = np.array(axs).reshape(panel_rows, panel_cols)
    legend_ax = None
    metric_ylim_candidates = {metric: [] for metric in panel_metrics}
    sigma_ylim_candidates = {metric: [] for metric in panel_metrics}

    for c_idx, param_name in enumerate(param_order):
        available_classes = [
            nc for nc in ["class_I", "class_IV"]
            if param_name in dict_data.get(nc, {}) and dict_data.get(nc, {}).get(param_name)
        ]
        if not available_classes:
            for r_idx in range(panel_rows):
                axs[r_idx, c_idx].axis("off")
            continue

        for r_idx, (metric, metric_name) in enumerate(zip(panel_metrics, panel_metric_names)):
            ax = axs[r_idx, c_idx]
            for neuron_class in available_classes:
                x_vals, mean_vals, std_vals, raw_points = _collect_metric_stats(
                    dict_data, neuron_class, param_name, metric, mode="last"
                )
                if not x_vals:
                    continue
                for x_val, values in raw_points:
                    ax.scatter(
                        [x_val] * len(values),
                        values,
                        color=colors[neuron_class],
                        alpha=0.35,
                        s=20,
                    )
                ax.errorbar(
                    x_vals,
                    mean_vals,
                    yerr=std_vals,
                    fmt="o-",
                    color=colors[neuron_class],
                    label=class_labels[neuron_class],
                    capsize=3,
                )
                
                metric_ylim_candidates[metric].extend(list(np.array(mean_vals) - np.array(std_vals)))
                metric_ylim_candidates[metric].extend(list(np.array(mean_vals) + np.array(std_vals)))
                for _, values in raw_points:
                    metric_ylim_candidates[metric].extend(values)
                    if param_name == "sigma":
                        sigma_ylim_candidates[metric].extend(values)
                if legend_ax is None:
                    legend_ax = ax

            ax.set_ylabel(metric_name)
            ax.set_xlabel(param_labels.get(param_name, param_name))
            ax.set_box_aspect(1)

    for r_idx, metric in enumerate(panel_metrics):
        candidates = [v for v in metric_ylim_candidates[metric] if np.isfinite(v)]
        if metric in {"total_length", "semi-minor_axis"}:
            sigma_vals = [v for v in sigma_ylim_candidates[metric] if np.isfinite(v)]
            if sigma_vals:
                # Exclude only the single most extreme sigma scatter value from y-limit computation.
                candidates_for_ref = np.array(candidates, dtype=float)
                sigma_vals_arr = np.array(sigma_vals, dtype=float)
                if candidates_for_ref.size > 0:
                    median_ref = float(np.median(candidates_for_ref))
                    idx = int(np.argmax(np.abs(sigma_vals_arr - median_ref)))
                    outlier_val = float(sigma_vals_arr[idx])
                    removed = False
                    filtered = []
                    for v in candidates:
                        if not removed and v == outlier_val:
                            removed = True
                            continue
                        filtered.append(v)
                    if filtered:
                        candidates = filtered
        if not candidates:
            continue
        y_min = float(np.min(candidates))
        y_max = float(np.max(candidates))
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            continue
        if y_min == y_max:
            pad = 0.05 * (abs(y_min) if y_min != 0 else 1.0)
            y_min -= pad
            y_max += pad
        else:
            pad = 0.05 * (y_max - y_min)
            y_min -= pad
            y_max += pad
        for c_idx in range(panel_cols):
            ax = axs[r_idx, c_idx]
            if not ax.has_data():
                continue
            ax.set_ylim(y_min, y_max)

    if legend_ax is not None and len(class_labels) > 1:
        legend_ax.legend(frameon=False)

    fig.tight_layout()
    _savefig_all_formats(fig, PLOTS_DIR, "parameter_variation_panel_last")
    plt.close(fig)
