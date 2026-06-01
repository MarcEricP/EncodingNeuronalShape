from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
import pickle
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tqdm
from dendrotree.nx_graph import nx_graph, metrics
from dendrotree.tree_structure import swc_rx
from scipy import stats

mpl.rcParams["hatch.linewidth"] = 0.3

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]


import simu_1D
from dendrogenesis import project_style

project_style.set_style()

RESULT_DIR = ROOT / "Result" / "9_SI_front_propagation"
PLOTS_DIR = RESULT_DIR / "plots"
RAW_DATA_DIR = RESULT_DIR / "raw_data"
OUT_DIR = PLOTS_DIR
CACHE_DIR = OUT_DIR / "cache"
SAVE_COMPUTE_PATH = RAW_DATA_DIR / "save_compute.json"
OVERLAY_SWC_DIR = os.path.join(os.getcwd(),
    "SimulationTree/simulation_result_exploration/ARFIMA_zero_drift_sparse_regression/class_IV/alpha_0p8__sig_x1__lam_x8/warm_start"
)
REALISTIC_SIM_ROOT = os.path.join(os.getcwd(),
    "SimulationTree/simulation_result_exploration/ARFIMA_zero_drift_sparse_regression"
)
N_ROWS = 3
N_COLS = 3
PANEL_SIZE_MM = 55.0
PANEL_SIZE = (project_style.mm_to_in(PANEL_SIZE_MM), project_style.mm_to_in(PANEL_SIZE_MM))
BASE_SCALING_PARAM_PAIRS = []
for i in range(11):
    for j in range(3):
        BASE_SCALING_PARAM_PAIRS.append((8 / 2 ** i, 0.01 * 2 ** (j + i)))

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def subdiff_ctrw_speed_prefactor(alpha):
    return((4-alpha)**((4-alpha)/(2+alpha))/(2**(2/(2+alpha)))/((2-alpha)**((2-alpha)/(2+alpha))))

def speed_subdiff_ctrw(alpha,sigma,lambd,dt):
    pref = subdiff_ctrw_speed_prefactor(alpha)
    speed = pref*(sigma/np.sqrt(2))**(4/(2+alpha))*lambd**((2-alpha)/(2+alpha))/(1)**(2*alpha/(2+alpha))
    # speed = pref*(sigma/np.sqrt(2))**(3/(2+alpha))*lambd**((2.5-alpha)/(2+alpha))/(1)**(2*alpha/(2+alpha))
    return(speed)

def fisher_tip_splitting_velocity(r, sigma, dt):
    diffusion = sigma ** 2 / 2.0
    velocity = 2.0 * np.sqrt(r * diffusion)
    return float(velocity), float(diffusion)


def side_branching_velocity(lamb, sigma, dt):
    diffusion = sigma ** 2 / 2.0
    velocity = (3.0 / (2.0 ** (2.0 / 3.0))) * (diffusion ** (2.0 / 3.0)) * (lamb ** (1.0 / 3.0))
    return float(velocity), float(diffusion)


def default_scaling_param_pairs_for_alpha(alpha):
    alpha = float(alpha)
    alpha_low = 0.2
    alpha_high = 1.0
    frac = np.clip((alpha_high - alpha) / (alpha_high - alpha_low), 0.0, 1.0)
    lambda_scale = 1.0 + 2.0 * frac
    sigma_scale = lambda_scale ** (-0.5)
    return tuple(
        (float(sigma * sigma_scale), float(lamb * lambda_scale))
        for sigma, lamb in BASE_SCALING_PARAM_PAIRS
    )


def resolve_sigma_lambda_pairs(alpha, sigma_lambda_pairs=None):
    if sigma_lambda_pairs is None:
        return default_scaling_param_pairs_for_alpha(alpha)
    if callable(sigma_lambda_pairs):
        return tuple(sigma_lambda_pairs(alpha))
    if isinstance(sigma_lambda_pairs, dict):
        if alpha in sigma_lambda_pairs:
            return tuple(sigma_lambda_pairs[alpha])
        alpha_key = f"{float(alpha):.6g}"
        if alpha_key in sigma_lambda_pairs:
            return tuple(sigma_lambda_pairs[alpha_key])
    return tuple(sigma_lambda_pairs)


def select_tip_ids_for_plot(list_tip_id, max_trajectories_to_plot):
    if max_trajectories_to_plot == 0:
        return np.array([], dtype=int)

    all_tip_ids = np.unique(np.concatenate(list_tip_id).astype(int))
    if max_trajectories_to_plot is None or max_trajectories_to_plot >= all_tip_ids.size:
        return all_tip_ids

    idx = np.linspace(
        0,
        all_tip_ids.size - 1,
        num=max_trajectories_to_plot,
        dtype=int,
    )
    return all_tip_ids[np.unique(idx)]


def build_individual_trajectories(all_ts, list_positions, list_tip_id, selected_tip_ids=None):
    if selected_tip_ids is not None:
        selected_tip_ids = set(np.asarray(selected_tip_ids, dtype=int).tolist())

    individual_trajs = {}
    for t, positions, tip_ids in zip(all_ts, list_positions, list_tip_id):
        for tip, pos in zip(tip_ids, positions):
            tip = int(tip)
            if selected_tip_ids is not None and tip not in selected_tip_ids:
                continue
            individual_trajs.setdefault(tip, []).append((float(t), float(pos)))
    return {
        tip: np.asarray(traj, dtype=float)
        for tip, traj in individual_trajs.items()
    }


def select_tip_ids_in_zoom(all_ts, list_positions, list_tip_id, zoom_xlim, zoom_ylim):
    if zoom_xlim is None or zoom_ylim is None:
        return np.array([], dtype=int)

    x_min, x_max = zoom_xlim
    y_min, y_max = zoom_ylim
    selected_tip_ids = set()

    for t, positions, tip_ids in zip(all_ts, list_positions, list_tip_id):
        if t < x_min or t > x_max:
            continue
        in_zoom = (
            (positions >= y_min) & (positions <= y_max)
        )
        if np.any(in_zoom):
            selected_tip_ids.update(np.asarray(tip_ids[in_zoom], dtype=int).tolist())

    if not selected_tip_ids:
        return np.array([], dtype=int)
    return np.array(sorted(selected_tip_ids), dtype=int)


def _freeze_for_cache(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_freeze_for_cache(v) for v in value]
    if isinstance(value, dict):
        return {k: _freeze_for_cache(v) for k, v in sorted(value.items())}
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_folder_params(folder_name):
    parts = str(folder_name).split("__")
    params = {}
    for part in parts:
        if part.startswith("alpha_"):
            value_str = part[len("alpha_"):]
            value_key = value_str.replace("p", ".")
            try:
                params["alpha"] = float(value_key)
            except ValueError:
                return None
        elif part.startswith("D_x"):
            value_str = part[len("D_x"):]
            value_key = value_str.replace("p", ".")
            try:
                params["D_factor"] = float(value_key)
            except ValueError:
                return None
        elif part.startswith("sig_x"):
            value_str = part[len("sig_x"):]
            value_key = value_str.replace("p", ".")
            try:
                params["sigma_factor"] = float(value_key)
            except ValueError:
                return None
        elif part.startswith("lam_x"):
            value_str = part[len("lam_x"):]
            value_key = value_str.replace("p", ".")
            try:
                params["lambda_factor"] = float(value_key)
            except ValueError:
                return None
        elif part.startswith("lambda_x"):
            value_str = part[len("lambda_x"):]
            value_key = value_str.replace("p", ".")
            try:
                params["lambda_factor"] = float(value_key)
            except ValueError:
                return None
    if "alpha" not in params:
        return None
    return params


def _list_param_folders(base_dir):
    param_folders = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if _parse_folder_params(entry) is not None:
            param_folders.append(entry)
    return param_folders


def _swc_centroid(swc_str):
    coords = []
    for line in swc_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
        except ValueError:
            continue
        coords.append((x, y, z))
    if not coords:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.mean(np.asarray(coords, float), axis=0)


def _front_from_swc_str(swc_str, centroid):
    max_r = np.nan
    for line in swc_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
        except ValueError:
            continue
        dx = x - centroid[0]
        dy = y - centroid[1]
        dz = z - centroid[2]
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        if not np.isfinite(max_r) or r > max_r:
            max_r = r
    return float(max_r)


def _compute_metrics_from_swc_file(swc_path, t_index, centroid):
    with open(swc_path, "r", encoding="utf-8") as f:
        swc_str = f.read()
    g_rx = swc_rx.swc2rx(swc_str)
    g = nx_graph.rx2nx(g_rx)
    tl = metrics.total_length(g)
    res_ellipse = metrics.tree_inertia_and_equivalent_ellipse(g)
    area = float(np.pi * res_ellipse["semi_minor"] * res_ellipse["semi_major"])
    front = _front_from_swc_str(swc_str, centroid)
    return {
        "time": int(t_index),
        "total_length": float(tl),
        "semi-minor_axis": float(res_ellipse["semi_minor"]),
        "semi-major_axis": float(res_ellipse["semi_major"]),
        "number_of_branches": int(metrics.n_branch_points(g)),
        "area": float(area),
        "front": float(front),
    }


def _compute_metrics_from_swc_folder(
    swc_dir,
    save_metric_path,
    neuron_class,
    alpha_key,
    sigma_value,
    lambda_value,
):
    identifiers = neuron_class, alpha_key, sigma_value, lambda_value
    if os.path.exists(save_metric_path):
        with open(save_metric_path, "r", encoding="utf-8") as f:
            return identifiers, json.load(f)

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
    with open(frames[0][1], "r", encoding="utf-8") as f:
        centroid = _swc_centroid(f.read())

    res = []
    for t_index, swc_path in frames:
        res.append(_compute_metrics_from_swc_file(swc_path, t_index, centroid))

    with open(save_metric_path, "w", encoding="utf-8") as f:
        json.dump(res, f)
    return identifiers, res


def _compute_save_compute_from_realistic_sims(sim_root=REALISTIC_SIM_ROOT, save_compute_path=SAVE_COMPUTE_PATH):
    sim_root = Path(sim_root)
    if not sim_root.exists():
        raise FileNotFoundError(f"Realistic simulation root not found: {sim_root}")

    dict_data = {}
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as ex:
        root_param_folders = _list_param_folders(str(sim_root))
        if root_param_folders:
            class_entries = [("default", str(sim_root), root_param_folders)]
        else:
            class_entries = []
            for neuron_class in os.listdir(sim_root):
                class_dir = os.path.join(sim_root, neuron_class)
                if not os.path.isdir(class_dir):
                    continue
                param_folders = _list_param_folders(class_dir)
                if not param_folders:
                    continue
                class_entries.append((neuron_class, class_dir, param_folders))

        for neuron_class, class_dir, param_folders in class_entries:
            dict_data.setdefault(neuron_class, {"combo": {}})
            for param_folder in param_folders:
                params = _parse_folder_params(param_folder)
                if params is None:
                    continue
                alpha_key = f"{params['alpha']:.6g}"
                if "D_factor" in params:
                    sigma_value = float(np.sqrt(params["D_factor"]))
                elif "sigma_factor" in params:
                    sigma_value = float(params["sigma_factor"])
                else:
                    sigma_value = 1.0
                lambda_value = float(params.get("lambda_factor", 1.0))
                param_key = f"sig_{sigma_value:.6g}__lam_{lambda_value:.6g}"
                dict_data[neuron_class]["combo"].setdefault(alpha_key, {})
                dict_data[neuron_class]["combo"][alpha_key].setdefault(
                    param_key,
                    {"sigma": sigma_value, "lambda": lambda_value, "runs": []},
                )

                param_dir = os.path.join(class_dir, param_folder)
                for movie_name in os.listdir(param_dir):
                    movie_dir = os.path.join(param_dir, movie_name)
                    if not os.path.isdir(movie_dir):
                        continue
                    for sim_number in os.listdir(movie_dir):
                        current_simu = os.path.join(movie_dir, sim_number)
                        if not os.path.isdir(current_simu):
                            continue
                        swc_dir = os.path.join(current_simu, "swc")
                        if not os.path.exists(swc_dir):
                            continue
                        save_metric_path = os.path.join(current_simu, "metrics_scaling.json")
                        futures.append(
                            ex.submit(
                                _compute_metrics_from_swc_folder,
                                swc_dir,
                                save_metric_path,
                                neuron_class,
                                alpha_key,
                                sigma_value,
                                lambda_value,
                            )
                        )

        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="building realistic cache"):
            identifiers, res = fut.result()
            if res is None:
                continue
            neuron_class, alpha_key, sigma_value, lambda_value = identifiers
            param_key = f"sig_{sigma_value:.6g}__lam_{lambda_value:.6g}"
            dict_data[neuron_class]["combo"][alpha_key][param_key]["runs"].append(res)

    save_compute_path = Path(save_compute_path)
    save_compute_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_compute_path, "w", encoding="utf-8") as f:
        json.dump(dict_data, f)
    return dict_data


def _cache_path(prefix, params):
    frozen = _freeze_for_cache(params)
    key = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{prefix}_{key}.pkl"


def load_or_run_simulation(prefix, params, run_fn, recompute=False):
    cache_path = _cache_path(prefix, params)
    if not recompute and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # Migrate legacy heavy subdiffusive cache (full trajectories)
        # to compact front-only payload in-place.
        if prefix == "subdiffusive_front" and isinstance(cached, (list, tuple)) and len(cached) >= 2:
            compact = {
                "all_ts": np.asarray(cached[0], dtype=float).tolist(),
                "front": np.asarray([np.max(pos) for pos in cached[1]], dtype=float).tolist(),
            }
            with open(cache_path, "wb") as f:
                pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
            return compact
        return cached

    result = run_fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def compute_front_velocity_from_trace(all_ts, front, t_fit_start):
    ts = np.asarray(all_ts, dtype=float)
    front = np.asarray(front, dtype=float)
    mask = ts >= float(t_fit_start)
    if np.count_nonzero(mask) < 2:
        raise ValueError("t_fit_start too large: need at least 2 time points.")
    slope, intercept = np.polyfit(ts[mask], front[mask], deg=1)
    return float(slope)


def run_scaling_exponent_sweep(
    *,
    alphas,
    sigma_lambda_pairs,
    t_max,
    dt,
    t_fit_start,
    n_repeat,
    top_n,
    seed,
    recompute,
):
    tasks = []
    for alpha in alphas:
        alpha_pairs = resolve_sigma_lambda_pairs(alpha, sigma_lambda_pairs)
        for pair_idx, (sigma, lamb) in enumerate(alpha_pairs):
            for repeat_idx in range(n_repeat):
                tasks.append((float(alpha), int(pair_idx), float(sigma), float(lamb), int(repeat_idx)))

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(len(tasks))
    run_seeds = [int(s.generate_state(1)[0]) for s in child_seeds]

    rows = []
    front_records = []
    total_tasks = len(tasks)
    for task_idx, ((alpha, pair_idx, sigma, lamb, repeat_idx), run_seed) in enumerate(zip(tasks, run_seeds), start=1):
        sim_params = {
            "simulator": "simu_arfima_front_only_v2_sigma_1min",
            "alpha": alpha,
            "sigma": sigma,
            "lamb": lamb,
            "dt": dt,
            "t_max": t_max,
            "t_fit_start": t_fit_start,
            "top_n": top_n,
            "seed": run_seed,
            "with_starting_point": False,
            "with_angle": None,
        }
        cache_path = _cache_path("scaling_velocity", sim_params)
        status = "run"
        cached_result = None
        if not recompute and cache_path.exists():
            with open(cache_path, "rb") as f:
                cached_result = pickle.load(f)
            if (
                isinstance(cached_result, dict)
                and "velocity" in cached_result
                and "front" in cached_result
                and "all_ts" in cached_result
            ):
                status = "cache"
            else:
                cached_result = None

        print(
            f"scaling exponents {task_idx}/{total_tasks}: "
            f"alpha={alpha:.1f}, sigma={sigma:g}, lambda={lamb:g}, "
            f"repeat={repeat_idx + 1}/{n_repeat} [{status}]"
        )

        def _run_one():
            np.random.seed(run_seed)
            lamb_schedule = ([0, t_max], [lamb, lamb])
            sigma_schedule = ([0, t_max], [sigma, sigma])
            alpha_schedule = ([0, t_max], [alpha, alpha])
            all_ts, front = simu_1D.simu_arfima_front_only(
                lamb_schedule,
                sigma_schedule,
                alpha_schedule,
                T_max=t_max,
                dt=dt,
                top_n=top_n,
                verbose=True,
                with_starting_point=False,
            )
            velocity = compute_front_velocity_from_trace(all_ts, front, t_fit_start)
            return {
                "velocity": float(velocity),
                "front": np.asarray(front, dtype=float).tolist(),
                "all_ts": np.asarray(all_ts, dtype=float).tolist(),
            }

        if cached_result is None:
            result = load_or_run_simulation(
                "scaling_velocity",
                sim_params,
                _run_one,
                recompute=recompute,
            )
        else:
            result = cached_result

        rows.append((alpha, sigma, lamb, float(result["velocity"])))
        front_records.append(
            {
                "alpha": float(alpha),
                "pair_idx": int(pair_idx),
                "sigma": float(sigma),
                "lambda": float(lamb),
                "repeat_idx": int(repeat_idx),
                "velocity": float(result["velocity"]),
                "all_ts": np.asarray(result["all_ts"], dtype=float),
                "front": np.asarray(result["front"], dtype=float),
            }
        )
    return rows, front_records


def save_scaling_fronts_figure(
    front_records,
    *,
    alphas,
    sigma_lambda_pairs,
    t_fit_start,
    base_name="Fig_SI_front_propagation_scaling_fronts",
):
    print("scaling fronts figure: plotting")
    n_rows = len(alphas)
    pairs_by_alpha = {
        float(alpha): tuple(resolve_sigma_lambda_pairs(alpha, sigma_lambda_pairs))
        for alpha in alphas
    }
    n_cols = max((len(pairs) for pairs in pairs_by_alpha.values()), default=1)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(project_style.mm_to_in(26) * n_cols, project_style.mm_to_in(26) * n_rows),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    axs = np.asarray(axs)
    if axs.ndim == 1:
        if n_rows == 1:
            axs = axs[None, :]
        else:
            axs = axs[:, None]

    grouped = {}
    for rec in front_records:
        key = (rec["alpha"], rec["pair_idx"])
        grouped.setdefault(key, []).append(rec)

    for row_idx, alpha in enumerate(alphas):
        alpha_pairs = pairs_by_alpha[float(alpha)]
        for col_idx in range(n_cols):
            ax = axs[row_idx, col_idx]
            if col_idx >= len(alpha_pairs):
                style_empty_panel(ax)
                continue

            sigma, lamb = alpha_pairs[col_idx]
            key = (float(alpha), int(col_idx))
            records = grouped.get(key, [])
            if not records:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            for rec in records:
                ax.plot(rec["all_ts"], rec["front"], color="0.5", alpha=0.35, lw=0.5)

            ref_ts = records[0]["all_ts"]
            fronts = np.vstack([rec["front"] for rec in records])
            front_mean = np.mean(fronts, axis=0)
            ax.plot(ref_ts, front_mean, color="black", lw=0.9)

            mask = ref_ts >= float(t_fit_start)
            if np.count_nonzero(mask) >= 2:
                slope, intercept = np.polyfit(ref_ts[mask], front_mean[mask], deg=1)
                fit_line = intercept + slope * ref_ts[mask]
                ax.plot(ref_ts[mask], fit_line, color="tab:red", ls="--", lw=0.9)
            ax.axvline(float(t_fit_start), color="tab:red", ls=":", lw=0.6)
            ax.set_title(rf"$\sigma={sigma:g}$, $\lambda={lamb:g}$")
            ax.set_box_aspect(1)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Time")
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(rf"$\alpha={alpha:.1f}$" "\nfront")
            else:
                ax.set_yticklabels([])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{base_name}.{ext}", dpi=300)
    plt.close(fig)


def add_trajectory_zoom_inset(ax, trajectories, zoom_xlim, zoom_ylim):
    if zoom_xlim is None or zoom_ylim is None or not trajectories:
        return

    axins = ax.inset_axes([0.57, 0.05, 0.38, 0.38])
    axins.set_box_aspect(1)

    for traj in trajectories:
        axins.plot(traj[:, 0], traj[:, 1], color="0.25", alpha=1.0, lw=0.5)
        axins.scatter(
            traj[0, 0],
            traj[0, 1],
            s=5,
            color="tab:orange",
            linewidths=0,
            zorder=3,
        )

    axins.set_xlim(*zoom_xlim)
    axins.set_ylim(*zoom_ylim)
    axins.set_xticks([])
    axins.set_yticks([])
    ax.indicate_inset_zoom(axins, edgecolor="0.35", alpha=0.8)


def make_panel_grid():
    fig, axs = plt.subplots(
        N_ROWS,
        N_COLS,
        figsize=(PANEL_SIZE[0] * N_COLS, PANEL_SIZE[1] * N_ROWS),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    axs = np.asarray(axs)
    for ax in axs.flat:
        ax.set_box_aspect(1)
    return fig, axs


def make_custom_panel_grid(n_rows, n_cols):
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(PANEL_SIZE[0] * n_cols, PANEL_SIZE[1] * n_rows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.asarray(axs)
    for ax in axs.flat:
        ax.set_box_aspect(1)
    return fig, axs


def style_empty_panel(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")


def add_panel_label(ax, label, x=-0.25, y=1.05):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        # fontweight="bold",
        clip_on=False,
    )


def add_identity_line_label(ax, diag_min, diag_max, text="x=y", frac=0.82, offset_factor=1.5):
    if not (np.isfinite(diag_min) and np.isfinite(diag_max) and diag_min > 0 and diag_max > diag_min):
        return

    log_x = np.log10(diag_min) + frac * (np.log10(diag_max) - np.log10(diag_min))
    x_pos = 10 ** log_x
    y_pos = x_pos * offset_factor

    p0 = ax.transData.transform((diag_min, diag_min))
    p1 = ax.transData.transform((diag_max, diag_max))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

    ax.text(
        x_pos,
        y_pos,
        text,
        rotation=angle,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        fontsize=6,
        color="0.25",
    )


def _swc_segments_for_overlay(path):
    if not path.exists():
        return []

    nodes = {}
    parents = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = s.split()
            if len(cols) < 7:
                continue
            nid = int(float(cols[0]))
            x = float(cols[2])
            y = float(cols[3])
            parent = int(float(cols[6]))
            nodes[nid] = (x, y)
            parents[nid] = parent

    if not nodes:
        return []

    roots = [n for n, p in parents.items() if p < 0 or p not in nodes]
    root = roots[0] if roots else next(iter(nodes))
    root_x, root_y = nodes[root]

    transformed = {}
    for nid, (x, y) in nodes.items():
        xx = y - root_y
        yy = x - root_x
        transformed[nid] = (xx, yy)

    segments = []
    for nid, parent in parents.items():
        if nid in transformed and parent in transformed:
            segments.append((transformed[parent], transformed[nid]))
    return segments


def _segment_length_inside_circle(p0, p1, center, radius):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    center = np.asarray(center, dtype=float)
    d = p1 - p0
    seg_len = float(np.hypot(d[0], d[1]))
    if seg_len == 0:
        return 0.0

    f = p0 - center
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(f, d))
    c = float(np.dot(f, f) - radius ** 2)
    disc = b ** 2 - 4.0 * a * c

    ts = [0.0, 1.0]
    if disc >= 0:
        sqrt_disc = float(np.sqrt(disc))
        ts.extend([(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)])

    ts = sorted(t for t in ts if 0.0 <= t <= 1.0)
    inside_length = 0.0
    for t0, t1 in zip(ts[:-1], ts[1:]):
        t_mid = 0.5 * (t0 + t1)
        p_mid = p0 + t_mid * d
        if np.sum((p_mid - center) ** 2) <= radius ** 2:
            inside_length += seg_len * (t1 - t0)
    return inside_length


def _prepare_segment_spatial_index(segments, cell_size):
    records = []
    grid = {}
    inv_cell = 1.0 / cell_size
    for idx, (p0, p1) in enumerate(segments):
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        xmin = min(x0, x1)
        xmax = max(x0, x1)
        ymin = min(y0, y1)
        ymax = max(y0, y1)
        records.append((p0, p1, xmin, xmax, ymin, ymax))
        ix0 = int(np.floor(xmin * inv_cell))
        ix1 = int(np.floor(xmax * inv_cell))
        iy0 = int(np.floor(ymin * inv_cell))
        iy1 = int(np.floor(ymax * inv_cell))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                grid.setdefault((ix, iy), []).append(idx)
    return {
        "records": records,
        "grid": grid,
        "cell_size": float(cell_size),
    }


def _candidate_segment_indices(index_data, center, radius):
    grid = index_data["grid"]
    inv_cell = 1.0 / index_data["cell_size"]
    cx, cy = float(center[0]), float(center[1])
    ix0 = int(np.floor((cx - radius) * inv_cell))
    ix1 = int(np.floor((cx + radius) * inv_cell))
    iy0 = int(np.floor((cy - radius) * inv_cell))
    iy1 = int(np.floor((cy + radius) * inv_cell))
    candidate_ids = set()
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            candidate_ids.update(grid.get((ix, iy), ()))
    return candidate_ids


def _axis_density_profile(indexed_segment_sets, axis, span, step=0.5, radius=0.5):
    coord_min, coord_max = span
    centers = np.arange(coord_min, coord_max + 0.5 * step, step, dtype=float)
    area = np.pi * radius ** 2
    densities = []
    for coord in centers:
        center = np.array([coord, 0.0], dtype=float) if axis == "x" else np.array([0.0, coord], dtype=float)
        total_length = 0.0
        for index_data in indexed_segment_sets:
            for idx in _candidate_segment_indices(index_data, center, radius):
                p0, p1, xmin, xmax, ymin, ymax = index_data["records"][idx]
                if (
                    center[0] < xmin - radius
                    or center[0] > xmax + radius
                    or center[1] < ymin - radius
                    or center[1] > ymax + radius
                ):
                    continue
                total_length += _segment_length_inside_circle(p0, p1, center, radius)
        densities.append(total_length / max(len(indexed_segment_sets), 1) / area)
    return centers, np.asarray(densities, dtype=float)


def _compute_overlay_extent(swc_root, frames, recompute=False):
    cache_path = _cache_path(
        "swc_overlay_extent",
        {"swc_root": str(swc_root), "frames": list(frames)},
    )
    if not recompute and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    all_x = []
    all_y = []
    for frame in tqdm.tqdm(frames, desc="SWC extent", leave=False):
        for swc_path in sorted(Path(swc_root).glob(f"*/swc/{frame}.swc")):
            segments = _swc_segments_for_overlay(swc_path)
            for (x0, y0), (x1, y1) in segments:
                all_x.extend((x0, x1))
                all_y.extend((y0, y1))

    if not all_x or not all_y:
        return None

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    pad_x = 0.03 * max(x_max - x_min, 1.0)
    pad_y = 0.03 * max(y_max - y_min, 1.0)
    extent = {
        "xlim": (x_min - pad_x, x_max + pad_x),
        "ylim": (y_min - pad_y, y_max + pad_y),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(extent, f, protocol=pickle.HIGHEST_PROTOCOL)
    return extent


def _load_or_compute_overlay_frame_payload(
    *,
    swc_root,
    frame,
    xlim,
    ylim,
    color,
    alpha,
    step,
    radius,
    raster_dpi,
    recompute=False,
):
    cache_path = _cache_path(
        "swc_overlay_frame_payload",
        {
            "swc_root": str(swc_root),
            "frame": int(frame),
            "xlim": tuple(float(v) for v in xlim),
            "ylim": tuple(float(v) for v in ylim),
            "color": tuple(float(v) for v in color),
            "alpha": float(alpha),
            "step": float(step),
            "radius": float(radius),
            "raster_dpi": int(raster_dpi),
            "density_method": "grid_bbox_v1",
            "image_format": "float_rgba_v1",
        },
    )
    if not recompute and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    swc_paths = sorted(Path(swc_root).glob(f"*/swc/{frame}.swc"))
    segment_sets = []
    for swc_path in tqdm.tqdm(swc_paths, desc=f"frame {frame}: read SWC", leave=False):
        segments = _swc_segments_for_overlay(swc_path)
        if segments:
            segment_sets.append(segments)

    if not segment_sets:
        payload = None
    else:
        cell_size = max(2.0 * radius, step)
        indexed_segment_sets = [
            _prepare_segment_spatial_index(segments, cell_size=cell_size)
            for segments in tqdm.tqdm(segment_sets, desc=f"frame {frame}: build index", leave=False)
        ]
        x_centers, x_density = _axis_density_profile(
            indexed_segment_sets,
            axis="x",
            span=xlim,
            step=step,
            radius=radius,
        )
        y_centers, y_density = _axis_density_profile(
            indexed_segment_sets,
            axis="y",
            span=ylim,
            step=step,
            radius=radius,
        )

        raster_fig = plt.Figure(figsize=PANEL_SIZE, dpi=raster_dpi, facecolor=(1, 1, 1, 0))
        FigureCanvasAgg(raster_fig)
        raster_ax = raster_fig.add_axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
        for segments in tqdm.tqdm(segment_sets, desc=f"frame {frame}: rasterize", leave=False):
            for (x0, y0), (x1, y1) in segments:
                raster_ax.plot([x0, x1], [y0, y1], color=color, lw=0.8, alpha=alpha)
        raster_ax.set_xlim(*xlim)
        raster_ax.set_ylim(*ylim)
        raster_ax.set_aspect("equal", adjustable="box")
        raster_ax.axis("off")
        raster_fig.patch.set_alpha(0.0)
        raster_ax.patch.set_alpha(0.0)
        raster_fig.canvas.draw()
        width, height = raster_fig.canvas.get_width_height()
        image = np.frombuffer(raster_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        image = np.asarray(image, dtype=np.float32) / 255.0
        plt.close(raster_fig)

        payload = {
            "image": image,
            "x_centers": x_centers,
            "x_density": x_density,
            "y_centers": y_centers,
            "y_density": y_density,
        }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def plot_overlay_swc_panel(
    ax,
    swc_root=OVERLAY_SWC_DIR,
    frames=(100, 200, 300),
    alpha=0.3,
    step=0.5,
    radius=3,
    raster_dpi=700,
    recompute=False,
    density_axis_fraction=0.28,
    hide_density_axis_decoration=True,
):
    extent = _compute_overlay_extent(swc_root, frames, recompute=recompute)
    if extent is None:
        ax.text(0.5, 0.5, "no SWC found", ha="center", va="center", transform=ax.transAxes)
        style_empty_panel(ax)
        return

    xlim = tuple(extent["xlim"])
    ylim = tuple(extent["ylim"])
    colors = plt.cm.magma(np.linspace(0.2, 0.85, len(frames)))
    payloads = []
    for frame, color in tqdm.tqdm(list(zip(frames, colors)), desc="SWC overlay frames"):
        payload = _load_or_compute_overlay_frame_payload(
            swc_root=swc_root,
            frame=frame,
            xlim=xlim,
            ylim=ylim,
            color=color,
            alpha=alpha,
            step=step,
            radius=radius,
            raster_dpi=raster_dpi,
            recompute=recompute,
        )
        if payload is not None:
            payloads.append((frame, color, payload))

    if not payloads:
        ax.text(0.5, 0.5, "empty SWC", ha="center", va="center", transform=ax.transAxes)
        style_empty_panel(ax)
        return

    for zorder, (_frame, _color, payload) in enumerate(sorted(payloads, key=lambda item: item[0], reverse=True), start=1):
        ax.imshow(
            payload["image"],
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="upper",
            zorder=zorder,
        )

    ax.set_title("")
    ax.set_xlabel(r"x (µm)")
    ax.set_ylabel(r"y (µm)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    x_span = max(xlim[1] - xlim[0], 1e-9)
    y_span = max(ylim[1] - ylim[0], 1e-9)
    ax.set_aspect("auto")
    ax.set_box_aspect(y_span / x_span)
    ax.axhline(0.0, color="0.25", ls="--",alpha = 0.7)
    ax.axvline(0.0, color="0.25", ls="--",alpha = 0.7)

    ax_top = ax.twinx()
    ax_top.set_xlim(*xlim)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_top.tick_params(axis="y")
    ax_top.set_ylabel(r"density (µm$^{-1}$)")

    ax_right = ax.twiny()
    ax_right.set_ylim(*ylim)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.tick_params(axis="x")
    ax_right.set_xlabel(r"density (µm$^{-1}$)")

    max_x_density = 2.5
    max_y_density = 2.5
    for frame, color, payload in payloads:
        ax_top.plot(payload["x_centers"], payload["x_density"], color=color, label=f"{frame} min", alpha=0.7)
        ax_right.plot(payload["y_density"], payload["y_centers"], color=color, alpha=0.7)
        max_x_density = max(max_x_density, float(np.max(payload["x_density"])))
        max_y_density = max(max_y_density, float(np.max(payload["y_density"])))

    density_axis_fraction = float(np.clip(density_axis_fraction, 0.05, 1.0))
    x_density_limit = (max_x_density * 1.05 / density_axis_fraction) if max_x_density > 0 else 1.0
    y_density_limit = (max_y_density * 1.05 / density_axis_fraction) if max_y_density > 0 else 1.0
    ax_top.set_ylim(0.0, x_density_limit)
    ax_right.set_xlim(0.0, y_density_limit)
    arrow_x = 23.0
    arrow_y = 0.2 * max_x_density
    arrow_dx = 10.0
    ax_top.annotate(
        "",
        xy=(arrow_x + arrow_dx, arrow_y),
        xytext=(arrow_x, arrow_y),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        annotation_clip=False,
    )
    ax_top.text(
        arrow_x + 0.4 * arrow_dx,
        arrow_y + 0.01 * x_density_limit,
        "c",
        color="black",
        ha="center",
        va="bottom",
    )
    if hide_density_axis_decoration:
        ax_top.set_ylabel("")
        ax_right.set_xlabel("")
        ax_top.set_yticks([])
        ax_right.set_xticks([])
    ax_top.legend(
        loc="upper right",
        bbox_to_anchor=(1, 1),
        borderaxespad=0.0,
        frameon=False,
        handlelength=1.0,
        fontsize=5,
    )


def plot_tip_splitting_panel(
    ax,
    *,
    r=0.03,
    sigma=1.0,
    alpha=1.0,
    dt=0.1,
    t_max=1000,
    length_start=0,#20.0,
    top_n=5000,
    seed=501,
    max_trajectories_to_plot=5000,
    zoom_xlim=(180, 190),
    zoom_ylim=(90, 10),
    recompute=False,
):
    panel_t0 = time.perf_counter()
    np.random.seed(seed)

    r_schedule = ([0, t_max], [r, r])
    sigma_schedule = ([0, t_max], [sigma, sigma])
    alpha_schedule = ([0, t_max], [alpha, alpha])

    sim_params = {
        "simulator": "simu_tip_splitting_v2_sigma_1min",
        "r": r,
        "sigma": sigma,
        "alpha": alpha,
        "dt": dt,
        "t_max": t_max,
        "length_start": length_start,
        "top_n": top_n,
        "seed": seed,
    }
    sim_t0 = time.perf_counter()
    all_ts, list_positions, list_tip_id = load_or_run_simulation(
        "tip_splitting",
        sim_params,
        lambda: simu_1D.simu_tip_splitting(
            r_schedule,
            sigma_schedule,
            alpha_schedule,
            T_max=t_max,
            dt=dt,
            top_n=top_n,
            verbose=True,
            length_start=length_start,
        ),
        recompute=recompute,
    )
    print(f"tip splitting simulation: {time.perf_counter() - sim_t0:.2f}s")

    front = np.asarray([np.max(pos) for pos in list_positions], dtype=float)
    c_min, diffusion = fisher_tip_splitting_velocity(r=r, sigma=sigma, dt=dt)
    min_t_theory = 80
    theory_front = length_start + c_min * (all_ts - min_t_theory)
    
    plot_t0 = time.perf_counter()
    trajectories_to_plot = []
    inset_trajectories = []
    if max_trajectories_to_plot != 0:
        selected_tip_ids = select_tip_ids_for_plot(
            list_tip_id,
            max_trajectories_to_plot=max_trajectories_to_plot,
        )
        individual_trajs = build_individual_trajectories(
            all_ts,
            list_positions,
            list_tip_id,
            selected_tip_ids=selected_tip_ids,
        )
        trajectories_to_plot = list(individual_trajs.values())
        for traj in trajectories_to_plot:
            ax.plot(traj[:, 0], traj[:, 1], color="0.25", alpha=0.08, lw=0.45)
    if zoom_xlim is not None and zoom_ylim is not None:
        selected_tip_ids_in_zoom = select_tip_ids_in_zoom(
            all_ts,
            list_positions,
            list_tip_id,
            zoom_xlim,
            zoom_ylim,
        )
        inset_trajectories = list(
            build_individual_trajectories(
                all_ts,
                list_positions,
                list_tip_id,
                selected_tip_ids=selected_tip_ids_in_zoom,
            ).values()
        )

    ax.plot(all_ts, front, color="black", lw=1.0, label="Front")
    ax.plot(
        all_ts[all_ts > min_t_theory],
        theory_front[all_ts > min_t_theory],
        color="tab:red",
        lw=1.2,
        ls="--",
        label=rf"$c^* = 2\sqrt{{rD}}$",# = {c_min:.2f}$",
    )

    ax.set_title("Tip Splitting")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Branch tip position (µm)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.8)
    add_trajectory_zoom_inset(ax, inset_trajectories, zoom_xlim, zoom_ylim)
    print(f"tip splitting plotting: {time.perf_counter() - plot_t0:.2f}s")
    print(f"tip splitting total: {time.perf_counter() - panel_t0:.2f}s")
    # ax.text(
    #     0.03,
    #     0.03,
    #     rf"$r={r:.3f}$, $\sigma={sigma:.2f}$, $D=\sigma^2/(2\Delta t)={diffusion:.2f}$",
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="bottom",
    # )


def plot_tip_splitting_example_panel(
    ax,
    *,
    r=0.077,
    sigma=1.0,
    alpha=1.0,
    dt=0.1,
    t_max=30,
    length_start=0.,
    top_n=None,
    seed_start=20001,
    max_seed_tries=200,
):
    selected_result = None

    for seed in range(seed_start, seed_start + max_seed_tries):
        np.random.seed(seed)
        all_ts, list_positions, list_tip_id = simu_1D.simu_tip_splitting(
            ([0, t_max], [r, r]),
            ([0, t_max], [sigma, sigma]),
            ([0, t_max], [alpha, alpha]),
            T_max=t_max,
            dt=dt,
            top_n=top_n,
            verbose=False,
            with_starting_point=False,
            length_start=length_start,
            take_abs_positions=False,
        )
        tip0_traj = build_individual_trajectories(
            all_ts,
            list_positions,
            list_tip_id,
            selected_tip_ids=np.array([0]),
        ).get(0)
        if tip0_traj is not None and tip0_traj.size > 0 and np.all(tip0_traj[:, 1] > 0):
            selected_result = (seed, all_ts, list_positions, list_tip_id)
            break

    if selected_result is None:
        raise RuntimeError("Could not find a tip-splitting example where the first trajectory stays positive.")

    seed, all_ts, list_positions, list_tip_id = selected_result
    trajectories = build_individual_trajectories(all_ts, list_positions, list_tip_id)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (_tip_id, traj) in enumerate(trajectories.items()):
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            lw=0.6,
            alpha=0.6,
            zorder=2,
        )
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            s=5,
            color=color,
            alpha=0.9,
            linewidths=0,
            zorder=3,
        )

    ax.set_title("Tip Splitting")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Particle position (µm)")
    ax.set_xlim(float(np.min(all_ts)), float(np.max(all_ts)))
    ymax = max(float(np.max(pos)) for pos in list_positions if len(pos) > 0)
    ax.set_ylim(0, ymax * 1.05)


def plot_side_branching_example_panel(
    ax,
    *,
    lamb=0.03,
    sigma=1.0,
    alpha=1.0,
    dt=0.1,
    t_max=30,
    initial_length=0,
    top_n=None,
    seed_start=21,
    max_seed_tries=200,
):
    selected_result = None
    for seed in range(seed_start, seed_start + max_seed_tries):
        np.random.seed(seed)
        all_ts, list_positions, list_tip_id = simu_1D.simu_arfima(
            ([0, t_max], [lamb, lamb]),
            ([0, t_max], [sigma, sigma]),
            ([0, t_max], [alpha, alpha]),
            T_max=t_max,
            dt=dt,
            top_n=top_n,
            with_angle=None,
            verbose=False,
            with_starting_point=False,
            initial_length=initial_length,
            take_abs_positions=False,
            branching_uses_abs_positions=True,
        )
        tip0_traj = build_individual_trajectories(
            all_ts,
            list_positions,
            list_tip_id,
            selected_tip_ids=np.array([0]),
        ).get(0)
        if tip0_traj is not None and tip0_traj.size > 0 and np.all(tip0_traj[:, 1] >= 0):
            selected_result = (all_ts, list_positions, list_tip_id)
            break

    if selected_result is None:
        raise RuntimeError("Could not find a side-branching example where the first trajectory stays nonnegative.")

    all_ts, list_positions, list_tip_id = selected_result

    trajectories = build_individual_trajectories(all_ts, list_positions, list_tip_id)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (_tip_id, traj) in enumerate(trajectories.items()):
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            lw=0.6,
            alpha=0.6,
            zorder=2,
        )
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            s=5,
            color=color,
            alpha=0.9,
            linewidths=0,
            zorder=3,
        )

    ax.set_title("Side Branching")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Branch tip position (µm)")
    ax.set_xlim(float(np.min(all_ts)), float(np.max(all_ts)))
    ymax = max(float(np.max(pos)) for pos in list_positions if len(pos) > 0)
    ax.set_ylim(0, ymax * 1.05)


def plot_side_branching_panel(
    ax,
    *,
    lamb=0.003,
    sigma=1.0,
    alpha=1.0,
    dt=0.1,
    t_max=1000,
    length_start=0,
    top_n=5000,
    seed=502,
    max_trajectories_to_plot=5000,
    zoom_xlim=(200, 260),
    zoom_ylim=(0, 35),
    recompute=False,
):
    panel_t0 = time.perf_counter()
    np.random.seed(seed)

    lamb_schedule = ([0, t_max], [lamb, lamb])
    sigma_schedule = ([0, t_max], [sigma, sigma])
    alpha_schedule = ([0, t_max], [alpha, alpha])

    sim_params = {
        "simulator": "simu_arfima_v2_sigma_1min",
        "lamb": lamb,
        "sigma": sigma,
        "alpha": alpha,
        "dt": dt,
        "t_max": t_max,
        "length_start": length_start,
        "top_n": top_n,
        "seed": seed,
    }
    sim_t0 = time.perf_counter()
    all_ts, list_positions, list_tip_id = load_or_run_simulation(
        "side_branching",
        sim_params,
        lambda: simu_1D.simu_arfima(
            lamb_schedule,
            sigma_schedule,
            alpha_schedule,
            T_max=t_max,
            dt=dt,
            top_n=top_n,
            verbose=True,
            with_angle=None,
            with_starting_point=False,
        ),
        recompute=recompute,
    )
    print(f"side branching simulation: {time.perf_counter() - sim_t0:.2f}s")

    front = np.asarray([np.max(pos) for pos in list_positions], dtype=float)
    c_theory = side_branching_velocity(lamb=lamb, sigma=sigma, dt=dt)[0]
    min_t_theory = 200
    theory_front = length_start + c_theory * (all_ts - min_t_theory)

    plot_t0 = time.perf_counter()
    trajectories_to_plot = []
    inset_trajectories = []
    if max_trajectories_to_plot != 0:
        selected_tip_ids = select_tip_ids_for_plot(
            list_tip_id,
            max_trajectories_to_plot=max_trajectories_to_plot,
        )
        individual_trajs = build_individual_trajectories(
            all_ts,
            list_positions,
            list_tip_id,
            selected_tip_ids=selected_tip_ids,
        )
        trajectories_to_plot = list(individual_trajs.values())
        for traj in trajectories_to_plot:
            ax.plot(traj[:, 0], traj[:, 1], color="0.25", alpha=0.08, lw=0.45)
    if zoom_xlim is not None and zoom_ylim is not None:
        selected_tip_ids_in_zoom = select_tip_ids_in_zoom(
            all_ts,
            list_positions,
            list_tip_id,
            zoom_xlim,
            zoom_ylim,
        )
        inset_trajectories = list(
            build_individual_trajectories(
                all_ts,
                list_positions,
                list_tip_id,
                selected_tip_ids=selected_tip_ids_in_zoom,
            ).values()
        )

    ax.plot(all_ts, front, color="black", lw=1.0, label="simulated front")
    ax.plot(
        all_ts[all_ts > min_t_theory],
        theory_front[all_ts > min_t_theory],
        color="tab:red",
        lw=1.2,
        ls="--",
        label=rf"$c^* = \frac{{3}}{{2^{{2/3}}}} D^{{2/3}} \lambda^{{1/3}}$",
    )

    ax.set_title("Side Branching")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Branch tip position (µm)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.8)
    add_trajectory_zoom_inset(ax, inset_trajectories, zoom_xlim, zoom_ylim)
    print(f"side branching plotting: {time.perf_counter() - plot_t0:.2f}s")
    print(f"side branching total: {time.perf_counter() - panel_t0:.2f}s")
    # ax.text(
    #     0.03,
    #     0.03,
    #     rf"$\lambda={lamb:.3f}$, $\sigma={sigma:.2f}$, $D=\sigma^2/(2\Delta t)={diffusion:.2f}$",
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="bottom",
    # )


def plot_subdiffusive_front_panel(
    ax,
    *,
    lamb=0.003,
    sigma=1.0,
    dt=0.1,
    t_max=1000,
    top_n=5000,
    seed=503,
    alphas=(0.2, 0.4, 0.6, 0.8, 1.0),
    recompute=False,
):
    panel_t0 = time.perf_counter()
    print("subdiffusive fronts: starting")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))
    for alpha, color in zip(alphas, colors):
        np.random.seed(seed)
        lamb_schedule = ([0, t_max], [lamb, lamb])
        sigma_schedule = ([0, t_max], [sigma, sigma])
        alpha_schedule = ([0, t_max], [alpha, alpha])

        sim_params = {
            "simulator": "simu_arfima_v2_sigma_1min",
            "lamb": lamb,
            "sigma": sigma,
            "alpha": alpha,
            "dt": dt,
            "t_max": t_max,
            "top_n": top_n,
            "seed": seed,
        }
        sim_t0 = time.perf_counter()
        cached = load_or_run_simulation(
            "subdiffusive_front",
            sim_params,
            lambda: (
                lambda out: {
                    "all_ts": np.asarray(out[0], dtype=float).tolist(),
                    "front": np.asarray([np.max(pos) for pos in out[1]], dtype=float).tolist(),
                }
            )(
                simu_1D.simu_arfima(
                    lamb_schedule,
                    sigma_schedule,
                    alpha_schedule,
                    T_max=t_max,
                    dt=dt,
                    top_n=top_n,
                    verbose=True,
                    with_angle=None,
                    with_starting_point=False,
                )
            ),
            recompute=recompute,
        )
        print(f"subdiffusive alpha={alpha:.1f} simulation: {time.perf_counter() - sim_t0:.2f}s")

        # Backward compatibility with old cache format (tuple with full trajectories).
        if isinstance(cached, (list, tuple)) and len(cached) >= 2:
            all_ts = np.asarray(cached[0], dtype=float)
            front = np.asarray([np.max(pos) for pos in cached[1]], dtype=float)
        else:
            all_ts = np.asarray(cached.get("all_ts", []), dtype=float)
            front = np.asarray(cached.get("front", []), dtype=float)
        ax.plot(all_ts, front, color=color, lw=1.0, label=rf"$\alpha={alpha:.1f}$",alpha = 0.5)
        temp_c = speed_subdiff_ctrw(alpha, sigma, lamb, dt)
        anchor_t = float(t_max) - 100.0
        tail_mask = (all_ts >= float(t_max) - 200.0) & (all_ts <= float(t_max))
        if not np.any(tail_mask):
            tail_mask = all_ts >= max(0.0, float(t_max) - 200.0)
        anchor_front = float(np.mean(front[tail_mask]))
        theory_front = anchor_front + temp_c * (all_ts - anchor_t)
        ax.plot(
            all_ts[theory_front>0],
            theory_front[theory_front>0],
            color=color,
            lw=1.0,
            ls="--",
            # label=rf"$\alpha={alpha:.1f}$ theory",
        )
    ax.set_title("Subdiffusive Fronts")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Front position (µm)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.8)
    print(f"subdiffusive fronts total: {time.perf_counter() - panel_t0:.2f}s")


def plot_scaling_exponent_panel(
    ax,
    *,
    alphas=(0.2, 0.4, 0.6, 0.8, 1.0),
    sigma_lambda_pairs=None,
    t_max=300,
    dt=0.1,
    t_fit_start=100.0,
    n_repeat=5,
    top_n=500,
    seed=504,
    recompute=False,
    randomize_scatter_zorder=False,
    random_zorder_seed=0,
):
    panel_t0 = time.perf_counter()
    print("speed comparison: starting")

    rows, front_records = run_scaling_exponent_sweep(
        alphas=alphas,
        sigma_lambda_pairs=sigma_lambda_pairs,
        t_max=t_max,
        dt=dt,
        t_fit_start=t_fit_start,
        n_repeat=n_repeat,
        top_n=top_n,
        seed=seed,
        recompute=recompute,
    )
    save_scaling_fronts_figure(
        front_records,
        alphas=alphas,
        sigma_lambda_pairs=sigma_lambda_pairs,
        t_fit_start=t_fit_start,
    )

    if len(rows) == 0:
        ax.text(0.5, 0.5, "no speed data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Speed Comparison")
        ax.set_xlabel(r"$\mathrm{c}_{\mathrm{theory}}$")
        ax.set_ylabel(r"$\mathrm{c}_{\mathrm{simu}}$ 1D")
        print(f"speed comparison total: {time.perf_counter() - panel_t0:.2f}s")
        return

    rows_arr = np.asarray(rows, dtype=float)
    sim_speeds = rows_arr[:, 3]
    theory_speeds = np.asarray(
        [speed_subdiff_ctrw(alpha, sigma, lamb, dt) for alpha, sigma, lamb, _ in rows],
        dtype=float,
    )
    alpha_values = rows_arr[:, 0]

    valid = (
        np.isfinite(sim_speeds)
        & np.isfinite(theory_speeds)
        & (sim_speeds > 0)
        & (theory_speeds > 0)
    )
    if np.count_nonzero(valid) == 0:
        ax.text(0.5, 0.5, "no valid speed data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Speed Comparison")
        ax.set_xlabel(r"$\mathrm{c}_{\mathrm{theory}}$ (µm/min)")
        ax.set_ylabel(r"$\mathrm{c}_{\mathrm{simu}}$ 1D (µm/min)")
        print(f"speed comparison total: {time.perf_counter() - panel_t0:.2f}s")
        return

    sim_speeds = sim_speeds[valid]
    theory_speeds = theory_speeds[valid]
    alpha_values = alpha_values[valid]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))
    alpha_color_map = {float(alpha): color for alpha, color in zip(alphas, colors)}
    if randomize_scatter_zorder:
        rng = np.random.default_rng(int(random_zorder_seed))
        draw_order = rng.permutation(theory_speeds.size)
        for idx in draw_order:
            alpha_val = float(alpha_values[idx])
            color = alpha_color_map.get(alpha_val, "0.3")
            ax.scatter(
                theory_speeds[idx],
                sim_speeds[idx],
                s=14,
                color=color,
                alpha=0.8,
                linewidths=0,
            )
    else:
        for alpha, color in zip(alphas, colors):
            mask = np.isclose(alpha_values, float(alpha))
            if np.any(mask):
                ax.scatter(
                    theory_speeds[mask],
                    sim_speeds[mask],
                    s=14,
                    color=color,
                    alpha=0.8,
                    label=rf"$\alpha={float(alpha):.1f}$",
                )

    diag_min = float(min(np.min(sim_speeds), np.min(theory_speeds)))
    diag_max = float(max(np.max(sim_speeds), np.max(theory_speeds)))
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color="0.3", ls="--", lw=0.9)
    ax.set_title("Speed Comparison")
    ax.set_xlabel(r"$\mathrm{c}_{\mathrm{theory}}$ (µm/min)")
    ax.set_ylabel(r"$\mathrm{c}_{\mathrm{simu}}$ 1D (µm/min)")
    if randomize_scatter_zorder:
        legend_handles = []
        for alpha in alphas:
            alpha_f = float(alpha)
            if np.any(np.isclose(alpha_values, alpha_f)):
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markerfacecolor=alpha_color_map[alpha_f],
                        markeredgewidth=0,
                        markersize=4.5,
                        alpha=0.8,
                        label=rf"$\alpha={alpha_f:.1f}$",
                    )
                )
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", frameon=False, handlelength=1.4)
    else:
        ax.legend(loc="upper left", frameon=False, handlelength=1.4)
    ax.set_yscale("log")
    ax.set_xscale("log")
    add_identity_line_label(ax, diag_min, diag_max)
    print(f"speed comparison total: {time.perf_counter() - panel_t0:.2f}s")


def sensitivity_ratio_abs(alpha, P):
    """
    Absolute ratio of relative sensitivities:
        |S_alpha / S_sigma|
    where P = sigma * lambda * dt and
        A(alpha) = ln(2 - alpha) - 3/2 ln(4 - alpha) + ln 2.
    """
    alpha = np.asarray(alpha, dtype=float)
    P = np.asarray(P, dtype=float)

    A_alpha = np.log(2.0 - alpha) - 1.5 * np.log(4.0 - alpha) + np.log(2.0)
    return np.abs((A_alpha - np.log(P)) / (2.0 + alpha))


def load_save_compute(results_path=SAVE_COMPUTE_PATH):
    results_path = Path(results_path)
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return _compute_save_compute_from_realistic_sims(
        sim_root=REALISTIC_SIM_ROOT,
        save_compute_path=results_path,
    )


def _growth_rate(sequence, metric, t_fit_start=None):
    if not sequence:
        return np.nan

    ts = []
    ys = []
    for row in sequence:
        t = row.get("time", np.nan)
        y = row.get(metric, np.nan)
        if not (np.isfinite(t) and np.isfinite(y)):
            continue
        if t_fit_start is not None and t < float(t_fit_start):
            continue
        ts.append(float(t))
        ys.append(float(y))

    if len(ts) < 2:
        return np.nan

    # Requested rule: regress all points if fewer than 1000, else last 1000 points.
    if len(ts) > 1000:
        x = np.asarray(ts[-1000:], dtype=float)
        y = np.asarray(ys[-1000:], dtype=float)
    else:
        x = np.asarray(ts, dtype=float)
        y = np.asarray(ys, dtype=float)

    if x.size < 2:
        return np.nan
    slope, _intercept = np.polyfit(x, y, deg=1)
    return float(slope)


def _front_displacement(sequence):
    if not sequence:
        return np.nan
    front_vals = []
    for row in sequence:
        y = row.get("front", np.nan)
        if np.isfinite(y):
            front_vals.append(float(y))
    if len(front_vals) < 2:
        return np.nan
    return float(front_vals[-1] - front_vals[0])


def _build_results_for_metric(
    class_data,
    metric,
    t_fit_start_by_alpha=None,
    default_t_fit_start=0,
    min_front_displacement_um=None,
    exclude_lambda_x64=False,
):
    results = {"combo": {}}
    combo_data = class_data.get("combo", {})
    for alpha_key, alpha_dict in combo_data.items():
        try:
            alpha_val = float(alpha_key)
        except ValueError:
            continue
        alpha_key_fmt = f"{alpha_val:.6g}"
        t_fit_start = default_t_fit_start
        if isinstance(t_fit_start_by_alpha, dict):
            if alpha_val in t_fit_start_by_alpha:
                t_fit_start = t_fit_start_by_alpha[alpha_val]
            elif alpha_key_fmt in t_fit_start_by_alpha:
                t_fit_start = t_fit_start_by_alpha[alpha_key_fmt]
            elif "default" in t_fit_start_by_alpha:
                t_fit_start = t_fit_start_by_alpha["default"]
        results["combo"].setdefault(alpha_key_fmt, {})
        for param_key, entry in alpha_dict.items():
            param_key_str = str(param_key)
            if exclude_lambda_x64 and (("lam_x64" in param_key_str) or ("lam_64" in param_key_str)):
                continue
            sigma_val = float(entry.get("sigma", np.nan))
            lambda_val = float(entry.get("lambda", np.nan))#/2000
            sequences = entry.get("runs", [])
            values = []
            for seq in sequences:
                if min_front_displacement_um is not None and metric == "front":
                    displacement = _front_displacement(seq)
                    if (not np.isfinite(displacement)) or (displacement < float(min_front_displacement_um)):
                        continue
                val = _growth_rate(seq, metric, t_fit_start=t_fit_start)
                if np.isfinite(val):
                    values.append(float(val))
            if not values:
                continue
            param_key_fmt = f"sig_{sigma_val:.6g}__lam_{lambda_val:.6g}"
            results["combo"][alpha_key_fmt][param_key_fmt] = {
                "sigma": sigma_val,
                "lambda": lambda_val,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
    return results


def _fit_global_interaction_from_rows(arr, confidence=0.95):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] != 4:
        return None
    alpha = arr[:, 0]
    sigma = arr[:, 1]
    lambd = arr[:, 2]
    y = np.log(arr[:, 3])
    log_sigma = np.log(sigma)
    log_lambda = np.log(lambd)

    X = np.column_stack(
        [
            np.ones(alpha.size, dtype=float),
            log_sigma,
            log_lambda,
            alpha,
            alpha * log_sigma,
            alpha * log_lambda,
        ]
    )
    coef_names = ["intercept", "log_sigma", "log_lambda", "alpha", "alpha_log_sigma", "alpha_log_lambda"]
    n_obs = int(X.shape[0])
    n_params = int(X.shape[1])
    if n_obs <= n_params:
        return None

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    dof = n_obs - n_params
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - rss / tss) if tss > 0 else np.nan
    xtx_inv = np.linalg.pinv(X.T @ X)
    sigma2 = rss / dof if dof > 0 else np.nan
    cov = xtx_inv * sigma2 if np.isfinite(sigma2) else np.full((n_params, n_params), np.nan)
    stderr = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    tvals = beta / stderr

    alpha_tail = 1.0 - confidence
    if stats is not None and dof > 0:
        tcrit = float(stats.t.ppf(1.0 - alpha_tail / 2.0, df=dof))
        pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df=dof))
    else:
        pvals = np.full_like(beta, np.nan, dtype=float)
    ci_low = beta - 1.96 * stderr
    ci_high = beta + 1.96 * stderr

    return {
        "coef_names": coef_names,
        "beta": beta,
        "stderr": stderr,
        "tvals": tvals,
        "pvals": pvals,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cov": cov,
        "n_obs": n_obs,
        "n_params": n_params,
        "dof": dof,
        "rss": rss,
        "r2": r2,
        "y": y,
        "y_hat": y_hat,
        "alpha": alpha,
    }


def _fit_global_interaction_model(results, confidence=0.95, n_bootstrap=200, rng_seed=0):
    rows = []
    for alpha_key, combo_block in results.get("combo", {}).items():
        try:
            alpha_val = float(alpha_key)
        except ValueError:
            continue
        for entry in combo_block.values():
            sigma_val = float(entry.get("sigma", np.nan))
            lambda_val = float(entry.get("lambda", np.nan))
            if not (np.isfinite(sigma_val) and np.isfinite(lambda_val)):
                continue
            if sigma_val <= 0 or lambda_val <= 0:
                continue
            for val in entry.get("values", []):
                v = float(val)
                if np.isfinite(v) and v > 0:
                    rows.append((alpha_val, sigma_val, lambda_val, v))
    if not rows:
        return None

    arr = np.asarray(rows, dtype=float)
    fit = _fit_global_interaction_from_rows(arr, confidence=confidence)
    if fit is None:
        return None

    boot_beta = []
    rng = np.random.default_rng(rng_seed)
    n_obs = arr.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_obs, size=n_obs)
        fit_boot = _fit_global_interaction_from_rows(arr[idx], confidence=confidence)
        if fit_boot is not None and np.all(np.isfinite(fit_boot["beta"])):
            boot_beta.append(fit_boot["beta"])
    if boot_beta:
        fit["boot_beta"] = np.asarray(boot_beta, dtype=float)
    else:
        fit["boot_beta"] = np.empty((0, fit["n_params"]), dtype=float)
    return fit


def build_realistic_front_results(
    save_compute,
    class_key=None,
    min_front_displacement_um=None,
    exclude_lambda_x64=False,
):
    if class_key is None:
        class_key = next(iter(save_compute.keys()))
    class_data = save_compute[class_key]
    return _build_results_for_metric(
        class_data,
        metric="front",
        t_fit_start_by_alpha={"default": 0},
        default_t_fit_start=0,
        min_front_displacement_um=min_front_displacement_um,
        exclude_lambda_x64=exclude_lambda_x64,
    )


def fit_realistic_front_scaling(
    save_compute,
    class_key=None,
    min_front_displacement_um=None,
    exclude_lambda_x64=False,
):
    results = build_realistic_front_results(
        save_compute,
        class_key=class_key,
        min_front_displacement_um=min_front_displacement_um,
        exclude_lambda_x64=exclude_lambda_x64,
    )
    return results, _fit_global_interaction_model(results, confidence=0.95)


def realistic_front_rows(results):
    rows = []
    for alpha_key, combo_block in results.get("combo", {}).items():
        try:
            alpha_val = float(alpha_key)
        except ValueError:
            continue
        for entry in combo_block.values():
            sigma_val = float(entry.get("sigma", np.nan))
            lambda_val = float(entry.get("lambda", np.nan))
            if not (np.isfinite(sigma_val) and np.isfinite(lambda_val)):
                continue
            if sigma_val <= 0 or lambda_val <= 0:
                continue
            for val in entry.get("values", []):
                v = float(val)
                if np.isfinite(v) and v > 0:
                    rows.append((alpha_val, sigma_val, lambda_val, v))
    return np.asarray(rows, dtype=float)


def print_per_alpha_logspeed_regression(results, confidence=0.99):
    alpha_tail = 1.0 - float(confidence)
    print("\nPer-alpha log-speed regression:")
    print("  ln(c_measured) = A + B*ln(sigma) + C*ln(lambda)")
    for alpha_key in sorted(results.get("combo", {}).keys(), key=lambda x: float(x)):
        combo_block = results["combo"].get(alpha_key, {})
        rows = []
        for entry in combo_block.values():
            sigma_val = float(entry.get("sigma", np.nan))
            lambda_val = float(entry.get("lambda", np.nan))
            if not (np.isfinite(sigma_val) and np.isfinite(lambda_val)):
                continue
            if sigma_val <= 0 or lambda_val <= 0:
                continue
            for val in entry.get("values", []):
                v = float(val)
                if np.isfinite(v) and v > 10**(-2):
                    rows.append((sigma_val, lambda_val, v))
        if len(rows) < 4:
            print(f"  alpha={float(alpha_key):.3g}: skipped (n={len(rows)} < 4)")
            continue

        arr = np.asarray(rows, dtype=float)
        y = np.log(arr[:, 2])
        X = np.column_stack(
            [
                np.ones(arr.shape[0], dtype=float),
                np.log(arr[:, 0]),
                np.log(arr[:, 1]),
            ]
        )
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat
        n_obs = X.shape[0]
        n_params = X.shape[1]
        dof = n_obs - n_params
        if dof <= 0:
            print(f"  alpha={float(alpha_key):.3g}: skipped (non-positive dof)")
            continue
        rss = float(np.sum(resid ** 2))
        sigma2 = rss / dof
        xtx_inv = np.linalg.pinv(X.T @ X)
        cov = xtx_inv * sigma2
        stderr = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
        if stats is not None:
            tcrit = float(stats.t.ppf(1.0 - alpha_tail / 2.0, df=dof))
        else:
            tcrit = 1.96
        ci_low = beta - tcrit * stderr
        ci_high = beta + tcrit * stderr

        alpha_val = float(alpha_key)
        b_theory = 4.0 / (2.0 + alpha_val)
        c_theory = (2.0 - alpha_val) / (2.0 + alpha_val)
        b_in_ci = bool(ci_low[1] <= b_theory <= ci_high[1])
        c_in_ci = bool(ci_low[2] <= c_theory <= ci_high[2])
        delta_diff = (beta[1] - b_theory) - (beta[2] - c_theory)
        var_delta = cov[1, 1] + cov[2, 2] - 2.0 * cov[1, 2]
        if np.isfinite(var_delta) and var_delta > 0:
            se_delta = float(np.sqrt(var_delta))
            t_delta = float(delta_diff / se_delta)
            ci_delta_low = float(delta_diff - tcrit * se_delta)
            ci_delta_high = float(delta_diff + tcrit * se_delta)
            delta_in_ci = bool(ci_delta_low <= 0.0 <= ci_delta_high)
            if stats is not None:
                p_delta = float(2.0 * (1.0 - stats.t.cdf(abs(t_delta), df=dof)))
            else:
                p_delta = np.nan
        else:
            se_delta = np.nan
            t_delta = np.nan
            p_delta = np.nan
            ci_delta_low = np.nan
            ci_delta_high = np.nan
            delta_in_ci = False

        print(
            f"  alpha={alpha_val:.3g} (n={n_obs}, dof={dof}): "
            f"A={beta[0]:+.4f} [{ci_low[0]:+.4f}, {ci_high[0]:+.4f}], "
            f"B={beta[1]:+.4f} [{ci_low[1]:+.4f}, {ci_high[1]:+.4f}], "
            f"C={beta[2]:+.4f} [{ci_low[2]:+.4f}, {ci_high[2]:+.4f}]"
        )
        print(
            f"    theory: B*= {b_theory:+.4f} (in CI: {b_in_ci}), "
            f"C*= {c_theory:+.4f} (in CI: {c_in_ci})"
        )
        if np.isfinite(delta_diff):
            p_text = f"{p_delta:.3g}" if np.isfinite(p_delta) else "nan"
            print(
                "    test H0: (B-B*) = (C-C*)  "
                f"=> delta={delta_diff:+.4f}, "
                f"CI=[{ci_delta_low:+.4f}, {ci_delta_high:+.4f}], "
                f"t={t_delta:+.3f}, p={p_text}, in CI(0): {delta_in_ci}"
            )


def _fit_per_alpha_exponents_from_results(results, confidence=0.99):
    alpha_tail = 1.0 - float(confidence)
    fit_rows = []
    for alpha_key in sorted(results.get("combo", {}).keys(), key=lambda x: float(x)):
        combo_block = results["combo"].get(alpha_key, {})
        rows = []
        for entry in combo_block.values():
            sigma_val = float(entry.get("sigma", np.nan))
            lambda_val = float(entry.get("lambda", np.nan))
            if not (np.isfinite(sigma_val) and np.isfinite(lambda_val)):
                continue
            if sigma_val <= 0 or lambda_val <= 0:
                continue
            for val in entry.get("values", []):
                v = float(val)
                if np.isfinite(v) and v > 1e-2:
                    rows.append((sigma_val, lambda_val, v))
        if len(rows) < 4:
            continue
        arr = np.asarray(rows, dtype=float)
        y = np.log(arr[:, 2])
        X = np.column_stack([np.ones(arr.shape[0], dtype=float), np.log(arr[:, 0]), np.log(arr[:, 1])])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - (X @ beta)
        dof = X.shape[0] - X.shape[1]
        if dof <= 0:
            continue
        sigma2 = float(np.sum(resid ** 2)) / dof
        cov = np.linalg.pinv(X.T @ X) * sigma2
        stderr = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
        if stats is not None:
            tcrit = float(stats.t.ppf(1.0 - alpha_tail / 2.0, df=dof))
        else:
            tcrit = 1.96
        ci_low = beta - tcrit * stderr
        ci_high = beta + tcrit * stderr
        fit_rows.append(
            {
                "alpha": float(alpha_key),
                "B": float(beta[1]),
                "C": float(beta[2]),
                "B_low": float(ci_low[1]),
                "B_high": float(ci_high[1]),
                "C_low": float(ci_low[2]),
                "C_high": float(ci_high[2]),
            }
        )
    return fit_rows


def plot_front_exponents_vs_alpha_panel(ax, results, confidence=0.99):
    fit_rows = _fit_per_alpha_exponents_from_results(results, confidence=confidence)
    if not fit_rows:
        ax.text(0.5, 0.5, "no valid exponent fits", ha="center", va="center", transform=ax.transAxes)
        style_empty_panel(ax)
        return
    fit_rows = sorted(fit_rows, key=lambda r: r["alpha"])
    alpha = np.asarray([r["alpha"] for r in fit_rows], dtype=float)
    b = np.asarray([r["B"] for r in fit_rows], dtype=float)
    c = np.asarray([r["C"] for r in fit_rows], dtype=float)
    b_low = np.asarray([r["B_low"] for r in fit_rows], dtype=float)
    b_high = np.asarray([r["B_high"] for r in fit_rows], dtype=float)
    c_low = np.asarray([r["C_low"] for r in fit_rows], dtype=float)
    c_high = np.asarray([r["C_high"] for r in fit_rows], dtype=float)
    b_theory = 4.0 / (2.0 + alpha)
    c_theory = (2.0 - alpha) / (2.0 + alpha)
    ax.fill_between(alpha, b_low, b_high, color="tab:blue", alpha=0.20)
    ax.fill_between(alpha, c_low, c_high, color="tab:orange", alpha=0.20)
    ax.plot(alpha, b, color="tab:blue", lw=1.3, label=r"fit $B(\alpha)$")
    ax.plot(alpha, c, color="tab:orange", lw=1.3, label=r"fit $C(\alpha)$")
    ax.plot(alpha, b_theory, color="tab:blue", ls="--", lw=1.0, label=r"theory $4/(2+\alpha)$")
    ax.plot(alpha, c_theory, color="tab:orange", ls="--", lw=1.0, label=r"theory $(2-\alpha)/(2+\alpha)$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Exponent")
    # ax.set_title("Front metric exponents vs alpha (99% CI)")
    ax.legend(frameon=False, fontsize=6.5, loc="best")


def theory_prefactor_from_speed(alpha):
    alpha = np.asarray(alpha, dtype=float)
    return subdiff_ctrw_speed_prefactor(alpha) * 2.0 ** (-2.0 / (2.0 + alpha))


def plot_xi_and_sim_prefactor_panel(ax, results_path=SAVE_COMPUTE_PATH, class_key=None):
    save_compute = load_save_compute(results_path)
    results, global_fit = fit_realistic_front_scaling(save_compute, class_key=class_key)
    if global_fit is None:
        ax.text(0.5, 0.5, "no prefactor fit", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"Theory vs fitted $K(\alpha)$")
        return

    alpha_unique = np.unique(np.asarray(global_fit["alpha"], dtype=float))
    alpha_unique = alpha_unique[np.isfinite(alpha_unique)]
    alpha_grid = np.linspace(float(np.min(alpha_unique)), float(np.max(alpha_unique)), 200)
    k_fit = np.exp(global_fit["beta"][0] + global_fit["beta"][3] * alpha_grid)
    k_points = np.exp(global_fit["beta"][0] + global_fit["beta"][3] * alpha_unique)
    theory_k = theory_prefactor_from_speed(alpha_grid)

    ax2 = ax.twinx()
    line_theory, = ax.plot(alpha_grid, theory_k, color="tab:blue", lw=1.5, label=r"theory $\Xi(\alpha)$")
    line_fit, = ax2.plot(alpha_grid, k_fit, color="tab:red", lw=1.5, label=r"fitted $\Xi(\alpha)$")
    pts = ax2.plot(
        alpha_unique,
        k_points,
        marker="o",
        linestyle="none",
        color="black",
        label=r"fitted points",
    )[0]

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"theory $\Xi(\alpha)$", color="tab:blue")
    ax2.set_ylabel(r"fitted $\Xi(\alpha)$", color="tab:red")
    ax.tick_params(axis="y", colors="tab:blue")
    ax2.tick_params(axis="y", colors="tab:red")
    ax.set_title(r"Theory vs fitted $K(\alpha)$")
    ax.set_xlim(float(np.min(alpha_grid)) - 0.02, float(np.max(alpha_grid)) + 0.02)
    ax2.set_yscale("log")

    handles = [line_theory, line_fit, pts]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="best", frameon=False, fontsize=8)


def plot_realistic_scaling_panel(ax, results_path=SAVE_COMPUTE_PATH, class_key=None):
    save_compute = load_save_compute(results_path)
    results, global_fit = fit_realistic_front_scaling(save_compute, class_key=class_key)
    if global_fit is None:
        ax.text(0.5, 0.5, "no scaling fit", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Realistic Scaling")
        return

    alpha_grid = np.linspace(
        float(np.nanmin(global_fit["alpha"])),
        float(np.nanmax(global_fit["alpha"])),
        200,
    )
    sigma_line = global_fit["beta"][1] + global_fit["beta"][4] * alpha_grid
    lambda_line = global_fit["beta"][2] + global_fit["beta"][5] * alpha_grid
    theory_sigma = 4.0 / (2.0 + alpha_grid)
    theory_lambda = (2.0 - alpha_grid) / (2.0 + alpha_grid)

    ax.plot(alpha_grid, sigma_line, color="tab:blue", lw=1.5, label=r"fit $\sigma$")
    ax.plot(alpha_grid, lambda_line, color="tab:orange", lw=1.5, label=r"fit $\lambda$")
    ax.plot(
        alpha_grid,
        theory_sigma,
        color="tab:blue",
        ls="--",
        lw=1.0,
        label=r"theory $\sigma$",
    )
    ax.plot(
        alpha_grid,
        theory_lambda,
        color="tab:orange",
        ls="--",
        lw=1.0,
        label=r"theory $\lambda$",
    )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Exponent")
    ax.set_title("Realistic Scaling")
    ax.set_xlim(float(np.min(alpha_grid)) - 0.02, float(np.max(alpha_grid)) + 0.02)
    ax.legend(loc="best", frameon=False, fontsize=8)


def plot_realistic_speed_scatter_panel(
    ax,
    results_path=SAVE_COMPUTE_PATH,
    class_key=None,
    lambda_scale=1 / 4622.228187341831,
    sigma_base=1.0,
    lambda_base=1.0,
    title=None,
    legend_mode="colorbar",
    annotation_text=None,
    randomize_scatter_zorder=False,
    random_zorder_seed=0,
    min_front_displacement_um=None,
    exclude_lambda_x64=True,
):
    save_compute = load_save_compute(results_path)
    results, _global_fit = fit_realistic_front_scaling(
        save_compute,
        class_key=class_key,
        min_front_displacement_um=min_front_displacement_um,
        exclude_lambda_x64=exclude_lambda_x64,
    )
    rows = realistic_front_rows(results)
    if rows.size == 0:
        ax.text(0.5, 0.5, "no speed data", ha="center", va="center", transform=ax.transAxes)
        if title is None:
            title = r"$\mathrm{c}_{\mathrm{theory}}$ vs $\mathrm{c}_{\mathrm{measured}}$"
        ax.set_title(title)
        return

    alpha_vals = rows[:, 0]
    sigma_vals = rows[:, 1]
    lambda_vals = rows[:, 2]
    measured = rows[:, 3]
    sigma_eff = sigma_vals * float(sigma_base)
    lambda_eff = lambda_vals * float(lambda_base) * float(lambda_scale)
    theory = speed_subdiff_ctrw(alpha_vals, sigma_eff, lambda_eff, dt=1.0)

    valid = (
        np.isfinite(alpha_vals)
        & np.isfinite(measured)
        & np.isfinite(theory)
        # & (measured > 0)
        & (measured > 10**(-2))
        & (theory > 0)
    )

    if np.count_nonzero(valid) == 0:
        ax.text(0.5, 0.5, "no valid speed data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(r"$\mathrm{c}_{\mathrm{theory}}$ vs $\mathrm{c}_{\mathrm{simu}}$ 2D")
        return

    alpha_vals = alpha_vals[valid]
    measured = measured[valid]
    theory = theory[valid]
    sigma_vals = sigma_vals[valid]
    lambda_vals = lambda_vals[valid]
    sigma_eff = sigma_eff[valid]
    lambda_eff = lambda_eff[valid]
    
    unique_alphas = np.unique(alpha_vals)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(unique_alphas), 1)))
    alpha_color_map = {float(alpha): color for alpha, color in zip(unique_alphas, colors)}
    scatter = None
    if randomize_scatter_zorder:
        rng = np.random.default_rng(int(random_zorder_seed))
        draw_order = rng.permutation(theory.size)
        for idx in draw_order:
            alpha_val = float(alpha_vals[idx])
            color = alpha_color_map.get(alpha_val, "0.3")
            scatter = ax.scatter(
                theory[idx],
                measured[idx],
                color=color,
                s=14,
                alpha=0.75,
                linewidths=0,
            )
    else:
        for zorder, (alpha, color) in enumerate(zip(unique_alphas, colors)):
            mask = np.isclose(alpha_vals, alpha)
            scatter = ax.scatter(
                theory[mask],
                measured[mask],
                color=color,
                s=14,
                alpha=0.75,
                linewidths=0,
                label=rf"$\alpha={float(alpha):.1f}$",
                zorder=zorder,
            )
    diag_min = float(min(np.min(theory), np.min(measured)))
    diag_max = float(max(np.max(theory), np.max(measured)))
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color="0.25", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{c}_{\mathrm{theory}}$ (µm/min)")
    ax.set_ylabel(r"$\mathrm{c}_{\mathrm{simu}}$ 2D (µm/min)")
    if not(title is None):
        ax.set_title(title)
    add_identity_line_label(ax, diag_min, diag_max)
    if legend_mode == "colorbar":
        cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\alpha$")
    elif legend_mode == "legend":
        if randomize_scatter_zorder:
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=alpha_color_map[float(alpha)],
                    markeredgewidth=0,
                    markersize=4.5,
                    alpha=0.75,
                    label=rf"$\alpha={float(alpha):.1f}$",
                )
                for alpha in unique_alphas
            ]
            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper left", frameon=False, handlelength=1.4)
        else:
            ax.legend(loc="upper left", frameon=False, handlelength=1.4)

    if annotation_text is not None:
        ax.text(
            0.03,
            0.97,
            annotation_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.7,
        )


def alpha_P_sampling_metric(alpha, P, dt=0.1, sigma_ref=1.0):
    alpha = np.asarray(alpha, dtype=float)
    P = np.asarray(P, dtype=float)
    lambda_eff = P / (sigma_ref * dt)
    c_star_eff = speed_subdiff_ctrw(alpha, sigma_ref, lambda_eff, dt)
    radicand = ((4.0 - alpha) / (2.0 - alpha)) * lambda_eff * c_star_eff
    return dt * np.sqrt(np.clip(radicand, 0.0, None))


def plot_alpha_P_sensitivity_map(
    alpha_min=0.01,
    alpha_max=0.999,
    P_min=1e-5,
    P_max=1e-1,#1e3
    n_alpha=400,
    n_P=400,
    dt=1,
    sigma_ref=1.0,
    sampling_threshold=0.1,
    show_hatch_regions=True,
    cmap="viridis",
    ax=None,
    show=True,
):
    """
    Plot in (alpha, P) space, where P = sigma * lambda * dt:
      - x-axis: alpha
      - y-axis: P
      - color: absolute sensitivity ratio |S_alpha / S_sigma|
      - overlay: ratio level curves
    """
    alphas = np.linspace(alpha_min, alpha_max, n_alpha)
    Ps = np.logspace(np.log10(P_min), np.log10(P_max), n_P)

    Alpha, P = np.meshgrid(alphas, Ps)
    Rabs = sensitivity_ratio_abs(Alpha, P)
    SamplingMetric = alpha_P_sampling_metric(Alpha, P, dt=dt, sigma_ref=sigma_ref)

    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    pcm = ax.pcolormesh(Alpha, P, Rabs, shading="auto", cmap=cmap)
    max_sampling_metric = float(np.max(SamplingMetric))
    if show_hatch_regions and max_sampling_metric > sampling_threshold:
        ax.contourf(
            Alpha,
            P,
            SamplingMetric,
            levels=[sampling_threshold, max_sampling_metric],
            colors="none",
            hatches=["//////"],
        )
        ax.contour(
            Alpha,
            P,
            SamplingMetric,
            levels=[sampling_threshold],
            colors="black",
            linewidths=0.5,
        )
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|S_{\alpha}/S_{\sigma}|$")
    contours = ax.contour(
        Alpha,
        P,
        Rabs,
        levels=[1.0, 2.0, 3.0, 4.0],
        colors=["red"] + ["black"]*3,
        linestyles="--",
    )
    
    # ax.clabel(contours, fmt="%g", fontsize=7, inline=True)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$P=\sigma\lambda\Delta t$")
    ax.set_yscale("log")
    ax.set_title(r"Absolute sensitivity ratio $|S_{\alpha}/S_{\sigma}|$")

    if created_ax:
        fig.tight_layout()
        if show:
            plt.show()

    return Alpha, P, Rabs


def plot_alpha_P_velocity_map(
    alpha_min=0.01,
    alpha_max=0.999,
    P_min=1e-5,
    P_max=1e-1,#1e3
    n_alpha=400,
    n_P=400,
    dt=1,
    sigma_ref=1.0,
    sampling_threshold=0.1,
    show_hatch_regions=True,
    cmap="viridis",
    vmin=None,
    vmax=None,
    ax=None,
    show=True,
):
    """
    Plot theoretical front velocity in (alpha, P) space using the sigma=sigma_ref
    slice, with P = sigma * lambda * dt and lambda = P / (sigma_ref * dt).
    """
    alphas = np.linspace(alpha_min, alpha_max, n_alpha)
    Ps = np.logspace(np.log10(P_min), np.log10(P_max), n_P)

    Alpha, P = np.meshgrid(alphas, Ps)
    Lambda = P / (sigma_ref * dt)
    Velocity = speed_subdiff_ctrw(Alpha, sigma_ref, Lambda, dt)
    SamplingMetric = alpha_P_sampling_metric(Alpha, P, dt=dt, sigma_ref=sigma_ref)

    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    positive_velocity = Velocity[np.isfinite(Velocity) & (Velocity > 0)]
    if positive_velocity.size > 0:
        if vmin is None:
            vmin = float(np.quantile(positive_velocity, 0.01))
        if vmax is None:
            vmax = float(np.quantile(positive_velocity, 0.99))
        if vmax <= vmin:
            vmax = float(np.max(positive_velocity))
            vmin = float(np.min(positive_velocity))
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    pcm = ax.pcolormesh(Alpha, P, Velocity, shading="auto", cmap=cmap, norm=norm)
    max_sampling_metric = float(np.max(SamplingMetric))
    if show_hatch_regions and max_sampling_metric > sampling_threshold:
        

        ax.contourf(
            Alpha,
            P,
            SamplingMetric,
            levels=[sampling_threshold, max_sampling_metric],
            colors="none",
            hatches=["//////"],
        )
        ax.contour(
            Alpha,
            P,
            SamplingMetric,
            levels=[sampling_threshold],
            colors="black",
            linewidths=0.50,
        )
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"c (µm/min)")

    if positive_velocity.size > 0:
        contour_levels = np.geomspace(
            float(norm.vmin),
            float(norm.vmax),
            6,
        )
        contour_levels = [
            1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4
        ]
        contour_levels = np.unique(contour_levels)
        if contour_levels.size >= 2:
            contours = ax.contour(
                Alpha,
                P,
                Velocity,
                levels=contour_levels,
                colors="black",
                linestyles="--",
            )
            # ax.clabel(contours, fmt="%.2g", fontsize=7, inline=True)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$P=\sigma\lambda\Delta t$")
    ax.set_yscale("log")
    ax.set_title(r"Front velocity ($\sigma = 1$ µm, $\Delta t = 1$ min)")

    if created_ax:
        fig.tight_layout()
        if show:
            plt.show()

    return Alpha, P, Velocity

def save_figure(fig, base_name="Fig_SI_front_propagation"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{base_name}.{ext}", dpi=300)


def build_figure_1(max_trajectories_to_plot=500, recompute=False):
    fig, axs = make_custom_panel_grid(1, 3)
    plot_tip_splitting_example_panel(axs[0, 0])
    plot_side_branching_example_panel(axs[0, 1])
    plot_side_branching_panel(
        axs[0, 2],
        max_trajectories_to_plot=max_trajectories_to_plot,
        recompute=recompute,
        sigma=1,
        lamb=0.01,
        zoom_xlim=None,
        zoom_ylim=None,
    )

    add_panel_label(axs[0, 0], "A")
    add_panel_label(axs[0, 1], "B")
    add_panel_label(axs[0, 2], "C")
    return fig, axs


def build_figure_2(
    max_trajectories_to_plot=500,
    recompute=False,
    randomize_scatter_zorder=False,
    random_zorder_seed=0,
    show_hatch_regions=True,
):
    fig, axs = make_custom_panel_grid(1, 3)
    plot_subdiffusive_front_panel(
        axs[0, 0],
        lamb=0.003,
        sigma=1.0,
        dt=0.1,
        t_max=1000,
        top_n=5000,
        seed=503,
        alphas=(0.2, 0.4, 0.6, 0.8, 1.0),
        recompute=recompute,
    )
    plot_scaling_exponent_panel(
        axs[0, 1],
        alphas=(0.2, 0.4, 0.6, 0.8, 1.0),
        sigma_lambda_pairs=BASE_SCALING_PARAM_PAIRS,
        t_max=300,
        dt=0.1,
        t_fit_start=150.0,
        n_repeat=5,
        top_n=500,
        seed=504,
        recompute=recompute,
        randomize_scatter_zorder=randomize_scatter_zorder,
        random_zorder_seed=random_zorder_seed,
    )
    plot_alpha_P_velocity_map(
        ax=axs[0, 2],
        show=False,
        sigma_ref=0.7,
        show_hatch_regions=show_hatch_regions,
    )

    add_panel_label(axs[0, 0], "A")
    add_panel_label(axs[0, 1], "B")
    add_panel_label(axs[0, 2], "C")
    return fig, axs


def build_figure_3(show_hatch_regions=True):
    fig, axs = make_custom_panel_grid(1, 1)
    plot_alpha_P_sensitivity_map(
        ax=axs[0, 0],
        show=False,
        show_hatch_regions=show_hatch_regions,
    )
    return fig, axs


def build_figure_4(
    recompute=False,
    realistic_sigma_base=0.756,
    realistic_lambda_base=0.029,
    randomize_scatter_zorder=False,
    random_zorder_seed=0,
    min_front_displacement_um=0.0,
):
    fig, axs = make_custom_panel_grid(1, 3)
    save_compute = load_save_compute(SAVE_COMPUTE_PATH)
    results, _ = fit_realistic_front_scaling(
        save_compute,
        min_front_displacement_um=min_front_displacement_um,
        exclude_lambda_x64=True,
    )
    print_per_alpha_logspeed_regression(results, confidence=0.99)
    plot_overlay_swc_panel(axs[0, 0], recompute=recompute)
    plot_realistic_speed_scatter_panel(
        axs[0, 1],
        lambda_scale=1.0,
        sigma_base=realistic_sigma_base,
        lambda_base=realistic_lambda_base,
        title="",#r"Speed comparison with $\lambda$",
        legend_mode="legend",
        randomize_scatter_zorder=randomize_scatter_zorder,
        random_zorder_seed=random_zorder_seed,
        min_front_displacement_um=min_front_displacement_um,
    )
    plot_front_exponents_vs_alpha_panel(
        axs[0, 2],
        results=results,
        confidence=0.99,
    )

    add_panel_label(axs[0, 0], "A")
    add_panel_label(axs[0, 1], "B")
    add_panel_label(axs[0, 2], "C")
    return fig, axs



def main(max_trajectories_to_plot=5000, recompute=False):
    fig, _ = build_figure_1(
        max_trajectories_to_plot=max_trajectories_to_plot,
        recompute=recompute,
    )
    save_figure(fig, base_name="Fig_SI_front_propagation_1")
    plt.close(fig)

    fig, _ = build_figure_2(
        max_trajectories_to_plot=max_trajectories_to_plot,
        recompute=recompute,
        randomize_scatter_zorder=True,
        show_hatch_regions=False,
    )
    save_figure(fig, base_name="Fig_SI_front_propagation_2")
    plt.close(fig)

    fig, _ = build_figure_3(show_hatch_regions=False)
    save_figure(fig, base_name="Fig_SI_front_propagation_3")
    plt.close(fig)

    fig, _ = build_figure_4(recompute=recompute,randomize_scatter_zorder=True)
    save_figure(fig, base_name="Fig_SI_front_propagation_4")
    plt.close(fig)



if __name__ == "__main__":
    main(recompute = False)
