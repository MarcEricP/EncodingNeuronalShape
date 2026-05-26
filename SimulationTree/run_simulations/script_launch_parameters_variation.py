from simulation_tree import run_simu, graphics
import json
import os
import numpy as np
from scipy.special import gamma, rgamma
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import datetime


LOCK_CI_CIV_TO_MEAN_WHEN_VARYING = True


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def should_skip(res_simu_dir, max_t, recompute):
    if recompute:
        return False
    return os.path.exists(os.path.join(res_simu_dir, f"{max_t}.swc"))


def sanitize_tag(x):
    s = f"{x:.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def gennorm2laplace(beta_gennorm, scale_gennorm):
    """
    Return scale of the equivalent Laplace distribution.
    """
    return scale_gennorm * np.sqrt(gamma(3 / beta_gennorm) * rgamma(1 / beta_gennorm) / 2)


def scale_schedule(schedule, factor):
    if schedule is None:
        return None
    times, values = schedule
    return (times, tuple(v * factor for v in values))


def mean_schedules(schedule_a, schedule_b):
    if schedule_a is None or schedule_b is None:
        return None
    times_a, values_a = schedule_a
    times_b, values_b = schedule_b
    if times_a != times_b:
        raise ValueError(f"Cannot average schedules with different time supports: {times_a} vs {times_b}")
    if len(values_a) != len(values_b):
        raise ValueError(f"Cannot average schedules with different value lengths: {len(values_a)} vs {len(values_b)}")
    return (times_a, tuple((va + vb) / 2.0 for va, vb in zip(values_a, values_b)))


def build_base_arfima_zero_drift_schedules(n_class, max_t):
    """
    Parameter schedules for ARFIMA zero drift
    """
    schedule_branching_lambda = ((0, max_t), (0.036, 0.022))
    branching_age_dependance = None

    if n_class == "class_I":
        schedule_contact_memory_time = ((0, max_t), (1, 1))
        schedule_contact_mean_retraction = ((0, max_t), (-1.1, -1.1))
        schedule_contact_std_retraction = ((0, max_t), (0., 0.))
        d = -0.27
        schedule_gennorm_beta = ((25, 75, 125, 175, 225, 275), (1.258, 1.214, 1.217, 1.178, 1.081, 0.996))
        schedule_gennorm_sigma = ((25, 75, 125, 175, 225, 275), (0.722, 0.657, 0.659, 0.564, 0.463, 0.356))
    elif n_class == "class_IV":
        schedule_contact_memory_time = ((0, max_t), (1, 1))
        schedule_contact_mean_retraction = ((0, max_t), (-1.7, -1.7))
        schedule_contact_std_retraction = ((0, max_t), (0, 0))
        d = -0.04
        schedule_gennorm_beta = ((25, 75, 125, 175, 225, 275), (1.533, 1.551, 1.490, 1.369, 1.349, 1.162))
        schedule_gennorm_sigma = ((25, 75, 125, 175, 225, 275), (1.055, 1.015, 0.902, 0.779, 0.725, 0.533))
    else:
        raise ValueError(f"Unsupported class: {n_class}")

    schedule_arfima_kappa = ((0, max_t), (1.0, 1.0))
    schedule_arfima_loc = ((0, max_t), (0.0, 0.0))  # zero drift
    scale_laplace = [gennorm2laplace(b, s) for b, s in zip(schedule_gennorm_beta[1], schedule_gennorm_sigma[1])]
    schedule_arfima_scale = ((25, 75, 125, 175, 225, 275), tuple(scale_laplace))

    return dict(
        schedule_branching_lambda=schedule_branching_lambda,
        branching_age_dependance=branching_age_dependance,
        schedule_contact_memory_time=schedule_contact_memory_time,
        schedule_contact_mean_retraction=schedule_contact_mean_retraction,
        schedule_contact_std_retraction=schedule_contact_std_retraction,
        schedule_gennorm_beta=None,
        schedule_gennorm_loc=None,
        schedule_gennorm_scale=None,
        arfima_d=d,
        schedule_arfima_kappa=schedule_arfima_kappa,
        schedule_arfima_loc=schedule_arfima_loc,
        schedule_arfima_scale=schedule_arfima_scale,
        schedule_palavalli_von=None,
        schedule_palavalli_voff=None,
        schedule_palavalli_kon=None,
        schedule_palavalli_koff=None,
    )


def run_one(task):
    (save_simu_base, n_class, start_name, n, swc_txt,
     max_t, dt, recompute, base_seed, generate_graphics,
     param_name, param_value, param_factor) = (
        task["save_simu_base"], task["n_class"], task["start_name"], task["n"], task["swc_txt"],
        task["max_t"], task["dt"], task["recompute"], task["base_seed"], task["generate_graphics"],
        task["param_name"], task["param_value"], task["param_factor"],
    )

    res_simu = os.path.join(save_simu_base, start_name, str(n), "swc")
    save_path_parameters = os.path.join(save_simu_base, start_name, str(n), "params.json")

    if should_skip(res_simu, max_t, recompute):
        return (n_class, param_name, param_value, start_name, n, "skipped")

    ensure_dir(res_simu)
    ensure_dir(os.path.dirname(save_path_parameters))

    schedules = build_base_arfima_zero_drift_schedules(n_class=n_class, max_t=max_t)
    if param_name in {"lambda_branch", "kappa", "sigma"}:
        target_schedule_key = {
            "lambda_branch": "schedule_branching_lambda",
            "kappa": "schedule_contact_mean_retraction",
            "sigma": "schedule_arfima_scale",
        }[param_name]

        if LOCK_CI_CIV_TO_MEAN_WHEN_VARYING:
            class_i_base = build_base_arfima_zero_drift_schedules(n_class="class_I", max_t=max_t)
            class_iv_base = build_base_arfima_zero_drift_schedules(n_class="class_IV", max_t=max_t)
            mean_schedule = mean_schedules(
                class_i_base[target_schedule_key],
                class_iv_base[target_schedule_key],
            )
            schedules[target_schedule_key] = scale_schedule(mean_schedule, param_factor)
        else:
            schedules[target_schedule_key] = scale_schedule(
                schedules[target_schedule_key], param_factor
            )
    elif param_name == "alpha":
        schedules["arfima_d"] = 0.5 * (param_value - 1.0)

    seed = (base_seed * 10_000_019 + hash((n_class, param_name, param_value, start_name, n))) % 2**32

    run_simu.run_sim(
        save_path=res_simu,
        save_path_parameters=save_path_parameters,
        max_t=max_t,
        dt=dt,
        seed=seed,
        swc_string=swc_txt,
        box_size_contact=0.5,
        new_branch_width=0.5,
        proba_overlap=0.0,
        persistence_length=30.0,
        new_branch_length_mean=0.1,
        new_branch_length_std=0.0,
        new_branch_angle_kappa=1.0,
        min_dist_node=0.1,
        program_type=run_simu.PROGRAM_ARFIMA,
        program_horizon=60,
        schedule_branching_lambda=schedules["schedule_branching_lambda"],
        schedule_contact_memory_time=schedules["schedule_contact_memory_time"],
        schedule_contact_mean_retraction=schedules["schedule_contact_mean_retraction"],
        schedule_contact_std_retraction=schedules["schedule_contact_std_retraction"],
        schedule_gennorm_beta=schedules["schedule_gennorm_beta"],
        schedule_gennorm_loc=schedules["schedule_gennorm_loc"],
        schedule_gennorm_scale=schedules["schedule_gennorm_scale"],
        arfima_d=schedules["arfima_d"],
        schedule_arfima_kappa=schedules["schedule_arfima_kappa"],
        schedule_arfima_loc=schedules["schedule_arfima_loc"],
        schedule_arfima_scale=schedules["schedule_arfima_scale"],
        schedule_palavalli_von=schedules["schedule_palavalli_von"],
        schedule_palavalli_voff=schedules["schedule_palavalli_voff"],
        schedule_palavalli_kon=schedules["schedule_palavalli_kon"],
        schedule_palavalli_koff=schedules["schedule_palavalli_koff"],
        branching_age_dependance=schedules["branching_age_dependance"],
    )

    if generate_graphics:
        try:
            folder_to_swcs = graphics.collect_swc_folders([os.path.dirname(save_path_parameters)])
            for folder, swc_paths in folder_to_swcs.items():
                graphics.make_movie_for_folder(folder, swc_paths)
        except Exception as e:
            print(f"[ERROR] Failed on folder {folder}: {e}")

    return (n_class, param_name, param_value, start_name, n, "done")


if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    with open(os.path.join(os.getcwd(), "run_simulations/all_swc.json"), "r") as f:
        all_starting_points = json.load(f)

    save_root = os.path.join(os.getcwd(), "simulation_result_parameters_variation")
    save_name_root = "ARFIMA_zero_drift"
    n_simu_per_start = 2
    max_t = 300
    dt = 1
    recompute = False
    generate_graphics = True
    MAX_WORKERS = 5

    param_factors = [0.25, 0.5, 1.0, 2.0, 4.0]
    alpha_values = [0.25, 0.5, 0.75, 1.0, 1.25]
    vary_params = ["lambda_branch", "kappa", "sigma"]

    tasks = []
    base_seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32).item()

    for n_class in ["class_I", "class_IV"]:
        for param_name in vary_params:
            for f in param_factors:
                tag = f"{param_name}_x{sanitize_tag(f)}"
                save_simu_base = os.path.join(save_root, save_name_root, n_class, tag)

                for start_name, swc_txt in all_starting_points[n_class].items():
                    for n in range(n_simu_per_start):
                        tasks.append(dict(
                            save_simu_base=save_simu_base,
                            n_class=n_class,
                            start_name=start_name,
                            n=n,
                            swc_txt=swc_txt,
                            max_t=max_t,
                            dt=dt,
                            recompute=recompute,
                            base_seed=base_seed,
                            generate_graphics=generate_graphics,
                            param_name=param_name,
                            param_value=f,
                            param_factor=f,
                        ))

        for alpha in alpha_values:
            tag = f"alpha_{sanitize_tag(alpha)}"
            save_simu_base = os.path.join(save_root, save_name_root, n_class, tag)

            for start_name, swc_txt in all_starting_points[n_class].items():
                for n in range(n_simu_per_start):
                    tasks.append(dict(
                        save_simu_base=save_simu_base,
                        n_class=n_class,
                        start_name=start_name,
                        n=n,
                        swc_txt=swc_txt,
                        max_t=max_t,
                        dt=dt,
                        recompute=recompute,
                        base_seed=base_seed,
                        generate_graphics=generate_graphics,
                        param_name="alpha",
                        param_value=alpha,
                        param_factor=1.0,
                    ))

    start_time = time.time()
    results = []
    print(f"Launching {len(tasks)} simulations across {MAX_WORKERS} workers...")

    if MAX_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(run_one, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    res = ("ERROR", str(e))
                results.append(res)
                if isinstance(res, tuple) and len(res) == 6:
                    n_class, param_name, param_value, start_name, n, status = res
                    print(f"[ARFIMA_zero_drift] {n_class} {param_name}={param_value} {start_name} run {n}: {status}")
                else:
                    print(res)
    else:
        for t in tasks:
            print(f"{t['save_simu_base']} | {t['n_class']} {t['param_name']}={t['param_value']} | {t['start_name']}")
            results.append(run_one(t))

    done = sum(1 for r in results if isinstance(r, tuple) and r[-1] == "done")
    skipped = sum(1 for r in results if isinstance(r, tuple) and r[-1] == "skipped")
    errors = [r for r in results if isinstance(r, tuple) and r[0] == "ERROR"]
    print(f"\nSummary: done={done}, skipped={skipped}, errors={len(errors)}, time={datetime.timedelta(seconds=time.time() - start_time)}")
    if errors:
        print("Example error:", errors[0])
