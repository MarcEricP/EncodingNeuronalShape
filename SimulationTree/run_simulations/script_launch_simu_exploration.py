import os, numpy as np, time, datetime

from simulation_tree import run_simu, graphics
from scipy.special import gamma, rgamma
from concurrent.futures import ProcessPoolExecutor, as_completed

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def movie_path_for_swc_dir(res_simu_dir):
    parent_dir = os.path.dirname(res_simu_dir)
    folder_tag = os.path.basename(res_simu_dir.rstrip(os.sep))
    return os.path.join(parent_dir, f"{folder_tag}_swc_sequence.mp4")

def should_skip(res_simu_dir, max_t, recompute, front_stop_enabled, min_save_t_recompute):
    if recompute:
        return False
    if front_stop_enabled and os.path.exists(movie_path_for_swc_dir(res_simu_dir)):
        return True
    if os.path.exists(os.path.join(res_simu_dir, f"{max_t}.swc")):
        return True
    if not os.path.isdir(res_simu_dir):
        return False
    try:
        for name in os.listdir(res_simu_dir):
            if not name.endswith(".swc"):
                continue
            stem = name[:-4]
            try:
                t = int(stem)
            except ValueError:
                continue
            if t >= min_save_t_recompute:
                return True
    except OSError:
        return False
    return False

def sanitize_tag(x):
    s = f"{x:.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def gennorm2laplace(beta_gennorm, scale_gennorm):
    """Return scale of the equivalent Laplace distribution."""
    return scale_gennorm * np.sqrt(gamma(3 / beta_gennorm) * rgamma(1 / beta_gennorm) / 2)


def average_schedule_values(values):
    return float(np.mean(values))


def build_class_iv_reference_params(max_t):
    """
    Reference params derived from class IV schedules.
    Lambda and Laplace scale are replaced by their time-average values.
    """
    schedule_branching_lambda = ((0, max_t), (0.036, 0.022))
    schedule_contact_mean_retraction = ((0, max_t), (-1.7, -1.7))
    schedule_gennorm_beta = ((25, 75, 125, 175, 225, 275), (1.533, 1.551, 1.490, 1.369, 1.349, 1.162))
    schedule_gennorm_sigma = ((25, 75, 125, 175, 225, 275), (1.055, 1.015, 0.902, 0.779, 0.725, 0.533))

    lambda_ref = average_schedule_values(schedule_branching_lambda[1])
    kappa_ref = average_schedule_values(schedule_contact_mean_retraction[1])

    scale_laplace = [
        gennorm2laplace(b, s) for b, s in zip(schedule_gennorm_beta[1], schedule_gennorm_sigma[1])
    ]
    laplace_scale_ref = average_schedule_values(scale_laplace)
    sigma_std_ref = laplace_scale_ref * np.sqrt(2.0)

    return dict(
        lambda_branch=lambda_ref,
        sigma_std=sigma_std_ref,
        kappa=kappa_ref,
    )


def build_vertical_line_swc(y_start=-15.0, y_end=15.0, radius=0.25):
    return "\n".join([
        f"1 1 0.0 {y_start:.6f} 0.0 {radius:.6f} -1",
        f"2 3 0.0 {y_end:.6f} 0.0 {radius:.6f} 1",
    ])


def force_swc_width(swc_text, target_width):
    """
    Force initial SWC branch widths to target_width.
    """
    target_radius = 0.5 * float(target_width)
    out_lines = []
    for ln in swc_text.splitlines():
        stripped = ln.strip()
        if (not stripped) or stripped.startswith("#"):
            out_lines.append(ln)
            continue
        parts = stripped.split()
        if len(parts) < 7:
            out_lines.append(ln)
            continue
        parts[5] = f"{target_radius:.6f}"  # radius
        if len(parts) >= 9:
            parts[8] = f"{float(target_width):.6f}"  # explicit width
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines)


def build_const_arfima_zero_drift_schedules(n_class, max_t, lambda_branch, sigma_std, kappa, alpha):
    """
    Schedules:
    - lambda_branch: branching rate (per min)
    - sigma_std: std of increments: internally Laplace scale b = sigma / sqrt(2)
    - kappa: constant step-back parameter
    - alpha: subdiffusion exponent: ARFIMA d = (alpha - 1)/2
    """
    # branching
    schedule_branching_lambda = ((0, max_t), (lambda_branch, lambda_branch))
    branching_age_dependance = None

    #fluctuation and subdiffusion
    d = 0.5 * (alpha - 1.0)
    laplace_scale = sigma_std / np.sqrt(2.0)

    schedule_contact_memory_time = ((0, max_t), (1, 1))
    schedule_contact_mean_retraction = ((0, max_t), (kappa, kappa))
    schedule_contact_std_retraction = ((0, max_t), (0., 0.))

    schedule_arfima_kappa = ((0, max_t), (1.0, 1.0))
    schedule_arfima_loc = ((0, max_t), (0.0, 0.0))  # zero drift
    schedule_arfima_scale = ((0, max_t), (laplace_scale, laplace_scale))

    schedule_gennorm_beta = None
    schedule_gennorm_loc = None
    schedule_gennorm_scale = None
    schedule_palavalli_von = None
    schedule_palavalli_voff = None
    schedule_palavalli_kon = None
    schedule_palavalli_koff = None

    return dict(
        # branching and contact
        schedule_branching_lambda=schedule_branching_lambda,
        branching_age_dependance=branching_age_dependance,
        schedule_contact_memory_time=schedule_contact_memory_time,
        schedule_contact_mean_retraction=schedule_contact_mean_retraction,
        schedule_contact_std_retraction=schedule_contact_std_retraction,
        # GENNORM (off)
        schedule_gennorm_beta=schedule_gennorm_beta,
        schedule_gennorm_loc=schedule_gennorm_loc,
        schedule_gennorm_scale=schedule_gennorm_scale,
        # ARFIMA (on)
        arfima_d=d,
        schedule_arfima_kappa=schedule_arfima_kappa,
        schedule_arfima_loc=schedule_arfima_loc,
        schedule_arfima_scale=schedule_arfima_scale,
        # PALAVALLI (off)
        schedule_palavalli_von=schedule_palavalli_von,
        schedule_palavalli_voff=schedule_palavalli_voff,
        schedule_palavalli_kon=schedule_palavalli_kon,
        schedule_palavalli_koff=schedule_palavalli_koff,
    )

def run_one(task):
    """
    task: dict with all parameters needed for one run
    """
    (save_simu_base, n_class, start_name, n, swc_txt,
     max_t, dt, recompute, base_seed, generate_graphics,
     lambda_branch, sigma_std, kappa, alpha, tag,
     new_branch_width,
     init_mode, boundary_mode, boundary_x0, boundary_y_min, boundary_y_max,
     front_only, front_only_bins, front_stop_enabled, front_stop_position,
     max_compartments_stop_enabled, max_compartments_stop_count, min_save_t_recompute) = (
        task["save_simu_base"], task["n_class"], task["start_name"], task["n"], task["swc_txt"],
        task["max_t"], task["dt"], task["recompute"], task["base_seed"], task["generate_graphics"],
        task["lambda_branch"], task["sigma_std"], task["kappa"], task["alpha"],
        task["tag"], task["new_branch_width"], task["init_mode"], task["boundary_mode"],
        task["boundary_x0"], task["boundary_y_min"], task["boundary_y_max"],
        task["front_only"],
        task["front_only_bins"],
        task["front_stop_enabled"],
        task["front_stop_position"],
        task["max_compartments_stop_enabled"],
        task["max_compartments_stop_count"],
        task["min_save_t_recompute"],
    )

    with_branch_dep = False
    res_simu = os.path.join(save_simu_base, start_name, str(n), "swc")
    save_path_parameters = os.path.join(save_simu_base, start_name, str(n), "params.json")

    if should_skip(res_simu, max_t, recompute, front_stop_enabled, min_save_t_recompute):
        return (n_class, with_branch_dep, "ARFIMA_zero_drift", start_name, n, "skipped")

    ensure_dir(res_simu)
    ensure_dir(os.path.dirname(save_path_parameters))

    # get schedules
    schedules = build_const_arfima_zero_drift_schedules(
        n_class=n_class, max_t=max_t,
        lambda_branch=lambda_branch, sigma_std=sigma_std, kappa=kappa, alpha=alpha
    )

    # Distinct seed per task
    seed = (base_seed * 10_000_019 + hash((n_class, start_name, n, lambda_branch, sigma_std, kappa, alpha))) % 2**32

    profile_time = run_simu.run_sim(
        save_path=res_simu,
        save_path_parameters=save_path_parameters,
        max_t=max_t,
        dt=dt,
        seed=seed,
        swc_string=swc_txt,
        front_only=front_only,
        front_only_bins=front_only_bins,
        front_stop_enabled=front_stop_enabled,
        front_stop_position=front_stop_position,
        max_compartments_stop_enabled=max_compartments_stop_enabled,
        max_compartments_stop_count=max_compartments_stop_count,
        print_profile_summary=True,
        init_mode=init_mode,
        boundary_mode=boundary_mode,
        boundary_x0=boundary_x0,
        boundary_y_min=boundary_y_min,
        boundary_y_max=boundary_y_max,
        box_size_contact=0.5,
        new_branch_width=new_branch_width,
        proba_overlap=0.0,
        persistence_length=30.0,
        new_branch_length_mean=0.0,#0.1
        new_branch_length_std=0.,
        new_branch_angle_kappa=1.0,    
        min_dist_node=0.1,#0.1
        program_type=run_simu.PROGRAM_ARFIMA,
        program_horizon=300,#60
        # branching and contact
        schedule_branching_lambda=schedules["schedule_branching_lambda"],
        schedule_contact_memory_time=schedules["schedule_contact_memory_time"],
        schedule_contact_mean_retraction=schedules["schedule_contact_mean_retraction"],
        schedule_contact_std_retraction=schedules["schedule_contact_std_retraction"],
        # GENNORM (off)
        schedule_gennorm_beta=schedules["schedule_gennorm_beta"],
        schedule_gennorm_loc=schedules["schedule_gennorm_loc"],
        schedule_gennorm_scale=schedules["schedule_gennorm_scale"],
        # ARFIMA
        arfima_d=schedules["arfima_d"],
        schedule_arfima_kappa=schedules["schedule_arfima_kappa"],
        schedule_arfima_loc=schedules["schedule_arfima_loc"],
        schedule_arfima_scale=schedules["schedule_arfima_scale"],
        # PALAVALLI (off)
        schedule_palavalli_von=schedules["schedule_palavalli_von"],
        schedule_palavalli_voff=schedules["schedule_palavalli_voff"],
        schedule_palavalli_kon=schedules["schedule_palavalli_kon"],
        schedule_palavalli_koff=schedules["schedule_palavalli_koff"],
        # Branching age dependence (None)
        branching_age_dependance=schedules["branching_age_dependance"],
    )

    if generate_graphics:
        try:
            folder_to_swcs = graphics.collect_swc_folders([os.path.dirname(save_path_parameters)])
            for folder, swc_paths in folder_to_swcs.items():
                graphics.make_movie_for_folder(folder, swc_paths)
        except Exception as e:
            print(f"[ERROR] Failed on folder {folder}: {e}")

    return (n_class, with_branch_dep, "ARFIMA_zero_drift", start_name, n, "done")

if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    
    #SET DIRECTORY WHERE ALL SIMULATIONS WILL BE SAVED HERE
    save_root = os.path.join(os.getcwd(), "simulation_result_exploration")

    n_simu_per_start = 5
    max_t = 500
    dt = 1
    recompute = False
    generate_graphics = True # if True, run with MPLBACKEND=Agg
    MAX_WORKERS = 1  # increase if you want parallel
    FRONT_ONLY = None  # 10
    FRONT_ONLY_BINS = None  # 360
    FRONT_STOP_ENABLED = True
    FRONT_STOP_POSITION = 140.0
    MAX_COMPARTMENTS_STOP_ENABLED = True
    MAX_COMPARTMENTS_STOP_COUNT = 50000
    MIN_SAVE_T_RECOMPUTE = 150
    NEW_BRANCH_WIDTH = 0.001

    # Initialization
    LIST_SIMU_CLASSES = ["class_IV"]

    # warm_start = start from an existing class IV
    START_MODE = "warm_start"  # "warm_start", "ring", "line", or "vertical_branch"
    save_name_root = "ARFIMA_zero_drift_sparse_regression"  # top-level folder for this sweep
    warm_start_path = os.path.join(os.getcwd(), "run_simulations/warm_start_realistic.swc")
    init_mode = "primary"
    boundary_mode = "off"
    boundary_x0 = 0.0
    boundary_y_min = 0.0
    boundary_y_max = 30.0
    if START_MODE == "warm_start":
        with open(warm_start_path, "r", encoding="utf-8") as f:
            swc_txt = f.read()
        start_name = "warm_start"
    elif START_MODE == "ring":
        swc_txt = "ring"
        start_name = "ring"
    elif START_MODE == "line":
        swc_txt = build_vertical_line_swc()
        start_name = "line"
    elif START_MODE == "vertical_branch":
        swc_txt = None
        start_name = "vertical_branch"
        init_mode = "vertical_branch"
        boundary_mode = "hard"
    else:
        raise ValueError(f"Unsupported START_MODE: {START_MODE}")

    if isinstance(swc_txt, str) and swc_txt != "ring":
        swc_txt = force_swc_width(swc_txt, NEW_BRANCH_WIDTH)


    ref_params = build_class_iv_reference_params(max_t=max_t)


    # CONFIGURATION OF PARAMETERS SWEEP
    alpha_values = [0.2, 0.4, 0.6, 0.8, 1.0]


    PAIR_CONFIG = {
        0.4 : [],
        0.6 : [],
        0.8 : [],
        1.0 : [],
    }

    pair_config_bounds = {
        0.4 : (4,-2,3),
        0.6 : (3,-2,3),
        0.8 : (2,-2,3),
        1.0 : (1,-2,3),
    }
   
    lambda_max = 5
    for alpha in PAIR_CONFIG.keys():
        lambda_basis,exp_min,exp_max = pair_config_bounds[alpha]
        for i in range(exp_min,exp_max):
            for j in range(-i -(lambda_max - lambda_basis),2):
                PAIR_CONFIG[alpha].append((2**i,2**(lambda_basis-i-j)))
    print(PAIR_CONFIG)
    tasks = []
    base_seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32).item()

    for n_class in LIST_SIMU_CLASSES:
        for alpha in alpha_values:
            pair_list = PAIR_CONFIG.get(alpha, [])
            for sigma_mult, lambda_mult in pair_list:
                sigma_std = ref_params["sigma_std"] * sigma_mult
                lambda_branch = ref_params["lambda_branch"] * lambda_mult
                if sigma_std <= 0 or lambda_branch <= 0:
                    continue
                tag = (
                    f"alpha_{sanitize_tag(alpha)}"
                    f"__sig_x{sanitize_tag(sigma_mult)}"
                    f"__lam_x{sanitize_tag(lambda_mult)}"
                )
                save_simu_base = os.path.join(save_root, save_name_root, n_class, tag)
                kappa = ref_params["kappa"]

                for n in range(n_simu_per_start):
                    tasks.append(dict(
                        save_simu_base=save_simu_base,
                        n_class=n_class,
                        start_name=start_name,
                        n=n,
                        swc_txt=swc_txt,
                        init_mode=init_mode,
                        boundary_mode=boundary_mode,
                        boundary_x0=boundary_x0,
                        boundary_y_min=boundary_y_min,
                        boundary_y_max=boundary_y_max,
                        front_only=FRONT_ONLY,
                        front_only_bins=FRONT_ONLY_BINS,
                        front_stop_enabled=FRONT_STOP_ENABLED,
                        front_stop_position=FRONT_STOP_POSITION,
                        max_compartments_stop_enabled=MAX_COMPARTMENTS_STOP_ENABLED,
                        max_compartments_stop_count=MAX_COMPARTMENTS_STOP_COUNT,
                        max_t=max_t,
                        dt=dt,
                        recompute=recompute,
                        base_seed=base_seed,
                        generate_graphics=generate_graphics,
                        lambda_branch=lambda_branch,
                        sigma_std=sigma_std,
                        kappa=kappa,
                        alpha=alpha,
                        new_branch_width=NEW_BRANCH_WIDTH,
                        tag=tag,
                        min_save_t_recompute=MIN_SAVE_T_RECOMPUTE,
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
                    n_class, with_branch_dep, pname, start_name, n, status = res
                    print(f"[{pname}] {n_class}/{start_name} run {n}: {status}")
                else:
                    print(res)
    else:
        for t in tasks:
            print(
                t["save_simu_base"],
                f"| alpha={t['alpha']:.3g}"
                f" | lambda={t['lambda_branch']:.4g}, sigma={t['sigma_std']:.4g}, kappa={t['kappa']:.4g}",
            )
            results.append(run_one(t))

    # Summary
    done = sum(1 for r in results if isinstance(r, tuple) and r[-1] == "done")
    skipped = sum(1 for r in results if isinstance(r, tuple) and r[-1] == "skipped")
    errors = [r for r in results if isinstance(r, tuple) and r[0] == "ERROR"]
    print(f"\nSummary: done={done}, skipped={skipped}, errors={len(errors)}, time={datetime.timedelta(seconds=time.time() - start_time)}")
    if errors:
        print("Example error:", errors[0])
