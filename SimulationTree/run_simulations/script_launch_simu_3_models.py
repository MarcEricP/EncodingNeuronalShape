from simulation_tree import run_simu,graphics
import json
import os
import numpy as np
from scipy.special import gamma, rgamma
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
import datetime

def gennorm2laplace(beta_gennorm, scale_gennorm):
    return scale_gennorm * np.sqrt(gamma(3 / beta_gennorm) * rgamma(1 / beta_gennorm) / 2)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def should_skip(res_simu_dir, max_t, recompute):
    if recompute:
        return False
    return os.path.exists(os.path.join(res_simu_dir, f"{max_t}.swc"))

def build_schedules(n_class, with_branch_dep, program_type, max_t,save_name):
    # Defaults
    schedule_branching_lambda = ((0, 300), (0.036, 0.022))
    if with_branch_dep:
        mini, maxi = 0.02, 0.035
        mini_2 = 0.029
        # branching_age_dependance = ((0, 64), (mini / maxi, 1))
        branching_age_dependance = ((0,16,64),(mini/maxi,mini_2/maxi,1))
    else:
        branching_age_dependance = None

    # schedule_contact_penalty = ((0, max_t), (1.2, 1.2))
    if n_class == "class_I":
        schedule_contact_memory_time = ((0, max_t), (3, 3))
        schedule_contact_mean_retraction =  ((0, max_t), (-1.1, -1.1))
        schedule_contact_std_retraction =  ((0, max_t), (0., 0.))
    elif n_class in ["class_IV","class_IV_MT_inh"]:
        schedule_contact_memory_time = ((0, max_t), (5, 5))
        schedule_contact_mean_retraction =  ((0, max_t), (-1.7, -1.7))
        schedule_contact_std_retraction =  ((0, max_t), (0., 0.))

    # Initialize all schedules to None
    schedule_gennorm_beta = None
    schedule_gennorm_loc = None
    schedule_gennorm_scale = None

    d = None
    schedule_arfima_kappa = None
    schedule_arfima_loc = None
    schedule_arfima_scale = None

    schedule_palavalli_von = None
    schedule_palavalli_voff = None
    schedule_palavalli_kon = None
    schedule_palavalli_koff = None

    if program_type == run_simu.PROGRAM_GENNORM:
        if save_name == "GENNORM_IID":
            if n_class == "class_I":
                schedule_gennorm_beta = ((25, 75, 125, 175, 225, 275), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
                schedule_gennorm_loc = ((25, 75, 125, 175, 225, 275), (0.005404, -0.008271, 0.002186, -0.022603, -0.014289, -0.011953))
                schedule_gennorm_std = ((25, 75, 125, 175, 225, 275), (0.725734, 0.690776, 0.692882, 0.618443, 0.569433, 0.508866))
                schedule_gennorm_scale = (schedule_gennorm_std[0], tuple(std / np.sqrt(2.0) for std in schedule_gennorm_std[1]))
            elif n_class == "class_IV":
                schedule_gennorm_beta = ((25, 75, 125, 175, 225, 275), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
                schedule_gennorm_loc = ((25, 75, 125, 175, 225, 275), (0.075622, 0.031648, 0.033263, 0.011507, 0.015589, 0.005736))
                schedule_gennorm_std = ((25, 75, 125, 175, 225, 275), (0.888271, 0.850668, 0.783095, 0.727901, 0.685876, 0.601251))
                schedule_gennorm_scale = (schedule_gennorm_std[0], tuple(std / np.sqrt(2.0) for std in schedule_gennorm_std[1]))
            elif n_class == "class_IV_MT_inh":
                schedule_gennorm_beta = ((25, 75, 125, 175, 225, 275), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
                schedule_gennorm_loc = ((25, 75, 125, 175, 225, 275), (0.077659, 0.053331, 0.029596, 0.010789, 0.009833, 0.024921))
                schedule_gennorm_std = ((25, 75, 125, 175, 225, 275), (1.017549, 0.952861, 0.888166, 0.825471, 0.795185, 0.792715))
                schedule_gennorm_scale = (schedule_gennorm_std[0], tuple(std / np.sqrt(2.0) for std in schedule_gennorm_std[1]))
        if save_name == "GENNORM_IID_zero_drift":
            if n_class == "class_I":
                schedule_gennorm_beta = ((25, 75, 125, 175, 225, 275), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
                schedule_gennorm_loc = ((25, 75, 125, 175, 225, 275), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                schedule_gennorm_std = ((25, 75, 125, 175, 225, 275), (0.725734, 0.690776, 0.692882, 0.618443, 0.569433, 0.508866))
                schedule_gennorm_scale = (schedule_gennorm_std[0], tuple(std / np.sqrt(2.0) for std in schedule_gennorm_std[1]))
            elif n_class == "class_IV":
                schedule_gennorm_beta = ((25, 75, 125, 175, 225, 275), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
                schedule_gennorm_loc = ((25, 75, 125, 175, 225, 275), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                schedule_gennorm_std = ((25, 75, 125, 175, 225, 275), (0.888271, 0.850668, 0.783095, 0.727901, 0.685876, 0.601251))
                schedule_gennorm_scale = (schedule_gennorm_std[0], tuple(std / np.sqrt(2.0) for std in schedule_gennorm_std[1]))
            elif n_class == "class_IV_MT_inh":
                schedule_gennorm_beta = ((25, 75, 125, 175, 225, 275), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
                schedule_gennorm_loc = ((25, 75, 125, 175, 225, 275), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                schedule_gennorm_std = ((25, 75, 125, 175, 225, 275), (1.017549, 0.952861, 0.888166, 0.825471, 0.795185, 0.792715))
                schedule_gennorm_scale = (schedule_gennorm_std[0], tuple(std / np.sqrt(2.0) for std in schedule_gennorm_std[1]))
        


    elif program_type == run_simu.PROGRAM_ARFIMA:
        if save_name == "ARFIMA":
            schedule_arfima_kappa = ((0, max_t), (1, 1))
            

            if n_class == "class_I":
                d = -0.27
                schedule_arfima_loc = ((25, 75, 125, 175, 225, 275), (0.005404, -0.008271, 0.002186, -0.022603, -0.014289, -0.011953))
                schedule_arfima_std = ((25, 75, 125, 175, 225, 275), (0.725734, 0.690776, 0.692882, 0.618443, 0.569433, 0.508866))
                schedule_arfima_scale = (schedule_arfima_std[0], tuple(std / np.sqrt(2.0) for std in schedule_arfima_std[1]))
                  
            elif n_class == "class_IV":
                d = -0.00239#-0.04
                schedule_arfima_loc = ((25, 75, 125, 175, 225, 275), (0.075622, 0.031648, 0.033263, 0.011507, 0.015589, 0.005736))
                schedule_arfima_std = ((25, 75, 125, 175, 225, 275), (0.888271, 0.850668, 0.783095, 0.727901, 0.685876, 0.601251))
                schedule_arfima_scale = (schedule_arfima_std[0], tuple(std / np.sqrt(2.0) for std in schedule_arfima_std[1]))
            elif n_class == "class_IV_MT_inh":
                d = -0.173673#-0.185
                schedule_arfima_loc = ((25, 75, 125, 175, 225, 275), (0.077659, 0.053331, 0.029596, 0.010789, 0.009833, 0.024921))
                schedule_arfima_std = ((25, 75, 125, 175, 225, 275), (1.017549, 0.952861, 0.888166, 0.825471, 0.795185, 0.792715))
                schedule_arfima_scale = (schedule_arfima_std[0], tuple(std / np.sqrt(2.0) for std in schedule_arfima_std[1]))
        if save_name == "ARFIMA_zero_drift":
            schedule_arfima_kappa = ((0, max_t), (1, 1))
            schedule_arfima_loc   = ((0, max_t), (0, 0))

            if n_class == "class_I":
                d = -0.27
                schedule_arfima_std = ((25, 75, 125, 175, 225, 275), (0.725734, 0.690776, 0.692882, 0.618443, 0.569433, 0.508866))
                schedule_arfima_scale = (schedule_arfima_std[0], tuple(std / np.sqrt(2.0) for std in schedule_arfima_std[1]))

            elif n_class == "class_IV":
                d = -0.00239#-0.04
                schedule_arfima_std = ((25, 75, 125, 175, 225, 275), (0.888271, 0.850668, 0.783095, 0.727901, 0.685876, 0.601251))
                schedule_arfima_scale = (schedule_arfima_std[0], tuple(std / np.sqrt(2.0) for std in schedule_arfima_std[1]))
            elif n_class == "class_IV_MT_inh":
                d = -0.175
                schedule_arfima_std = ((25, 75, 125, 175, 225, 275), (1.017549, 0.952861, 0.888166, 0.825471, 0.795185, 0.792715))
                schedule_arfima_scale = (schedule_arfima_std[0], tuple(std / np.sqrt(2.0) for std in schedule_arfima_std[1]))

    elif program_type == run_simu.PROGRAM_PALAVALLI:
        if save_name == "PALAVALLI":
            if n_class == "class_I":
                schedule_palavalli_von  = ((0, 150, 300), (1.56, 1.08, 0.764))
                schedule_palavalli_voff = ((0, 150, 300), (1.2, 0.76, 0.544))
                schedule_palavalli_kon  = ((0, 150, 300), (0.638, 0.466, 0.509))
                schedule_palavalli_koff = ((0, 150, 300), (0.849, 0.692, 0.704))
            elif n_class == "class_IV":
                schedule_palavalli_von  = ((0, 150, 300), (1.28, 1.17, 1.18))
                schedule_palavalli_voff = ((0, 150, 300), (1.03, 1.1, 1.12))
                schedule_palavalli_kon  = ((0, 150, 300), (0.781, 0.653, 0.694))
                schedule_palavalli_koff = ((0, 150, 300), (0.803, 0.654, 0.696))
        elif save_name == "PALAVALLI_zero_drift":
            if n_class == "class_I":
                schedule_palavalli_von  = ((0, 150, 300), (1.56, 1.08, 0.764))
                schedule_palavalli_voff = ((0, 150, 300), (1.2, 0.76, 0.544))
                schedule_palavalli_kon  = ((0, 150, 300), (0.638, 0.466, 0.509))
                schedule_palavalli_koff = ((0, 150, 300), (0.849, 0.692, 0.704))
                drift = [(kn*vn - kf*vf)/(kn+kf) for vn,vf,kn,kf in zip(schedule_palavalli_von[1],schedule_palavalli_voff[1],schedule_palavalli_kon[1],schedule_palavalli_koff[1])]
                schedule_palavalli_von = (schedule_palavalli_von[0],tuple([a-d for a,d in zip(schedule_palavalli_von[1],drift)])) 
                schedule_palavalli_voff = (schedule_palavalli_voff[0],tuple([a+d for a,d in zip(schedule_palavalli_voff[1],drift)])) 
            elif n_class == "class_IV":
                schedule_palavalli_von  = ((0, 150, 300), (1.28, 1.17, 1.18))
                schedule_palavalli_voff = ((0, 150, 300), (1.03, 1.1, 1.12))
                schedule_palavalli_kon  = ((0, 150, 300), (0.781, 0.653, 0.694))
                schedule_palavalli_koff = ((0, 150, 300), (0.803, 0.654, 0.696))
                drift = [(kn*vn - kf*vf)/(kn+kf) for vn,vf,kn,kf in zip(schedule_palavalli_von[1],schedule_palavalli_voff[1],schedule_palavalli_kon[1],schedule_palavalli_koff[1])]
                schedule_palavalli_von = (schedule_palavalli_von[0],tuple([a-d for a,d in zip(schedule_palavalli_von[1],drift)])) 
                schedule_palavalli_voff = (schedule_palavalli_voff[0],tuple([a+d for a,d in zip(schedule_palavalli_voff[1],drift)])) 

    return dict(
        schedule_branching_lambda=schedule_branching_lambda,
        branching_age_dependance=branching_age_dependance,
        schedule_contact_memory_time = schedule_contact_memory_time,
        schedule_contact_mean_retraction =  schedule_contact_mean_retraction,
        schedule_contact_std_retraction =  schedule_contact_std_retraction,
        schedule_gennorm_beta=schedule_gennorm_beta,
        schedule_gennorm_loc=schedule_gennorm_loc,
        schedule_gennorm_scale=schedule_gennorm_scale,
        arfima_d=d,
        schedule_arfima_kappa=schedule_arfima_kappa,
        schedule_arfima_loc=schedule_arfima_loc,
        schedule_arfima_scale=schedule_arfima_scale,
        schedule_palavalli_von=schedule_palavalli_von,
        schedule_palavalli_voff=schedule_palavalli_voff,
        schedule_palavalli_kon=schedule_palavalli_kon,
        schedule_palavalli_koff=schedule_palavalli_koff,
    )

def run_one(task):
    """
    task: dict with all parameters needed for one run
    """
    # Unpack
    (save_simu_base, n_class, with_branch_dep, program_type, start_name, n, n_simu_per_start,
     swc_txt, max_t, dt, recompute, base_seed,generate_graphics,save_name) = (
        task["save_simu_base"], task["n_class"], task["with_branch_dep"], task["program_type"],
        task["start_name"], task["n"], task["n_simu_per_start"], task["swc_txt"],
        task["max_t"], task["dt"], task["recompute"], task["base_seed"],task["generate_graphics"],task["save_name"],
    )

    target_name = start_name + ( "_branch_dep" if with_branch_dep else "" )
    res_simu = os.path.join(save_simu_base, n_class, target_name, str(n), "swc")
    save_path_parameters = os.path.join(save_simu_base, n_class, target_name, str(n), "params.json")

    if should_skip(res_simu, max_t, recompute):
        return (n_class, with_branch_dep, program_type, start_name, n, "skipped")

    ensure_dir(res_simu)
    ensure_dir(os.path.dirname(save_path_parameters))

    schedules = build_schedules(n_class, with_branch_dep, program_type, max_t,save_name)

    # Distinct seed per task for reproducibility across processes
    seed = (base_seed * 10_000_019 + hash((n_class, with_branch_dep, program_type, start_name, n)) ) % 2**32

    profile_time = run_simu.run_sim(
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
        new_branch_length_std=0.,
        new_branch_angle_kappa=1.0,  # 0 = uniform branch angle, inf = dirac
        min_dist_node=0.1,
        program_type=program_type,   
        program_horizon=60,
        # branching and contact
        schedule_branching_lambda=schedules["schedule_branching_lambda"],
        # schedule_contact_penalty=schedules["schedule_contact_penalty"],
        schedule_contact_memory_time = schedules["schedule_contact_memory_time"],
        schedule_contact_mean_retraction =  schedules["schedule_contact_mean_retraction"],
        schedule_contact_std_retraction =  schedules["schedule_contact_std_retraction"],
        # GENNORM
        schedule_gennorm_beta=schedules["schedule_gennorm_beta"],
        schedule_gennorm_loc=schedules["schedule_gennorm_loc"],
        schedule_gennorm_scale=schedules["schedule_gennorm_scale"],
        # ARFIMA
        arfima_d=schedules["arfima_d"],
        schedule_arfima_kappa=schedules["schedule_arfima_kappa"],
        schedule_arfima_loc=schedules["schedule_arfima_loc"],
        schedule_arfima_scale=schedules["schedule_arfima_scale"],
        # PALAVALLI
        schedule_palavalli_von=schedules["schedule_palavalli_von"],
        schedule_palavalli_voff=schedules["schedule_palavalli_voff"],
        schedule_palavalli_kon=schedules["schedule_palavalli_kon"],
        schedule_palavalli_koff=schedules["schedule_palavalli_koff"],
        # Branching age dependence
        branching_age_dependance=schedules["branching_age_dependance"],
    )

    if generate_graphics:
        print("find swc")
        folder_to_swcs = graphics.collect_swc_folders([os.path.dirname(save_path_parameters)])
        for folder, swc_paths in folder_to_swcs.items():
            print("make movie")
            try:
                graphics.make_movie_for_folder(folder, swc_paths)
            except Exception as e:
                print(f"[ERROR] Failed on folder {folder}: {e}")
                
    return (n_class, with_branch_dep, program_type, start_name, n, "done")

if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    with open(os.path.join(os.getcwd(), "run_simulations/all_swc.json"), "r") as f:
        all_starting_points = json.load(f)
        print(all_starting_points.keys())

    save_simu_base_all = os.path.join(os.getcwd(), "simulation_result_3_models")
    n_simu_per_start = 10#10
    max_t = 300
    dt = 1
    recompute = False
    
    generate_graphics = True

    list_program_type = [  # (program_type, save_dir_name)
        (run_simu.PROGRAM_GENNORM, "GENNORM_IID"),
        (run_simu.PROGRAM_ARFIMA,   "ARFIMA"),
        (run_simu.PROGRAM_PALAVALLI,"PALAVALLI"),
        (run_simu.PROGRAM_ARFIMA,   "ARFIMA_zero_drift"),
        (run_simu.PROGRAM_GENNORM, "GENNORM_IID_zero_drift"),
        (run_simu.PROGRAM_PALAVALLI,"PALAVALLI_zero_drift"),
    ]

    # Make a flat list of tasks
    tasks = []
    base_seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32).item()

    for (program_type, save_name) in list_program_type:
        if program_type in [run_simu.PROGRAM_PALAVALLI]:
            list_simu = list_simu = [  # neuron_class, with_branch_age_dependency
                        ("class_I", False),
                        ("class_IV", False),
                    ]
        elif program_type in [run_simu.PROGRAM_GENNORM]:
            list_simu = [  # neuron_class, with_branch_age_dependency
                    ("class_I", False),
                    ("class_IV", False),
                    ("class_IV_MT_inh",False)
                ]
        elif program_type in [run_simu.PROGRAM_ARFIMA]:
            list_simu = [  # neuron_class, with_branch_age_dependency
                    ("class_I", False),
                    ("class_IV", False),
                    ("class_IV_MT_inh",False)
                ]
        for (n_class, with_branch_dep) in list_simu:
            save_simu_base = os.path.join(save_simu_base_all, save_name)
            # Loop through all start points of this class
            n_class_start = n_class

            for start_name, swc_txt in all_starting_points[n_class_start].items():
                for n in range(n_simu_per_start):
                    temp_generate_graphics = False
                    if generate_graphics :
                        temp_generate_graphics = True
                    tasks.append(dict(
                        save_simu_base=save_simu_base,
                        n_class=n_class,
                        with_branch_dep=with_branch_dep,
                        program_type=program_type,
                        start_name=start_name,
                        n=n,
                        n_simu_per_start=n_simu_per_start,
                        swc_txt=swc_txt,
                        max_t=max_t,
                        dt=dt,
                        recompute=recompute,
                        base_seed=base_seed,
                        generate_graphics=temp_generate_graphics,
                        save_name = save_name,
                    ))


    MAX_WORKERS = max(1, os.cpu_count()-5)
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
                    n_class, with_branch_dep, program_type, start_name, n, status = res
                    dep = "branch_dep" if with_branch_dep else "no_branch_dep"
                    pname = {run_simu.PROGRAM_GENNORM: "GENNORM",
                            run_simu.PROGRAM_ARFIMA: "ARFIMA",
                            run_simu.PROGRAM_PALAVALLI: "PALAVALLI"}.get(program_type, str(program_type))
                    print(f"[{pname}] {n_class}/{dep}/{start_name} run {n}: {status}")
                else:
                    print(res)
    else:
        for t in tasks:
            print(t["save_simu_base"])
            results.append(run_one(t))

    done = sum(1 for r in results if isinstance(r, tuple) and r[-1] == "done")
    skipped = sum(1 for r in results if isinstance(r, tuple) and r[-1] == "skipped")
    errors = [r for r in results if isinstance(r, tuple) and r[0] == "ERROR"]
    print(f"\nSummary: done={done}, skipped={skipped}, errors={len(errors)}, time={datetime.timedelta(seconds=time.time() - start_time)}")
    if errors:
        print("Example error:", errors[0])
