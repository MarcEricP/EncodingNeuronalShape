import os
from numba import types
from numba.typed import Dict, List
import numpy as np
import tqdm
import json
import time

from simulation_tree.sim_core import (
    init_state_primary,
    init_state_ring,
    init_state_vertical_branch,
    init_state_from_swc_string,
    compact_state,
    step_kernel,
    init_angle_bins,
    recompute_max_dist_alive,
    recompute_max_r_per_bin,
)
from .branch_program import (
    PROGRAM_ARFIMA,
    PROGRAM_PALAVALLI,
    PROGRAM_GENNORM,
    get_next_increment,
)


from . import swc
from . import side_branching as sb
import time

def interp_none(schedule,time_simu):
    if schedule is None:
        return(np.zeros_like(time_simu))
    else:
        return(np.interp(time_simu,schedule[0],schedule[1]))
    
def to_numba_float_list(py_list):
    nb_list = List.empty_list(types.float64)
    for x in py_list:
        nb_list.append(float(x))  # ensures dtype is float64
    return nb_list   

def to_numba_bool_list(py_list):
    nb_list = List.empty_list(types.boolean)
    for x in py_list:
        nb_list.append(bool(x))
    return nb_list
 
#Save parameters to JSON if requested
def _to_serializable(x):
    # numba typed containers are not serializable; schedules and tuples handled
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    # Fallback to string
    return str(x)

f64 = types.float64
i64 = types.int64
arr_f64 = types.float64[:] 

def run_sim(max_t=50.0, dt=1, seed=1234,
            swc_string=None,
            front_only=None,
            front_only_bins=None,
            front_stop_enabled=False,
            front_stop_position=140.0,
            max_compartments_stop_enabled=False,
            max_compartments_stop_count=30000,
            box_size_contact=0.5,
            max_length_merge=0.25,
            max_merge_angle_deg=180.0,
            new_branch_width=0.5,
            proba_overlap=0.0,
            persistence_length=30.0,
            new_branch_length_mean=0.3,
            new_branch_length_std=0.1,
            new_branch_angle_kappa = 0,#0 = uniform distirbution, np.inf = dirac in 0
            # branching_angle_kappa=1.0,  # unused here uniform branch
            min_dist_node=0.1,
            degree_exempt_collision=1,
            save_path = None,
            program_type = PROGRAM_ARFIMA,
            program_horizon = 100,

            #Scheduled parameters (each can be None or (list_t, list_values))
            #BRANCHING
            schedule_branching_lambda = None,
            #CONTACT
            # schedule_contact_penalty = None,
            schedule_contact_memory_time = None,#time for which the contact memory is kept
            schedule_contact_mean_retraction = None,#mean retraction after contact (should be negative)
            schedule_contact_std_retraction = None,

            #ARFIMA
            arfima_d = None,#this one is a float, not an array
            schedule_arfima_kappa = None,
            schedule_arfima_loc = None,
            schedule_arfima_scale = None,
            arfima_p_mb:float = 0,
            arfima_d_mb:float = 0,

            #PALAVALLI
            schedule_palavalli_von = None,
            schedule_palavalli_voff = None,
            schedule_palavalli_kon = None,
            schedule_palavalli_koff = None,

            #GENNORM
            schedule_gennorm_beta = None,
            schedule_gennorm_loc = None,
            schedule_gennorm_scale = None,

            branching_age_dependance=None,
            save_path_parameters=None,
            save_every_n_steps=1,
            print_profile_summary=False,
            init_mode="primary",
            boundary_mode="off",
            boundary_x0=0.0,
            boundary_y_min=0.0,
            boundary_y_max=30.0,
            compaction_min_interval_steps=100
            ):
    """
    Notes on schedules:
      - Each schedule_... accepts a tuple (list_t, list_values) of equal length.
      - Values are linearly interpolated in time.
    Init/boundary:
      - init_mode: "primary" (default) or "vertical_branch".
      - boundary_mode: "off" or "hard".
    Front-only:
      - front_only: None (off) or float. If set, only tips within the
        band [max_dist - front_only, max_dist] (dist from initial centroid)
        are updated; deeper tips are skipped for speed.
      - front_only_bins: None/0 (global max_dist) or int>0 to enable
        directional bins, max_dist is computed per angle bin each step.
    Front-stop:
      - front_stop_enabled: if True, stop early when the farthest alive
        compartment (from the initial centroid) reaches front_stop_position.
      - front_stop_position: distance threshold for early stop.
    Compartments-stop:
      - max_compartments_stop_enabled: if True, stop early when the number
        of alive compartments reaches max_compartments_stop_count.
      - max_compartments_stop_count: threshold for early stop.
    Output:
      - save_every_n_steps: when save_path is set, save SWC every N iterations.
        Use 1 to keep previous behavior.
    Profiling:
      - print_profile_summary: print average timings per loop block at the end.
    Merge-after-retraction:
      - max_length_merge controls the max gp->child distance allowed for splicing.
      - max_merge_angle_deg controls alignment (smaller = stricter).
    Compaction:
      - Dead-node compaction runs when len(alive_nodes)/len(alive) < 0.6.
      - compaction_min_interval_steps enforces a minimum number of iterations
        between two compactions.
    """
    np.random.seed(seed)


    _params = dict(locals())
    if save_path_parameters is not None:
        os.makedirs(os.path.dirname(save_path_parameters), exist_ok=True)
        with open(save_path_parameters, "w", encoding="utf-8") as _fjson:
            json.dump({k: _to_serializable(v) for k, v in _params.items()}, _fjson, ensure_ascii=False, indent=2)


    # init tree state
    if swc_string == "ring":
        res = init_state_ring(box_size_contact, new_branch_width, 2)
    elif swc_string is not None:
        res = init_state_from_swc_string(box_size_contact, new_branch_width, swc_string)
    elif init_mode == "vertical_branch":
        res = init_state_vertical_branch(
            box_size_contact, new_branch_width,
            boundary_x0, boundary_y_min, boundary_y_max
        )
    else:
        res = init_state_primary(box_size_contact, new_branch_width)
    (unit_size, elem2box, box2elem,
     position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
     alive_nodes, alive_pos,
     tip_list, simulation_time, max_tip_id, root_idx,
     center, dist_to_center, angle_bin, max_dist_list) = res 
    
    # Per-tip-id program buffers
    prog_data = dict([])

    if save_path is not None:
        os.makedirs(save_path,exist_ok=True)

    front_only_bins_int = int(front_only_bins) if front_only_bins is not None else 0
    if front_only_bins_int < 0:
        front_only_bins_int = 0
    if front_only_bins_int > 0:
        init_angle_bins(center, position, angle_bin, front_only_bins_int)
        max_r_per_bin = np.zeros(front_only_bins_int, dtype=np.float64)
    else:
        max_r_per_bin = np.zeros(1, dtype=np.float64)

    # main loop
    num_iter = int(np.round(max_t/dt))
    time_simu = np.arange(num_iter + 2)*dt
    # transform scheduled parameters in arrays with the value of the parameter at each time step of the simulation
    branching_lambda_of_time = interp_none(schedule_branching_lambda, time_simu)
    if branching_age_dependance is None:
        branching_age_ages = np.zeros(0, dtype=np.float64)
        branching_age_mults = np.zeros(0, dtype=np.float64)
        use_branching_age_dependence = False
    else:
        try:
            _ages, _mults = branching_age_dependance
            branching_age_ages = np.asarray(_ages, dtype=np.float64)
            branching_age_mults = np.asarray(_mults, dtype=np.float64)
            if branching_age_ages.shape[0] == 0 or branching_age_mults.shape[0] == 0:
                use_branching_age_dependence = False
                branching_age_ages = np.zeros(0, dtype=np.float64)
                branching_age_mults = np.zeros(0, dtype=np.float64)
            else:
                use_branching_age_dependence = True
        except Exception:
            branching_age_ages = np.zeros(0, dtype=np.float64)
            branching_age_mults = np.zeros(0, dtype=np.float64)
            use_branching_age_dependence = False

    # CONTACT
    # contact_penalty_of_time  = interp_none(schedule_contact_penalty, time_simu)
    contact_memory_time = interp_none(schedule_contact_memory_time, time_simu)#time for which the contact memory is kept
    contact_mean_retraction = interp_none(schedule_contact_mean_retraction, time_simu)#mean retraction after contact (should be negative)
    contact_std_retraction = interp_none(schedule_contact_std_retraction, time_simu)

    # ARFIMA
    arfima_kappa_of_time = interp_none(schedule_arfima_kappa, time_simu)
    arfima_loc_of_time   = interp_none(schedule_arfima_loc,   time_simu)
    arfima_scale_of_time = interp_none(schedule_arfima_scale, time_simu)

    # PALAVALLI
    palavalli_von_of_time  = interp_none(schedule_palavalli_von,  time_simu)
    palavalli_voff_of_time = interp_none(schedule_palavalli_voff, time_simu)
    palavalli_kon_of_time  = interp_none(schedule_palavalli_kon,  time_simu)
    palavalli_koff_of_time = interp_none(schedule_palavalli_koff, time_simu)

    # GENNORM
    gennorm_beta_of_time  = interp_none(schedule_gennorm_beta,  time_simu)
    gennorm_loc_of_time   = interp_none(schedule_gennorm_loc,   time_simu)
    gennorm_scale_of_time = interp_none(schedule_gennorm_scale, time_simu)

    # save_increments = dict([])
    profile_time = []
    boundary_mode_int = {"off": 0, "hard": 1}.get(str(boundary_mode).lower(), 0)
    front_stop_enabled = bool(front_stop_enabled)
    front_stop_position = float(front_stop_position)
    max_compartments_stop_enabled = bool(max_compartments_stop_enabled)
    max_compartments_stop_count = int(max_compartments_stop_count)
    max_merge_angle_rad = float(np.deg2rad(float(max_merge_angle_deg)))
    save_every_n_steps = max(1, int(save_every_n_steps))
    compaction_min_interval_steps = max(0, int(compaction_min_interval_steps))
    last_compaction_iter = -compaction_min_interval_steps
    for i in tqdm.trange(num_iter):
        temp_profile_time = []
        #print("start loop", time.time() - t)
        t = time.time()
        simulation_time = time_simu[i]
        should_save_step = (save_path is not None) and ((i % save_every_n_steps) == 0)
        if should_save_step:
            # Export SWC with lineage ids (tip_id) per compartment
            save_is_contact = np.zeros_like(tip_id)
            for idx in tip_list:
                save_is_contact[idx] = 1
            swc.save_tree_to_swc_extended(
                os.path.join(save_path,f"{i}.swc"),
                position, width, parent, alive,
                birth_time, tip_id,save_is_contact,
                header_comment = f"time (min): {simulation_time}"
            )

        
        if front_stop_enabled:
            recompute_max_dist_alive(dist_to_center, alive, max_dist_list)
            if max_dist_list[0] >= front_stop_position:
                break
        if max_compartments_stop_enabled:
            if len(alive_nodes) >= max_compartments_stop_count:
                break

        temp_profile_time.append(time.time()-t)
        t = time.time()
        # Derive current tip lineage ids in same order as tip_list
        tip_keys = [tip_id[idx] for idx in tip_list]
        # print(tip_list)
        # if i < 5 and i > 0:
        #     print(tip_keys)
        #     print(prog_data.keys())
        #     print(step_increments)
        step_increments,prog_data = get_next_increment(
            program_type,
            tip_keys,          # numba.List of tip_id (lineage)
            prog_data,         # Dict[int -> float64[:]]
            is_contact,  
            # common
            program_horizon,
            dt,
            # contact:
            # contact_penalty_of_time[i:i+program_horizon],
            contact_memory_time[i:i+program_horizon],#time for which the contact memory is kept
            contact_mean_retraction[i:i+program_horizon],#mean retraction after contact
            contact_std_retraction[i:i+program_horizon],#std retraction after contact
            # GENORM PARAMS
            gennorm_beta_of_time[i:i+program_horizon],#array with the value taken by the parameter for the next horizon steps
            gennorm_loc_of_time[i:i+program_horizon] ,#array with the value taken by the parameter for the next horizon steps
            gennorm_scale_of_time[i:i+program_horizon] ,#array with the value taken by the parameter for the next horizon steps
            # ARFIMA params/state
            arfima_d,
            arfima_kappa_of_time[i:i+program_horizon],#array with the value taken by the parameter for the next horizon steps
            arfima_scale_of_time[i:i+program_horizon],#array with the value taken by the parameter for the next horizon steps
            arfima_loc_of_time[i:i+program_horizon],
            arfima_p_mb,
            arfima_d_mb,
            # Palavalli telegraph params/state
            palavalli_von_of_time[i:i+program_horizon],#array with the value taken by the parameter for the next horizon steps
            palavalli_voff_of_time[i:i+program_horizon],#array with the value taken by the parameter for the next horizon steps
            palavalli_kon_of_time[i:i+program_horizon],#array with the value taken by the parameter for the next horizon steps
            palavalli_koff_of_time[i:i+program_horizon],#array with the value taken by the parameter for the next horizon steps
            i==0,#if it is the first time point initialization
        )
        temp_profile_time.append(time.time()-t)
        t = time.time()
    
        # Branching decision
        if front_only is None:
            front_only_value = -1.0
        else:
            if front_only_bins_int > 0:
                recompute_max_r_per_bin(dist_to_center, angle_bin, alive, front_only_bins_int, max_r_per_bin)
            else:
                recompute_max_dist_alive(dist_to_center, alive, max_dist_list)
            front_only_value = float(front_only)
        res = sb.decide_branches(
                position, parent, alive, alive_nodes,
                dt, branching_lambda_of_time[i],
                new_branch_length_mean, new_branch_length_std,
                new_branch_angle_kappa,
                max_tip_id,
                birth_time,
                simulation_time,
                branching_age_ages,
                branching_age_mults,
                use_branching_age_dependence,
                dist_to_center,
                angle_bin,
                front_only_value,
                front_only_bins_int,
                max_r_per_bin,
                max_dist_list,
            )
        (branch_parent_nodes, branch_prop_splits, branch_angles, branch_lengths,branch_new_tip_ids, max_tip_id) = res
        temp_profile_time.append(time.time()-t)
        t = time.time()

        # Simulation step
        tip_list, max_tip_id, is_contact = step_kernel(
            unit_size, elem2box, box2elem, max_length_merge,
            position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
            alive_nodes, alive_pos,
            center, dist_to_center, angle_bin, max_dist_list,
            tip_list,
            dt,
            simulation_time,
            proba_overlap,
            persistence_length,
            new_branch_width,
            min_dist_node,
            degree_exempt_collision,
            max_tip_id,
            step_increments,
            branch_parent_nodes,
            branch_prop_splits,
            branch_angles,
            branch_lengths,
            branch_new_tip_ids,
            boundary_mode_int,
            boundary_x0,
            boundary_y_min,
            boundary_y_max,
            front_only_value,
            front_only_bins_int,
            max_r_per_bin,
            max_merge_angle_rad,
        )

        ratio_trigger = (len(alive) > 0) and (len(alive_nodes) / len(alive) < 0.6)
        interval_ok = (i - last_compaction_iter) >= compaction_min_interval_steps
        if ratio_trigger and interval_ok:
            (unit_size, elem2box, box2elem,
             position, width, birth_time, tip_id, _is_contact_nodes, is_protected_scaffold, parent, children, alive,
             alive_nodes, alive_pos, tip_list, root_idx,
             dist_to_center, angle_bin, max_dist_list) = compact_state(
                unit_size,
                position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                alive_nodes, alive_pos,
                tip_list,
                center,
                front_only_bins_int,
            )
            # contact flags are per-tip in this simulation loop; reset after reindexing
            is_contact = to_numba_bool_list([False] * len(tip_list))
            # drop extinct lineages from program cache
            active_tip_ids = set(int(tip_id[idx]) for idx in tip_list)
            prog_data = {k: v for k, v in prog_data.items() if int(k) in active_tip_ids}
            last_compaction_iter = i

        simulation_time += dt
        temp_profile_time.append(time.time()-t)
        t = time.time()

        profile_time.append(temp_profile_time)
        if front_stop_enabled:
            recompute_max_dist_alive(dist_to_center, alive, max_dist_list)
            if max_dist_list[0] >= front_stop_position:
                break

        # # Save increments per lineage
        # for k, key in enumerate(tip_keys):
        #     if key not in save_increments:
        #         save_increments[key] = []
        #     if (len(save_increments[key]) > 0 and (np.isnan(save_increments[key][-1]))):
        #         save_increments[key].append(step_increments[k])
        #     else:
        #         save_increments[key].append(np.nan)

    save_is_contact = np.zeros_like(tip_id)
    for idx in tip_list:
        save_is_contact[idx] = 1
    # Export SWC with lineage ids (tip_id) per compartment
    if save_path is not None:
        swc.save_tree_to_swc_extended(
            os.path.join(save_path,f"{i+1}.swc"),
            position, width, parent, alive,
            birth_time, tip_id,save_is_contact,
            header_comment = None
        )
    # import matplotlib.pyplot as plt
    # for k in save_increments.keys():
    #     if len(save_increments[k])>8:
    #         plt.plot(np.cumsum(save_increments[k]),alpha=0.2)
    # plt.show()
    profile_time = np.array(profile_time)
    if print_profile_summary and profile_time.size > 0:
        avg = np.mean(profile_time, axis=0)
        total = float(np.sum(avg))
        print(
            "[PROFILE] avg_step_seconds "
            f"| save+checks={avg[0]:.6f} "
            f"| increments={avg[1]:.6f} "
            f"| branching_decision={avg[2]:.6f} "
            f"| kernel+compaction={avg[3]:.6f} "
            f"| total={total:.6f} "
            f"| n_steps={profile_time.shape[0]}"
        )
    return profile_time

if __name__ == "__main__":
    swc_txt = """\
            # id type x y z radius parent [birth_time width tip_id]
            1 1 0.0 0.0 0 0.25 -1
            2 3 15.0 0.0 0 0.20 1
            3 3 7.0 7.0 0 0.20 1
            4 3 -8.0 7.0 0 0.20 1
            5 3 -10 4 0 0.2 4
            6 3 -4 10 0 0.2 4
            """
    for d,save_d,sigma,program in zip(
        [-1/4,0,-0.05,0],
        ["cI","diff_cI","CIV","diff_cIV"],
        [np.sqrt(2)/4,np.sqrt(2)/4,np.sqrt(2)/2,np.sqrt(2)/2],
        [PROGRAM_PALAVALLI,PROGRAM_ARFIMA,PROGRAM_ARFIMA,PROGRAM_ARFIMA],#[PROGRAM_PALAVALLI,PROGRAM_PALAVALLI,PROGRAM_PALAVALLI,PROGRAM_PALAVALLI]#[PROGRAM_ARFIMA,PROGRAM_PALAVALLI,PROGRAM_ARFIMA,PROGRAM_PALAVALLI]
    ):
        res_simu = os.path.join(os.getcwd(),f"example/test_simu/{save_d}/swc")
        save_path_parameters = os.path.join(os.getcwd(),f"example/test_simu/{save_d}/params.json")
        _ = run_sim(save_path = res_simu, max_t = 500, new_branch_width=0.5, d=d,
                    program_type=program,
                    new_branch_length_mean=0.1, new_branch_length_std=0,
                    contact_penalty=1.2, iid_sigma=sigma, program_horizon=60, M=128, seed=421,branching_lambda=0.02,swc_string = swc_txt,
                    schedule_branching_lambda=((0,500),(0.035,0.015)),schedule_arfima_mu=((0,500),(0.2, -0.2)),
                    palavalli_von=1,
                    palavalli_voff=0.7,
                    palavalli_kon=0.7,
                    palavalli_koff=1,
                    branching_age_dependance=None,
                    save_path_parameters=save_path_parameters)
