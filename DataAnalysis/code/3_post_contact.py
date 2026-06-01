import os
import pandas
import matplotlib.pyplot as plt
import sys
from pathlib import Path


from dendrogenesis import project_style
import numpy as np
import tqdm
from scipy import stats
from dendrogenesis import load_tracks

project_style.set_style()
# seaborn.set_theme(context="talk", style="ticks")
# plt.rcParams.update({'font.size': 22})

# --- helpers for consistent sizing & saving ---
basic_figsize = (project_style.mm_to_in(55),project_style.mm_to_in(50))
share_axis = "all"  # for figures that used 'all' originally
SEED = 2

def make_figsize(nrows, ncols):
    return (1.6*basic_figsize[0] , 1.8*basic_figsize[1] * nrows/ncols)


def _safe_name(s):
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(s)).strip("_")

def save_all_ext(fig, path_wo_ext, dpi=300):
    for ext in [".png", ".svg", ".pdf"]:
        fig.savefig(f"{path_wo_ext}{ext}", dpi=dpi)


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


def _welch_summary(a, b, sidedness="two-sided", alpha=0.05):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n_a = int(a.size)
    n_b = int(b.size)
    mean_a = float(np.mean(a)) if n_a else np.nan
    mean_b = float(np.mean(b)) if n_b else np.nan
    se_a = float(np.std(a, ddof=1) / np.sqrt(n_a)) if n_a > 1 else np.nan
    se_b = float(np.std(b, ddof=1) / np.sqrt(n_b)) if n_b > 1 else np.nan
    mean_diff = mean_a - mean_b if (np.isfinite(mean_a) and np.isfinite(mean_b)) else np.nan
    if n_a < 2 or n_b < 2:
        return {
            "mean_a": mean_a, "se_a": se_a, "n_a": n_a,
            "mean_b": mean_b, "se_b": se_b, "n_b": n_b,
            "statistic": np.nan, "p_value": np.nan, "welch_df": np.nan,
            "mean_diff": mean_diff, "mean_diff_ci95_low": np.nan, "mean_diff_ci95_high": np.nan,
            "sidedness": sidedness,
        }
    s2a = float(np.var(a, ddof=1))
    s2b = float(np.var(b, ddof=1))
    v1 = s2a / n_a
    v2 = s2b / n_b
    se_diff = float(np.sqrt(v1 + v2))
    denom = (v1 ** 2) / (n_a - 1) + (v2 ** 2) / (n_b - 1)
    welch_df = float(((v1 + v2) ** 2) / denom) if denom > 0 else np.nan
    if not (np.isfinite(se_diff) and se_diff > 0 and np.isfinite(welch_df)):
        return {
            "mean_a": mean_a, "se_a": se_a, "n_a": n_a,
            "mean_b": mean_b, "se_b": se_b, "n_b": n_b,
            "statistic": np.nan, "p_value": np.nan, "welch_df": welch_df,
            "mean_diff": mean_diff, "mean_diff_ci95_low": np.nan, "mean_diff_ci95_high": np.nan,
            "sidedness": sidedness,
        }
    t_stat = float(mean_diff / se_diff)
    if sidedness == "one-sided":
        p_value = float(stats.t.sf(t_stat, welch_df))
    else:
        p_value = float(2 * stats.t.sf(abs(t_stat), welch_df))
    tcrit = float(stats.t.ppf(1 - alpha / 2.0, welch_df))
    return {
        "mean_a": mean_a, "se_a": se_a, "n_a": n_a,
        "mean_b": mean_b, "se_b": se_b, "n_b": n_b,
        "statistic": t_stat, "p_value": p_value, "welch_df": welch_df,
        "mean_diff": mean_diff,
        "mean_diff_ci95_low": float(mean_diff - tcrit * se_diff),
        "mean_diff_ci95_high": float(mean_diff + tcrit * se_diff),
        "sidedness": sidedness,
    }

def memtime_mean_increment(trajs, dt=1.0, B=200, sustain=3, min_n=5, seed=SEED, max_lag=None):
    """
    Minimal memory-time via mean post-contact increment.

    Inputs
    ------
    trajs   : list of 1D arrays (aligned at contact; length>=2)
    dt      : sampling step
    B       : bootstrap resamples over trajectories (cluster bootstrap)
    sustain : need this many consecutive lags with CI covering 0
    min_n   : min #trajectories needed at a lag
    seed    : RNG seed
    max_lag : optional cap on lags (int)

    Returns (minimal)
    -------
    dict with keys:
      lags        : int array of lags k>=1
      mean        : mean increment at lag k
      se          : bootstrap standard error at lag k
      ci_lo/hi    : 95% band per lag (mean ± 1.96*se)
      t_mem_mean  : time when band first contains 0 and then stays for `sustain` lags
      ci95_t      : ~95% CI for t_mem_mean from bootstrap (plug-in), or (nan,nan) if undefined
    """
    rng = np.random.default_rng(seed)

    # -> increments
    incs = []
    for x in trajs:
        if x is None or len(x) < 2: continue
        dx = np.diff(np.asarray(x, float))
        if dx.size >= 1: incs.append(dx)
    if not incs:  # no data
        return dict(lags=np.array([]), mean=np.array([]), se=np.array([]),
                    ci_lo=np.array([]), ci_hi=np.array([]),
                    t_mem_mean=np.nan, ci95_t=(np.nan, np.nan))

    # per-lag samples
    Kmax = max(len(dx) for dx in incs) - 0
    if max_lag is not None:
        Kmax = min(Kmax, int(max_lag)+1)
    lags, samples = [], []
    for k in range(1, Kmax):
        vals = [dx[k] for dx in incs if len(dx) > k]
        if len(vals) >= min_n:
            lags.append(k)
            samples.append(np.asarray(vals, float))
    if not lags:
        return dict(lags=np.array([]), mean=np.array([]), se=np.array([]),
                    ci_lo=np.array([]), ci_hi=np.array([]),
                    t_mem_mean=np.nan, ci95_t=(np.nan, np.nan))
    lags = np.asarray(lags, int)

    # full-sample mean per lag
    mean = np.array([v.mean() for v in samples])

    # cluster bootstrap SE per lag
    T = len(incs)
    B = int(B)
    mu_boot = np.full((B, len(lags)), np.nan)
    for b in tqdm.trange(B):
        idx = rng.integers(0, T, size=T)   # resample trajectories
        subset = [incs[j] for j in idx]
        for i, k in enumerate(lags):
            vals = [dx[k] for dx in subset if len(dx) > k]
            if len(vals) >= min_n:
                mu_boot[b, i] = np.mean(vals)
    se = np.nanstd(mu_boot, axis=0, ddof=1)
    ci_lo, ci_hi = mean - 1.96*se, mean + 1.96*se

    # memory time: first sustained inclusion of 0 in CI
    def first_sustained_in(ci_lo, ci_hi, sustain):
        inside = (ci_lo <= 0.0) & (0.0 <= ci_hi)
        run = 0
        for i, ok in enumerate(inside):
            run = run + 1 if ok else 0
            if run >= sustain:
                return lags[i - sustain + 1] * dt
        return np.nan

    t_mem_mean = first_sustained_in(ci_lo, ci_hi, sustain)

    # rough CI for t_mem_mean: reuse mu_boot with a normal band around each resample
    t_b = []
    for b in range(B):
        mu_b = mu_boot[b]
        if np.all(np.isnan(mu_b)):
            continue
        # plug-in band around this resample (same se used for all)
        lo_b = mu_b - 1.96*se
        hi_b = mu_b + 1.96*se
        t_b.append(first_sustained_in(lo_b, hi_b, sustain))
    t_b = np.asarray([t for t in t_b if np.isfinite(t)], float)
    ci95_t = (np.nan, np.nan) if t_b.size == 0 else tuple(np.percentile(t_b, [2.5, 97.5]).astype(float))

    return dict(lags=lags, mean=mean, se=se, ci_lo=ci_lo, ci_hi=ci_hi,
                t_mem_mean=t_mem_mean, ci95_t=ci95_t)

if __name__ == "__main__":
    base_dir_candidates = [
        "/mnt/c/Users/menir/Documents/0000NeuronsData/clean_movies_1_min",
    ]
    base_dir = None
    for _p in base_dir_candidates:
        if os.path.isdir(_p):
            base_dir = _p
            break

    pixels_per_microns = 16
    repo_root = Path(__file__).resolve().parents[2]
    result_dir = os.path.join(repo_root, "DataAnalysis", "Result", "3_post_contact")
    plots_dir = os.path.join(result_dir, "plots")
    raw_data_dir = os.path.join(result_dir, "raw_data")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)

    hours_array = np.array([13, 16, 18, 20])  # np.array([13,16,18,19,20])

    dataset_post_contact = dict([])
    if base_dir is not None:
        dataset_path = base_dir + "/branch_tracking/all_branches_length.pickle"
        dataset = load_tracks.load_dataset_from_pickle(dataset_path)
        min_len = 0
        max_incr = 3
        dataset = load_tracks.filter_tracks(dataset, max_incr, min_len, None)
        for n_class in dataset.keys():
            for n_name in dataset[n_class].keys():
                post_contact_file = os.path.join(base_dir, n_class, n_name, "flow_track_opened_skel", "post_contact.txt")
                if os.path.exists(post_contact_file):
                    with open(post_contact_file, "r") as f:
                        content = f.read()
                    if not (n_class in dataset_post_contact.keys()):
                        dataset_post_contact[n_class] = dict([])
                    if not (n_name in dataset_post_contact[n_class].keys()):
                        dataset_post_contact[n_class][n_name] = dict([])
                    lines = content.split("\n")[1:]
                    for l in lines:
                        try:
                            cols = l.split(",")
                            branch = int(cols[0].replace(" ", ""))
                            t_start_contact = int(cols[1].replace(" ", "")) - 1
                            t_end_contact = int(cols[2].replace(" ", "")) - 1
                            track_time = np.asarray(dataset[n_class][n_name][branch]["time"], dtype=float)
                            track_len = np.asarray(dataset[n_class][n_name][branch]["length"], dtype=float)
                            dataset_post_contact[n_class][n_name][branch] = {
                                "time": dataset[n_class][n_name][branch]["time"],
                                "length": dataset[n_class][n_name][branch]["length"],
                                "t_start_contact": t_start_contact,
                                "t_end_contact": t_end_contact,
                                "n_class": n_class,
                                "n_name": n_name,
                                "raw_table_df": pandas.DataFrame({
                                    "time after 16 h AEL (min)": track_time,
                                    "length to reference branch point (µm)": track_len,
                                    "t_start_contact": np.full(track_time.shape[0], t_start_contact, dtype=float),
                                    "t_end_contact": np.full(track_time.shape[0], t_end_contact, dtype=float),
                                }),
                            }
                        except Exception:
                            pass
    else:
        xlsx_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".xlsx")]
        class_prefixes = ["class_IV_MT_inh_", "class_IV_", "class_I_"]
        for fname in xlsx_files:
            root = fname[:-5]
            n_class = None
            n_name = None
            for pref in class_prefixes:
                if root.startswith(pref):
                    n_class = pref[:-1]
                    n_name = root[len(pref):]
                    break
            if n_class is None or n_name is None:
                continue
            xlsx_path = os.path.join(raw_data_dir, fname)
            xl = pandas.ExcelFile(xlsx_path)
            for sheet in xl.sheet_names:
                df_sheet = pandas.read_excel(xlsx_path, sheet_name=sheet)
                required = ["time after 16 h AEL (min)", "length to reference branch point (µm)"]
                if any(c not in df_sheet.columns for c in required):
                    continue
                try:
                    branch = int(str(sheet).replace("branch", ""))
                except Exception:
                    branch = 0
                if n_class not in dataset_post_contact:
                    dataset_post_contact[n_class] = {}
                if n_name not in dataset_post_contact[n_class]:
                    dataset_post_contact[n_class][n_name] = {}
                arr_time = np.asarray(df_sheet["time after 16 h AEL (min)"], dtype=float)
                arr_len = np.asarray(df_sheet["length to reference branch point (µm)"], dtype=float)
                keep = np.isfinite(arr_time) & np.isfinite(arr_len)
                arr_time = arr_time[keep]
                arr_len = arr_len[keep]
                if "t_start_contact" in df_sheet.columns:
                    t_start_contact = float(np.asarray(df_sheet["t_start_contact"], dtype=float)[0])
                else:
                    t_start_contact = np.nan
                if "t_end_contact" in df_sheet.columns:
                    t_end_contact = float(np.asarray(df_sheet["t_end_contact"], dtype=float)[0])
                else:
                    t_end_contact = np.nan
                dataset_post_contact[n_class][n_name][branch] = {
                    "time": arr_time,
                    "length": arr_len,
                    "t_start_contact": t_start_contact,
                    "t_end_contact": t_end_contact,
                    "n_class": n_class,
                    "n_name": n_name,
                }
        if not any(len(v) for v in dataset_post_contact.values()):
            raise RuntimeError(
                "No raw dataset found and no cached raw acquisition workbooks found.\n"
                f"Tried dataset roots: {base_dir_candidates}\n"
                f"and cache directory: {raw_data_dir}"
            )
    # Save raw per-branch tables (one sheet per branch)
    if base_dir is not None:
        for n_class in dataset_post_contact:
            for n_name in dataset_post_contact[n_class]:
                branches_to_write = []
                for branch, d in dataset_post_contact[n_class][n_name].items():
                    df_sheet = d.get("raw_table_df", None)
                    if df_sheet is None or df_sheet.empty:
                        continue
                    branches_to_write.append((branch, df_sheet))
                if not branches_to_write:
                    continue
                xlsx_name = f"{n_class}_{_safe_name(n_name)}.xlsx"
                xlsx_path = os.path.join(raw_data_dir, xlsx_name)
                with pandas.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                    for branch, df_sheet in branches_to_write:
                        sheet_name = f"branch{int(branch)}"[:31]
                        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    tau_post_contact = 20
    share_axis = "row"
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=make_figsize(nrows, ncols),
        sharex=share_axis, 
        sharey=share_axis
    )

    tmem_by_class = {}
    tmem_ci95_by_class = {}
    mean_step_at_tmem_by_class = {}
    step_values_at_tmem_by_class = {}
    n_traj_by_class = {}
    mean_disp_by_class = {}
    std_disp_by_class = {}

    # plot trajectories
    for i, (n_class, color) in enumerate(zip(["class_I", "class_IV"], ["tab:green", "tab:pink"])):
        temp_all_trajs = []
        temp_all_contact_duration = []
        for n_name in dataset_post_contact[n_class].keys():
            for branch in dataset_post_contact[n_class][n_name].keys():
                time = dataset_post_contact[n_class][n_name][branch]["time"]
                length = dataset_post_contact[n_class][n_name][branch]["length"]
                t_start_contact = dataset_post_contact[n_class][n_name][branch]["t_start_contact"]
                t_end_contact = dataset_post_contact[n_class][n_name][branch]["t_end_contact"]
                if t_end_contact in time:
                    idx_end_contact = np.nonzero(time == t_end_contact)[0][0]
                elif t_end_contact < time[0]:
                    idx_end_contact = 0
                temp_all_contact_duration.append(t_end_contact - t_start_contact)
                axs[0, i].plot(length[idx_end_contact:idx_end_contact + tau_post_contact] - length[idx_end_contact],
                               c=color, alpha=0.3)
                temp_all_trajs.append(length[idx_end_contact:] - length[idx_end_contact])

        res_dict = memtime_mean_increment(temp_all_trajs, dt=1.0, B=200, sustain=3, min_n=5, seed=SEED, max_lag=None)

        square_trajs = np.zeros((len(temp_all_trajs), tau_post_contact))
        for j, a in enumerate(temp_all_trajs):
            right_stop = min(tau_post_contact, a.shape[0])
            square_trajs[j, 1:right_stop] = a[1:right_stop] - a[:right_stop - 1]
        square_trajs = np.cumsum(square_trajs, axis=1)
        mean_step_back = np.mean(square_trajs, axis=0)
        std_step_back = np.std(square_trajs, axis=0)

        axs[0, i].plot(mean_step_back, c="black")
        mean_disp_by_class[n_class] = mean_step_back
        std_disp_by_class[n_class] = std_step_back

        # index corresponding to t_mem_mean (in minutes, dt=1)
        t_mem = res_dict["t_mem_mean"]
        t_idx = int(np.round(t_mem)) if np.isfinite(t_mem) else None

        if t_idx is not None and 0 <= t_idx < tau_post_contact:
            # store per-class values for the between-class test
            tmem_by_class[n_class] = res_dict["t_mem_mean"]
            tmem_ci95_by_class[n_class] = res_dict["ci95_t"]
            mean_step_at_tmem_by_class[n_class] = mean_step_back[t_idx]
            step_values_at_tmem_by_class[n_class] = square_trajs[:, t_idx].copy()
            n_traj_by_class[n_class] = len(temp_all_trajs)
        else:
            # Still store t_mem and CI for potential reporting
            tmem_by_class[n_class] = res_dict["t_mem_mean"]
            tmem_ci95_by_class[n_class] = res_dict["ci95_t"]
            n_traj_by_class[n_class] = len(temp_all_trajs)

        axs[1, i].plot(res_dict["lags"][:tau_post_contact], res_dict["mean"][:tau_post_contact], c=color)
        axs[1, i].fill_between(res_dict["lags"][:tau_post_contact],
                               res_dict["ci_lo"][:tau_post_contact],
                               res_dict["ci_hi"][:tau_post_contact],
                               color=color, alpha=0.3)
        axs[1, i].plot(res_dict["lags"][:tau_post_contact],
                       0 * res_dict["lags"][:tau_post_contact],
                       linestyle=":", c='black')

        axs[1, i].set_xlabel("Time after contact ends $min$")
        axs[0, i].set_xlabel("Time after contact ends $min$")

        axs[1, i].set_ylabel(r"Mean increment $\mu m$")
        axs[0, i].set_ylabel(r"Displacement $\mu m$")
        axs[0, i].set_ylim(bottom=-4)

    # cosmetics
    for ax in axs.flatten():
        ax.tick_params(labelbottom=True, labelleft=True)
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(True)
    plt.tight_layout()
    save_all_ext(fig, os.path.join(plots_dir, "post-contact"), dpi=300)

    # Precompute mean_step_back between-class test text for the combined-plot legend.
    legend_test_text = "mean_step_back test: n/a"
    if "class_I" in step_values_at_tmem_by_class and "class_IV" in step_values_at_tmem_by_class:
        vals_I_leg = np.asarray(step_values_at_tmem_by_class["class_I"], float)
        vals_IV_leg = np.asarray(step_values_at_tmem_by_class["class_IV"], float)
        vals_I_leg = vals_I_leg[np.isfinite(vals_I_leg)]
        vals_IV_leg = vals_IV_leg[np.isfinite(vals_IV_leg)]
        if len(vals_I_leg) > 1 and len(vals_IV_leg) > 1:
            t_stat_leg, p_val_leg = stats.ttest_ind(
                vals_I_leg, vals_IV_leg, equal_var=False, nan_policy="omit"
            )
            legend_test_text = f"p={p_val_leg:.2g}"

    fig2, ax2 = plt.subplots(1, 1, figsize=basic_figsize)
    time_axis = np.arange(tau_post_contact)
    for n_class, color, label in zip(["class_I", "class_IV"], ["tab:green", "tab:pink"], ["cI", "cIV"]):
        if n_class not in mean_disp_by_class:
            continue
        mean_disp = mean_disp_by_class[n_class]
        std_disp = std_disp_by_class[n_class]
        ax2.plot(time_axis, mean_disp[:tau_post_contact], c=color, label=label)
        ax2.fill_between(
            time_axis,
            (mean_disp - std_disp)[:tau_post_contact],
            (mean_disp + std_disp)[:tau_post_contact],
            color=color, alpha=0.3
        )
    ax2.axhline(0, linestyle="--", color="0.5")
    if "class_I" in mean_disp_by_class and 3 < tau_post_contact:
        y_ci = mean_disp_by_class["class_I"][3]
        ax2.annotate(
            "",
            # r"$\kappa$",
            # color = "tab:green",
            xy=(3, y_ci),
            xytext=(3, 0),
            arrowprops=dict(arrowstyle="<->", linestyle="--", color="tab:green", lw=1.5),
        )
    if "class_IV" in mean_disp_by_class and 5 < tau_post_contact:
        y_civ = mean_disp_by_class["class_IV"][5]
        ax2.annotate(
            "",
            # r"$\kappa$",
            # color = "tab:pink",
            xy=(5, y_civ),
            xytext=(5, 0),
            arrowprops=dict(arrowstyle="<->", linestyle="--", color="tab:pink", lw=1.5),
        )
    ax2.set_xlabel("Time after contact ends (min)")
    ax2.set_ylabel(r"Displacement ($\mu m$)")
    ax2.set_ylim(bottom=-4)
    ax2.legend(title=legend_test_text)
    plt.tight_layout()
    save_all_ext(fig2, os.path.join(plots_dir, "post-contact-mean-std"), dpi=300)
    plt.close(fig2)

    class_I = "class_I"
    class_IV = "class_IV"

    #Test on mean_step_back[int(round(t_mem_mean))]:
    if class_I in step_values_at_tmem_by_class and class_IV in step_values_at_tmem_by_class:
        vals_I = np.asarray(step_values_at_tmem_by_class[class_I], float)
        vals_IV = np.asarray(step_values_at_tmem_by_class[class_IV], float)

        vals_I = vals_I[np.isfinite(vals_I)]
        vals_IV = vals_IV[np.isfinite(vals_IV)]

        t_stat, p_val = stats.ttest_ind(vals_I, vals_IV, equal_var=False, nan_policy="omit")

        mean_I = vals_I.mean()
        se_I = vals_I.std(ddof=1) / np.sqrt(len(vals_I)) if len(vals_I) > 1 else np.nan
        mean_IV = vals_IV.mean()
        se_IV = vals_IV.std(ddof=1) / np.sqrt(len(vals_IV)) if len(vals_IV) > 1 else np.nan

        pass
    else:
        pass

    #Test on t_mem_mean:
    def ci_to_se(ci):
        lo, hi = ci
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return np.nan
        return (hi - lo) / (2 * 1.96)

    if class_I in tmem_by_class and class_IV in tmem_by_class:
        t_I = tmem_by_class[class_I]
        t_IV = tmem_by_class[class_IV]
        ci_I = tmem_ci95_by_class.get(class_I, (np.nan, np.nan))
        ci_IV = tmem_ci95_by_class.get(class_IV, (np.nan, np.nan))

        se_t_I = ci_to_se(ci_I)
        se_t_IV = ci_to_se(ci_IV)

        if np.isfinite(t_I) and np.isfinite(t_IV) and np.isfinite(se_t_I) and np.isfinite(se_t_IV):
            diff = t_I - t_IV
            se_diff = np.sqrt(se_t_I**2 + se_t_IV**2)
            z = diff / se_diff
            p_tmem = 2 * stats.norm.sf(np.abs(z))

            pass
        else:
            pass
    else:
        pass


    class_rows = []
    for c in ["class_I", "class_IV"]:
        tmem = tmem_by_class.get(c, np.nan)
        ci = tmem_ci95_by_class.get(c, (np.nan, np.nan))
        ci_lo = float(ci[0]) if len(ci) == 2 else np.nan
        ci_hi = float(ci[1]) if len(ci) == 2 else np.nan
        se_t = (ci_hi - ci_lo) / (2 * 1.96) if (np.isfinite(ci_lo) and np.isfinite(ci_hi)) else np.nan
        vals_kappa = np.asarray(step_values_at_tmem_by_class.get(c, []), float)
        vals_kappa = vals_kappa[np.isfinite(vals_kappa)]
        if vals_kappa.size >= 2:
            kappa_mean = float(np.mean(vals_kappa))
            kappa_se = float(np.std(vals_kappa, ddof=1) / np.sqrt(vals_kappa.size))
            kappa_ci95 = f"[{kappa_mean - 1.96 * kappa_se}, {kappa_mean + 1.96 * kappa_se}]"
        else:
            kappa_ci95 = "NA"
        class_rows.append(
            {
                "class": c,
                "n_trajectories": int(n_traj_by_class.get(c, 0)),
                "t_mem_mean_min": float(tmem) if np.isfinite(tmem) else np.nan,
                "t_mem_ci95_lo": ci_lo,
                "t_mem_ci95_hi": ci_hi,
                "t_mem_se_from_ci": se_t,
                "mean_step_at_t_mem": float(mean_step_at_tmem_by_class.get(c, np.nan)),
                "kappa_ci95": kappa_ci95,
            }
        )
    class_df = pandas.DataFrame(class_rows)

    vals_I = np.asarray(step_values_at_tmem_by_class.get(class_I, []), float)
    vals_IV = np.asarray(step_values_at_tmem_by_class.get(class_IV, []), float)
    vals_I = vals_I[np.isfinite(vals_I)]
    vals_IV = vals_IV[np.isfinite(vals_IV)]
    welch_disp = _welch_summary(vals_I, vals_IV, sidedness="two-sided", alpha=0.05)

    t_I = float(tmem_by_class.get(class_I, np.nan))
    t_IV = float(tmem_by_class.get(class_IV, np.nan))
    ci_I = tmem_ci95_by_class.get(class_I, (np.nan, np.nan))
    ci_IV = tmem_ci95_by_class.get(class_IV, (np.nan, np.nan))
    se_t_I = (ci_I[1] - ci_I[0]) / (2 * 1.96) if (len(ci_I) == 2 and np.isfinite(ci_I[0]) and np.isfinite(ci_I[1])) else np.nan
    se_t_IV = (ci_IV[1] - ci_IV[0]) / (2 * 1.96) if (len(ci_IV) == 2 and np.isfinite(ci_IV[0]) and np.isfinite(ci_IV[1])) else np.nan
    z = np.nan
    p_tmem = np.nan
    if np.isfinite(t_I) and np.isfinite(t_IV) and np.isfinite(se_t_I) and np.isfinite(se_t_IV):
        se_diff = np.sqrt(se_t_I**2 + se_t_IV**2)
        if np.isfinite(se_diff) and se_diff > 0:
            z = (t_I - t_IV) / se_diff
            p_tmem = 2 * stats.norm.sf(np.abs(z))

    between_df = pandas.DataFrame(
        [
            {
                "test": "displacement_at_own_t_mem",
                "group_a": "class_I",
                "group_b": "class_IV",
                "mean_a": welch_disp["mean_a"],
                "se_a": welch_disp["se_a"],
                "n_a": welch_disp["n_a"],
                "mean_b": welch_disp["mean_b"],
                "se_b": welch_disp["se_b"],
                "n_b": welch_disp["n_b"],
                "mean_diff_a_minus_b": welch_disp["mean_diff"],
                "mean_diff_ci95_low": welch_disp["mean_diff_ci95_low"],
                "mean_diff_ci95_high": welch_disp["mean_diff_ci95_high"],
                "welch_df": welch_disp["welch_df"],
                "statistic": welch_disp["statistic"],
                "sidedness": welch_disp["sidedness"],
                "hypothesis_H0": "mean_a - mean_b = 0",
                "hypothesis_H1": "mean_a - mean_b != 0",
                "ci_method": "Welch t CI for mean difference",
                "test_name": "Welch t test",
                "p_value": welch_disp["p_value"],
                "test_type": "Welch_t_test",
            },
            {
                "test": "t_mem_difference",
                "group_a": "class_I",
                "group_b": "class_IV",
                "mean_a": t_I,
                "se_a": se_t_I,
                "n_a": int(n_traj_by_class.get(class_I, 0)),
                "mean_b": t_IV,
                "se_b": se_t_IV,
                "n_b": int(n_traj_by_class.get(class_IV, 0)),
                "mean_diff_a_minus_b": (t_I - t_IV) if (np.isfinite(t_I) and np.isfinite(t_IV)) else np.nan,
                "mean_diff_ci95_low": np.nan,
                "mean_diff_ci95_high": np.nan,
                "welch_df": np.nan,
                "statistic": z,
                "sidedness": "two-sided",
                "hypothesis_H0": "t_mem_a - t_mem_b = 0",
                "hypothesis_H1": "t_mem_a - t_mem_b != 0",
                "ci_method": "not computed in this row",
                "test_name": "Approximate z test from bootstrap-derived SE",
                "p_value": p_tmem,
                "test_type": "approx_z_test_from_bootstrap_CI",
            },
        ]
    )

    kappa_rows = []
    if "class_I" in mean_disp_by_class and 3 < tau_post_contact:
        kappa_rows.append(
            {
                "class": "class_I",
                "kappa_marker_time_min": 3,
                "kappa_marker_displacement_um": float(mean_disp_by_class["class_I"][3]),
                "definition": "vertical arrow from y=0 to mean displacement at t=3 min",
            }
        )
    if "class_IV" in mean_disp_by_class and 5 < tau_post_contact:
        kappa_rows.append(
            {
                "class": "class_IV",
                "kappa_marker_time_min": 5,
                "kappa_marker_displacement_um": float(mean_disp_by_class["class_IV"][5]),
                "definition": "vertical arrow from y=0 to mean displacement at t=5 min",
            }
        )
    kappa_df = pandas.DataFrame(kappa_rows)

    class_display = {"class_I": "Class I", "class_IV": "Class IV"}
    neurons_per_class = {
        "class_I": len(dataset_post_contact.get("class_I", {})),
        "class_IV": len(dataset_post_contact.get("class_IV", {})),
    }
    class_table = pandas.DataFrame({
        "Neuronal class": [class_display.get(c, c) for c in class_df["class"]],
        "Number of trajectories": [
            f"{int(n)} from {int(neurons_per_class.get(c, 0))} neurons"
            for n, c in zip(class_df["n_trajectories"], class_df["class"])
        ],
        "Mean memory time (min)": class_df["t_mem_mean_min"],
        "95% CI for memory time (min) (Bootstrap 200 rep)": [
            f"[{lo},{hi}]" if (np.isfinite(lo) and np.isfinite(hi)) else "NA"
            for lo, hi in zip(class_df["t_mem_ci95_lo"], class_df["t_mem_ci95_hi"])
        ],
        "Memory time standard error from CI (min)": class_df["t_mem_se_from_ci"],
        "Mean displacement at memory time κ (µm)": class_df["mean_step_at_t_mem"],
        "Mean displacement at memory time κ 95% CI (µm) (mean ± SE)": class_df["kappa_ci95"],
    })

    row_disp = between_df[between_df["test"] == "displacement_at_own_t_mem"].iloc[0]
    row_tmem = between_df[between_df["test"] == "t_mem_difference"].iloc[0]
    between_table = pandas.DataFrame([
        {
            "Post contact metric tested": "Displacement at each class memory time (µm)",
            "Group A": "Class I",
            "Group B": "Class IV",
            "Mean in group A": row_disp["mean_a"],
            "Standard error in group A": row_disp["se_a"],
            "Number in group A": int(row_disp["n_a"]),
            "Mean in group B": row_disp["mean_b"],
            "Standard error in group B": row_disp["se_b"],
            "Number in group B": int(row_disp["n_b"]),
            "Mean difference A - B": row_disp["mean_diff_a_minus_b"],
            "Mean difference 95% CI": f"[{row_disp['mean_diff_ci95_low']}, {row_disp['mean_diff_ci95_high']}]",
            "Test statistic": row_disp["statistic"],
            "Degrees of freedom": row_disp["welch_df"],
            "Two-sided test on difference (H0: difference = 0) p-value": f"p = {row_disp['p_value']:.3g}" if np.isfinite(row_disp["p_value"]) else "NA",
            "Statistical test": "Welch t test",
        },
        {
            "Post contact metric tested": "Difference in memory time (min)",
            "Group A": "Class I",
            "Group B": "Class IV",
            "Mean in group A": row_tmem["mean_a"],
            "Standard error in group A": row_tmem["se_a"],
            "Number in group A": int(row_tmem["n_a"]),
            "Mean in group B": row_tmem["mean_b"],
            "Standard error in group B": row_tmem["se_b"],
            "Number in group B": int(row_tmem["n_b"]),
            "Mean difference A - B": row_tmem["mean_diff_a_minus_b"],
            "Mean difference 95% CI": "NA",
            "Test statistic": row_tmem["statistic"],
            "Degrees of freedom": "NA",
            "Two-sided test on difference (H0: difference = 0) p-value": f"p = {row_tmem['p_value']:.3g}" if np.isfinite(row_tmem["p_value"]) else "NA",
            "Statistical test": "approximate z test from bootstrap confidence interval",
        },
    ])

    _write_df_markdown(class_table, os.path.join(result_dir, "post_contact_class_summary.md"))
    _write_df_csv(class_table, os.path.join(result_dir, "post_contact_class_summary.csv"))
    _write_df_markdown(between_table, os.path.join(result_dir, "post_contact_between_class_tests.md"))
    _write_df_csv(between_table, os.path.join(result_dir, "post_contact_between_class_tests.csv"))
