import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress

import sys

from dendrogenesis import project_style

project_style.set_style()


WORKING_DIR = os.getcwd()
RESULT_DIR = os.path.join(WORKING_DIR, "DataAnalysis", "Result", "4_von_voff_ton_toff_slope_duration")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
RAW_DATA_DIR = os.path.join(RESULT_DIR, "raw_data")
INPUT_JSON_PATH = os.path.join(RAW_DATA_DIR, "results.json")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

CLASSES = ["class_I", "class_IV"]
HOURS = [13, 16, 18, 20]
SIDES = ["Growth", "Shrinkage"]
MAX_ABS_V = 6.0
MERGE_SAME_SIGN = True
SIG_FIGS = 3
GLOBAL_RANDOM_SEED = 12345


def _write_df_markdown(df, path, preamble_lines=None):
    with open(path, "w") as f:
        if preamble_lines:
            for line in preamble_lines:
                f.write(str(line).rstrip() + "\n")
            f.write("\n")
        headers = [str(c) for c in df.columns]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in df.to_numpy():
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def _write_df_csv(df, path):
    df.to_csv(path, index=False)


def _fmt_num(x, sig=SIG_FIGS):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        xv = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xv):
        return "NA"
    return f"{xv:.{sig}g}"


def _fmt_pm(mean, se, sig=SIG_FIGS):
    try:
        m = float(mean)
    except Exception:
        return str(mean)
    try:
        s = float(se)
    except Exception:
        s = np.nan
    if not np.isfinite(m):
        return "NA"
    if np.isfinite(s):
        return f"{m:.{sig}g} ± {s:.{sig}g}"
    return f"{m:.{sig}g}"


def _fmt_ci95(mean, se, sig=SIG_FIGS):
    try:
        m = float(mean)
        s = float(se)
    except Exception:
        return "NA"
    if (not np.isfinite(m)) or (not np.isfinite(s)) or s < 0:
        return "NA"
    lo = m - 1.96 * s
    hi = m + 1.96 * s
    return f"[{lo:.{sig}g}, {hi:.{sig}g}]"


def _stringify_numeric_df(df):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(_fmt_num)
    return out


def _format_slope_duration_table(df):
    return pd.DataFrame(
        {
            "class": df["class"],
            "hour": df["hour"],
            "side": df["side"],
            "n_phase": df["n"].map(_fmt_num),
            "slope": [_fmt_pm(m, s) for m, s in zip(df["slope_mean"], df["slope_std"])],
            "duration": [_fmt_pm(m, s) for m, s in zip(df["duration_mean"], df["duration_std"])],
        }
    )


def _format_independent_table(df):
    drift_test = "Wald z-test (normal approximation)"
    drift_h0 = "H0: drift = 0"
    return pd.DataFrame(
        {
            "class": df["class"],
            "hour": df["hour"],
            "n_phase_on": df["n_phase_on"].map(_fmt_num),
            "n_phase_off": df["n_phase_off"].map(_fmt_num),
            "von": [_fmt_pm(m, s) for m, s in zip(df["von_mean"], df["von_std_or_se"])],
            "voff": [_fmt_pm(m, s) for m, s in zip(df["voff_mean"], df["voff_std_or_se"])],
            "kon": [_fmt_pm(m, s) for m, s in zip(df["kon_mean"], df["kon_std_or_se"])],
            "koff": [_fmt_pm(m, s) for m, s in zip(df["koff_mean"], df["koff_std_or_se"])],
            "drift": [_fmt_pm(m, s) for m, s in zip(df["drift_mean"], df["drift_std_or_se"])],
            "drift_95CI": [_fmt_ci95(m, s) for m, s in zip(df["drift_mean"], df["drift_se_for_ci"])],
            "drift_p": df["drift_p"].map(_fmt_num),
            "drift_p_test": drift_test,
            "drift_p_sidedness": "two-sided",
            "drift_p_hypothesis": drift_h0,
            "D": [_fmt_pm(m, s) for m, s in zip(df["D_mean"], df["D_std_or_se"])],
        }
    )


def _format_dependent_table(df):
    return pd.DataFrame(
        {
            "class": df["class"],
            "hour": df["hour"],
            "side": df["side"],
            "n_phase_on": df["n_phase_on"].map(_fmt_num),
            "n_phase_off": df["n_phase_off"].map(_fmt_num),
            "v0": [_fmt_pm(m, s) for m, s in zip(df["v0_mean"], df["v0_std_or_se"])],
            "v1": [_fmt_pm(m, s) for m, s in zip(df["v1_mean"], df["v1_std_or_se"])],
            "p_v1": df["p_v1"].map(_fmt_num),
            "model_equation": df["model_equation"],
            "tested_coefficient": df["tested_coefficient"],
            "b_estimate": df["b_estimate"].map(_fmt_num),
            "b_95CI": [f"[{_fmt_num(lo)}, {_fmt_num(hi)}]" if (np.isfinite(lo) and np.isfinite(hi)) else "NA"
                       for lo, hi in zip(df["b_ci95_low"], df["b_ci95_high"])],
            "test_statistic_t": df["test_statistic_t"].map(_fmt_num),
            "test_df": df["test_df"].map(_fmt_num),
            "p_value_test": df["p_value_test"],
            "p_value_h0": df["p_value_h0"],
            "p_value_sidedness": df["p_value_sidedness"],
            "tau0": [_fmt_pm(m, s) for m, s in zip(df["tau0_mean"], df["tau0_std_or_se"])],
            "sigma": [_fmt_pm(m, s) for m, s in zip(df["sigma_mean"], df["sigma_std_or_se"])],
        }
    )


def _save_all_fig(base, exts=("png", "pdf", "svg")):
    for e in exts:
        plt.savefig(f"{base}.{e}", bbox_inches="tight", dpi=200)


def _closed_box(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
    ax.tick_params(direction="in", top=True, right=True)


def _set_axes_square(axs):
    arr = np.asarray(axs, dtype=object)
    for ax in arr.ravel():
        if ax is not None:
            ax.set_box_aspect(1)


def _class_label(nclass):
    return nclass.replace("_", " ")


def _palette_for(_nclass):
    return {"Growth": "tab:blue", "Shrinkage": "tab:orange"}


def _build_plot_phase_df(phase_df):
    out = phase_df.copy()
    out["hour_i"] = out["hour"].astype(int)
    out["class_pretty"] = out["class"].map(_class_label)
    out["slope_signed"] = np.where(out["side"] == "Growth", out["slope_abs"], -out["slope_abs"])
    out["slope_abs"] = np.abs(out["slope_abs"])
    out["hour_label"] = out["hour_i"].map(lambda h: f"{int(h)}h AEL")
    return out


def _plot_duration_vs_slope_linabs_grid(plot_df, out_dir):
    hours_plot = [16, 18, 20]
    basic_figsize = (project_style.mm_to_in(50), project_style.mm_to_in(50))
    fig, axs = plt.subplots(
        len(CLASSES),
        len(hours_plot),
        figsize=(len(hours_plot) * basic_figsize[0], len(CLASSES) * basic_figsize[1]),
        sharex=False,
        sharey=False,
    )

    for r, nclass in enumerate(CLASSES):
        for c, hour in enumerate(hours_plot):
            ax = axs[r, c]
            sub = plot_df[(plot_df["class"] == nclass) & (plot_df["hour_i"] == hour)].copy()
            pal = _palette_for(nclass)

            for side in SIDES:
                sside = sub[sub["side"] == side].copy()
                m_sc = (
                    np.isfinite(sside["slope_signed"])
                    & np.isfinite(sside["slope_abs"])
                    & np.isfinite(sside["duration"])
                    & (sside["duration"] > 0)
                )
                if np.any(m_sc):
                    ax.scatter(
                        sside.loc[m_sc, "slope_signed"],
                        sside.loc[m_sc, "duration"],
                        s=7,
                        alpha=0.15,
                        color=pal[side],
                    )

                fit = sside[np.isfinite(sside["slope_abs"]) & np.isfinite(sside["duration"]) & (sside["duration"] > 0)]
                if fit.shape[0] >= 3 and np.unique(fit["slope_abs"]).size >= 2:
                    x_abs = fit["slope_abs"].to_numpy(float)
                    y_log = np.log(fit["duration"].to_numpy(float))
                    lr = linregress(x_abs, y_log)
                    xg_abs = np.linspace(x_abs.min(), min(x_abs.max(), MAX_ABS_V), 200)
                    yg = np.exp(float(lr.intercept) + float(lr.slope) * xg_abs)
                    label = f"{lr.slope:.2g} p={lr.pvalue:.3f}, $r^2$={lr.rvalue**2:.2g}"
                    if side == "Growth":
                        ax.plot(xg_abs, yg, "-", lw=1, color=pal[side], label=label)
                    else:
                        ax.plot(-xg_abs, yg, "-", lw=1, color=pal[side], label=label)

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(handles=handles, labels=labels, frameon=False, handlelength=1.0)
            ax.set_ylim(bottom=0.02)
            ax.set_yscale("log")
            ax.set_xlim(-MAX_ABS_V, MAX_ABS_V)
            ax.set_title(f"{_class_label(nclass)} {hour}h AEL")
            ax.set_xlabel(r"Velocity (µm/min)")
            ax.set_ylabel("Duration (min)")
            _closed_box(ax)

    _set_axes_square(axs)
    plt.tight_layout()
    _save_all_fig(os.path.join(out_dir, "duration_vs_slope_linABS_GRID_16_18_20"))
    plt.close(fig)


def _plot_amrutha_carac_time_lesscomp(plot_df, out_dir):
    import seaborn as sns
    from statannotations.Annotator import Annotator

    # Use unified phase data but keep legacy plotting style/logic.
    df = pd.DataFrame(
        {
            "Hour": plot_df["hour_label"],
            "neuron class": plot_df["class_pretty"],
            "slope": plot_df["slope_abs"],
            "duration": plot_df["duration"],
            "is_growing": plot_df["side"],
        }
    )
    df = df[np.isfinite(df["slope"]) & np.isfinite(df["duration"]) & (df["duration"] > 0)].copy()

    n_rows = 2
    n_cols = 2
    basic_figsize = (project_style.mm_to_in(50), project_style.mm_to_in(50))
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(basic_figsize[0] * n_cols, basic_figsize[1] * n_rows),
        sharey=None,
    )

    legacy_hour_order = {
        "class I": ["16h AEL", "18h AEL", "20h AEL"],
        "class IV": ["16h AEL", "18h AEL", "20h AEL"],
    }

    for i, (n_class,) in enumerate(zip(["class I", "class IV"])):
        temp_df = df[df["neuron class"] == n_class]
        hour_order = legacy_hour_order[n_class]
        palette = {"Growth": "tab:blue", "Shrinkage": "tab:orange"}
        hue_order = ["Growth", "Shrinkage"]
        axs[0, i] = sns.boxplot(
            data=temp_df,
            x="Hour",
            y="slope",
            hue="is_growing",
            order=hour_order,
            ax=axs[0, i],
            hue_order=hue_order,
            palette=palette,
            legend=False,
            fill=False,
            fliersize=3,
            flierprops={"markeredgewidth": 0.5},
        )
        axs[1, i] = sns.boxplot(
            data=temp_df,
            x="Hour",
            y="duration",
            hue="is_growing",
            order=hour_order,
            ax=axs[1, i],
            hue_order=hue_order,
            palette=palette,
            legend=False,
            fill=False,
            fliersize=3,
            flierprops={"markeredgewidth": 0.5},
        )

        pairs = []
        for h in hour_order:
            temp_pair = []
            for is_growing in hue_order:
                temp_pair.append((h, is_growing))
            if len(temp_pair) == 2:
                pairs.append(temp_pair)

        if len(pairs):
            annot = Annotator(
                axs[0, i],
                pairs,
                data=temp_df,
                x="Hour",
                y="slope",
                hue="is_growing",
                order=hour_order,
                hue_order=hue_order,
            )
            annot.configure(test="t-test_welch", text_format="star", comparisons_correction="holm")
            annot.apply_and_annotate()

            annot = Annotator(
                axs[1, i],
                pairs,
                data=temp_df,
                x="Hour",
                y="duration",
                hue="is_growing",
                order=hour_order,
                hue_order=hue_order,
            )
            annot.configure(test="t-test_welch", text_format="star", comparisons_correction="holm")
            annot.apply_and_annotate()

    axs[0, 0].set_ylabel(r"$v_{on}$ vs $v_{off}$ (µm/min)")
    axs[0, 1].set_ylabel(r"$v_{on}$ vs $v_{off}$ (µm/min)")
    axs[1, 0].set_ylabel(r"$\tau_{on}$ vs $\tau_{off}$ (min)")
    axs[1, 1].set_ylabel(r"$\tau_{on}$ vs $\tau_{off}$ (min)")
    axs[0, 1].yaxis.label.set_visible(True)
    axs[1, 1].yaxis.label.set_visible(True)

    axs[0, 0].set_title("Class I")
    axs[0, 1].set_title("Class IV")
    axs[1, 0].set_title("Class I")
    axs[1, 1].set_title("Class IV")

    for ax in axs.flatten():
        xt = [t.get_text().replace("h AEL", "").replace(" AEL", "") for t in ax.get_xticklabels()]
        ax.set_xticklabels(xt)

    axs[0, 0].set_xlabel("Time (hours AEL)")
    axs[0, 1].set_xlabel("Time (hours AEL)")
    axs[1, 0].set_xlabel("Time (hours AEL)")
    axs[1, 1].set_xlabel("Time (hours AEL)")

    m_i_sh_16 = (
        (df["neuron class"] == "class I")
        & (df["is_growing"] == "Shrinkage")
        & (df["Hour"] == "16h AEL")
    )
    if np.any(m_i_sh_16):
        y = float(np.nanmean(df.loc[m_i_sh_16, "duration"].to_numpy(float)))
        axs[1, 0].text(1.05, y, r"$\langle\tau_{off}\rangle=1/k_{on}$", ha="left", va="center")

    m_iv_gr_16 = (
        (df["neuron class"] == "class IV")
        & (df["is_growing"] == "Growth")
        & (df["Hour"] == "16h AEL")
    )
    if np.any(m_iv_gr_16):
        y = float(np.nanmean(df.loc[m_iv_gr_16, "duration"].to_numpy(float)))
        axs[1, 1].text(0.95, y, r"$\langle\tau_{on}\rangle=1/k_{off}$", ha="right", va="center")

    _set_axes_square(axs)
    plt.tight_layout()
    _save_all_fig(os.path.join(out_dir, "amrutha-carac-time-lesscomp-None"))
    plt.close(fig)


def _build_indep_param_phase_df(phase_df):
    rows = []
    for nclass in CLASSES:
        for hour in [16, 18, 20]:
            sub = phase_df[(phase_df["class"] == nclass) & (phase_df["hour"] == hour)]
            if sub.empty:
                continue
            on = sub[sub["side"] == "Growth"]
            off = sub[sub["side"] == "Shrinkage"]
            # Use phase-level values to match distribution style of amrutha-carac-time-lesscomp.
            von_vals = on["slope_abs"].to_numpy(float)
            voff_vals = off["slope_abs"].to_numpy(float)
            ton_vals = on["duration"].to_numpy(float)
            toff_vals = off["duration"].to_numpy(float)

            von_vals = von_vals[np.isfinite(von_vals)]
            voff_vals = voff_vals[np.isfinite(voff_vals)]
            ton_vals = ton_vals[np.isfinite(ton_vals) & (ton_vals > 0)]
            toff_vals = toff_vals[np.isfinite(toff_vals) & (toff_vals > 0)]
            tau_on_vals = ton_vals
            tau_off_vals = toff_vals

            for v in von_vals:
                rows.append({"Hour": f"{int(hour)}h AEL", "Class": _class_label(nclass), "acq": "", "von": float(v), "voff": np.nan, "tau_on": np.nan, "tau_off": np.nan})
            for v in voff_vals:
                rows.append({"Hour": f"{int(hour)}h AEL", "Class": _class_label(nclass), "acq": "", "von": np.nan, "voff": float(v), "tau_on": np.nan, "tau_off": np.nan})
            for v in tau_on_vals:
                rows.append({"Hour": f"{int(hour)}h AEL", "Class": _class_label(nclass), "acq": "", "von": np.nan, "voff": np.nan, "tau_on": float(v), "tau_off": np.nan})
            for v in tau_off_vals:
                rows.append({"Hour": f"{int(hour)}h AEL", "Class": _class_label(nclass), "acq": "", "von": np.nan, "voff": np.nan, "tau_on": np.nan, "tau_off": float(v)})
    return pd.DataFrame(rows)


def _plot_indep_param_class_compare(phase_df, out_dir):
    import seaborn as sns
    from statannotations.Annotator import Annotator

    dfi = _build_indep_param_phase_df(phase_df)
    if dfi.empty:
        return

    hour_order = ["16h AEL", "18h AEL", "20h AEL"]
    class_order = ["class I", "class IV"]
    palette = {"class I": "tab:green", "class IV": "tab:pink"}
    params = [("von", r"$v_{on}$"), ("voff", r"$v_{off}$"), ("tau_on", r"$\tau_{on}$"), ("tau_off", r"$\tau_{off}$")]

    basic_figsize = (project_style.mm_to_in(50), project_style.mm_to_in(50))
    fig, axs = plt.subplots(2, 2, figsize=(2 * basic_figsize[0], 2 * basic_figsize[1]), sharey=False)

    for ax, (pcol, ptitle) in zip(axs.flatten(), params):
        sns.boxplot(
            data=dfi,
            x="Hour",
            y=pcol,
            hue="Class",
            order=hour_order,
            hue_order=class_order,
            palette=palette,
            legend=False,
            fill=False,
            fliersize=3,
            flierprops={"markeredgewidth": 0.5},
            ax=ax,
        )
        pairs = [[(h, "class I"), (h, "class IV")] for h in hour_order]
        annot = Annotator(
            ax,
            pairs,
            data=dfi,
            x="Hour",
            y=pcol,
            hue="Class",
            order=hour_order,
            hue_order=class_order,
        )
        annot.configure(test="t-test_welch", text_format="star", comparisons_correction="holm")
        annot.apply_and_annotate()

        xt = [t.get_text().replace("h AEL", "").replace(" AEL", "") for t in ax.get_xticklabels()]
        ax.set_xticklabels(xt)
        ax.set_title(ptitle)
        ax.set_xlabel("Time (hours AEL)")
        _closed_box(ax)
        ax.tick_params(top=False, right=False)

    axs[0, 0].set_ylabel(r"$v_{on}$ (µm/min)")
    axs[0, 1].set_ylabel(r"$v_{off}$ (µm/min)")
    axs[1, 0].set_ylabel(r"$\tau_{on}$ (min)")
    axs[1, 1].set_ylabel(r"$\tau_{off}$ (min)")

    _set_axes_square(axs)
    plt.tight_layout()
    _save_all_fig(os.path.join(out_dir, "amrutha-kparams-class-lesscomp-16_18_20"))
    plt.close(fig)


def _plot_param_distributions_grid(phase_df, out_dir):
    # Rows: class/time combinations. Cols: von, voff, tau_on, tau_off.
    row_spec = [("class_I", 13), ("class_I", 16), ("class_I", 18), ("class_I", 20),
                ("class_IV", 16), ("class_IV", 18), ("class_IV", 20)]
    col_spec = [
        ("von", "Growth", "slope_abs", "tab:blue"),
        ("voff", "Shrinkage", "slope_abs", "tab:orange"),
        ("tau_on", "Growth", "duration", "tab:blue"),
        ("tau_off", "Shrinkage", "duration", "tab:orange"),
    ]
    xlabels = {
        "von": r"$\mathrm{v}_{\mathrm{on}}$ (µm/min)",
        "voff": r"$\mathrm{v}_{\mathrm{off}}$ (µm/min)",
        "tau_on": r"$\mathrm{\tau}_{\mathrm{on}}$ (min)",
        "tau_off": r"$\mathrm{\tau}_{\mathrm{off}}$ (min)",
    }

    basic_figsize = (project_style.mm_to_in(50), project_style.mm_to_in(50))
    n_rows = len(row_spec)
    n_cols = len(col_spec)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * basic_figsize[0], n_rows * basic_figsize[1]),
        sharex=False,
        sharey=False,
    )
    axs = np.atleast_2d(axs)

    for r, (nclass, hour) in enumerate(row_spec):
        sub = phase_df[(phase_df["class"] == nclass) & (phase_df["hour"] == hour)]
        row_label = f"{_class_label(nclass)} {int(hour)}h AEL"
        for c, (pname, side, value_col, color) in enumerate(col_spec):
            ax = axs[r, c]
            arr = sub[sub["side"] == side][value_col].to_numpy(float) if not sub.empty else np.array([])
            arr = arr[np.isfinite(arr) & (arr > 0)]

            if arr.size >= 1:
                ax.hist(arr, bins=24, density=True, color=color, alpha=0.45)
                mean = float(np.mean(arr))
                se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else np.nan
                if np.isfinite(mean) and mean > 0:
                    xg = np.linspace(max(np.min(arr), 1e-9), np.max(arr), 300)
                    pdf = (1.0 / mean) * np.exp(-xg / mean)
                    ax.plot(xg, pdf, color="red", lw=1.5, label=f"Exp mean: {mean:.3g} ± {se:.3g}" if np.isfinite(se) else f"Exp mean: {mean:.3g}")
                if ax.get_legend_handles_labels()[1]:
                    ax.legend(frameon=False, fontsize=6, loc="best")
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=7)

            ax.set_yscale("log")
            ax.set_xlabel(xlabels.get(pname, pname))
            if c == 0:
                ax.set_ylabel(f"{row_label}\nDensity")
            _closed_box(ax)

    _set_axes_square(axs)
    plt.tight_layout()
    _save_all_fig(os.path.join(out_dir, "distribution_von_voff_tauon_tauoff_grid"))
    plt.close(fig)


def _split_trajectories_10min(t, y, window=10.0, max_jump=1.0):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if t.size == 0:
        return []
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]
    start = t[0]
    segments = []
    while start <= t[-1]:
        end = start + window
        mask = (t >= start) & (t < end)
        if np.any(mask):
            tt = t[mask] - start
            yy = y[mask] - y[mask][0]
            if tt.size >= 2 and np.nanmax(np.abs(np.diff(yy))) <= max_jump:
                segments.append((tt, yy))
        start = end
    return segments


def _displacement_at_time(segments, t_query=4.0):
    vals = []
    for tt, yy in segments:
        if tt.size < 2:
            continue
        if t_query < tt[0] or t_query > tt[-1]:
            continue
        vals.append(float(np.interp(t_query, tt, yy)))
    return np.asarray(vals, dtype=float)


def _plot_traj_with_inset(ax, segments, t_ref=4.0, title="", inset_loc="lower right"):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    for tt, yy in segments:
        ax.plot(tt, yy, color="0.25", alpha=0.25, lw=0.8)
    ax.set_title(title)
    ax.set_ylim(bottom=-12)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Displacement (µm)")
    disp = _displacement_at_time(segments, t_query=t_ref)
    disp = disp[np.isfinite(disp)]
    if disp.size >= 2:
        dmin = float(np.min(disp))
        dmax = float(np.max(disp))
        margin = 0.1 * max(dmax - dmin, 1e-6)
        y0 = dmin - margin
        y1 = dmax + margin
        ax.vlines(t_ref, y0, y1, color="k", lw=1, alpha=0.7, linestyles="--")
    else:
        ax.axvline(t_ref, color="k", lw=1, alpha=0.7, linestyle="--")
        return
    axins = inset_axes(ax, width="35%", height="35%", loc=inset_loc, borderpad=0.6)
    axins.hist(disp, bins=25, color="0.6", alpha=0.8)
    axins.set_xlim(left=-5, right=5)
    mu = float(np.mean(disp))
    sd = float(np.std(disp, ddof=1))
    y_arrow = 0.9 * axins.get_ylim()[1]
    axins.annotate(
        "",
        xy=(mu + sd, y_arrow),
        xytext=(mu - sd, y_arrow),
        arrowprops=dict(arrowstyle="<->", lw=1, color="red"),
    )
    axins.text(mu, y_arrow, r"$\sigma_{4min}$", ha="center", va="bottom", color="red")
    axins.set_xticks([])
    axins.set_yticks([])
    x_right = float(ax.get_xlim()[1])
    x0 = t_ref
    x1 = x_right - 0.40 * (x_right - float(ax.get_xlim()[0]))
    ax.annotate(
        "",
        xy=(x1, y0),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", linestyle="--", lw=1, color="k", alpha=0.7),
    )


def _real_tracks_envelope(time_axis, list_branch, target_class, target_hour, min_duration_min=8.0):
    tracks = []
    for br in list_branch:
        if br.get("neuron_class") != target_class:
            continue
        hour = br.get("inner_time_fallback", np.nan)
        if not np.isfinite(hour) or int(round(float(hour))) != int(target_hour):
            continue
        t = br.get("length_time", None)
        y = br.get("length_length", None)
        if t is None or y is None or len(t) == 0 or len(y) == 0:
            continue
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        if t.size != y.size:
            continue
        if (t[-1] - t[0]) < min_duration_min:
            continue
        t = t - t[0]
        y = y - y[0]
        arr = np.full(time_axis.shape, np.nan, dtype=float)
        m = time_axis <= t[-1]
        if np.any(m):
            arr[m] = np.interp(time_axis[m], t, y)
        tracks.append(arr)
    if len(tracks) == 0:
        return None, None, None
    X = np.vstack(tracks)
    return np.nanmean(X, axis=0), np.nanstd(X, axis=0), X


def _phase_alternance(phase_succession, time_axis, tracks):
    jumps_plus = phase_succession[:, :, 0] * phase_succession[:, :, 1]
    jumps_minus = -phase_succession[:, :, 2] * phase_succession[:, :, 3]
    n_tracks, n_phases_max = phase_succession.shape[:2]
    alternance_bool = np.tile(np.mod(np.arange(n_phases_max), 2), (n_tracks, 1)).astype(bool)
    start_growth = np.random.random((n_tracks, 1)) > 0.5
    alternance_bool = (start_growth * alternance_bool + ~start_growth * (1 - alternance_bool)).astype(bool)
    phases_length = jumps_plus * alternance_bool + jumps_minus * (~alternance_bool)
    phases_time_axis = phase_succession[:, :, 1] * alternance_bool + phase_succession[:, :, 3] * (~alternance_bool)
    phase_start = np.cumsum(phases_time_axis, axis=1)
    branch_length = np.cumsum(phases_length, axis=1)
    for i in range(n_tracks):
        tracks[i] = np.interp(
            time_axis,
            np.concatenate([[0], phase_start[i]]),
            np.concatenate([[0], branch_length[i]]),
        )
    return tracks, time_axis


def _linear_phases_indep(max_T, dt, n_tracks, von, voff, kon, koff, mode="sample_velocity"):
    time_axis = np.arange(0, max_T + dt, dt)
    tracks = np.zeros((n_tracks, time_axis.shape[0]))
    n_phases_max = int(5 * max_T * (kon + koff) / 2)
    phase_succession = np.zeros((n_tracks, n_phases_max, 4))
    if mode == "mean_velocity":
        phase_succession[:, :, 0] = von
        phase_succession[:, :, 2] = voff
    else:
        phase_succession[:, :, 0] = np.random.exponential(1 / von, size=(n_tracks, n_phases_max))
        phase_succession[:, :, 2] = np.random.exponential(1 / voff, size=(n_tracks, n_phases_max))
    phase_succession[:, :, 1] = np.random.exponential(koff, size=(n_tracks, n_phases_max))
    phase_succession[:, :, 3] = np.random.exponential(kon, size=(n_tracks, n_phases_max))
    return _phase_alternance(phase_succession, time_axis, tracks)


def _linear_phases_dep(max_T, dt, n_tracks, v0G, v0S, v1G, v1S, tau0G, tau0S, sigmaG, sigmaS):
    time_axis = np.arange(0, max_T + dt, dt)
    tracks = np.zeros((n_tracks, time_axis.shape[0]))
    n_phases_max = int(5 * max_T / (tau0G + tau0S) * 2)
    phase_succession = np.zeros((n_tracks, n_phases_max, 4))
    phase_succession[:, :, 0] = np.random.exponential(1 / v0G, size=(n_tracks, n_phases_max))
    phase_succession[:, :, 2] = np.random.exponential(1 / v0S, size=(n_tracks, n_phases_max))
    phase_succession[:, :, 1] = np.random.lognormal(tau0G + phase_succession[:, :, 0] / v1G, sigmaG)
    phase_succession[:, :, 3] = np.random.lognormal(tau0S + phase_succession[:, :, 2] / v1S, sigmaS)
    return _phase_alternance(phase_succession, time_axis, tracks)


def _build_model_params_from_unified(indep_df, dep_df):
    params = {}
    for _, r in indep_df.iterrows():
        key = f"{r['class']}_{str(r['hour']).replace('h','')}h"
        params.setdefault(key, {})
        params[key]["indep"] = {
            "von": float(r["von_mean"]),
            "voff": float(r["voff_mean"]),
            "kon": float(r["kon_mean"]),
            "koff": float(r["koff_mean"]),
        }
    for key, g in dep_df.groupby(["class", "hour"]):
        cls, hour = key
        hour_i = str(hour).replace("h", "")
        name = f"{cls}_{hour_i}h"
        params.setdefault(name, {})
        gg = g[g["side"] == "Growth"]
        gs = g[g["side"] == "Shrinkage"]
        if gg.empty or gs.empty:
            continue
        params[name]["dep"] = {
            "v0G": float(gg["v0_mean"].iloc[0]),
            "v0S": float(gs["v0_mean"].iloc[0]),
            "v1G": float(gg["v1_mean"].iloc[0]),
            "v1S": float(gs["v1_mean"].iloc[0]),
            "tau0G": float(gg["tau0_mean"].iloc[0]),
            "tau0S": float(gs["tau0_mean"].iloc[0]),
            "sigmaG": float(gg["sigma_mean"].iloc[0]),
            "sigmaS": float(gs["sigma_mean"].iloc[0]),
        }
    return params


def _plot_legacy_track_figures(list_branch, indep_df, dep_df, out_dir, stats_out_dir=None):
    if stats_out_dir is None:
        stats_out_dir = out_dir
    params = _build_model_params_from_unified(indep_df, dep_df)
    basic_figsize = (project_style.mm_to_in(50), project_style.mm_to_in(50))
    T_max = 8
    dt = 1 / 12
    n_tracks = 1000
    max_track_plot = 1000
    color_indep = "tab:purple"
    color_dep = "tab:cyan"
    real_class = "class_IV"
    real_hour = 18

    # trajectories_GRID_16_18_20
    fig, axs = plt.subplots(2, 3, figsize=(3 * basic_figsize[0], 2 * basic_figsize[1]), sharex=False, sharey=False)
    for r, cls in enumerate(["class_I", "class_IV"]):
        for c, hour in enumerate([16, 18, 20]):
            ax = axs[r, c]
            any_traj = False
            for br in list_branch:
                if br.get("neuron_class", "") != cls:
                    continue
                h = br.get("inner_time_fallback", np.nan)
                if not np.isfinite(h) or int(round(float(h))) != int(hour):
                    continue
                t = br.get("length_time", None)
                y = br.get("length_length", None)
                if t is None or y is None or len(t) == 0 or len(y) == 0:
                    continue
                for tt, yy in _split_trajectories_10min(t, y, window=10.0):
                    ax.plot(tt, yy, color="0.25", alpha=0.25, lw=0.8)
                    any_traj = True
            ax.set_title(f"{_class_label(cls)} {hour}h AEL")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Displacement (µm)")
            _closed_box(ax)
            ax.tick_params(left=False, top=False, right=False)
            if not any_traj:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=7)
    _set_axes_square(axs)
    plt.tight_layout()
    _save_all_fig(os.path.join(out_dir, "trajectories_GRID_16_18_20"))
    plt.close(fig)

    # Need class IV 18h params for simulations
    key_main = f"{real_class}_{real_hour}h"
    if key_main not in params or "indep" not in params[key_main] or "dep" not in params[key_main]:
        return
    tr_indep, time_axis = _linear_phases_indep(T_max, dt, n_tracks, **params[key_main]["indep"], mode="sample_velocity")
    tr_dep, _ = _linear_phases_dep(T_max, dt, n_tracks, **params[key_main]["dep"])

    # trajectories_real_vs_dep_inset_4min
    fig_inset, axs_inset = plt.subplots(1, 2, figsize=(2 * basic_figsize[0], basic_figsize[0]), sharex=False, sharey=False)
    segments_real = []
    for br in list_branch:
        if br.get("neuron_class") != real_class:
            continue
        hour = br.get("inner_time_fallback", np.nan)
        if not np.isfinite(hour) or int(round(float(hour))) != int(real_hour):
            continue
        t = br.get("length_time", None)
        y = br.get("length_length", None)
        if t is None or y is None or len(t) == 0 or len(y) == 0:
            continue
        segments_real.extend(_split_trajectories_10min(t, y, window=10.0, max_jump=1.0))
    _plot_traj_with_inset(
        axs_inset[0], segments_real, t_ref=4.0, title=f"{real_class.replace('_', ' ')} {real_hour}h AEL (real)", inset_loc="lower right"
    )
    segments_dep = []
    for i in range(min(tr_dep.shape[0], max_track_plot)):
        segments_dep.extend(_split_trajectories_10min(time_axis, tr_dep[i], window=10.0, max_jump=1.0))
    _plot_traj_with_inset(axs_inset[1], segments_dep, t_ref=4.0, title="class IV 18h AEL (dep sim)", inset_loc="lower right")
    _set_axes_square(axs_inset)
    plt.tight_layout()
    _save_all_fig(os.path.join(out_dir, "trajectories_real_vs_dep_inset_4min"))
    plt.close(fig_inset)

    def _diffusion_from_msd_nan(tracks, t, detrend=False):
        x = np.asarray(tracks, float)
        if x.size == 0:
            return np.nan
        if detrend:
            mean_x = np.nanmean(x, axis=0)
            m = np.isfinite(t) & np.isfinite(mean_x)
            if np.sum(m) < 2:
                return np.nan
            drift, _ = np.polyfit(t[m], mean_x[m], 1)
            x = x - drift * t[None, :]
        x0 = x[:, [0]]
        msd = np.nanmean((x - x0) ** 2, axis=0)
        m = np.isfinite(t) & np.isfinite(msd)
        if np.sum(m) < 2:
            return np.nan
        slope, _ = np.polyfit(t[m], msd[m], 1)
        return float(slope / 2.0)

    def _bootstrap_ci(tracks, t, func, n_boot=200, alpha=0.05, seed=0):
        x = np.asarray(tracks, float)
        if x.size == 0:
            return np.nan, np.nan
        rng = np.random.default_rng(seed)
        n = x.shape[0]
        vals = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            vals[i] = func(x[idx], t)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan, np.nan
        lo = np.quantile(vals, alpha / 2)
        hi = np.quantile(vals, 1 - alpha / 2)
        return float(lo), float(hi)

    def _bootstrap_diff_pvalue(tracks_a, tracks_b, t, func_a, func_b, n_boot=200, seed=0):
        xa = np.asarray(tracks_a, float)
        xb = np.asarray(tracks_b, float)
        if xa.size == 0 or xb.size == 0:
            return np.nan
        rng = np.random.default_rng(seed)
        na = xa.shape[0]
        nb = xb.shape[0]
        diffs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            ia = rng.integers(0, na, size=na)
            ib = rng.integers(0, nb, size=nb)
            da = func_a(xa[ia], t)
            db = func_b(xb[ib], t)
            diffs[i] = da - db
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            return np.nan
        return float(2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0)))

    # single_tracks_comparison_envelop (3 panels)
    n_boot_grid_stats = 200
    p_floor_grid_stats = 2.0 / float(n_boot_grid_stats)

    def _fmt_boot_p(p, p_floor=p_floor_grid_stats):
        if not np.isfinite(p):
            return "NA"
        if p <= p_floor:
            return f"<{p_floor:.2g}"
        return f"{p:.2g}"

    def _fmt_boot_p_label(p, p_floor=p_floor_grid_stats):
        ptxt = _fmt_boot_p(p, p_floor=p_floor)
        if ptxt == "NA":
            return "p=NA"
        if ptxt.startswith("<"):
            return f"p{ptxt}"
        return f"p={ptxt}"
    fig_env, axs_env = plt.subplots(1, 3, figsize=(3 * basic_figsize[0], basic_figsize[1]), sharex=False, sharey=False)
    mean_ind = np.mean(tr_indep, axis=0)
    std_ind = np.std(tr_indep, axis=0)
    mean_dep = np.mean(tr_dep, axis=0)
    std_dep = np.std(tr_dep, axis=0)
    rm, rs, real_tracks_main = _real_tracks_envelope(time_axis, list_branch, real_class, real_hour, min_duration_min=8.0)
    d_ind_main = _diffusion_from_msd_nan(tr_indep, time_axis, detrend=True)
    d_dep_main = _diffusion_from_msd_nan(tr_dep, time_axis, detrend=True)
    if real_tracks_main is not None:
        d_real_main = _diffusion_from_msd_nan(real_tracks_main, time_axis, detrend=True)
        p_ind_real_main = _bootstrap_diff_pvalue(
            tr_indep,
            real_tracks_main,
            time_axis,
            lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
            lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
            n_boot=n_boot_grid_stats,
            seed=10,
        )
        p_dep_real_main = _bootstrap_diff_pvalue(
            tr_dep,
            real_tracks_main,
            time_axis,
            lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
            lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
            n_boot=n_boot_grid_stats,
            seed=11,
        )
    else:
        d_real_main = np.nan
        p_ind_real_main = np.nan
        p_dep_real_main = np.nan

    axs_env[0].plot(
        time_axis,
        mean_ind,
        color=color_indep,
        label=f"Simu. indep. D={d_ind_main:.3g} $\\mathrm{{\\mu m^{{2}}/min}}$ {_fmt_boot_p_label(p_ind_real_main)}" if np.isfinite(d_ind_main) else "Simu. indep. D=NA $\\mathrm{{\\mu m^{{2}}/min}}$ p=NA",
    )
    axs_env[0].fill_between(time_axis, mean_ind - std_ind, mean_ind + std_ind, color=color_indep, alpha=0.3)
    axs_env[0].plot(
        time_axis,
        mean_dep,
        color=color_dep,
        label=f"Simu. dep. D={d_dep_main:.3g} $\\mathrm{{\\mu m^{{2}}/min}}$ {_fmt_boot_p_label(p_dep_real_main)}" if np.isfinite(d_dep_main) else "Simu. dep. D=NA $\\mathrm{{\\mu m^{{2}}/min}}$ p=NA",
    )
    axs_env[0].fill_between(time_axis, mean_dep - std_dep, mean_dep + std_dep, color=color_dep, alpha=0.3)
    if rm is not None and rs is not None:
        axs_env[0].plot(time_axis, rm, color="black", label=f"Data WT class {real_class.split('_')[-1]} {real_hour}h D={d_real_main:.3g} $\\mathrm{{\\mu m^{{2}}/min}}$")
        axs_env[0].fill_between(time_axis, rm - rs, rm + rs, color="black", alpha=0.15)
    axs_env[0].set_ylabel("Displacement (µm)")
    axs_env[0].set_xlabel("Time (min)")
    axs_env[0].legend(frameon=False, handlelength=1.0, fontsize=4)
    _closed_box(axs_env[0])
    axs_env[0].tick_params(left=False, top=False, right=False)

    def _simulate_rw(n, t, drift, diffusion, seed=0):
        rng = np.random.default_rng(seed)
        dt_local = float(np.mean(np.diff(t)))
        steps = rng.normal(loc=drift * dt_local, scale=np.sqrt(2 * diffusion * dt_local), size=(n, t.size - 1))
        x = np.zeros((n, t.size), dtype=float)
        x[:, 1:] = np.cumsum(steps, axis=1)
        return x

    D_fixed = 0.5
    for drift, col in [(0.0, "black"), (-0.2, "tab:red")]:
        rw = _simulate_rw(n_tracks, time_axis, drift=drift, diffusion=D_fixed, seed=1)
        m = np.mean(rw, axis=0)
        s = np.std(rw, axis=0)
        axs_env[1].plot(time_axis, m, color=col, label=f"drift={drift:g}")
        axs_env[1].fill_between(time_axis, m - s, m + s, color=col, alpha=0.2)
    axs_env[1].set_ylabel("Displacement (µm)")
    axs_env[1].set_xlabel("Time (min)")
    axs_env[1].legend(frameon=False)
    _closed_box(axs_env[1])
    axs_env[1].tick_params(left=False, top=False, right=False)

    for D, col in [(0.2, "tab:blue"), (0.5, "tab:orange"), (1.0, "tab:purple")]:
        rw = _simulate_rw(n_tracks, time_axis, drift=0.0, diffusion=D, seed=2)
        m = np.mean(rw, axis=0)
        s = np.std(rw, axis=0)
        axs_env[2].plot(time_axis, m, color=col, label=f"D={D:g}")
        axs_env[2].fill_between(time_axis, m - s, m + s, color=col, alpha=0.2)
    axs_env[2].set_ylabel("Displacement (µm)")
    axs_env[2].set_xlabel("Time (min)")
    axs_env[2].legend(frameon=False)
    _closed_box(axs_env[2])
    axs_env[2].tick_params(left=False, top=False, right=False)
    _set_axes_square(axs_env)
    plt.tight_layout()
    _save_all_fig(os.path.join(out_dir, "single_tracks_comparison_envelop"))
    plt.close(fig_env)

    # single_tracks_comparison_envelop_GRID_16_18_20
    classes = ["class_I", "class_IV"]
    hours = [16, 18, 20]
    fig_grid, axs_grid = plt.subplots(2, 3, figsize=(3 * basic_figsize[0], 2 * basic_figsize[1]), sharex=False, sharey=False)
    stats_rows = []
    for r, cls in enumerate(classes):
        for c, hour in enumerate(hours):
            ax = axs_grid[r, c]
            key = f"{cls}_{hour}h"
            if key not in params or "indep" not in params[key] or "dep" not in params[key]:
                ax.set_title(f"{_class_label(cls)} {hour}h AEL ($\\mathrm{{n}}_{{\\mathrm{{trajs}}}}=0$, missing)")
                _closed_box(ax)
                stats_rows.append(
                    {
                        "class": cls,
                        "hour": hour,
                        "n_tracks": 0,
                        "n_neurons": 0,
                        "D_indep": np.nan,
                        "CI_indep": "NA",
                        "D_dep": np.nan,
                        "CI_dep": "NA",
                        "D_real": np.nan,
                        "CI_real": "NA",
                        "p(indep vs real)": np.nan,
                        "p(dep vs real)": np.nan,
                        "p(indep vs dep)": np.nan,
                    }
                )
                continue
            np.random.seed(1234567)
            tracks_ind, _ = _linear_phases_indep(T_max, dt, n_tracks, **params[key]["indep"], mode="sample_velocity")
            tracks_dep2, _ = _linear_phases_dep(T_max, dt, n_tracks, **params[key]["dep"])
            m_i = np.mean(tracks_ind, axis=0)
            s_i = np.std(tracks_ind, axis=0)
            m_d = np.mean(tracks_dep2, axis=0)
            s_d = np.std(tracks_dep2, axis=0)
            ax.plot(time_axis, m_i, color=color_indep, label="Simu. indep.")
            ax.fill_between(time_axis, m_i - s_i, m_i + s_i, color=color_indep, alpha=0.3)
            ax.plot(time_axis, m_d, color=color_dep, label="Simu. dep.")
            ax.fill_between(time_axis, m_d - s_d, m_d + s_d, color=color_dep, alpha=0.3)
            rm2, rs2, _ = _real_tracks_envelope(time_axis, list_branch, cls, hour, min_duration_min=8.0)
            if rm2 is not None and rs2 is not None:
                ax.plot(time_axis, rm2, color="black", label="Data")
                ax.fill_between(time_axis, rm2 - rs2, rm2 + rs2, color="black", alpha=0.15)
                _, _, real_tracks_panel = _real_tracks_envelope(time_axis, list_branch, cls, hour, min_duration_min=8.0)
            else:
                real_tracks_panel = None
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Displacement (µm)")
            _closed_box(ax)
            ax.tick_params(left=False, top=False, right=False)

            names = set()
            n_tracks_real = 0
            for br in list_branch:
                if br.get("neuron_class") != cls:
                    continue
                hour_br = br.get("inner_time_fallback", np.nan)
                if not np.isfinite(hour_br) or int(round(float(hour_br))) != int(hour):
                    continue
                t = br.get("length_time", None)
                y = br.get("length_length", None)
                if t is None or y is None or len(t) == 0 or len(y) == 0:
                    continue
                t = np.asarray(t, float)
                if (t[-1] - t[0]) < 8.0:
                    continue
                n_tracks_real += 1
                name = br.get("name", None)
                if name is not None:
                    names.add(name)
            ax.set_title(f"C{_class_label(cls)[1:]} {hour}h ($\\mathrm{{n}}_{{\\mathrm{{trajs}}}}={n_tracks_real}$)")

            d_ind = _diffusion_from_msd_nan(tracks_ind, time_axis, detrend=True)
            d_dep = _diffusion_from_msd_nan(tracks_dep2, time_axis, detrend=True)
            d_real = (
                _diffusion_from_msd_nan(real_tracks_panel, time_axis, detrend=True)
                if real_tracks_panel is not None
                else np.nan
            )
            ci_ind = _bootstrap_ci(
                tracks_ind,
                time_axis,
                lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
                n_boot=n_boot_grid_stats,
                alpha=0.05,
                seed=1,
            )
            ci_dep = _bootstrap_ci(
                tracks_dep2,
                time_axis,
                lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
                n_boot=n_boot_grid_stats,
                alpha=0.05,
                seed=2,
            )
            p_ind_dep = _bootstrap_diff_pvalue(
                tracks_ind,
                tracks_dep2,
                time_axis,
                lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
                lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
                n_boot=n_boot_grid_stats,
                seed=12,
            )
            if real_tracks_panel is not None:
                ci_real = _bootstrap_ci(
                    real_tracks_panel,
                    time_axis,
                    lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
                    n_boot=n_boot_grid_stats,
                    alpha=0.05,
                    seed=3,
                )
                p_ind_real = _bootstrap_diff_pvalue(
                    tracks_ind,
                    real_tracks_panel,
                    time_axis,
                    lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
                    lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
                    n_boot=n_boot_grid_stats,
                    seed=10,
                )
                p_dep_real = _bootstrap_diff_pvalue(
                    tracks_dep2,
                    real_tracks_panel,
                    time_axis,
                    lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
                    lambda x, t: _diffusion_from_msd_nan(x, t, detrend=True),
                    n_boot=n_boot_grid_stats,
                    seed=11,
                )
            else:
                ci_real = (np.nan, np.nan)
                p_ind_real = np.nan
                p_dep_real = np.nan

            def _ci_str(ci):
                lo, hi = ci
                if np.isfinite(lo) and np.isfinite(hi):
                    return f"[{lo:.3g}, {hi:.3g}]"
                return "NA"

            stats_rows.append(
                {
                    "class": cls,
                    "hour": hour,
                    "n_tracks": int(n_tracks_real),
                    "n_neurons": int(len(names)),
                    "D_indep": d_ind,
                    "CI_indep": _ci_str(ci_ind),
                    "D_dep": d_dep,
                    "CI_dep": _ci_str(ci_dep),
                    "D_real": d_real,
                    "CI_real": _ci_str(ci_real),
                    "p(indep vs real)": p_ind_real,
                    "p(dep vs real)": p_dep_real,
                    "p(indep vs dep)": p_ind_dep,
                }
            )
            label_ind = f"Simu. indep. D={d_ind:.3g} $\\mathrm{{\\mu m^{{2}}/min}}$ {_fmt_boot_p_label(p_ind_real)}" if np.isfinite(d_ind) else "Simu. indep. D=NA $\\mathrm{{\\mu m^{{2}}/min}}$ p=NA"
            label_dep = f"Simu. dep. D={d_dep:.3g} $\\mathrm{{\\mu m^{{2}}/min}}$ {_fmt_boot_p_label(p_dep_real)}" if np.isfinite(d_dep) else "Simu. dep. D=NA $\\mathrm{{\\mu m^{{2}}/min}}$ p=NA"
            label_real = (
                f"Data WT class {cls.split('_')[-1]} {hour}h D={d_real:.3g} $\\mathrm{{\\mu m^{{2}}/min}}$"
                if np.isfinite(d_real)
                else f"Data WT class {cls.split('_')[-1]} {hour}h D=NA $\\mathrm{{\\mu m^{{2}}/min}}$"
            )
            ax.lines[0].set_label(label_ind)
            ax.lines[1].set_label(label_dep)
            if np.isfinite(d_real):
                ax.lines[2].set_label(label_real)
            ax.legend(frameon=False, fontsize=4, handlelength=1.0)
    _set_axes_square(axs_grid)
    plt.tight_layout()
    _save_all_fig(os.path.join(out_dir, "single_tracks_comparison_envelop_GRID_16_18_20"))
    plt.close(fig_grid)

    # Save stats table for grid figure (.md/.tex/.svg)
    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df[
        [
            "class",
            "hour",
            "n_tracks",
            "n_neurons",
            "D_indep",
            "CI_indep",
            "D_dep",
            "CI_dep",
            "D_real",
            "CI_real",
            "p(indep vs real)",
            "p(dep vs real)",
            "p(indep vs dep)",
        ]
    ]
    stats_df["hour"] = stats_df["hour"].map(lambda h: f"{int(h)}")
    stats_disp = _stringify_numeric_df(stats_df)
    # p-values here come from bootstrap proportions with n_boot=n_boot_grid_stats. so minimal resolution for pv_alue is 2/n_boot.
    p_floor = p_floor_grid_stats
    for pcol in ["p(indep vs real)", "p(dep vs real)", "p(indep vs dep)"]:
        if pcol in stats_df.columns and pcol in stats_disp.columns:
            pvals = stats_df[pcol].to_numpy(float)
            mfin = np.isfinite(pvals)
            mbelow = mfin & (pvals <= p_floor)
            mabove = mfin & (~mbelow)
            if np.any(mbelow):
                stats_disp.loc[mbelow, pcol] = f"<{p_floor:.2g}"
            if np.any(mabove):
                stats_disp.loc[mabove, pcol] = [f"{v:.2g}" for v in pvals[mabove]]
    _write_df_markdown(stats_disp, os.path.join(stats_out_dir, "single_tracks_comparison_envelop_GRID_16_18_20_stats.md"))
    _write_df_csv(stats_disp, os.path.join(stats_out_dir, "single_tracks_comparison_envelop_GRID_16_18_20_stats.csv"))


def _export_plot_test_tables(plot_phase_df, phase_df, out_dir):
    #amrutha-carac-time-lesscomp-None
    df = pd.DataFrame(
        {
            "Hour": plot_phase_df["hour_label"],
            "neuron class": plot_phase_df["class_pretty"],
            "slope": plot_phase_df["slope_abs"],
            "duration": plot_phase_df["duration"],
            "is_growing": plot_phase_df["side"],
        }
    )
    df = df[np.isfinite(df["slope"]) & np.isfinite(df["duration"]) & (df["duration"] > 0)].copy()

    rows = []
    families = {}
    legacy_hour_order = {
        "class I": ["16h AEL", "18h AEL", "20h AEL"],
        "class IV": ["16h AEL", "18h AEL", "20h AEL"],
    }
    for n_class in ["class I", "class IV"]:
        hour_order = legacy_hour_order[n_class]
        for metric in ["slope", "duration"]:
            fam_key = f"carac_time::{n_class}::{metric}"
            fam_idx = []
            for hour in hour_order:
                g1 = df[(df["neuron class"] == n_class) & (df["Hour"] == hour) & (df["is_growing"] == "Growth")][metric].to_numpy(float)
                g2 = df[(df["neuron class"] == n_class) & (df["Hour"] == hour) & (df["is_growing"] == "Shrinkage")][metric].to_numpy(float)
                g1 = g1[np.isfinite(g1)]
                g2 = g2[np.isfinite(g2)]
                welch = _welch_stats_with_ci(g1, g2, alpha=0.05)
                rows.append(
                    {
                        "plot": "amrutha-carac-time-lesscomp-None",
                        "family": fam_key,
                        "class": n_class,
                        "metric": metric,
                        "hour": hour,
                        "group_a": "Growth",
                        "group_b": "Shrinkage",
                        "n_a": int(g1.size),
                        "n_b": int(g2.size),
                        "mean_a": welch["mean_a"],
                        "mean_b": welch["mean_b"],
                        "mean_diff_a_minus_b": welch["mean_diff"],
                        "t_stat": welch["t_stat"],
                        "welch_df": welch["welch_df"],
                        "mean_diff_ci95_low": welch["mean_diff_ci95_low"],
                        "mean_diff_ci95_high": welch["mean_diff_ci95_high"],
                        "p_raw": welch["p_raw"],
                    }
                )
                fam_idx.append(len(rows) - 1)
            families[fam_key] = fam_idx

    #amrutha-kparams-class-lesscomp-16_18_20
    dfi = _build_indep_param_phase_df(phase_df)
    if not dfi.empty:
        hour_order = ["16h AEL", "18h AEL", "20h AEL"]
        for metric in ["von", "voff", "tau_on", "tau_off"]:
            fam_key = f"kparams_class::{metric}"
            fam_idx = []
            for hour in hour_order:
                g1 = dfi[(dfi["Class"] == "class I") & (dfi["Hour"] == hour)][metric].to_numpy(float)
                g2 = dfi[(dfi["Class"] == "class IV") & (dfi["Hour"] == hour)][metric].to_numpy(float)
                g1 = g1[np.isfinite(g1)]
                g2 = g2[np.isfinite(g2)]
                welch = _welch_stats_with_ci(g1, g2, alpha=0.05)
                rows.append(
                    {
                        "plot": "amrutha-kparams-class-lesscomp-16_18_20",
                        "family": fam_key,
                        "class": "both",
                        "metric": metric,
                        "hour": hour,
                        "group_a": "class I",
                        "group_b": "class IV",
                        "n_a": int(g1.size),
                        "n_b": int(g2.size),
                        "mean_a": welch["mean_a"],
                        "mean_b": welch["mean_b"],
                        "mean_diff_a_minus_b": welch["mean_diff"],
                        "t_stat": welch["t_stat"],
                        "welch_df": welch["welch_df"],
                        "mean_diff_ci95_low": welch["mean_diff_ci95_low"],
                        "mean_diff_ci95_high": welch["mean_diff_ci95_high"],
                        "p_raw": welch["p_raw"],
                    }
                )
                fam_idx.append(len(rows) - 1)
            families[fam_key] = fam_idx

    tests_df = pd.DataFrame(rows)
    if tests_df.empty:
        return
    tests_df["p_holm"] = np.nan
    for _, idxs in families.items():
        pvals = tests_df.loc[idxs, "p_raw"].to_numpy(float)
        tests_df.loc[idxs, "p_holm"] = _holm(pvals)

    tests_df = tests_df[
        [
            "plot",
            "family",
            "class",
            "metric",
            "hour",
            "group_a",
            "group_b",
            "n_a",
            "n_b",
            "mean_a",
            "mean_b",
            "mean_diff_a_minus_b",
            "t_stat",
            "welch_df",
            "mean_diff_ci95_low",
            "mean_diff_ci95_high",
            "p_raw",
            "p_holm",
        ]
    ]
    _write_df_csv(tests_df, os.path.join(out_dir, "unified_plot_stat_tests.csv"))
    tests_disp = _stringify_numeric_df(tests_df)
    if "welch_df" in tests_disp.columns:
        tests_disp["welch_df"] = tests_df["welch_df"].map(
            lambda x: str(int(round(float(x)))) if np.isfinite(x) else "NA"
        )
    _write_df_markdown(tests_disp, os.path.join(out_dir, "unified_plot_stat_tests.md"))


def _load_results():
    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"Missing input results.json at: {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, "r") as f:
        data = json.load(f)
    for i in range(len(data)):
        for k in list(data[i].keys()):
            if "length" in k or "phases" in k:
                data[i][k] = np.asarray(data[i][k])
    return data, INPUT_JSON_PATH


def _merge_same_sign_phases(durations, slopes):
    d = np.asarray(durations, dtype=float)
    s = np.asarray(slopes, dtype=float)
    if d.size == 0 or s.size == 0 or d.size != s.size:
        return np.array([], dtype=float), np.array([], dtype=float)
    out_d = [float(d[0])]
    out_s = [float(s[0])]
    for i in range(1, d.size):
        if s[i] * out_s[-1] >= 0:
            new_d = out_d[-1] + d[i]
            if new_d > 0:
                out_s[-1] = (out_s[-1] * out_d[-1] + s[i] * d[i]) / new_d
            out_d[-1] = new_d
        else:
            out_d.append(float(d[i]))
            out_s.append(float(s[i]))
    return np.asarray(out_d, float), np.asarray(out_s, float)


def _phase_mask(n):
    idx = np.arange(int(n))
    return np.ones(int(n), dtype=bool)


def _parse_hour_class(br):
    h = br.get("inner_time_fallback", np.nan)
    if not np.isfinite(h):
        return None, None
    hour = int(round(float(h)))
    nclass = br.get("neuron_class", "")
    if hour not in HOURS or nclass not in CLASSES:
        return None, None
    return hour, nclass


def _to_phase_df(list_branch):
    rows = []
    for traj_idx, br in enumerate(list_branch):
        hour, nclass = _parse_hour_class(br)
        if hour is None:
            continue
        acq = br.get("name", "")
        traj_id = f"{nclass}|{hour}|{acq}|{traj_idx}"
        d = br.get("phases_duration (min)")
        s = br.get("phases_slopes (micron/min)")
        if d is None or s is None:
            continue
        d = np.asarray(d, float)
        s = np.asarray(s, float)
        m = np.isfinite(d) & np.isfinite(s) & (d > 0) & (np.abs(s) <= MAX_ABS_V)
        d = d[m]
        s = s[m]
        if d.size == 0:
            continue
        if MERGE_SAME_SIGN:
            d, s = _merge_same_sign_phases(d, s)
        if d.size == 0:
            continue
        keep = _phase_mask(d.size)
        d = d[keep]
        s = s[keep]
        if d.size == 0:
            continue
        for si, di in zip(s, d):
            rows.append(
                {
                    "class": nclass,
                    "hour": hour,
                    "acq": acq,
                    "traj_id": traj_id,
                    "side": "Growth" if si >= 0 else "Shrinkage",
                    "slope_abs": abs(float(si)),
                    "duration": float(di),
                }
            )
    return pd.DataFrame(rows)


def _mean_std_n(x):
    arr = np.asarray(x, float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return np.nan, np.nan, 0
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if n > 1 else np.nan, n


def _mean_se_n(x):
    m, sd, n = _mean_std_n(x)
    se = (sd / np.sqrt(n)) if (n > 1 and np.isfinite(sd)) else np.nan
    return m, se, n


def _holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not np.any(finite):
        return out
    idx = np.where(finite)[0]
    p2 = p[idx]
    m = len(p2)
    order = np.argsort(p2)
    adj = np.empty_like(p2)
    for rank, j in enumerate(order):
        adj[j] = min((m - rank) * p2[j], 1.0)
    # enforce monotone
    running = 0.0
    for j in order:
        running = max(running, adj[j])
        adj[j] = running
    out[idx] = adj
    return out


def _delta_se(func, x, ses, rel_eps=1e-6):
    x0 = {k: float(v) for k, v in x.items()}
    var = 0.0
    for k in x0:
        sk = float(ses.get(k, np.nan))
        if (not np.isfinite(sk)) or sk == 0:
            continue
        h = rel_eps * max(1.0, abs(x0[k]))
        xp = dict(x0)
        xm = dict(x0)
        xp[k] += h
        xm[k] -= h
        gk = (func(**xp) - func(**xm)) / (2 * h)
        var += (gk * sk) ** 2
    return float(np.sqrt(var))


def _welch_stats_with_ci(g1, g2, alpha=0.05):
    a = np.asarray(g1, float)
    b = np.asarray(g2, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1 = int(a.size)
    n2 = int(b.size)
    mean_a = float(np.mean(a)) if n1 else np.nan
    mean_b = float(np.mean(b)) if n2 else np.nan
    mean_diff = mean_a - mean_b if (np.isfinite(mean_a) and np.isfinite(mean_b)) else np.nan
    if n1 < 2 or n2 < 2:
        return {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": mean_diff,
            "t_stat": np.nan,
            "p_raw": np.nan,
            "welch_df": np.nan,
            "mean_diff_ci95_low": np.nan,
            "mean_diff_ci95_high": np.nan,
        }

    s1 = float(np.std(a, ddof=1))
    s2 = float(np.std(b, ddof=1))
    v1 = (s1 ** 2) / n1
    v2 = (s2 ** 2) / n2
    se = float(np.sqrt(v1 + v2))
    denom = (v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1)
    welch_df = float(((v1 + v2) ** 2) / denom) if denom > 0 else np.nan

    if np.isfinite(welch_df) and np.isfinite(se) and se > 0:
        t_stat = float(mean_diff / se)
        p_raw = float(2 * stats.t.sf(abs(t_stat), df=welch_df))
        tcrit = float(stats.t.ppf(1 - alpha / 2.0, df=welch_df))
        ci_low = float(mean_diff - tcrit * se)
        ci_high = float(mean_diff + tcrit * se)
    else:
        t_stat = np.nan
        p_raw = np.nan
        ci_low = np.nan
        ci_high = np.nan

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_raw": p_raw,
        "welch_df": welch_df,
        "mean_diff_ci95_low": ci_low,
        "mean_diff_ci95_high": ci_high,
    }


def _k_from_tau(mean_tau, se_tau):
    if not (np.isfinite(mean_tau) and mean_tau > 0):
        return np.nan, np.nan
    mean_k = 1.0 / mean_tau
    se_k = (se_tau / (mean_tau**2)) if np.isfinite(se_tau) else np.nan
    return float(mean_k), float(se_k)


def _slope_duration_summary(phase_df):
    rows = []
    for nclass in CLASSES:
        for hour in HOURS:
            for side in SIDES:
                sub = phase_df[
                    (phase_df["class"] == nclass) & (phase_df["hour"] == hour) & (phase_df["side"] == side)
                ]
                if sub.empty:
                    rows.append(
                        {
                            "class": nclass,
                            "hour": hour,
                            "side": side,
                            "n": 0,
                            "slope_mean": np.nan,
                            "slope_std": np.nan,
                            "duration_mean": np.nan,
                            "duration_std": np.nan,
                        }
                    )
                    continue
                sm, ss, n = _mean_std_n(sub["slope_abs"].values)
                dm, ds, _ = _mean_std_n(sub["duration"].values)
                rows.append(
                    {
                        "class": nclass,
                        "hour": hour,
                        "side": side,
                        "n": n,
                        "slope_mean": sm,
                        "slope_std": ss,
                        "duration_mean": dm,
                        "duration_std": ds,
                    }
                )
    return pd.DataFrame(rows)


def _estimate_independent(phase_df):
    rows = []
    for nclass in CLASSES:
        for hour in HOURS:
            sub = phase_df[(phase_df["class"] == nclass) & (phase_df["hour"] == hour)]
            on = sub[sub["side"] == "Growth"]
            off = sub[sub["side"] == "Shrinkage"]
            n_phase_on = int(on.shape[0])
            n_phase_off = int(off.shape[0])
            if on.empty or off.empty:
                continue
            von_m, von_se, _ = _mean_se_n(on["slope_abs"])
            voff_m, voff_se, _ = _mean_se_n(off["slope_abs"])
            ton_m, ton_se, _ = _mean_se_n(on["duration"])
            toff_m, toff_se, _ = _mean_se_n(off["duration"])
            kon_m, kon_se = _k_from_tau(toff_m, toff_se)
            koff_m, koff_se = _k_from_tau(ton_m, ton_se)

            def _drift(von, voff, kon, koff):
                return (von * kon - voff * koff) / (kon + koff)

            def _D(von, voff, kon, koff):
                return kon * koff * (von + voff) ** 2 / ((kon + koff) ** 3)

            drift_m = _drift(von_m, voff_m, kon_m, koff_m)
            D_m = _D(von_m, voff_m, kon_m, koff_m)
            drift_s = _delta_se(
                _drift,
                {"von": von_m, "voff": voff_m, "kon": kon_m, "koff": koff_m},
                {"von": von_se, "voff": voff_se, "kon": kon_se, "koff": koff_se},
            )
            D_s = _delta_se(
                _D,
                {"von": von_m, "voff": voff_m, "kon": kon_m, "koff": koff_m},
                {"von": von_se, "voff": voff_se, "kon": kon_se, "koff": koff_se},
            )
            z = drift_m / drift_s if (np.isfinite(drift_s) and drift_s > 0) else np.nan
            p_drift = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
            drift_se_for_ci = drift_s
            von_s, voff_s, kon_s, koff_s = von_se, voff_se, kon_se, koff_se
            rows.append(
                {
                    "class": nclass,
                    "hour": f"{hour}h",
                    "n_phase_on": n_phase_on,
                    "n_phase_off": n_phase_off,
                    "von_mean": von_m,
                    "von_std_or_se": von_s,
                    "voff_mean": voff_m,
                    "voff_std_or_se": voff_s,
                    "kon_mean": kon_m,
                    "kon_std_or_se": kon_s,
                    "koff_mean": koff_m,
                    "koff_std_or_se": koff_s,
                    "drift_mean": drift_m,
                    "drift_std_or_se": drift_s,
                    "drift_se_for_ci": drift_se_for_ci,
                    "drift_p": p_drift,
                    "D_mean": D_m,
                    "D_std_or_se": D_s,
                }
            )
    return pd.DataFrame(rows).sort_values(["class", "hour"])


def _fit_dep_block(sub):
    v = np.asarray(sub["slope_abs"], float)
    t = np.asarray(sub["duration"], float)
    m = np.isfinite(v) & np.isfinite(t) & (t > 0)
    v = v[m]
    t = t[m]
    if v.size < 3 or np.unique(v).size < 2:
        return None
    y = np.log(t)
    lr = stats.linregress(v, y)
    b = float(lr.slope)
    a = float(lr.intercept)
    yhat = a + b * v
    resid = y - yhat
    sigma = float(np.sqrt(np.mean(resid**2)))
    v0 = float(np.mean(v))
    v1 = float(1.0 / b) if b != 0 else np.nan
    n = int(v.size)
    v0_se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    b_se = float(getattr(lr, "stderr", np.nan))
    df_b = float(max(n - 2, 0))
    t_b = float(b / b_se) if (np.isfinite(b_se) and b_se > 0) else np.nan
    if np.isfinite(df_b) and df_b > 0 and np.isfinite(b_se):
        tcrit = float(stats.t.ppf(1 - 0.05 / 2.0, df_b))
        b_ci95_low = float(b - tcrit * b_se)
        b_ci95_high = float(b + tcrit * b_se)
    else:
        b_ci95_low, b_ci95_high = np.nan, np.nan
    v1_se = float(abs(b_se / (b**2))) if np.isfinite(b_se) and b != 0 else np.nan
    tau0_se = float(getattr(lr, "intercept_stderr", np.nan))
    sigma_se = float(sigma / np.sqrt(2 * max(n - 2, 1))) if n > 1 else np.nan
    return {
        "v0": v0,
        "v1": v1,
        "tau0": a,
        "sigma": sigma,
        "v0_se": v0_se,
        "v1_se": v1_se,
        "tau0_se": tau0_se,
        "sigma_se": sigma_se,
        "n": n,
        "b": b,
        "t_b": t_b,
        "df_b": df_b,
        "b_ci95_low": b_ci95_low,
        "b_ci95_high": b_ci95_high,
        "p_b": float(lr.pvalue),
    }


def _estimate_dependent(phase_df):
    rows = []
    for nclass in CLASSES:
        for hour in HOURS:
            sub_all = phase_df[(phase_df["class"] == nclass) & (phase_df["hour"] == hour)]
            n_phase_on = int((sub_all["side"] == "Growth").sum())
            n_phase_off = int((sub_all["side"] == "Shrinkage").sum())
            for side in SIDES:
                sub = sub_all[sub_all["side"] == side]
                if sub.empty:
                    continue
                out = _fit_dep_block(sub)
                if out is None:
                    continue
                v0_m, v1_m, tau0_m, sigma_m = out["v0"], out["v1"], out["tau0"], out["sigma"]
                v0_s, v1_s, tau0_s, sigma_s = (
                    out["v0_se"],
                    out["v1_se"],
                    out["tau0_se"],
                    out["sigma_se"],
                )
                p_v1 = out["p_b"]
                t_b = out["t_b"]
                df_b = out["df_b"]
                b_est = out["b"]
                b_ci95_low = out["b_ci95_low"]
                b_ci95_high = out["b_ci95_high"]
                rows.append(
                    {
                        "class": nclass,
                        "hour": f"{hour}h",
                        "side": side,
                        "n_phase_on": n_phase_on,
                        "n_phase_off": n_phase_off,
                        "v0_mean": v0_m,
                        "v0_std_or_se": v0_s,
                        "v1_mean": v1_m,
                        "v1_std_or_se": v1_s,
                        "p_v1": p_v1,
                        "model_equation": "log(duration) = a + b*|v| + error",
                        "tested_coefficient": "b (slope of log(duration) vs |v|; v1=1/b)",
                        "b_estimate": b_est,
                        "b_ci95_low": b_ci95_low,
                        "b_ci95_high": b_ci95_high,
                        "test_statistic_t": t_b,
                        "test_df": df_b,
                        "p_value_test": "Wald t-test on b",
                        "p_value_h0": "H0: b = 0",
                        "p_value_sidedness": "two-sided",
                        "tau0_mean": tau0_m,
                        "tau0_std_or_se": tau0_s,
                        "sigma_mean": sigma_m,
                        "sigma_std_or_se": sigma_s,
                    }
                )
    return pd.DataFrame(rows).sort_values(["class", "hour", "side"])


if __name__ == "__main__":

    np.random.seed(GLOBAL_RANDOM_SEED)

    list_branch, input_path = _load_results()
    phase_df = _to_phase_df(list_branch)
    if phase_df.empty:
        raise RuntimeError("Dataset empty after filtering")

    slope_duration_df = _slope_duration_summary(phase_df)
    indep_df = _estimate_independent(phase_df)
    dep_df = _estimate_dependent(phase_df)
    slope_duration_display_df = _format_slope_duration_table(slope_duration_df)
    indep_display_df = _format_independent_table(indep_df)
    dep_display_df = _format_dependent_table(dep_df)

    indep_md_notes = [
        "± reports mean ± standard error.",
        "drift_95CI is computed as mean ± 1.96xSE.",
    ]
    _write_df_markdown(
        indep_display_df,
        os.path.join(RESULT_DIR, "unified_kon_koff_von_voff_drift_diffusion.md"),
        preamble_lines=indep_md_notes,
    )
    _write_df_csv(indep_display_df, os.path.join(RESULT_DIR, "unified_kon_koff_von_voff_drift_diffusion.csv"))

    _write_df_markdown(dep_display_df, os.path.join(RESULT_DIR, "unified_duration_dependent_parameters.md"))
    _write_df_csv(dep_display_df, os.path.join(RESULT_DIR, "unified_duration_dependent_parameters.csv"))

    plot_phase_df = _build_plot_phase_df(phase_df)
    _plot_duration_vs_slope_linabs_grid(plot_phase_df, PLOTS_DIR)
    _plot_amrutha_carac_time_lesscomp(plot_phase_df, PLOTS_DIR)
    _plot_indep_param_class_compare(phase_df, PLOTS_DIR)
    _plot_param_distributions_grid(phase_df, PLOTS_DIR)
    _plot_legacy_track_figures(list_branch, indep_df, dep_df, PLOTS_DIR, stats_out_dir=RESULT_DIR)

    _export_plot_test_tables(plot_phase_df, phase_df, RESULT_DIR)
