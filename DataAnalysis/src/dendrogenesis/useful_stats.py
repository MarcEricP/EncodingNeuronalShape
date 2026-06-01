import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

def slope_intercept_with_robust_and_cluster(df, y="y", a="a", exp=None, alpha=0.05):
    # Fit y ~ a and return models + structured stats.
    models = {}

    # Base OLS model
    m = smf.ols(f"{y} ~ {a}", data=df).fit(use_t=True)
    models["base"] = m

    # HC3 robust model
    m_HC3 = m.get_robustcov_results(cov_type="HC3")
    models["hc3"] = m_HC3

    # FE + clustered SE by experiment
    if exp is not None and exp in df.columns:
        fe = smf.ols(f"{y} ~ {a} + C({exp})", data=df).fit(use_t=True)
        fe_clu = fe.get_robustcov_results(cov_type="cluster", groups=df[exp])
        models["fe_cluster"] = fe_clu

    def _tt(result, term):
        t = result.t_test(f"{term} = 0")
        ci = np.asarray(t.conf_int(alpha=alpha), dtype=float).ravel()
        return {
            "estimate": float(np.asarray(t.effect).item()),
            "se": float(np.asarray(t.sd).item()),
            "t_or_z": float(np.asarray(t.tvalue).item()),
            "p_two_sided": float(np.asarray(t.pvalue).item()),
            "ci_low": float(ci[0]) if ci.size >= 2 else np.nan,
            "ci_high": float(ci[1]) if ci.size >= 2 else np.nan,
        }

    stats_dict = {
        "base": _tt(m, a),
        "hc3": _tt(m_HC3, a),
        "fe_cluster": _tt(models["fe_cluster"], a) if "fe_cluster" in models else None,
        "alpha": alpha,
    }
    return models, stats_dict



def compare_two_series(Y1, Y2, a, *, alpha=0.05, center_at=None, return_per_time=False):
    # Compare two unpaired series across time with FE + cluster-robust stats.

    # Inputs
    Y1 = np.asarray(Y1, float)
    Y2 = np.asarray(Y2, float)
    a = np.asarray(a, float)
    n1, T1 = Y1.shape
    n2, T2 = Y2.shape
    assert T1 == T2 == a.size, "time dimension mismatch between Y1, Y2, and a"

    # Optional anchor centering
    if center_at is None:
        a_fit = a.copy()
        anchor_note = "a=0"
    else:
        a_fit = a - float(center_at)
        anchor_note = f"a={center_at:g}"

    # Per-time output is disabled in scripts 2 and 5.
    per_time = None

    # Overall FE regression
    lab1, lab2 = "S1", "S2"
    # Build long dataframe
    df1 = pd.DataFrame({
        "y": Y1.ravel(),
        "a": np.tile(a_fit, n1),
        "exp": np.repeat(np.arange(n1), a.size),
        "series": lab1,
    })
    df2 = pd.DataFrame({
        "y": Y2.ravel(),
        "a": np.tile(a_fit, n2),
        "exp": np.repeat(np.arange(n2), a.size),   # always consistent with Y2 length
        "series": lab2,
    })
    df = pd.concat([df1, df2], ignore_index=True)

    # Unpaired FE model with exp-within-series clustering
    df["exp_series"] = df["series"].astype(str) + "_" + df["exp"].astype(str)
    fe = smf.ols("y ~ a + C(series) + a:C(series) + C(exp):C(series)", data=df).fit(use_t=True)
    model_overall = fe.get_robustcov_results(cov_type="cluster", groups=df["exp_series"])
    cluster_note = f"unpaired; clusters={df['exp_series'].nunique()} exp-within-series"

    # Omnibus + contrasts
    names = list(model_overall.model.exog_names)
    idx = {nm: i for i, nm in enumerate(names)}
    level_term = f"C(series)[T.{lab2}]"
    cand1 = f"a:C(series)[T.{lab2}]"
    cand2 = f"C(series)[T.{lab2}]:a"
    slope_term = cand1 if cand1 in idx else (cand2 if cand2 in idx else None)

    terms = [t for t in (level_term, slope_term) if t and t in idx]
    Ftest = None
    if terms:
        R = np.zeros((len(terms), len(names)))
        for r, term in enumerate(terms):
            R[r, idx[term]] = 1.0
        Ftest = model_overall.f_test(R)

    level_test = model_overall.t_test(level_term) if level_term in idx else None
    slope_test = model_overall.t_test(slope_term) if (slope_term and slope_term in idx) else None

    def _safe_float(x):
        try:
            return float(np.asarray(x).item())
        except Exception:
            return np.nan
    stats_dict = {
        "p_level_diff": _safe_float(level_test.pvalue) if level_test is not None else np.nan,
        "p_slope_diff": _safe_float(slope_test.pvalue) if slope_test is not None else np.nan,
        "p_omnibus": _safe_float(Ftest.pvalue) if Ftest is not None else np.nan,
        "anchor_note": anchor_note,
        "cluster_note": cluster_note,
        "alpha": alpha,
    }
    return per_time, model_overall, stats_dict
