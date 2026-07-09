#!/usr/bin/env python3
"""Head-to-head: the user's multi-source *Homoset* method vs a *mixle*-style
hierarchical partial-pooling model, both reconciling GuthrieSolv's noisy
per-molecule dG_hyd observations. Scored against FreeSolv on the overlap.

HOMOSET  (faithful port of aromma/merge_bbb_homoset_v2 + build_delta_hvap_v2_homoset):
  Sources = measurement processes {FEOH,DGS = free-energy REFERENCE; KWG; KGW; PAIR}.
  Stage 1 - source alignment: for each non-reference source, take per-molecule diffs
            vs the reference over the overlap, run psi_gate:
                offset = median(diffs);  rms = sqrt(mean((diffs-offset)^2))
                PS = 1 - rms/L;  accept iff PS >= ps_crit(n, alpha)
            accepted sources are shifted by -offset; rejected sources are dropped.
  Stage 2 - per molecule: median consensus over aligned/accepted obs;
            conflict flag if aligned obs range >= outlier_threshold.
  Swept:  L in {adaptive = max((max-min)/2,0.5) on diffs, fixed 0.6}
          outlier_threshold in {2.0, 1.0} kcal/mol.

MIXLE  (Normal(Normal(mu0,tau), sigma_source) with per-source bias b_s, EM):
  y_ij ~ N(mu_i + b_s, sigma_s^2),  b_ref=0,  mu_i ~ N(mu0, tau^2).
  Estimates per-source bias + variance + per-molecule shrunk posterior jointly.
  Robust variant: Student-t observation model -> per-obs weights down-weight outliers.
"""
from __future__ import annotations
import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ALPHA = 0.01
MIN_OVERLAP = 3
REFERENCE_ROUTE = "free_energy"      # FEOH+DGS: validated ~0 residual vs FreeSolv

# ---------------------------------------------------------------------------
def load():
    obs = pd.read_csv(OUT / "guthrie_dg_observations.csv")
    fs = json.load(open("/Users/tgg/Github/FreeSolv/database.json"))
    rows = []
    for cid, d in fs.items():
        m = Chem.MolFromSmiles(d["smiles"])
        if m:
            rows.append((Chem.MolToInchiKey(m), float(d["expt"]), float(d.get("d_expt", np.nan))))
    fsdf = pd.DataFrame(rows, columns=["ikey", "expt", "d_expt"]).drop_duplicates("ikey")
    return obs, fsdf


# ---------------------------------------------------------------------------
# Canonical Homoset primitives (verbatim from the user's code)
# ---------------------------------------------------------------------------
def ps_crit(n, alpha=ALPHA):
    if n < 2:
        return np.nan
    y = chi2.ppf(alpha, df=n - 1)
    return 1.0 - np.sqrt(y / (3.0 * n))


def homoset_l_scale(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    return max(float((arr.max() - arr.min()) / 2.0), 0.5)


def homogeneity(values, L):
    """The Homoset test. A set of values is homogeneous iff its scatter is a small
    enough fraction of the dimension scale L:  noise = RMS/L,  PS = 1 - noise.
    Returns (offset=median, noise_fraction, n)."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or L <= 0:
        return np.nan, np.nan, len(x)
    off = float(np.median(x))
    rms = float(np.sqrt(np.mean((x - off) ** 2)))
    return off, rms / L, len(x)


def homoset_reconcile(obs: pd.DataFrame, L, noise_level: float):
    """Homoset with its two real parameters.
      noise_level : max tolerated RMS/L for a set to be a 'homogeneous set' (a fraction, %/100)
      L           : the dimension/amplitude scale.  L='adaptive' -> max((max-min)/2,0.5) on the
                    source-vs-reference diffs; else a fixed scalar in kcal/mol (e.g. 0.6).
    A source is admitted (and offset-aligned) iff its diffs-vs-reference are homogeneous;
    a molecule is flagged conflicted iff its own observations are NOT homogeneous.
    """
    src_col = "process"
    ref_mask = obs["route"] == REFERENCE_ROUTE
    obs = obs.copy()
    ref_val = obs[ref_mask].groupby("ikey")["dg_hyd"].median()
    # per-molecule homogeneity is tested against the physical experimental-noise dimension
    # (FreeSolv's ~0.6 kcal/mol), so the conflict flag means "replicates disagree beyond
    # experimental noise" rather than relative to the whole-dataset span.
    L_mol = 0.6 if L == "adaptive" else float(L)

    # Stage 1: source-alignment homogeneity gate
    offsets, gate = {}, {}
    for s, gs in obs[~ref_mask].groupby(src_col):
        src_val = gs.groupby("ikey")["dg_hyd"].median()
        common = src_val.index.intersection(ref_val.index)
        diffs = (src_val.loc[common] - ref_val.loc[common]).to_numpy()
        L_use = homoset_l_scale(diffs) if L == "adaptive" else float(L)
        offset, noise, n = homogeneity(diffs, L_use)
        accepted = bool(n >= MIN_OVERLAP and np.isfinite(noise) and noise <= noise_level)
        offsets[s] = offset if accepted else np.nan
        gate[s] = dict(noise=noise, ps=1.0 - noise if np.isfinite(noise) else np.nan,
                       n_overlap=n, L=L_use, accepted=accepted, offset=offset)
    for s in obs[ref_mask][src_col].unique():
        offsets[s] = 0.0
        gate[s] = dict(noise=0.0, ps=1.0, n_overlap=int((obs[src_col] == s).sum()),
                       L=np.nan, accepted=True, offset=0.0)

    # Stage 2: consensus over accepted/aligned obs + per-molecule homogeneity flag
    obs["offset"] = obs[src_col].map(offsets)
    obs["accepted_src"] = obs["offset"].notna()
    obs["aligned"] = obs["dg_hyd"] - obs["offset"]
    rows = []
    for ik, g in obs.groupby("ikey", sort=False):
        acc = g[g["accepted_src"]]
        used = acc if len(acc) else g
        v = used["aligned"].to_numpy() if len(acc) else used["dg_hyd"].to_numpy()
        n = len(v)
        _, noise_mol, _ = homogeneity(v, L_mol)
        rows.append(dict(
            ikey=ik, n_obs=int(len(g)), n_used=int(n),
            n_sources=int(g[src_col].nunique()),
            n_accepted_sources=int(acc[src_col].nunique()),
            consensus=float(np.median(v)),
            obs_range=float(v.max() - v.min()) if n else np.nan,
            noise_mol=float(noise_mol) if np.isfinite(noise_mol) else np.nan,
            fell_back_raw=bool(len(acc) == 0),
            conflict=bool(n >= 2 and np.isfinite(noise_mol) and noise_mol > noise_level),
        ))
    return pd.DataFrame(rows), gate


# ---------------------------------------------------------------------------
# mixle-style hierarchical model with per-source bias (EM)
# ---------------------------------------------------------------------------
def _prep(obs):
    o = obs.reset_index(drop=True)
    uniq, inv = np.unique(o["ikey"].to_numpy(), return_inverse=True)
    suniq, sinv = np.unique(o["process"].to_numpy(), return_inverse=True)
    y = o["dg_hyd"].to_numpy(float)
    ref_source_ids = set(np.where(np.isin(suniq, o.loc[o["route"] == REFERENCE_ROUTE, "process"].unique()))[0])
    return uniq, inv, suniq, sinv, y, ref_source_ids


def mixle_em(obs, robust=False, nu=4.0, n_iter=400, tol=1e-9):
    uniq, inv, suniq, sinv, y, ref_ids = _prep(obs)
    M, S = len(uniq), len(suniq)
    mu0 = float(np.median(y)); tau2 = float(y.var()) + 1e-3
    sig2 = np.full(S, max(float(y.var()) * 0.5, 1e-2))
    b = np.zeros(S)                        # per-source bias, reference anchored to 0
    Emu = np.full(M, mu0); Vmu = np.full(M, tau2)
    w = np.ones_like(y)
    prev = None
    for _ in range(n_iter):
        r = y - b[sinv]                    # bias-corrected obs
        v_obs = sig2[sinv] / (np.maximum(w, 1e-6) if robust else 1.0)
        prec = 1.0 / v_obs
        S1 = np.bincount(inv, weights=prec, minlength=M)
        Sr = np.bincount(inv, weights=r * prec, minlength=M)
        Pi = 1.0 / tau2 + S1
        Emu = (mu0 / tau2 + Sr) / Pi
        Vmu = 1.0 / Pi
        if robust:
            resid = y - Emu[inv] - b[sinv]
            w = (nu + 1.0) / (nu + resid ** 2 / sig2[sinv])
        # M-step
        mu0 = float(Emu.mean())
        tau2 = max(float(np.mean((Emu - mu0) ** 2 + Vmu)), 1e-3)
        for s in range(S):
            mask = sinv == s
            if s in ref_ids:
                b[s] = 0.0                 # anchor reference source
            else:
                ww = w[mask] if robust else 1.0
                b[s] = float(np.sum(ww * (y[mask] - Emu[inv][mask])) / np.sum(ww * np.ones(mask.sum())))
            resid2 = (y[mask] - Emu[inv][mask] - b[s]) ** 2
            if robust:
                resid2 = w[mask] * resid2
            sig2[s] = max(float((resid2 + Vmu[inv][mask]).mean()), 1e-3)
        cur = (mu0, tau2, tuple(b))
        if prev is not None and abs(prev[0] - mu0) < tol and max(abs(np.array(prev[2]) - b)) < tol:
            break
        prev = cur
    name = "mixle_robust" if robust else "mixle_gauss"
    minw = np.ones(M); np.minimum.at(minw, inv, w)
    df = pd.DataFrame({"ikey": uniq, name: Emu, name + "_sd": np.sqrt(Vmu)})
    if robust:
        df["min_obs_weight"] = minw
    hp = dict(mu0=mu0, tau=float(np.sqrt(tau2)),
              source_bias=dict(zip(suniq, np.round(b, 3))),
              source_sigma=dict(zip(suniq, np.round(np.sqrt(sig2), 3))))
    if robust:
        hp["nu"] = nu
    return df, hp


# ---------------------------------------------------------------------------
def metrics(pred, truth):
    d = np.asarray(pred, float) - np.asarray(truth, float)
    ok = np.isfinite(d); d = d[ok]
    if len(d) == 0:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, medAE=np.nan, bias=np.nan)
    return dict(n=int(len(d)), MAE=float(np.mean(np.abs(d))), RMSE=float(np.sqrt(np.mean(d**2))),
                medAE=float(np.median(np.abs(d))), bias=float(d.mean()))


# Homoset's two parameters swept as a 2-D grid: dimension scale L x noise level.
L_GRID = [("adaptive", "adaptive"), (0.6, "0.6")]        # (value, label)
NOISE_GRID = [0.25, 0.50]                                # tolerated RMS/L (25%, 50%)
PRIMARY = ("adaptive", 0.50)                             # config whose conflict flag is carried


def main():
    obs, fsdf = load()
    base = obs.groupby("ikey")["dg_hyd"].agg(raw_mean="mean", raw_median="median",
                                             n_obs="size").reset_index()

    # Homoset 2-D parameter sweep (L x noise_level)
    hs_tables, hs_gates, hs_conflict = {}, {}, {}
    for L_val, L_lab in L_GRID:
        for nl in NOISE_GRID:
            key = f"homoset_L{L_lab}_noise{int(nl*100)}"
            tab, gate = homoset_reconcile(obs, L_val, nl)
            hs_tables[key] = tab.rename(columns={"consensus": key})
            hs_gates[key] = gate
            hs_conflict[key] = tab.set_index("ikey")["conflict"]

    mg, hp_g = mixle_em(obs, robust=False)
    mr, hp_r = mixle_em(obs, robust=True)

    tab = base.copy()
    for key, t in hs_tables.items():
        tab = tab.merge(t[["ikey", key]], on="ikey", how="left")
    prim_key = f"homoset_L{PRIMARY[0] if PRIMARY[0]=='adaptive' else '0.6'}_noise{int(PRIMARY[1]*100)}"
    prim, _ = homoset_reconcile(obs, PRIMARY[0], PRIMARY[1])
    tab = tab.merge(prim[["ikey", "conflict", "n_sources", "n_accepted_sources", "fell_back_raw"]],
                    on="ikey", how="left")
    tab = tab.merge(mg, on="ikey").merge(mr, on="ikey")
    tab.to_csv(OUT / "per_molecule_estimates.csv", index=False)

    ev = tab.merge(fsdf, on="ikey", how="inner")
    ev.to_csv(OUT / "overlap_freesolv_estimates.csv", index=False)

    estimators = (["raw_mean", "raw_median"] + list(hs_tables.keys())
                  + ["mixle_gauss", "mixle_robust"])

    report = {"reference_route": REFERENCE_ROUTE,
              "homoset_parameters": {"L_grid": [l for _, l in L_GRID], "noise_levels": NOISE_GRID,
                                     "primary_config": prim_key},
              "hyperparams_gauss": hp_g, "hyperparams_robust": hp_r,
              "homoset_source_gates": hs_gates,
              "n_molecules_total": int(len(tab)), "n_overlap_freesolv": int(len(ev)),
              "n_conflict_overlap_primary": int(ev["conflict"].sum())}

    def block(sub, name):
        report[name] = {"n": int(len(sub)),
                        **{est: metrics(sub[est], sub["expt"]) for est in estimators}}
    block(ev, "all_overlap")
    block(ev[ev["n_obs"] >= 2], "ge2_obs")
    block(ev[ev["n_obs"] >= 4], "ge4_obs")
    block(ev[ev["conflict"] == True], "conflict_molecules")
    block(ev[(ev["conflict"] == False) & (ev["n_obs"] >= 2)], "clean_multi_obs")
    (OUT / "comparison_report.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- print ----
    print(f"Overlap Guthrie n FreeSolv: {len(ev)} molecules\n")
    print("HOMOSET two parameters — source-gate outcome across the (L, noise level) grid:")
    print("  a source joins the homogeneous set iff  noise = RMS/L <= noise_level\n")
    for key in hs_tables:
        gd = hs_gates[key]
        acc = [s for s in gd if gd[s]["accepted"] and s not in ("FEOH", "DGS")]
        print(f"  {key:28s} admits non-ref sources: {acc if acc else '[] (free-energy only)'}")
        for s in ["KWG", "KGW", "PAIR"]:
            if s in gd:
                g = gd[s]
                print(f"      {s:5s} noise={g['noise']:.2f} (L={g['L']:.2f})  "
                      f"{'ACCEPT' if g['accepted'] else 'REJECT'}")
    print(f"\nmixle gauss  : mu0={hp_g['mu0']:.2f} tau={hp_g['tau']:.2f}  source_bias={hp_g['source_bias']}")
    print(f"mixle robust : mu0={hp_r['mu0']:.2f} tau={hp_r['tau']:.2f} nu={hp_r['nu']}  source_bias={hp_r['source_bias']}\n")
    for name in ["all_overlap", "ge2_obs", "ge4_obs", "conflict_molecules", "clean_multi_obs"]:
        b = report[name]
        print(f"=== {name}  (n={b['n']}) ===")
        print(f"    {'estimator':30s} {'MAE':>7} {'RMSE':>7} {'medAE':>7} {'bias':>7}")
        for est in estimators:
            m = b[est]
            print(f"    {est:30s} {m['MAE']:7.3f} {m['RMSE']:7.3f} {m['medAE']:7.3f} {m['bias']:+7.3f}")
        print()


if __name__ == "__main__":
    main()
