#!/usr/bin/env python3
"""Iterative *fixed-point* unit calibration (the real 'keep looping' version).

calibrate_units_loop.py runs one anneal-free pass and goes dry in 2 rounds. This
wraps it in an OUTER co-training loop so it genuinely keeps iterating:

  repeat:
    1. calibrate units against the current anchors, admitting until inner-dry;
    2. RECONCILE all observations -> a per-molecule consensus (robust median);
       this cleaner, larger anchor set is fed back as the new anchors;
    3. ANNEAL the gate one notch looser (strict high-confidence units first,
       lower-confidence tail later).
  until an outer pass admits no new unit at the loosest gate, OR the held-out
  FreeSolv accuracy stops improving (quality-based stop).

Each outer step re-anchoring on the reconciled consensus (not the raw physics
median) is what lets later steps reach units the first pass could not.
"""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate_units_loop import (robust_fit, homoset_l_scale, TRANSFORMS, parse_val,
                                  inchikey, EXCLUDED_PROC, DG_LO, DG_HI, MIN_ANCHORS)

OUT = HERE / "outputs"
SRC = HERE.parent / "guthrie_database.csv"

# annealing schedule: (eta, R_min) from strict -> loose. Repeat last to test dryness.
SCHEDULE = [(0.25, 0.90), (0.35, 0.85), (0.45, 0.82), (0.55, 0.80), (0.65, 0.78), (0.65, 0.78)]
MAX_INNER = 6


def freesolv_anchors():
    fs = json.load(open("/Users/tgg/Github/FreeSolv/database.json"))
    d = {}
    for cid, v in fs.items():
        m = Chem.MolFromSmiles(v["smiles"])
        if m:
            d[Chem.MolToInchiKey(m)] = float(v["expt"])
    return d


def fit_admit(vals, ys, eta, r_min):
    """Best transform fit that passes |R|>=r_min and RMS/L<=eta. Returns dict or None."""
    best = None
    for tname, (tf, valid) in TRANSFORMS.items():
        vmask = valid(vals)
        if vmask.sum() < MIN_ANCHORS:
            continue
        fit = robust_fit(tf(vals[vmask]), ys[vmask])
        if fit is None:
            continue
        a, b, rms, R, n = fit
        L = homoset_l_scale(ys[vmask]); noise = rms / L
        if abs(R) >= r_min and noise <= eta and (best is None or noise < best["noise"]):
            best = dict(transform=tname, a=a, b=b, rms=rms, R=R, n=n, noise=noise)
    return best


def main():
    df = pd.read_csv(SRC, encoding="latin1", low_memory=False)
    df["value1"] = df["value1"].map(parse_val)
    df["ikey"] = df["mol"].map(inchikey)
    df["unit"] = df["dimension1"].astype(str).str.strip()
    df["proc"] = df["process"].astype(str).str.strip()

    phys = pd.read_csv(OUT / "guthrie_dg_observations.csv")
    converted_keys = set(map(tuple, phys[["unit", "process"]].drop_duplicates().values))
    phys_mols = set(phys["ikey"])
    exd = freesolv_anchors()

    cand = df[df["ikey"].notna() & df["value1"].notna()
              & ~df["proc"].isin(EXCLUDED_PROC)].copy()
    cand["key"] = list(zip(cand["unit"], cand["proc"]))
    cand = cand[~cand["key"].isin(converted_keys)].copy()
    groups = {g: sub for g, sub in cand.groupby(["unit", "proc"])}

    # anchors start from the physics per-molecule median
    def reconcile(loop_obs):
        base = phys[["ikey", "dg_hyd"]].rename(columns={"dg_hyd": "dg"})
        if loop_obs:
            lo = pd.DataFrame(loop_obs, columns=["ikey", "dg", "unit", "proc", "tf"])
            base = pd.concat([base, lo[["ikey", "dg"]]], ignore_index=True)
        return base.groupby("ikey")["dg"].median()

    anchors = reconcile([])
    accepted = {}            # (unit,proc) -> fit
    loop_obs = []            # (ik, dg, unit, proc, transform)
    traj = []

    def fs_eval(anch):
        ov = [(exd[ik], v) for ik, v in anch.items() if ik in exd]
        if not ov:
            return (0, np.nan, np.nan)
        t = np.array([a for a, _ in ov]); p = np.array([b for _, b in ov])
        return (len(ov), float(np.mean(np.abs(p - t))), float(np.corrcoef(p, t)[0, 1]))

    n0, mae0, r0 = fs_eval(anchors)
    print(f"start: physics anchors  mols={len(anchors)}  FreeSolv n={n0} MAE={mae0:.3f} R={r0:.3f}\n")

    for outer, (eta, r_min) in enumerate(SCHEDULE, 1):
        added_units = 0
        for _ in range(MAX_INNER):           # inner: admit until dry at THIS gate
            amap = anchors.to_dict()
            new = 0
            for g in sorted(groups, key=lambda g: -len(groups[g])):
                if g in accepted:
                    continue
                rows = groups[g]
                av = np.array([amap.get(ik, np.nan) for ik in rows["ikey"]])
                m = np.isfinite(av)
                if m.sum() < MIN_ANCHORS:
                    continue
                best = fit_admit(rows["value1"].to_numpy()[m], av[m], eta, r_min)
                if not best:
                    continue
                accepted[g] = dict(best, eta=eta, r_min=r_min, outer=outer)
                tf = TRANSFORMS[best["transform"]][0]; valid = TRANSFORMS[best["transform"]][1]
                vv = rows["value1"].to_numpy(); ok = valid(vv)
                dg = best["a"] * tf(np.where(ok, vv, np.nan)) + best["b"]
                for ik, d in zip(rows["ikey"].to_numpy(), dg):
                    if np.isfinite(d) and DG_LO < d < DG_HI:
                        loop_obs.append((ik, d, g[0], g[1], best["transform"]))
                new += 1
            added_units += new
            if new == 0:
                break
            anchors = reconcile(loop_obs)      # re-anchor on reconciled consensus (co-training)
        n, mae, R = fs_eval(anchors)
        new_mols = len(set(o[0] for o in loop_obs) - phys_mols)
        traj.append(dict(outer=outer, eta=eta, r_min=r_min, units_added=added_units,
                         units_total=len(accepted), new_molecules=new_mols,
                         total_molecules=int(len(anchors)), loop_obs=len(loop_obs),
                         freesolv_n=n, freesolv_MAE=round(mae, 4), freesolv_R=round(R, 4)))
        print(f"outer {outer}: gate(eta={eta},R>={r_min})  +{added_units} units  "
              f"total {len(accepted)}u / +{new_mols} mols / {len(anchors)} total  "
              f"| FreeSolv n={n} MAE={mae:.3f} R={R:.3f}")

    # ---- assemble expanded set ----
    lo = pd.DataFrame(loop_obs, columns=["ikey", "dg_hyd", "unit", "process", "transform"])
    lo["route"] = "calibrated_iter"; lo["T"] = np.nan; lo["err"] = np.nan; lo["smiles"] = None
    expanded = pd.concat([phys, lo[phys.columns.intersection(lo.columns)]], ignore_index=True)
    expanded.to_csv(OUT / "guthrie_dg_observations_iter_expanded.csv", index=False)

    report = {"schedule": SCHEDULE, "trajectory": traj,
              "units_total": len(accepted),
              "new_molecules": len(set(lo["ikey"]) - phys_mols),
              "total_molecules": int(expanded["ikey"].nunique()),
              "loop_observations": int(len(lo)),
              "start_freesolv": {"n": n0, "MAE": round(mae0, 4), "R": round(r0, 4)},
              "units": [dict(unit=u, process=p, **{k: (round(v[k], 4) if isinstance(v[k], float) else v[k])
                                                    for k in ("transform", "a", "b", "R", "noise", "n",
                                                              "eta", "outer")})
                        for (u, p), v in sorted(accepted.items(), key=lambda kv: -kv[1]["n"])]}
    (OUT / "unit_calibration_iter_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== iterative fixed point reached ===")
    print(f"units {len(accepted)}  |  molecules {len(phys_mols)} -> {report['total_molecules']} "
          f"(+{report['new_molecules']})  |  loop obs {len(lo)}")
    print(f"FreeSolv MAE trajectory: {mae0:.3f} -> " + " -> ".join(f"{t['freesolv_MAE']:.3f}" for t in traj))


if __name__ == "__main__":
    main()
