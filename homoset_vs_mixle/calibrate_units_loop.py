#!/usr/bin/env python3
"""Active-learning / retro-loop unit calibration for GuthrieSolv.

The physics switchboard in harmonize_guthrie.py converts the high-frequency unit
strings; ~12k rows in the long tail of the 172 units stay unconverted. This loop
recovers many of them WITHOUT hand-coding each conversion:

  1. Anchors = molecules with a known dG_hyd (from the physics routes).
  2. For each unconverted (unit, process) group, fit the best monotone map
     dG ~ a * transform(value) + b  (transform in {log10, -log10, ln, identity})
     against the anchor dG of the molecules it shares.
  3. Accept the calibration IFF it passes the Homoset noise gate
     (noise = RMS/L <= eta) AND correlates (|R| >= R_MIN) -- the same
     homogeneity criterion used to admit a source, now used to admit a unit.
  4. Convert ALL rows of an accepted unit -> new dG observations -> new anchors.
  5. Loop: newly-anchored molecules let the next round calibrate units that only
     overlapped them. Stop when a round adds nothing (dry).

FreeSolv is NOT used as an anchor, so the newly-converted molecules that happen to
be in FreeSolv are a held-out check on the loop's calibrations.
"""
from __future__ import annotations
import json, math
import numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SRC = HERE.parent / "guthrie_database.csv"

MIN_ANCHORS = 6            # need this many overlapping molecules to fit a unit
R_MIN = 0.85              # correlation floor for a genuine monotone relationship
ETA = 0.35               # Homoset noise level: accept iff RMS/L <= ETA
MAX_ROUNDS = 10
EXCLUDED_PROC = {"DGT", "DGTX", "DGB", "DGV", "PKA", "KA", "LAQACTCO"}
DG_LO, DG_HI = -60.0, 12.0

TRANSFORMS = {
    "log10":    (lambda v: np.log10(v),  lambda v: v > 0),
    "neglog10": (lambda v: -np.log10(v), lambda v: v > 0),
    "ln":       (lambda v: np.log(v),    lambda v: v > 0),
    "identity": (lambda v: v.astype(float), lambda v: np.isfinite(v)),
}


def homoset_l_scale(arr):
    arr = np.asarray(arr, float); arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    return max(float((arr.max() - arr.min()) / 2.0), 0.5)


def parse_val(x):
    try:
        return float(x)
    except Exception:
        import re
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(x))
        return float(m.group(0)) if m else math.nan


_ik = {}
def inchikey(smi):
    if smi in _ik:
        return _ik[smi]
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    out = Chem.MolToInchiKey(m) if m else None
    _ik[smi] = out
    return out


def robust_fit(x, y):
    """Least-squares line with one trimming pass; returns (a, b, rms, R, n)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < MIN_ANCHORS or np.ptp(x) == 0:
        return None
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    keep = np.abs(resid) <= 3.5 * mad
    if keep.sum() >= MIN_ANCHORS and keep.sum() < len(x):
        a, b = np.polyfit(x[keep], y[keep], 1)
        x, y = x[keep], y[keep]
    pred = a * x + b
    rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    R = float(np.corrcoef(x, y)[0, 1]) if (np.std(x) > 0 and np.std(y) > 0) else 0.0
    return a, b, rms, R, len(x)


def main():
    df = pd.read_csv(SRC, encoding="latin1", low_memory=False)
    df["value1"] = df["value1"].map(parse_val)
    df["ikey"] = df["mol"].map(inchikey)
    df["unit"] = df["dimension1"].astype(str).str.strip()
    df["proc"] = df["process"].astype(str).str.strip()

    phys = pd.read_csv(OUT / "guthrie_dg_observations.csv")
    converted_keys = set(map(tuple, phys[["unit", "process"]].drop_duplicates().values))
    # anchor dG per molecule from the physics routes only
    anchor = phys.groupby("ikey")["dg_hyd"].median().to_dict()
    phys_mols = set(anchor)

    # candidate rows: not physics-converted, not excluded process, finite value, has ikey
    cand = df[df["ikey"].notna() & df["value1"].notna()
              & ~df["proc"].isin(EXCLUDED_PROC)].copy()
    key = list(zip(cand["unit"], cand["proc"]))
    cand["converted"] = [k in converted_keys for k in key]
    cand = cand[~cand["converted"]].copy()

    accepted = {}           # (unit,proc) -> dict(transform,a,b,rms,R,n)
    loop_obs = []           # new observations produced by the loop
    round_log = []

    for rnd in range(1, MAX_ROUNDS + 1):
        new_this_round = []
        # order groups by number of anchorable molecules (descending) for stable growth
        groups = cand.groupby(["unit", "proc"])
        order = sorted(groups.groups, key=lambda g: -len(groups.get_group(g)))
        for g in order:
            if g in accepted:
                continue
            rows = groups.get_group(g)
            av = np.array([anchor.get(ik, np.nan) for ik in rows["ikey"]])
            m = np.isfinite(av)
            if m.sum() < MIN_ANCHORS:
                continue
            vals = rows["value1"].to_numpy()[m]
            ys = av[m]
            best = None
            for tname, (tf, valid) in TRANSFORMS.items():
                vmask = valid(vals)
                if vmask.sum() < MIN_ANCHORS:
                    continue
                fit = robust_fit(tf(vals[vmask]), ys[vmask])
                if fit is None:
                    continue
                a, b, rms, R, n = fit
                L = homoset_l_scale(ys[vmask])
                noise = rms / L
                score = (abs(R) >= R_MIN) and (noise <= ETA)
                if score and (best is None or noise < best["noise"]):
                    best = dict(transform=tname, a=a, b=b, rms=rms, R=R, n=n,
                                noise=noise, L=L)
            if best:
                accepted[g] = best
                new_this_round.append((g, best))
                # convert ALL rows of this (unit,proc)
                tf = TRANSFORMS[best["transform"]][0]
                vv = groups.get_group(g)["value1"].to_numpy()
                valid = TRANSFORMS[best["transform"]][1](vv)
                dg = best["a"] * tf(np.where(valid, vv, np.nan)) + best["b"]
                iks = groups.get_group(g)["ikey"].to_numpy()
                for ik, d in zip(iks, dg):
                    if np.isfinite(d) and DG_LO < d < DG_HI:
                        loop_obs.append((ik, d, best["transform"], g[0], g[1]))
        if not new_this_round:
            break
        # retro step: rebuild anchors with physics + all loop obs so far
        lo = pd.DataFrame(loop_obs, columns=["ikey", "dg", "transform", "unit", "proc"])
        merged = pd.concat([phys[["ikey", "dg_hyd"]].rename(columns={"dg_hyd": "dg"}),
                            lo[["ikey", "dg"]]], ignore_index=True)
        anchor = merged.groupby("ikey")["dg"].median().to_dict()
        round_log.append({"round": rnd, "units_added": len(new_this_round),
                          "obs_so_far": len(loop_obs),
                          "molecules_so_far": int(pd.Series([o[0] for o in loop_obs]).nunique())})
        print(f"round {rnd}: +{len(new_this_round)} units  "
              f"| total loop obs {len(loop_obs)}  "
              f"| loop molecules {round_log[-1]['molecules_so_far']}")

    # ---- assemble & validate ----
    lo = pd.DataFrame(loop_obs, columns=["ikey", "dg_hyd", "transform", "unit", "process"])
    lo["route"] = "calibrated_loop"
    lo["T"] = np.nan; lo["err"] = np.nan; lo["smiles"] = None
    new_mols = set(lo["ikey"]) - phys_mols
    expanded = pd.concat([phys, lo[phys.columns.intersection(lo.columns)]], ignore_index=True)
    expanded.to_csv(OUT / "guthrie_dg_observations_expanded.csv", index=False)

    # FreeSolv held-out check (FreeSolv never used as an anchor)
    fs = json.load(open("/Users/tgg/Github/FreeSolv/database.json"))
    exd = {}
    for cid, d in fs.items():
        mm = Chem.MolFromSmiles(d["smiles"])
        if mm:
            exd[Chem.MolToInchiKey(mm)] = float(d["expt"])
    loo_med = lo.groupby("ikey")["dg_hyd"].median()
    newmol_med = loo_med[loo_med.index.isin(new_mols)]
    fs_new = [(exd[ik], v) for ik, v in newmol_med.items() if ik in exd]
    if fs_new:
        t, p = np.array([a for a, _ in fs_new]), np.array([b for _, b in fs_new])
        newmol_mae = float(np.mean(np.abs(p - t)))
        newmol_r = float(np.corrcoef(p, t)[0, 1])
    else:
        newmol_mae = newmol_r = float("nan")

    report = {
        "params": {"MIN_ANCHORS": MIN_ANCHORS, "R_MIN": R_MIN, "eta": ETA},
        "n_units_calibrated": len(accepted),
        "rounds": round_log,
        "physics_molecules": len(phys_mols),
        "loop_new_molecules": len(new_mols),
        "total_molecules_after_loop": int(expanded["ikey"].nunique()),
        "loop_observations": len(lo),
        "freesolv_heldout_newmolecules": {"n": len(fs_new), "MAE": newmol_mae, "R": newmol_r},
        "calibrated_units": [
            {"unit": u, "process": p, "transform": v["transform"],
             "a": round(v["a"], 4), "b": round(v["b"], 4),
             "R": round(v["R"], 3), "noise_RMS_over_L": round(v["noise"], 3), "n_anchors": v["n"],
             # high confidence = physically-convertible Henry/free-energy process with tight fit
             "confidence": ("high" if (p in {"KWG", "KGW"} and abs(v["R"]) >= 0.95
                                       and v["noise"] <= 0.15) else "medium")}
            for (u, p), v in sorted(accepted.items(), key=lambda kv: -kv[1]["n"])
        ],
    }
    report["n_high_confidence_units"] = sum(1 for c in report["calibrated_units"] if c["confidence"] == "high")
    (OUT / "unit_calibration_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== active-learning unit calibration ===")
    print(f"units calibrated: {len(accepted)}   (over {len(round_log)} rounds)")
    print(f"molecules: physics {len(phys_mols)}  ->  +{len(new_mols)} new  "
          f"=  {report['total_molecules_after_loop']} total")
    print(f"loop observations added: {len(lo)}")
    print(f"FreeSolv held-out check on brand-new molecules: n={len(fs_new)}  "
          f"MAE={newmol_mae:.3f}  R={newmol_r:.3f}")
    print("\ntop calibrated units (unit | process | transform | R | noise | n):")
    for c in report["calibrated_units"][:25]:
        print(f"  {c['unit'][:22]:22s} {c['process']:6s} {c['transform']:9s} "
              f"R={c['R']:+.2f} noise={c['noise_RMS_over_L']:.2f} n={c['n_anchors']}")


if __name__ == "__main__":
    main()
