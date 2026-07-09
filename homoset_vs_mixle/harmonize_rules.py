#!/usr/bin/env python3
"""Re-harmonise GuthrieSolv with the declarative rule engine (unit_rules.py) and
compare coverage + FreeSolv accuracy against the regex switchboard."""
from __future__ import annotations
import sys, json, math
import numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harmonize_guthrie import to_kelvin, parse_val, inchikey
from unit_rules import convert, parse_unit, classify, R_SI, kwa_to_dg

OUT = HERE / "outputs"
SRC = HERE.parent / "guthrie_database.csv"
EXCLUDED = {"DGT", "DGTX", "DGB", "DGV", "PKA", "KA", "LAQACTCO"}


def main():
    df = pd.read_csv(SRC, encoding="latin1", low_memory=False)
    df["value1"] = df["value1"].map(parse_val)
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")
    df["T"] = df["te53p"].map(to_kelvin)
    df["ikey"] = df["mol"].map(inchikey)
    df["unit"] = df["dimension1"].astype(str).str.strip()
    df["proc"] = df["process"].astype(str).str.strip()
    df = df[df["ikey"].notna() & df["value1"].notna() & ~df["proc"].isin(EXCLUDED)].copy()

    direct, vp, sol = [], [], []
    fail_units = {}
    for r in df.itertuples(index=False):
        kind, val, note = convert(r.unit, r.value1, r.proc, r.T, r.MW)
        if kind == "dg" and math.isfinite(val) and -60 < val < 12:
            direct.append((r.ikey, val, r.unit, r.proc, note))
        elif kind == "vp_pa" and math.isfinite(val) and val > 0:
            vp.append((r.ikey, val, r.T))
        elif kind == "sol_molL" and math.isfinite(val) and val > 0:
            sol.append((r.ikey, val, r.T))
        else:
            fail_units[(r.unit, r.proc)] = fail_units.get((r.unit, r.proc), 0) + 1

    # pair VP x solubility
    vpd = pd.DataFrame(vp, columns=["ikey", "Pa", "T"])
    sold = pd.DataFrame(sol, columns=["ikey", "molL", "T"])
    paired = []
    if len(vpd) and len(sold):
        vg = vpd.groupby("ikey").agg(Pa=("Pa", lambda s: float(np.exp(np.median(np.log(s))))),
                                     T=("T", "median"))
        sg = sold.groupby("ikey").agg(molL=("molL", lambda s: float(np.exp(np.median(np.log(s))))))
        both = vg.join(sg, how="inner")
        for ik, row in both.iterrows():
            T = row["T"] if math.isfinite(row["T"]) else 298.15
            kwa = (row["molL"] * 1e3) / (row["Pa"] / (R_SI * T))
            g = kwa_to_dg(kwa, T)
            if math.isfinite(g) and -60 < g < 12:
                paired.append((ik, g, "VP+AQSOL", "PAIR", "pair"))

    obs = pd.DataFrame(direct + paired, columns=["ikey", "dg_hyd", "unit", "process", "note"])

    # --- FreeSolv ---
    fs = json.load(open("/Users/tgg/Github/FreeSolv/database.json"))
    exd = {}
    for cid, d in fs.items():
        m = Chem.MolFromSmiles(d["smiles"])
        if m:
            exd[Chem.MolToInchiKey(m)] = float(d["expt"])
    g = obs.groupby("ikey")["dg_hyd"].median()
    ov = [(exd[ik], v) for ik, v in g.items() if ik in exd]
    t, p = np.array([a for a, _ in ov]), np.array([b for _, b in ov])
    mae = float(np.mean(np.abs(p - t))); R = float(np.corrcoef(p, t)[0, 1])

    # compare to the regex switchboard result
    old = pd.read_csv(OUT / "guthrie_dg_observations.csv")
    old_units = set(map(tuple, old[["unit", "process"]].drop_duplicates().values))
    new_units = set(map(tuple, obs[["unit", "process"]].drop_duplicates().values))

    # top still-failing units that are NOT vp/sol (those are genuinely un-pairable, not failures)
    fails = sorted(fail_units.items(), key=lambda kv: -kv[1])[:20]

    print(f"RULE ENGINE:  {len(obs)} obs / {obs['ikey'].nunique()} molecules / {len(paired)} paired")
    print(f"  FreeSolv overlap {len(ov)}  median-MAE {mae:.3f}  R {R:.3f}")
    print(f"SWITCHBOARD:  {len(old)} obs / {old['ikey'].nunique()} molecules")
    print(f"  distinct (unit,process) converted:  switchboard {len(old_units)}  ->  rule engine {len(new_units)}")
    gained = new_units - old_units
    print(f"  NEW (unit,process) the rule engine converts that the switchboard did not: {len(gained)}")
    for u, pr in sorted(gained)[:25]:
        print(f"      + {u:24s} {pr}")
    print("\ntop still-unconverted (unit,process) [mostly VP/AQSOL single-observable or exotic]:")
    for (u, pr), n in fails:
        print(f"      {u[:26]:26s} {pr:6s} n={n}")

    obs.to_csv(OUT / "guthrie_dg_observations_rules.csv", index=False)
    rep = {"rule_engine": {"obs": len(obs), "molecules": int(obs["ikey"].nunique()),
                           "paired": len(paired), "units_converted": len(new_units),
                           "freesolv": {"n": len(ov), "MAE": round(mae, 3), "R": round(R, 3)}},
           "switchboard": {"obs": len(old), "molecules": int(old["ikey"].nunique()),
                           "units_converted": len(old_units)},
           "new_units_gained": [{"unit": u, "process": p} for u, p in sorted(gained)]}
    (OUT / "rule_engine_report.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
