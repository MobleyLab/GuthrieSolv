#!/usr/bin/env python3
"""Why the unit loop goes dry, and how to get more anyway.

The remaining ~35k unconverted rows are vapour pressure (VP) and aqueous
solubility (AQSOL) -- NOT unit conversions of dG_hyd but *different physical
observables*. No molecule-independent math maps a lone VP or solubility to
dG_hyd. The only route is PAIRING: for a molecule with both,
    K_wa = C_aq_sat / C_gas_sat = S / (P_vap / R T)  ->  dG = -RT ln K_wa.
This pairs every VP+solubility pair we can parse (not just the subset the main
harmoniser reached), and reports the extra coverage + what stays truly lost.
"""
from __future__ import annotations
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harmonize_guthrie import (to_pascal, to_molL, kwa_to_dg, to_kelvin, parse_val,
                               inchikey, R_SI)
OUT = HERE / "outputs"
SRC = HERE.parent / "guthrie_database.csv"


def main():
    df = pd.read_csv(SRC, encoding="latin1", low_memory=False)
    df["value1"] = df["value1"].map(parse_val)
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")
    df["T"] = df["te53p"].map(to_kelvin)
    df["ikey"] = df["mol"].map(inchikey)
    df["unit"] = df["dimension1"].astype(str).str.strip()
    df["proc"] = df["process"].astype(str).str.strip()
    df = df[df["ikey"].notna() & df["value1"].notna()].copy()

    # convert every VP row -> Pa, every AQSOL row -> mol/L (as far as the parsers reach)
    vp = df[df["proc"] == "VP"].copy()
    vp["Pa"] = [to_pascal(u, v) for u, v in zip(vp["unit"], vp["value1"])]
    vp = vp[np.isfinite(vp["Pa"]) & (vp["Pa"] > 0)]
    sol = df[df["proc"] == "AQSOL"].copy()
    sol["molL"] = [to_molL(u, v, mw) for u, v, mw in zip(sol["unit"], sol["value1"], sol["MW"])]
    sol = sol[np.isfinite(sol["molL"]) & (sol["molL"] > 0)]

    # per-molecule geometric-median VP and solubility, then pair
    vg = vp.groupby("ikey").agg(Pa=("Pa", lambda s: float(np.exp(np.median(np.log(s))))),
                                T=("T", "median"))
    sg = sol.groupby("ikey").agg(molL=("molL", lambda s: float(np.exp(np.median(np.log(s))))),
                                 T=("T", "median"))
    both = vg.join(sg, how="inner", lsuffix="_vp", rsuffix="_sol")
    T = np.where(np.isfinite(both["T_vp"]), both["T_vp"], 298.15)
    c_gas = both["Pa"].to_numpy() / (R_SI * T)      # mol/m3
    c_aq = both["molL"].to_numpy() * 1e3            # mol/m3
    dg = np.array([kwa_to_dg(ca / cg, t) for ca, cg, t in zip(c_aq, c_gas, T)])
    paired = pd.DataFrame({"ikey": both.index, "dg_hyd": dg})
    paired = paired[(paired["dg_hyd"] > -60) & (paired["dg_hyd"] < 12)]

    # coverage vs the current physics+loop expanded set
    phys = pd.read_csv(OUT / "guthrie_dg_observations.csv")
    iter_exp = OUT / "guthrie_dg_observations_iter_expanded.csv"
    have = set(pd.read_csv(iter_exp)["ikey"]) if iter_exp.exists() else set(phys["ikey"])
    new_mols = set(paired["ikey"]) - have

    # FreeSolv held-out check on the freshly-paired molecules
    fs = json.load(open("/Users/tgg/Github/FreeSolv/database.json"))
    exd = {}
    for cid, d in fs.items():
        m = Chem.MolFromSmiles(d["smiles"])
        if m:
            exd[Chem.MolToInchiKey(m)] = float(d["expt"])
    pm = paired.set_index("ikey")["dg_hyd"]
    ov = [(exd[ik], pm[ik]) for ik in pm.index if ik in exd]
    t = np.array([a for a, _ in ov]); p = np.array([b for _, b in ov])
    mae = float(np.mean(np.abs(p - t))); R = float(np.corrcoef(p, t)[0, 1])
    ovn = [(exd[ik], pm[ik]) for ik in new_mols if ik in exd]
    if ovn:
        tn = np.array([a for a, _ in ovn]); pn = np.array([b for _, b in ovn])
        mae_new = float(np.mean(np.abs(pn - tn)))
    else:
        mae_new = float("nan")

    # truly-lost: molecules with only one of the two observables
    vp_mols, sol_mols = set(vp["ikey"]), set(sol["ikey"])
    vp_only = vp_mols - sol_mols
    sol_only = sol_mols - vp_mols

    rep = {
        "paired_molecules_total": int(len(paired)),
        "already_had_dG": int(len(set(paired['ikey']) & have)),
        "NEW_molecules_from_pairing": int(len(new_mols)),
        "coverage_after_pairing": int(len(have | set(paired["ikey"]))),
        "freesolv_all_paired": {"n": len(ov), "MAE": round(mae, 3), "R": round(R, 3)},
        "freesolv_new_paired": {"n": len(ovn), "MAE": round(mae_new, 3)},
        "truly_unconvertible": {
            "VP_only_molecules": int(len(vp_only)),
            "AQSOL_only_molecules": int(len(sol_only)),
            "reason": "single observable; dG_hyd needs BOTH volatility and solubility",
        },
    }
    (OUT / "pairing_expansion_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nSUMMARY: pairing VP x solubility converts {len(paired)} molecules "
          f"(+{len(new_mols)} beyond the unit routes).")
    print(f"Truly unconvertible = single-observable molecules: "
          f"{len(vp_only)} VP-only + {len(sol_only)} solubility-only "
          f"(a DATA gap, not a math gap).")


if __name__ == "__main__":
    main()
