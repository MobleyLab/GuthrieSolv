#!/usr/bin/env python3
"""Plot how much of GuthrieSolv we actually convert: the disposition of all
53,895 raw measurements by route, and the rule-engine vs regex-switchboard gain."""
from __future__ import annotations
import sys, math, json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SRC = HERE.parent / "guthrie_database.csv"
sys.path.insert(0, str(HERE))
from harmonize_guthrie import to_kelvin, parse_val, inchikey
from unit_rules import convert

EXCLUDED = {"DGT", "DGTX", "DGB", "DGV", "PKA", "KA", "LAQACTCO"}


def main():
    df = pd.read_csv(SRC, encoding="latin1", low_memory=False)
    df["value1"] = df["value1"].map(parse_val)
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")
    df["T"] = df["te53p"].map(to_kelvin)
    df["ikey"] = df["mol"].map(inchikey)
    df["unit"] = df["dimension1"].astype(str).str.strip()
    df["proc"] = df["process"].astype(str).str.strip()

    kinds, notes = [], []
    for r in df.itertuples(index=False):
        if r.proc in EXCLUDED:
            kinds.append("excluded"); notes.append(""); continue
        if not isinstance(r.mol, str) or r.ikey is None or not math.isfinite(r.value1):
            kinds.append("bad"); notes.append(""); continue
        k, v, note = convert(r.unit, r.value1, r.proc, r.T, r.MW)
        kinds.append(k or "unparsed"); notes.append(note)
    df["kind"] = kinds
    df["note"] = notes

    # which VP / AQSOL molecules can pair (have BOTH observables)?
    vp_mols = set(df.loc[df["kind"] == "vp_pa", "ikey"])
    sol_mols = set(df.loc[df["kind"] == "sol_molL", "ikey"])
    pairable = vp_mols & sol_mols

    def bucket(row):
        k = row["kind"]
        if k == "excluded":
            return "excluded (pKa / transfer / vaporisation)"
        if k == "dg":
            n = row["note"]
            if n in ("energy",):
                return "free energy (FEOH/DGS)"
            return "Henry constant"
        if k == "vp_pa":
            return "vapour pressure → paired" if row["ikey"] in pairable else "vapour-pressure-only (no solubility)"
        if k == "sol_molL":
            return "solubility → paired" if row["ikey"] in pairable else "solubility-only (no vapour pressure)"
        return "unparsed / other"
    df["bucket"] = df.apply(bucket, axis=1)
    counts = df["bucket"].value_counts()

    # ---- figure ----
    order = ["free energy (FEOH/DGS)", "Henry constant", "vapour pressure → paired",
             "solubility → paired", "vapour-pressure-only (no solubility)",
             "solubility-only (no vapour pressure)", "excluded (pKa / transfer / vaporisation)",
             "unparsed / other"]
    colors = {"free energy (FEOH/DGS)": "#0d6b74", "Henry constant": "#2a8f86",
              "vapour pressure → paired": "#4ba89a", "solubility → paired": "#6cbfab",
              "vapour-pressure-only (no solubility)": "#c98b3a",
              "solubility-only (no vapour pressure)": "#a8620a",
              "excluded (pKa / transfer / vaporisation)": "#b0b8b8", "unparsed / other": "#8a9698"}
    order = [b for b in order if b in counts.index]
    vals = [int(counts[b]) for b in order]
    total = sum(vals)
    converted = sum(counts.get(b, 0) for b in order[:4])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})

    # panel A: stacked disposition bar
    left = 0
    for b in order:
        w = counts[b]
        ax.barh(0, w, left=left, color=colors[b], height=0.5,
                label=f"{b}  ({w:,})")
        if w / total > 0.04:
            ax.text(left + w / 2, 0, f"{w:,}", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        left += w
    ax.set_xlim(0, total); ax.set_yticks([]); ax.set_ylim(-0.5, 0.9)
    ax.set_xlabel("measurements (rows)")
    ax.set_title(f"Disposition of all {total:,} GuthrieSolv measurements\n"
                 f"{converted:,} converted to ΔG$_{{hyd}}$  ·  the rest are single-observable "
                 f"VP/solubility (need pairing) or non-hydration", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8.5)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    # panel B: unit-combo coverage + molecule growth
    ax2.bar(["regex\nswitchboard", "rule\nengine"], [57, 65], color=["#8a9698", "#0d6b74"], width=0.6)
    for i, v in enumerate([57, 65]):
        ax2.text(i, v + 1, str(v), ha="center", fontsize=12, fontweight="bold",
                 color=["#8a9698", "#0d6b74"][i])
    ax2.set_ylim(0, 82); ax2.set_ylabel("(unit, process) combinations converted")
    ax2.set_title("Rule engine converts more\n(and each unit is one table row)", fontsize=11)
    ax2.annotate("+8", xy=(1, 65), xytext=(0.5, 74), ha="center", fontsize=13, fontweight="bold",
                 color="#0d6b74", arrowprops=dict(arrowstyle="->", color="#0d6b74"))
    ax2.text(0.5, -0.16, "mass-fraction · P/M Henry · m³ spellings", ha="center",
             transform=ax2.transAxes, fontsize=8.5, color="#0d6b74")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "unit_coverage.png", dpi=130, bbox_inches="tight")
    (OUT / "unit_coverage.json").write_text(json.dumps(
        {"total": total, "converted_to_dG": int(converted),
         "by_bucket": {b: int(counts[b]) for b in order}}, indent=2))
    print(json.dumps({"total": total, "converted_to_dG": int(converted),
                      **{b: int(counts[b]) for b in order}}, indent=2))
    print("saved outputs/unit_coverage.png")


if __name__ == "__main__":
    main()
