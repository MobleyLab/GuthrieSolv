#!/usr/bin/env python3
"""Harmonize the raw GuthrieSolv literature dump into Ben-Naim hydration free
energies (kcal/mol), keyed by InChIKey, using the `process` column as the
authority on measurement type and a per-unit conversion switchboard.

Target: dG*_hyd = -R T ln(K_wa),  K_wa = C_aq / C_gas  (Ben-Naim, gas 1 M -> aq 1 M).
This is the same standard state FreeSolv reports, so converted values are directly
comparable to FreeSolv `expt`.

Convertible routes:
  FEOH / DGS            : value already a hydration/solvation free energy.
  KWG (water/gas Henry) : dimensionless ratio, or P*V/mol, or P/mole-fraction.
  KGW (gas/water Henry) : C_aq/P solubility-form, dimensionless ratio, Ostwald/Bunsen.
  VP + AQSOL (paired)   : K_wa = C_aq_sat / C_gas_sat from vapour pressure + solubility.
Excluded (not hydration): DGT/DGTX/DGB/DGV (transfer/vaporisation), PKA, KA, LAQACTCO.
"""
from __future__ import annotations
import math, re, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "guthrie_database.csv"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

R_KCAL = 1.987204e-3          # kcal/mol/K
R_SI = 8.314462              # J/mol/K   -> P in Pa, V in m3
R_LATM = 0.082057            # L atm / mol / K
KCAL_PER_KJ = 1.0 / 4.184
WATER_M = 55.345             # mol/L pure water at 25 C

# ---------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------
def to_kelvin(x) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 298.15
    s = str(x).strip().replace("\xa0", "")
    m = re.match(r"^(-?\d+(?:\.\d+)?)", s)
    if not m:
        return 298.15
    v = float(m.group(1))
    if s.endswith("K") or v > 200:      # already Kelvin
        return v
    return v + 273.15                    # Celsius


# ---------------------------------------------------------------------------
# value parsing
# ---------------------------------------------------------------------------
def parse_val(x) -> float:
    try:
        return float(x)
    except Exception:
        if x is None:
            return math.nan
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(x))
        return float(m.group(0)) if m else math.nan


# ---------------------------------------------------------------------------
# unit -> canonical physical quantity converters
# each returns a value in the canonical unit, or NaN if unhandled
# ---------------------------------------------------------------------------
def _log_wrap(u: str, v: float):
    """Return (linear_value, base_unit) handling log/ln/-log prefixes, or (v,u)."""
    s = u.strip()
    low = s.lower()
    # (-)log / -log  -> 10^(-v)
    m = re.match(r"^\(?-\)?log\s*\(?(.*?)\)?$", low)
    if low.startswith("(-)log") or low.startswith("-log") or low.startswith("(-) log"):
        inner = re.sub(r"^\(?-\)?\s*log", "", s, flags=re.I).strip(" ()")
        return 10.0 ** (-v), inner
    if low.startswith("ln("):
        inner = s[3:].rstrip(")")
        return math.exp(v), inner
    if low.startswith("ln "):
        inner = s[3:].strip(" ()")
        return math.exp(v), inner
    if low.startswith("log10") or low.startswith("log("):
        inner = re.sub(r"^log(?:10)?", "", s, flags=re.I).strip(" ()")
        return 10.0 ** v, inner
    if low.startswith("log "):
        inner = s[4:].strip(" ()")
        return 10.0 ** v, inner
    if low.startswith("log p,") or low.startswith("log p ") or low.startswith("logk"):
        inner = s.split(",")[-1].strip() if "," in s else s.split()[-1]
        return 10.0 ** v, inner
    return v, s


# pressure units -> Pascal
_PA = {
    "pa": 1.0, "kpa": 1e3, "mpa": 1e6, "hpa": 1e2, "mbar": 1e2, "bar": 1e5,
    "atm": 101325.0, "torr": 133.322, "mm hg": 133.322, "mmhg": 133.322,
    "cm hg": 1333.22, "cmhg": 1333.22, "mtorr": 0.133322, "mpa_milli": 1e-3,
    "kn/(m^2)": 1e3, "kn/m2": 1e3, "kn/m^2": 1e3, "n/m2": 1.0,
    "um hg": 0.133322, "\xb5m hg": 0.133322, "micrometers h": 0.133322,
    "mpa ": 1e-3,  # milliPa (ambiguous with megaPa; handled below by exact key)
}
# explicit small/large pressure prefixes (exact keys, case-sensitive-ish)
_PA_EXACT = {
    "mPa": 1e-3, "microPa": 1e-6, "muPa": 1e-6, "\xb5Pa": 1e-6, "nPa": 1e-9,
    "pPa": 1e-12, "MPa": 1e6, "kPa": 1e3, "hPa": 1e2, "Pa": 1.0,
}

def to_pascal(u: str, v: float) -> float:
    lin, base = _log_wrap(u, v)
    b = base.strip()
    if b in _PA_EXACT:
        return lin * _PA_EXACT[b]
    bl = b.lower().strip()
    bl = bl.replace("(", "").replace(")", "")
    if bl in _PA:
        return lin * _PA[bl]
    # a few spellings
    aliases = {"torr": 133.322, "mm hg": 133.322, "um hg": 0.133322,
               "cm hg": 1333.22, "micrometers h": 0.133322, "atm": 101325.0,
               "bar": 1e5, "mbar": 1e2}
    if bl in aliases:
        return lin * aliases[bl]
    return math.nan


# solubility units -> mol/L  (needs MW g/mol)
def to_molL(u: str, v: float, mw: float) -> float:
    lin, base = _log_wrap(u, v)
    b = base.lower().strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    if b in ("mf", "mole fraction", "molefraction", "mf soln", "x"):
        return lin * WATER_M                       # aqueous mole fraction -> mol/L (dilute)
    if b in ("mol/l", "m", "mol/dm3", "molar"):
        return lin
    if b in ("mmol/l", "mm"):
        return lin * 1e-3
    if b in ("micro m", "microm", "um", "µm", "\xb5m", "mumol/l"):
        return lin * 1e-6
    if b in ("mol/m3", "mmol/l atm"):
        return lin * 1e-3
    if b in ("mol/kg", "mol/kg water"):
        return lin            # ~ water density 1
    if not math.isfinite(mw) or mw <= 0:
        return math.nan
    # mass-based -> need MW
    if b in ("mg/l", "g/m3", "microgm/ml", "ug/ml", "µg/ml"):
        return (lin * 1e-3) / mw            # mg/L
    if b in ("g/l", "mg/ml", "g/dm3"):
        return lin / mw
    if b in ("microgm/l", "ug/l", "µg/l", "ng/ml"):
        return (lin * 1e-6) / mw
    if b in ("ppm", "ppm mol", "mg/kg"):
        return (lin * 1e-3) / mw            # ~ mg/L in dilute water
    if b in ("ppb", "ppbvbyv"):
        return (lin * 1e-6) / mw
    if b in ("g/100g", "g/100ml", "g/dl", "wt%", "mass fraction", "g/100 g"):
        # g per 100 g(mL) water -> g/L = v*10
        return (lin * 10.0) / mw
    if b in ("g/kg", "g/kg water"):
        return lin / mw
    if b in ("g/g", "g/mL".lower(), "g/cm^3", "g/ml", "kg/m3"):
        return (lin) / mw * (1e3 if b == "kg/m3" else 1e3)  # g/g*1000 g/L approx
    return math.nan


# mole-fraction -> molarity (dilute aqueous)
def mf_to_molL(x_aq: float) -> float:
    return x_aq * WATER_M


# ---------------------------------------------------------------------------
# Henry conversions -> K_wa (dimensionless C_aq/C_gas)
# ---------------------------------------------------------------------------
def henry_to_kwa(process: str, u: str, v: float, T: float) -> float:
    # strip surrounding brackets so log/ln prefixes inside [] are seen
    u2 = u.strip()
    if u2.startswith("[") and u2.endswith("]"):
        u2 = u2[1:-1].strip()
    lin, base = _log_wrap(u2, v)
    b = base.strip().lower().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    b = b.replace("  ", " ").strip()

    # --- dimensionless concentration ratios ---
    # EMPIRICAL (validated vs FreeSolv): despite the KWG/KGW label, the stored
    # dimensionless ratio is the standard air/water Henry constant H_cc = C_gas/C_aq,
    # so K_wa = C_aq/C_gas = 1/value  (both KWG and KGW).
    if b in ("m/m", "molar/molar", "mol/l/mol/l", "dimensionless", "ca/cw",
             "log molar/molar", "mf/mf", "mf/mf and", "cw/ca"):
        r = lin
        if not (r > 0):
            return math.nan
        if b == "cw/ca":            # explicitly water/air -> already K_wa
            return r
        # EMPIRICAL (validated vs FreeSolv): the stored ratio's direction depends on the
        # process tag. KWG rows store the air/water constant (invert -> K_wa=1/r); KGW rows
        # store water/gas directly (K_wa=r). The labels are effectively swapped in the source.
        return r if process == "KGW" else 1.0 / r

    # --- Ostwald / Bunsen / Kuenen solubility coefficients (K_wa-like) ---
    if b in ("ostwald",):
        return lin                                  # L = C_aq/C_gas
    if b in ("bunsen", "kuenen", "kuenen ", "bunsen "):
        # Bunsen alpha: vol gas(STP)/vol liquid at 1 atm -> Ostwald L = alpha*T/273.15
        return lin * T / 273.15

    # --- P*V/mol Henry (H = P/C_aq); K_wa = R_SI*T / H[Pa m3/mol] ---
    pv = {
        "atm m3/mol": 101325.0, "atm m^3/mol": 101325.0, "m3 atm/mol": 101325.0,
        "pa m3/mol": 1.0, "kpa m3/mol": 1e3, "pa l/mol": 1e-3,
        "atm cm3/mol": 101325.0 * 1e-6, "l atm/mol": 101325.0 * 1e-3,
        "atm l/mol": 101325.0 * 1e-3,
    }
    if b in pv:
        H = lin * pv[b]
        return R_SI * T / H if H != 0 else math.nan

    # --- C_aq/P solubility-form Henry (M/atm etc); K_wa = K*R_LATM*T ---
    cp = {
        "m/atm": 1.0, "mol/l atm": 1.0, "mol/dm3 atm": 1.0, "mol/kg atm": 1.0,
        "mm/atm": 1e-3, "mol/m3 atm": 1e-3, "mmol/l atm": 1e-3,
        "m/pa": 101325.0, "mol/m3 pa": 101325.0 * 1e-3,
    }
    if b in cp:
        K = lin * cp[b]
        return K * R_LATM * T

    # --- P/mole-fraction Henry (Pa/mf etc); K_wa = WATER_M*R_SI*T / K[Pa/mf] ---
    pmf = {
        "pa/mf": 1.0, "kpa/mf": 1e3, "mpa/mf": 1e6, "atm/mf": 101325.0,
        "mmhg/mf": 133.322, "torr/mf": 133.322, "bar/mf": 1e5,
    }
    if b in pmf:
        K = lin * pmf[b]
        # C_aq[mol/L]=x*WATER_M ; C_gas[mol/L]=P/(R_SI T)/1000 ; K_wa=WATER_M*1000*R_SI*T/K
        return WATER_M * 1000.0 * R_SI * T / K if K != 0 else math.nan

    # --- mole-fraction / pressure solubility (mf/atm): x_aq = v*P ... treat as C_aq/P
    if b in ("mf/atm",):
        # x per atm -> C_aq/P = v*WATER_M mol/L/atm
        K = lin * WATER_M
        return K * R_LATM * T

    # --- H/RT dimensionless (already ~ C_gas/C_aq) ---
    if b in ("h/rt",):
        r = lin
        return 1.0 / r if (process == "KWG" and r > 0) else (r if r > 0 else math.nan)

    return math.nan


def dg_kcal(u: str, v: float) -> float:
    b = u.strip().lower()
    if b in ("kcal/mol", "kcal/mole"):
        return v
    if b == "cal/mol":
        return v * 1e-3
    if b in ("kj/mol", "kj/mole"):
        return v * KCAL_PER_KJ
    if b == "j/mol":
        return v * 1e-3 * KCAL_PER_KJ
    return math.nan


def kwa_to_dg(kwa: float, T: float) -> float:
    if not (isinstance(kwa, float) and math.isfinite(kwa)) or kwa <= 0:
        return math.nan
    return -R_KCAL * T * math.log(kwa)


# ---------------------------------------------------------------------------
# InChIKey
# ---------------------------------------------------------------------------
_ikey_cache: dict[str, str | None] = {}
def inchikey(smiles: str):
    if smiles in _ikey_cache:
        return _ikey_cache[smiles]
    out = None
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is not None:
        try:
            out = Chem.MolToInchiKey(m)
        except Exception:
            out = None
    _ikey_cache[smiles] = out
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
_DHVAP = None
def _dhvap_map() -> dict:
    """InChIKey -> dHvap (J/mol) from the cached meta37 table (empty if absent)."""
    global _DHVAP
    if _DHVAP is None:
        _DHVAP = {}
        f = OUT / "meta37_dhvap_by_inchikey.csv"
        try:
            if f.exists():
                dd = pd.read_csv(f)
                _DHVAP = {k: float(x) * 1000.0 for k, x in zip(dd["ikey"], dd["dHvap_kJmol"])}
        except Exception:
            pass
    return _DHVAP


def main() -> None:
    df = pd.read_csv(SRC, encoding="latin1", low_memory=False)
    df["value1"] = df["value1"].map(parse_val)
    df["error1"] = df["error1"].map(parse_val)
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")
    df["T"] = df["te53p"].map(to_kelvin)
    df["ikey"] = df["mol"].map(inchikey)

    direct_rows = []        # rows giving a dG directly
    vp_rows = []            # (ikey, smiles, Pa, T, err)
    sol_rows = []           # (ikey, smiles, molL, T, err)
    n_by_route = {"FEOH_DGS": 0, "henry": 0, "vp": 0, "aqsol": 0,
                  "unconverted": 0, "excluded_process": 0, "no_mol": 0}
    excluded = {"DGT", "DGTX", "DGB", "DGV", "PKA", "KA", "LAQACTCO", "DGV "}

    for row in df.itertuples(index=False):
        smi = row.mol
        ik = row.ikey
        proc = str(row.process).strip() if row.process is not None else ""
        unit = str(row.dimension1).strip() if row.dimension1 is not None else ""
        v = row.value1
        T = row.T
        err = row.error1
        if ik is None or not isinstance(smi, str):
            n_by_route["no_mol"] += 1
            continue
        if not math.isfinite(v):
            continue
        if proc in excluded:
            n_by_route["excluded_process"] += 1
            continue

        if proc in ("FEOH", "DGS"):
            g = dg_kcal(unit, v)
            if math.isfinite(g):
                direct_rows.append((ik, smi, g, T, err, unit, proc, "free_energy"))
                n_by_route["FEOH_DGS"] += 1
            else:
                n_by_route["unconverted"] += 1
        elif proc in ("KWG", "KGW"):
            kwa = henry_to_kwa(proc, unit, v, T)
            g = kwa_to_dg(kwa, T)
            if math.isfinite(g):
                direct_rows.append((ik, smi, g, T, err, unit, proc, "henry"))
                n_by_route["henry"] += 1
            else:
                n_by_route["unconverted"] += 1
        elif proc == "VP":
            pa = to_pascal(unit, v)
            if math.isfinite(pa) and pa > 0:
                # Clausius-Clapeyron correct VP to 25 C when meta37 dHvap is known
                # (many rows were measured at/near the boiling point -> VP~=1 atm artifact).
                Tvp = T
                dh = _dhvap_map().get(ik)
                if dh is not None and math.isfinite(T) and T > 0:
                    pa *= 10.0 ** (-(dh / R_SI / math.log(10.0)) * (1.0 / 298.15 - 1.0 / T))
                    Tvp = 298.15
                vp_rows.append((ik, smi, pa, Tvp, err))
                n_by_route["vp"] += 1
            else:
                n_by_route["unconverted"] += 1
        elif proc == "AQSOL":
            c = to_molL(unit, v, row.MW)
            if math.isfinite(c) and c > 0:
                sol_rows.append((ik, smi, c, T, err))
                n_by_route["aqsol"] += 1
            else:
                n_by_route["unconverted"] += 1
        else:
            n_by_route["unconverted"] += 1

    # --- pair VP + AQSOL per molecule -> dG ---
    vp = pd.DataFrame(vp_rows, columns=["ikey", "smi", "Pa", "T", "err"])
    sol = pd.DataFrame(sol_rows, columns=["ikey", "smi", "molL", "T", "err"])
    paired_rows = []
    if len(vp) and len(sol):
        # geometric-median-ish: use median (log) per molecule
        vp_g = vp.groupby("ikey").agg(Pa=("Pa", lambda s: float(np.exp(np.median(np.log(s))))),
                                      smi=("smi", "first"),
                                      T=("T", "median"), n=("Pa", "size"))
        sol_g = sol.groupby("ikey").agg(molL=("molL", lambda s: float(np.exp(np.median(np.log(s))))),
                                        T=("T", "median"), n=("molL", "size"))
        both = vp_g.join(sol_g, how="inner", lsuffix="_vp", rsuffix="_sol")
        for ik, r in both.iterrows():
            T = float(np.nanmean([r["T_vp"], r["T_sol"]])) if math.isfinite(r["T_vp"]) else 298.15
            c_gas = r["Pa"] / (R_SI * T)          # mol/m3
            c_aq = r["molL"] * 1e3                  # mol/m3
            kwa = c_aq / c_gas
            g = kwa_to_dg(kwa, T)
            if math.isfinite(g):
                paired_rows.append((ik, r["smi"], g, T, np.nan, "VP+AQSOL", "PAIR", "vp_sol_pair"))

    cols = ["ikey", "smiles", "dg_hyd", "T", "err", "unit", "process", "route"]
    obs = pd.DataFrame(direct_rows + paired_rows, columns=cols)

    # sanity clip: physical dG_hyd for small molecules roughly [-60, 10]
    before = len(obs)
    obs = obs[(obs["dg_hyd"] > -60) & (obs["dg_hyd"] < 12)].reset_index(drop=True)
    n_clipped = before - len(obs)

    obs.to_csv(OUT / "guthrie_dg_observations.csv", index=False)

    summary = {
        "n_rows_total": int(len(df)),
        "n_observations_converted": int(len(obs)),
        "n_molecules": int(obs["ikey"].nunique()),
        "n_clipped_unphysical": int(n_clipped),
        "route_counts": {k: int(v) for k, v in n_by_route.items()},
        "obs_route_counts": obs["route"].value_counts().to_dict(),
        "molecules_with_ge2_obs": int((obs.groupby("ikey").size() >= 2).sum()),
        "molecules_with_ge3_obs": int((obs.groupby("ikey").size() >= 3).sum()),
        "n_vp_molecules": int(vp["ikey"].nunique()) if len(vp) else 0,
        "n_sol_molecules": int(sol["ikey"].nunique()) if len(sol) else 0,
        "n_paired_molecules": int(len(paired_rows)),
        "dg_hyd_stats": {
            "min": float(obs["dg_hyd"].min()), "median": float(obs["dg_hyd"].median()),
            "max": float(obs["dg_hyd"].max()), "mean": float(obs["dg_hyd"].mean()),
        },
    }
    (OUT / "harmonize_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
