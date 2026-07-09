#!/usr/bin/env python3
"""A declarative *rule engine* for GuthrieSolv units — not a pile of regexes.

A unit string is parsed by dimensional analysis:
  1. peel off a log/ln/-log wrapper and any brackets (WRAPPERS);
  2. tokenise the remainder into a product/quotient of  <prefix><base>^power  factors
     plus bare numeric scale factors (e.g. the 100 in "g/100mL");
  3. each base token contributes a dimension vector and an SI factor (BASE table);
     each prefix a power of ten (PREFIX table);
  4. sum exponents -> a net dimension signature + net SI factor + numeric scale;
  5. CLASSIFY the signature into a physical quantity (pressure / concentration /
     mole-fraction / the four Henry forms / free energy / dimensionless ratio);
  6. CONVERT that quantity (with process, T, MW) to Ben-Naim dG_hyd, or to an
     intermediate (Pa for VP, mol/L for solubility) for pairing.

Extending coverage = adding a row to BASE or WRAPPERS, never touching the parser.
Dimensions tracked: P pressure, N amount(mol), V volume(m^3), G mass(g),
X mole-fraction, E energy(kcal/mol).
"""
from __future__ import annotations
import math, re
from dataclasses import dataclass, field

R_SI = 8.314462          # J/mol/K
R_LATM = 0.082057        # L atm / mol / K
R_KCAL = 1.987204e-3     # kcal/mol/K
WATER_M = 55.345         # mol/L

# ---------------------------------------------------------------------------
# RULE TABLES  (the whole "knowledge" of the system lives here)
# ---------------------------------------------------------------------------
PREFIX = {
    "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "h": 1e2, "da": 1e1,
    "d": 1e-1, "c": 1e-2, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "mu": 1e-6,
    "micro": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}
# base token -> (dimension dict, SI factor to the dimension's base unit)
#   pressure base = Pa ; amount = mol ; volume = m^3 ; mass = g ; energy = kcal/mol
BASE = {
    # pressure
    "pa": ({"P": 1}, 1.0), "bar": ({"P": 1}, 1e5), "atm": ({"P": 1}, 101325.0),
    "torr": ({"P": 1}, 133.322), "mmhg": ({"P": 1}, 133.322), "cmhg": ({"P": 1}, 1333.22),
    "umhg": ({"P": 1}, 0.133322), "psi": ({"P": 1}, 6894.76), "hpa": ({"P": 1}, 100.0),
    # amount / volume / mass
    "mol": ({"N": 1}, 1.0), "mole": ({"N": 1}, 1.0),
    "l": ({"V": 1}, 1e-3), "litre": ({"V": 1}, 1e-3), "liter": ({"V": 1}, 1e-3),
    "dm3": ({"V": 1}, 1e-3), "m3": ({"V": 1}, 1.0), "cm3": ({"V": 1}, 1e-6),
    "cc": ({"V": 1}, 1e-6), "ml": ({"V": 1}, 1e-6),
    "g": ({"G": 1}, 1.0), "kg": ({"G": 1}, 1e3),
    # composite/atomic concentration & mole fraction
    "m": ({"N": 1, "V": -1}, 1000.0),          # molar = mol/L = 1000 mol/m^3
    "molar": ({"N": 1, "V": -1}, 1000.0),
    "mf": ({"X": 1}, 1.0), "molefraction": ({"X": 1}, 1.0),
    # energy (free energy per mole)
    "kcal": ({"E": 1}, 1.0), "cal": ({"E": 1}, 1e-3),
    "kj": ({"E": 1}, 0.239006), "j": ({"E": 1}, 0.239006e-3),
    # dimensionless / gas-solubility coefficients (handled specially in convert)
    "dimensionless": ({}, 1.0), "ostwald": ({"OSTWALD": 1}, 1.0),
    "bunsen": ({"BUNSEN": 1}, 1.0), "kuenen": ({"BUNSEN": 1}, 1.0),
}
# aliases normalised before tokenising
ALIAS = {
    "mm hg": "mmhg", "cm hg": "cmhg", "um hg": "umhg", "µm hg": "umhg",
    "mmhg": "mmhg", "torr": "torr", "kn/m2": "kpa", "kn/(m^2)": "kpa", "kn/m^2": "kpa",
    "n/m2": "pa", "mbar": "mbar", "mole fraction": "mf", "molar/molar": "m/m",
    "mol/mol": "m/m", "[mf]": "mf", "mole/mole": "m/m",
    "wt%": "g/g", "mass fraction": "g/g", "[mf/mf]and": "mf/mf",
    "ppm": "mg/l", "ppm (mol)": "mg/l", "ppb": "ug/l", "ppbvbyv": "ug/l",
    "microgm/l": "ug/l", "microgm/ml": "ug/ml", "mg%": "mg/dl",
}
# substring normalisations applied to the (de-wrapped) expression before tokenising.
# Fixes ambiguous spellings: caret cubic-metre, multi-word mercury, molality~molarity.
NORMALISE = [
    ("m^3", "m3"), ("cm^3", "cm3"), ("dm^3", "dm3"), ("m ^3", "m3"),
    ("(cm^3)", "cm3"), ("mm hg", "mmhg"), ("cm hg", "cmhg"), ("um hg", "umhg"),
    ("µm hg", "umhg"), ("mol/kg", "mol/l"), ("mol/(kg", "mol/(l"),
    ("/kg atm", "/l atm"), ("g/kg", "g/l"),
]
# named non-dimensional Henry expressions the dimensional parser can't infer
SPECIAL = {
    "ca/cw": ("RATIO", "gw"),   # C_air/C_water -> invert
    "cw/ca": ("RATIO", "wg"),   # C_water/C_air -> keep
    "h/rt": ("RATIO", "gw"),    # dimensionless air/water Henry
    "[dimensionless]": ("RATIO", "auto"), "dimensionless": ("RATIO", "auto"),
}


@dataclass
class Parsed:
    dim: dict = field(default_factory=dict)     # net dimension exponents
    si: float = 1.0                              # net SI factor (to base dims)
    scale: float = 1.0                           # numeric scale (e.g. /100)
    transform: str = "lin"                       # lin | log10 | ln | neglog10
    ok: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# wrappers (log / ln / -log / brackets)  -- generic, order matters
# ---------------------------------------------------------------------------
def peel_wrapper(u: str):
    s = u.strip()
    # strip matched outer SQUARE brackets only (parens can belong to "(-)log")
    for _ in range(3):
        s2 = s.strip()
        if len(s2) >= 2 and s2[0] == "[" and s2[-1] == "]":
            s = s2[1:-1].strip()
        else:
            break
    low = s.lower()
    # "log p, Torr" / "log p atm" style
    m = re.match(r"^log\s*p[ ,]+(.*)$", low)
    if m:
        return "log10", m.group(1).strip()
    m = re.match(r"^log\s*\(?s\)?\s+(.*)$", low)   # "log(s) M"
    if m:
        return "log10", m.group(1).strip()
    if low.startswith("(-)log") or low.startswith("-log") or low.startswith("(-) log"):
        inner = re.sub(r"^\(?-\)?\s*log\d*", "", s, flags=re.I).strip(" ()[]")
        return "neglog10", inner
    if low.startswith("ln"):
        inner = re.sub(r"^ln", "", s, flags=re.I).strip(" ()[]")
        return "ln", inner
    if low.startswith("log"):
        inner = re.sub(r"^log\d*", "", s, flags=re.I).strip(" ()[]")
        return "log10", inner
    return "lin", s


# ---------------------------------------------------------------------------
# tokeniser: split into numerator / denominator, then factors
# ---------------------------------------------------------------------------
def _split_factors(expr: str):
    """Yield (token, power_sign) over a * / separated expression, honouring parens."""
    expr = expr.replace("per ", "/").replace(" per", "/")
    # normalise separators; keep '/' and '*' and spaces as multiply/divide
    out, sign, buf, depth = [], 1, "", 0
    i = 0
    tokens = []
    # first split on top-level '/' into num (sign +1) and rest (sign -1 thereafter)
    parts = re.split(r"/", expr)
    for pi, part in enumerate(parts):
        s = 1 if pi == 0 else -1
        part = part.strip().strip("()[]")
        for f in re.split(r"[*\s]+", part):
            f = f.strip()
            if f:
                tokens.append((f, s))
    return tokens


def _factor_dim(tok: str):
    """Return (dim, si, scale, ok) for a single factor token like 'atm', 'kpa', 'm3', '100ml'."""
    t = tok.strip().lower()
    if not t:
        return {}, 1.0, 1.0, True
    # bare number -> numeric scale
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return {}, 1.0, float(t), True
    # leading number glued to unit, e.g. 100ml, 100g
    m = re.match(r"^(\d+(?:\.\d+)?)([a-zµ].*)$", t)
    scale = 1.0
    if m:
        scale = float(m.group(1)); t = m.group(2)
    # direct base hit FIRST (so m3, cm3, dm3 are not mis-read as molar^3)
    if t in BASE:
        dim, si = BASE[t]
        return dict(dim), si, scale, True
    # prefix + base (longest prefix first)
    for p in sorted(PREFIX, key=len, reverse=True):
        if t.startswith(p) and t[len(p):] in BASE:
            base = t[len(p):]
            dim, si = BASE[base]
            return dict(dim), PREFIX[p] * si, scale, True
    # explicit exponent a^b / a2 as last resort (e.g. m^3 written oddly)
    m = re.match(r"^(.*?)\^(-?\d+)$", t)
    if m and m.group(1) in BASE:
        dim, si = BASE[m.group(1)]; power = int(m.group(2))
        return {k: v * power for k, v in dim.items()}, si ** power, scale, True
    return {}, 1.0, 1.0, False


def parse_unit(u: str) -> Parsed:
    if u is None:
        return Parsed(ok=False, note="none")
    raw = str(u).strip()
    low = raw.lower()
    if low in ("nan", ""):
        return Parsed(ok=False, note="empty")
    if low in ALIAS:
        raw = ALIAS[low]
    transform, inner = peel_wrapper(raw)
    inner_low = inner.lower().strip()
    if inner_low in ALIAS:
        inner = ALIAS[inner_low]; inner_low = inner.lower().strip()
    # named special dimensionless Henry expressions
    if inner_low in SPECIAL:
        _, hint = SPECIAL[inner_low]
        return Parsed(dim={}, transform=transform, ok=True, note=f"special:{hint}")
    # apply substring normalisations (caret cubic-metre, multi-word Hg, molality)
    for a, b in NORMALISE:
        if a in inner_low:
            inner = re.sub(re.escape(a), b, inner, flags=re.I)
    inner_low = inner.lower().strip()
    if inner_low in ALIAS:
        inner = ALIAS[inner_low]
    # special dimensionless coefficient names
    if inner.lower().strip() in ("ostwald", "bunsen", "kuenen"):
        dim, si = BASE[inner.lower().strip()]
        return Parsed(dim=dict(dim), si=si, transform=transform, ok=True, note="coeff")
    dim, si, scale, ok, seen = {}, 1.0, 1.0, True, set()
    for tok, sgn in _split_factors(inner):
        d, f, sc, o = _factor_dim(tok)
        if not o:
            return Parsed(ok=False, transform=transform, note=f"unparsed:{tok}")
        for k, v in d.items():
            dim[k] = dim.get(k, 0) + v * sgn
            seen.add(k)
        si *= f ** sgn
        scale *= sc ** sgn
    dim = {k: v for k, v in dim.items() if v != 0}
    # a same-dimension ratio (mass/mass, mole/mole) cancels to dimensionless; record the
    # basis so a solubility fraction (AQSOL) can be told from a Henry ratio (KWG/KGW).
    note = ""
    if not dim:
        if "G" in seen:
            note = "frac:mass"
        elif "X" in seen or "N" in seen:
            note = "frac:mole"
    return Parsed(dim=dim, si=si, scale=scale, transform=transform, ok=True, note=note)


# ---------------------------------------------------------------------------
# classify a parsed dimension into a physical quantity
# ---------------------------------------------------------------------------
def classify(p: Parsed) -> str:
    d = p.dim
    if "OSTWALD" in d:
        return "OSTWALD"
    if "BUNSEN" in d:
        return "BUNSEN"
    if d.get("E"):
        return "ENERGY"
    P, N, V, G, X = (d.get(k, 0) for k in ("P", "N", "V", "G", "X"))
    if not d:
        return "RATIO"                              # dimensionless (Henry ratio)
    if P == 1 and N == 0 and V == 0 and G == 0 and X == 0:
        return "PRESSURE"                           # vapour pressure
    if P == 0 and ((N == 1 and V == -1) or (G == 1 and V == -1)):
        return "CONCENTRATION"                      # solubility (molar or mass)
    if P == 0 and X == 1 and N == 0:
        return "MOLEFRAC"                           # mole-fraction solubility
    # Henry forms
    if P == 1 and V == 1 and N == -1:
        return "HENRY_PV"                           # P*V/mol  (= P / C)
    if P == -1 and ((N == 1 and V == -1) or (G == 1 and V == -1)):
        return "HENRY_CP"                           # C / P
    if P == 1 and X == -1:
        return "HENRY_PX"                           # P / mole-fraction
    if X == 1 and P == -1:
        return "HENRY_XP"                           # mole-fraction / P
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# convert to dG_hyd (or intermediate). Returns (kind, value):
#   ("dg", kcal)  | ("vp_pa", Pa) | ("sol_molL", mol/L) | (None, nan)
# ---------------------------------------------------------------------------
def _lin(p: Parsed, v: float) -> float:
    if p.transform == "log10":
        v = 10.0 ** v
    elif p.transform == "neglog10":
        v = 10.0 ** (-v)
    elif p.transform == "ln":
        v = math.exp(v)
    return v * p.scale


def kwa_to_dg(kwa: float, T: float) -> float:
    if not (isinstance(kwa, float) and math.isfinite(kwa)) or kwa <= 0:
        return math.nan
    return -R_KCAL * T * math.log(kwa)


def convert(unit: str, value: float, process: str, T: float, mw: float):
    p = parse_unit(unit)
    if not p.ok or not math.isfinite(value):
        return None, math.nan, p.note if not p.ok else "badval"
    q = classify(p)
    lin = _lin(p, value)
    proc = (process or "").upper()

    if q == "ENERGY":
        return "dg", lin * p.si, "energy"        # si already scales cal/kJ -> kcal
    if q == "PRESSURE":
        return "vp_pa", lin * p.si, "vp"          # Pa
    if q == "CONCENTRATION":
        # to mol/m^3 then mol/L
        if "G" in p.dim:                          # mass/volume -> need MW
            if not (math.isfinite(mw) and mw > 0):
                return None, math.nan, "need_mw"
            c_molm3 = lin * p.si / mw             # (g/m^3)/(g/mol)=mol/m^3
        else:
            c_molm3 = lin * p.si                  # already mol/m^3
        return "sol_molL", c_molm3 / 1000.0, "sol"
    if q == "MOLEFRAC":
        return "sol_molL", lin * WATER_M, "sol_mf"

    # ---- Henry -> K_wa (dimensionless C_aq/C_gas) ----
    if q == "HENRY_PV":                            # H = P/C in Pa*m^3/mol
        H = lin * p.si
        return "dg", kwa_to_dg(R_SI * T / H, T), "henry_pv"
    if q == "HENRY_CP":                            # C/P
        if "G" in p.dim:
            if not (math.isfinite(mw) and mw > 0):
                return None, math.nan, "need_mw"
            K = lin * p.si / mw                    # (mol/m^3)/Pa
        else:
            K = lin * p.si                         # (mol/m^3)/Pa
        # K_wa = C_aq/C_gas = (K*P)/(P/(R_SI T)) = K*R_SI*T
        return "dg", kwa_to_dg(K * R_SI * T, T), "henry_cp"
    if q == "HENRY_PX":                            # P/x  (Pa per mole fraction)
        K = lin * p.si
        kwa = WATER_M * 1000.0 * R_SI * T / K if K != 0 else math.nan
        return "dg", kwa_to_dg(kwa, T), "henry_px"
    if q == "HENRY_XP":                            # x/P
        K = lin * p.si                             # mole-fraction per Pa
        kwa = K * WATER_M * 1000.0 * R_SI * T
        return "dg", kwa_to_dg(kwa, T), "henry_xp"
    if q == "OSTWALD":
        return "dg", kwa_to_dg(lin, T), "ostwald"
    if q == "BUNSEN":
        return "dg", kwa_to_dg(lin * T / 273.15, T), "bunsen"
    if q == "RATIO":
        r = lin
        if not (r > 0):
            return None, math.nan, "ratio_nonpos"
        # AQSOL + a same-dimension fraction => a solubility fraction, not a Henry ratio
        if proc == "AQSOL" and p.note.startswith("frac:"):
            if p.note == "frac:mass":                 # w (g/g) -> mol/L via water density
                if not (math.isfinite(mw) and mw > 0):
                    return None, math.nan, "need_mw"
                return "sol_molL", r * 1000.0 / mw, "sol_massfrac"
            return "sol_molL", r * WATER_M, "sol_molefrac"
        # dimensionless C-ratio; direction is process-dependent (validated vs FreeSolv),
        # unless a named special fixes it explicitly (ca/cw, cw/ca, h/rt).
        hint = p.note.split(":")[1] if p.note.startswith("special:") else "auto"
        if hint == "wg":
            kwa = r
        elif hint == "gw":
            kwa = 1.0 / r
        else:
            kwa = r if proc == "KGW" else 1.0 / r
        return "dg", kwa_to_dg(kwa, T), "ratio"
    return None, math.nan, f"unclassified:{q}:{p.dim}"


if __name__ == "__main__":
    tests = ["log(M/M)", "atm m3/mol", "M/atm", "kcal/mol", "kJ/mol", "log(atm)",
             "(-)log(mol/L)", "mg/L", "g/100mL", "[mf]", "Pa/mf", "mol/(L atm)",
             "BUNSEN", "Ostwald", "log((Pa m3)/mol)", "kN/(m^2)", "mmHg", "ppm"]
    for u in tests:
        p = parse_unit(u)
        print(f"{u:20s} dim={p.dim} tf={p.transform} class={classify(p) if p.ok else 'FAIL'}")
