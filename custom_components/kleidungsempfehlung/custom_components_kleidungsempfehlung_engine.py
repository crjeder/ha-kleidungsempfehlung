"""
Engine: PMV/PPD und clo-Recommendation (aus dem Gist, leicht angepasst).

Dieses Modul enthält die Kernfunktionen:
- pmv_ppd(...)
- required_clo(...)
- du_bois_bsa(...)
- effective_met(...)
- recommend_clo_with_overlays(...)
"""
from typing import Tuple, Dict, Any
import math

SIGMA = 5.670367e-8
EMISS = 0.95

def pmv_ppd(tdb: float, tr: float, vr: float, rh: float, met: float, clo: float, wme: float = 0.0) -> Tuple[float, float]:
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (tdb + 235.0))
    icl = 0.155 * clo
    m = met * 58.15
    w = wme * 58.15
    mw = m - w

    fcl = (1.0 + 1.29 * icl) if icl <= 0.078 else (1.05 + 0.645 * icl)
    tcla = tdb + (35.5 - tdb) / (3.5 * (icl + 0.1))

    for _ in range(150):
        h_c_forced = 12.1 * math.sqrt(max(vr, 0.0))
        h_c_nat = 2.38 * abs(tcla - tdb) ** 0.25
        hc = max(h_c_forced, h_c_nat)

        tcl_k = tcla + 273.15
        tr_k = tr + 273.15
        hr = 4.0 * EMISS * SIGMA * ((tcl_k + tr_k) / 2.0) ** 3

        num = (35.7 - 0.028 * mw) + (hr * tr + hc * tdb) * icl * fcl
        den = 1.0 + icl * fcl * (hr + hc)
        tcla_new = num / den
        if abs(tcla_new - tcla) < 1e-4:
            tcla = tcla_new
            break
        tcla = tcla_new

    r_cl = fcl * hr * (tcla - tr)
    c_cl = fcl * hc * (tcla - tdb)

    e_dif = 3.05 * max(0.0, 5.733 - 0.007 * mw - pa / 1000.0)
    e_sw  = max(0.0, 0.42 * (mw - 58.15))
    e_res = 1.7e-5 * m * (5867.0 - pa)
    c_res = 0.0014 * m * (34.0 - tdb)

    tl = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = tl * (mw - (r_cl + c_cl) - e_dif - e_sw - e_res - c_res)
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)

    return pmv, ppd

def required_clo(tdb: float, tr: float, vr: float, rh: float, met: float, target_pmv: float = 0.0,
                 clo_bounds=(0.0, 6.0), iters: int = 60) -> float:
    lo, hi = clo_bounds
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pmv, _ = pmv_ppd(tdb, tr, vr, rh, met, mid)
        if pmv > target_pmv:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def du_bois_bsa(height_m: float, mass_kg: float) -> float:
    height_cm = max(height_m, 0.0) * 100.0
    mass = max(mass_kg, 0.0)
    return 0.007184 * (mass ** 0.425) * (height_cm ** 0.725)

def effective_met(met_tab: float,
                  height_m: float,
                  mass_kg: float,
                  body_fat_perc: float | None = None,
                  bsa_ref: float = 1.8,
                  apply_ffm_adjust: bool = True,
                  ffm_weight: float = 0.2) -> float:
    bsa = du_bois_bsa(height_m, mass_kg)
    scale_bsa = bsa / bsa_ref if bsa_ref > 0 else 1.0
    met_eff = met_tab * scale_bsa

    if apply_ffm_adjust and (body_fat_perc is not None) and mass_kg > 0:
        kfa = max(0.0, min(70.0, body_fat_perc))
        ffm_kg = mass_kg * (1.0 - kfa / 100.0)
        met_eff *= (0.8 + ffm_weight * (ffm_kg / mass_kg))
    return met_eff

def recommend_clo_with_overlays(
    tdb: float,
    tr: float,
    vr: float,
    rh: float,
    met_tab: float,
    height_m: float,
    mass_kg: float,
    alpha_age: float = 0.0,
    body_fat_perc: float | None = None,
    k_f: float = 0.007,
    delta_pmv_pref: float = 0.0,
    delta_clo_pref: float = 0.0,
    bsa_ref: float = 1.8,
    apply_ffm_adjust: bool = True,
    ffm_weight: float = 0.2,
    clo_bounds=(0.0, 6.0),
    iters: int = 60
) -> (float, Dict[str, Any]):
    met_eff = effective_met(
        met_tab=met_tab,
        height_m=height_m,
        mass_kg=mass_kg,
        body_fat_perc=body_fat_perc,
        bsa_ref=bsa_ref,
        apply_ffm_adjust=apply_ffm_adjust,
        ffm_weight=ffm_weight
    )

    alpha = max(0.0, min(0.5, alpha_age))
    met_eff *= (1.0 - alpha)

    pmv_target = 0.0 + float(delta_pmv_pref)

    clo_star = required_clo(
        tdb=tdb, tr=tr, vr=vr, rh=rh,
        met=met_eff,
        target_pmv=pmv_target,
        clo_bounds=clo_bounds,
        iters=iters
    )

    delta_clo_fat = 0.0
    if body_fat_perc is not None:
        delta_clo_fat = float(k_f) * (float(body_fat_perc) - 20.0)

    clo_req = clo_star - delta_clo_fat + float(delta_clo_pref)

    details = {
        "met_tab_in": met_tab,
        "met_eff_out": met_eff,
        "alpha_age": alpha,
        "bsa_m2": du_bois_bsa(height_m, mass_kg),
        "pmv_target": pmv_target,
        "clo_star": clo_star,
        "delta_clo_fat": delta_clo_fat,
        "delta_clo_pref": float(delta_clo_pref),
        "clo_req": clo_req
    }
    return clo_req, details