"""Host-microbe rescue model with density regulation and QME reduction.

Paper notation
--------------
    p_m   marginal frequency of microbial carriers
    p_g   marginal frequency of carriers of the focal host allele
    D     host-microbe association, D = p_gm - p_g p_m
    s_m   microbial selection coefficient
    s_g   host-allele selection coefficient
    e     non-additive gene-microbe interaction coefficient

The joint frequency p_gm is needed only to define D. Thereafter the model is
written in (p_m, p_g, D). Common competition -cN enters every host class
equally, so it bounds abundance without changing frequency dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    beta: float = 1.0
    ell0: float = 0.40
    phi: float = 0.0  # phi = f(1-v)
    sm: float = 0.03
    sg: float = 0.05
    e: float = 0.04
    r: float = 0.04  # decline magnitude; baseline post-change growth is -r
    c: float = 0.05

    @property
    def phat_m(self) -> float:
        return 1.0 - self.ell0 / self.beta



# ---------------------------------------------------------------------------
# Core expressions.
#
# These take plain scalars or numpy arrays so that grid sweeps and single
# trajectories use exactly the same algebra. Nothing anywhere else in the
# project may re-implement these formulas: every figure calls down to here.
# ---------------------------------------------------------------------------

def pm_qme_core(pg, *, phat, beta, sm, sg, e, phi):
    """First-order QME microbial prevalence, conditional on p_g."""
    return phat + ((sm + e * pg) * (1.0 - phat)
                   - phi * (sm + (sg + e) * pg)) / beta


def D_qme_core(pg, *, phat, beta, sg, e, phi):
    """First-order QME host-microbe association, conditional on p_g."""
    return (phat * pg * (1.0 - pg) / beta
            * (e * (1.0 - phat) - phi * (sg + e)))


def pg_dot_core(pg, pm, D, *, sg, sm, e):
    """Exact host-allele equation, evaluated at supplied (p_m, D)."""
    return sg * pg * (1.0 - pg) + sm * D + e * (pg * pm + D) * (1.0 - pg)


def growth_core(pg, pm, D, *, r, sg, sm, e):
    """Low-density (competition-free) host growth rate."""
    return -r + sg * pg + sm * pm + e * (pg * pm + D)


def pm_qme(pg, p: Params):
    return pm_qme_core(np.asarray(pg), phat=p.phat_m, beta=p.beta,
                       sm=p.sm, sg=p.sg, e=p.e, phi=p.phi)


def D_qme(pg, p: Params):
    return D_qme_core(np.asarray(pg), phat=p.phat_m, beta=p.beta,
                      sg=p.sg, e=p.e, phi=p.phi)


def phi_critical(p: Params):
    """Dilution threshold where d p_m~/d p_g and D~ change sign."""
    den = p.sg + p.e
    if den <= 0:
        return np.inf
    return p.e * (1.0 - p.phat_m) / den


def pg_dot_qme(pg, p: Params):
    return pg_dot_core(pg, pm_qme(pg, p), D_qme(pg, p),
                       sg=p.sg, sm=p.sm, e=p.e)


def low_density_growth_qme(pg, p: Params):
    return growth_core(pg, pm_qme(pg, p), D_qme(pg, p),
                       r=p.r, sg=p.sg, sm=p.sm, e=p.e)


def qme_rhs(_t, state, p: Params):
    pg, N = state
    m = low_density_growth_qme(pg, p)
    return np.array([pg_dot_qme(pg, p), N * (m - p.c * N)])


def host_only_rhs(_t, state, p: Params):
    pg, N = state
    m = -p.r + p.sg * pg
    return np.array([p.sg * pg * (1.0 - pg), N * (m - p.c * N)])


def frequency_rhs(_t, state, p: Params):
    pm, pg, D = state
    beta, ell0, phi, sm, sg, e = p.beta, p.ell0, p.phi, p.sm, p.sg, p.e
    joint = pm * pg + D
    pmdot = (
        sm * pm * (1.0 - pm)
        + sg * D
        + e * joint * (1.0 - pm)
        + beta * pm * (1.0 - pm)
        - ell0 * pm
        - phi * (sm * pm + (sg + e) * joint)
    )
    pgdot = pg_dot_core(pg, pm, D, sg=sg, sm=sm, e=e)
    Ddot = (
        e * (pm * pg * (1.0 - pm) * (1.0 - pg) + D * (1.0 - pm - pg) - D**2)
        + D * (sm * (1.0 - 2.0 * pm) + sg * (1.0 - 2.0 * pg) - (beta * pm + ell0))
        - phi * ((sg + e) * (1.0 - pg) * joint + sm * D)
    )
    return np.array([pmdot, pgdot, Ddot])


def fast_rhs_fixed_pg(_t, state, p: Params, pg_fixed):
    """Fast (p_m,D) subsystem conditional on a fixed slow p_g."""
    pm, D = state
    dpm, _, dD = frequency_rhs(_t, np.array([pm, pg_fixed, D]), p)
    return np.array([dpm, dD])


def full_rhs(t, state, p: Params):
    pm, pg, D, N = state
    dpm, dpg, dD = frequency_rhs(t, state[:3], p)
    m = growth_core(pg, pm, D, r=p.r, sg=p.sg, sm=p.sm, e=p.e)
    return np.array([dpm, dpg, dD, N * (m - p.c * N)])


def _solve(fun, y0, t_end, n=1200):
    sol = solve_ivp(
        fun, (0.0, t_end), y0, method="LSODA", dense_output=True,
        rtol=2e-9, atol=1e-11, max_step=t_end / 1000.0,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    t = np.linspace(0.0, t_end, n)
    return t, sol.sol(t)


def solve_qme(p: Params, pg0=0.02, N0=0.8, t_end=350.0, n=1600):
    return _solve(lambda t, z: qme_rhs(t, z, p), [pg0, N0], t_end, n)


def solve_host_only(p: Params, pg0=0.02, N0=0.8, t_end=350.0, n=1600):
    return _solve(lambda t, z: host_only_rhs(t, z, p), [pg0, N0], t_end, n)


def solve_fast_fixed_pg(p: Params, pg_fixed=0.35, pm0=0.15, D0=0.0,
                        t_end=12.0, n=500):
    return _solve(lambda t, z: fast_rhs_fixed_pg(t, z, p, pg_fixed),
                  [pm0, D0], t_end, n)


def solve_full(p: Params, pg0=0.02, N0=0.8, t_end=350.0, n=1600):
    pm0 = float(pm_qme(pg0, p))
    D0 = float(D_qme(pg0, p))
    return _solve(lambda t, z: full_rhs(t, z, p), [pm0, pg0, D0, N0], t_end, n)

# ---------------------------------------------------------------------------
# Validation utilities used by the retained Sobol robustness analysis.
# These are defined here so every manuscript figure and validation uses the
# same model implementation and notation.
# ---------------------------------------------------------------------------

def covariance_bounds(pm: float | np.ndarray, pg: float | np.ndarray):
    """Physical bounds for D given two Bernoulli marginals p_m and p_g."""
    pm = np.asarray(pm)
    pg = np.asarray(pg)
    lower = np.maximum(-pm * pg, -(1.0 - pm) * (1.0 - pg))
    upper = np.minimum(pm * (1.0 - pg), (1.0 - pm) * pg)
    return lower, upper


def is_physical_state(pm: float, pg: float, D: float, tol: float = 1e-9) -> bool:
    """Return True when (p_m,p_g,D) corresponds to valid four-class frequencies."""
    if not (-tol <= pm <= 1.0 + tol and -tol <= pg <= 1.0 + tol):
        return False
    lo, hi = covariance_bounds(float(pm), float(pg))
    return bool(lo - tol <= D <= hi + tol)


def solve_frequency_trajectory(
    p: Params,
    pg0: float = 0.05,
    pm0: float | None = None,
    D0: float | None = None,
    t_end: float | None = None,
    pg_target: float | None = 0.95,
    n: int = 800,
):
    """Integrate the full frequency system, optionally stopping at a p_g target.

    If ``pm0`` and ``D0`` are omitted, the trajectory begins on the first-order
    QME manifold. This helper is used by the broader Sobol robustness analysis.
    """
    if pm0 is None:
        pm0 = float(pm_qme(pg0, p))
    if D0 is None:
        D0 = float(D_qme(pg0, p))
    if not is_physical_state(pm0, pg0, D0):
        raise ValueError("Initial state is not biologically admissible")

    leading = p.sg + p.e * p.phat_m
    if t_end is None:
        scale = max(abs(leading), abs(p.sg), abs(p.sm), abs(p.e), 1e-3)
        t_end = 12.0 / scale

    events = None
    if pg_target is not None:
        direction = 1.0 if pg_target > pg0 else -1.0

        def target_event(_t, z):
            return z[1] - pg_target

        target_event.terminal = True
        target_event.direction = direction
        events = target_event

    sol = solve_ivp(
        lambda t, z: frequency_rhs(t, z, p),
        (0.0, float(t_end)),
        [pm0, pg0, D0],
        method="LSODA",
        dense_output=True,
        events=events,
        rtol=2e-9,
        atol=1e-11,
        max_step=max(float(t_end) / 500.0, 1e-4),
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    if events is not None and sol.t_events and len(sol.t_events[0]):
        stop = float(sol.t_events[0][0])
    else:
        stop = float(sol.t[-1])
    t = np.linspace(0.0, stop, n)
    pm, pg, D = sol.sol(t)
    return t, pm, pg, D


def microbial_adaptation_effect(pm, pg, D, p: Params):
    """Microbial contribution A_m to effective selection on the rescue allele."""
    denom = pg * (1.0 - pg)
    return p.e * (pm + D / pg) + p.sm * D / denom


def microbial_growth_contribution(pm, pg, D, p: Params):
    """Microbial contribution G_m to host low-density growth."""
    return p.sm * pm + p.e * (pm * pg + D)

