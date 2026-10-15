from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.optimize import root
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from qme_rescue_model import (
    Params, pm_qme, D_qme, phi_critical, fast_rhs_fixed_pg, frequency_rhs,
    pm_qme_core, D_qme_core, pg_dot_core, growth_core,
    solve_qme, solve_host_only, solve_full, solve_fast_fixed_pg,
    covariance_bounds, is_physical_state, solve_frequency_trajectory,
    microbial_adaptation_effect, microbial_growth_contribution,
)

HOST = "#4C78A8"      # host genetics
MICROBE = "#59A14F"   # additive microbial effect (buffering)
BOTH = "#D65F5F"      # non-additive gene-microbe effect (acceleration)
COMBINED = "#E17C3A"  # both microbial routes acting together
DILUTION = "#B279A2"  # inheritance failure / loss of assistance
NEUTRAL = "#575757"   # microbe-free reference and annotation
NOFLOW = "#DCDCDC"    # masked "no recovery" fill

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.2,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.5,
    "axes.linewidth": 0.65,
    "legend.fontsize": 6.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 2.7,
    "ytick.major.size": 2.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(FIG / f"{name}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter, dx=-26, dy=10):
    """Panel letter placed in points relative to the axes top-left corner.

    Point offsets do not rescale with axes width, so a long left-aligned
    title in one column can no longer run underneath the letter of the
    column to its right.
    """
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=10.5, fontweight="bold", va="bottom", ha="left",
                annotation_clip=False)


def scenario_title(ax, name, coords=None, pad=3.0):
    """Left-aligned title with an optional smaller coordinate subtitle."""
    ax.set_title(name, loc="left", pad=(13.0 if coords else pad))
    if coords:
        ax.text(0.0, 1.012, coords, transform=ax.transAxes,
                fontsize=6.3, color=NEUTRAL, va="bottom", ha="left")


def draw_host(ax, xy, gene=False, microbe=False, scale=1.0):
    x, y = xy
    ax.add_patch(Ellipse((x, y), .16*scale, .11*scale, fc="white", ec="#222", lw=1.0))
    for dx, dy in [(-.065,.045),(.065,.045),(-.065,-.045),(.065,-.045)]:
        ax.add_patch(Circle((x+dx*scale, y+dy*scale), .021*scale, fc="white", ec="#222", lw=1.0))
    badge_scale = max(scale, .95)
    badge_r = .015 * badge_scale
    badge_fs = max(4.5, 5.1*scale)
    if gene:
        gx = x-.030*scale
        ax.add_patch(Circle((gx,y), badge_r, fc=HOST, ec="none", zorder=4))
        ax.text(gx, y, "g", ha="center", va="center", color="white",
                fontsize=badge_fs, fontweight="bold", zorder=5)
    if microbe:
        mx = x+.030*scale
        ax.add_patch(Circle((mx,y), badge_r, fc=MICROBE, ec="none", zorder=4))
        ax.text(mx, y, "m", ha="center", va="center", color="white",
                fontsize=badge_fs, fontweight="bold", zorder=5)


# ---------------------------------------------------------------------------
# Figure 1 uses a local Okabe-Ito palette. Colour here labels a STATE VARIABLE
# (rescue allele, microbial carriage, association), not a mechanism as it does
# in Figs. 2-4, and red/green are avoided entirely. Kept local so Figs. 2-4 are
# untouched.
# ---------------------------------------------------------------------------
F1_ALLELE = "#0072B2"   # rescue allele / p_g
F1_MICROBE = "#E69F00"  # microbial carriage / p_m
F1_ASSOC = "#CC79A7"    # host-microbe association / D
F1_GREY = "#444444"     # loss, reference, annotation
F1_FIELD = "#C4C4C4"    # fast-flow streamlines

# Illustrative coefficients and rates for Fig. 1. Rates are per host
# generation. beta = 10 and ell0 = 4 preserve p_hat = 0.6 and make the
# fast-slow separation visible on the scale of one host generation.
F1_SG, F1_SM, F1_E = 0.03, 0.05, 0.10
F1_PG = 0.50       # slow variable at which the fast field is shown
F1_DT = 0.12       # host generations between early-approach markers
F1_TEND = 6.0      # host generations followed


def f1_params(beta=10.0, phi=0.0):
    return Params(beta=beta, ell0=beta*0.4, phi=phi,
                  sg=F1_SG, sm=F1_SM, e=F1_E, r=.04, c=.05)


def badge(ax, xy, kind, present, s=1.0):
    """Carriage badge: filled g/m when present, empty-set symbol when absent."""
    x, y = xy
    if present:
        col = F1_ALLELE if kind == "g" else F1_MICROBE
        ax.add_patch(Circle((x, y), .026*s, fc=col, ec="none", zorder=4))
        ax.text(x, y, kind, ha="center", va="center", color="white",
                fontsize=8.2*s, fontweight="bold", zorder=5)
    else:
        ax.text(x, y, r"$\varnothing$", ha="center", va="center",
                color="#9A9A9A", fontsize=11.0*s, zorder=5)


def _bracket_v(ax, x, y0, y1, tick=.016, **kw):
    ax.plot([x, x], [y0, y1], **kw)
    ax.plot([x-tick, x], [y0, y0], **kw)
    ax.plot([x-tick, x], [y1, y1], **kw)


def _bracket_h(ax, x0, x1, y, tick=.020, **kw):
    ax.plot([x0, x1], [y, y], **kw)
    ax.plot([x0, x0], [y, y+tick], **kw)
    ax.plot([x1, x1], [y, y+tick], **kw)


def _solve_local(p, pm0, pg0, D0, t_end, n):
    """Integrate the full frequency system; used only by Fig. 1B."""
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda t, z: frequency_rhs(t, z, p), (0., t_end),
                    [pm0, pg0, D0], method="LSODA", dense_output=True,
                    rtol=2e-10, atol=1e-14, max_step=t_end/2000.)
    tt = np.linspace(0., t_end, n)
    return tt, sol.sol(tt)


def figure_framework():
    fig = plt.figure(figsize=(7.15, 4.10))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.12, 1], height_ratios=[1, 1],
                           wspace=.38, hspace=.48)

    # =====================================================================
    # A. four host classes, their fitness effects, and microbial turnover
    # =====================================================================
    ax = fig.add_subplot(gs[:, 0]); ax.set_axis_off(); panel(ax, "A", dx=-10, dy=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Four host classes and microbial turnover", loc="left", pad=3, fontsize=7.8)

    CL, CR, CW, CH = .165, .690, .235, .250
    cx = {"l": CL+CW/2, "r": CR+CW/2}
    cy = {"t": .740, "b": .410}

    ax.text(cx["l"], .898, "microbe absent", ha="center", fontsize=6.9, color=F1_GREY)
    ax.text(cx["r"], .898, "microbe present", ha="center", fontsize=6.9, color=F1_MICROBE)
    ax.text(.150, cy["t"], "no rescue\nallele", ha="right", va="center",
            fontsize=6.9, color=F1_GREY, linespacing=1.25)
    ax.text(.150, cy["b"], "rescue\nallele", ha="right", va="center",
            fontsize=6.9, color=F1_ALLELE, linespacing=1.25)

    cells = [("l", "t", False, False, r"$\Delta r=0$"),
             ("r", "t", False, True,  r"$\Delta r=s_m$"),
             ("l", "b", True,  False, r"$\Delta r=s_g$"),
             ("r", "b", True,  True,  r"$\Delta r=s_g\!+\!s_m\!+\!e$")]
    for c, rw, g, m, eff in cells:
        x0 = CL if c == "l" else CR
        ax.add_patch(FancyBboxPatch((x0, cy[rw]-CH/2), CW, CH,
                     boxstyle="round,pad=0.004,rounding_size=0.014",
                     fc="white", ec=F1_GREY, lw=1.6 if (g and m) else .9, zorder=2))
        badge(ax, (cx[c]-.034, cy[rw]+.072), "g", g, s=1.25)
        badge(ax, (cx[c]+.034, cy[rw]+.072), "m", m, s=1.25)
        ax.text(cx[c], cy[rw]-.078, eff, ha="center", va="center",
                fontsize=6.8, color=F1_GREY)
    ax.text(cx["r"], cy["b"]-.008, r"$p_{gm}$", ha="center", va="center", fontsize=7.6)

    gapc = (CL+CW + CR)/2
    for rw, loss, lw, fs in [("t", r"$\ell_0+f(1-v)\,s_m$", 1.1, 6.1),
                             ("b", "$\\ell_0+f(1-v)\\,\\times$\n$(s_g\\!+\\!s_m\\!+\\!e)$", 2.1, 6.1)]:
        y = cy[rw]
        ax.add_patch(FancyArrowPatch((CL+CW+.010, y+.034), (CR-.010, y+.034),
                     arrowstyle="->", mutation_scale=8, lw=1.2, color=F1_MICROBE))
        ax.text(gapc, y+.048, r"$\beta p_m$", ha="center", fontsize=6.8, color=F1_MICROBE)
        ax.add_patch(FancyArrowPatch((CR-.010, y-.030), (CL+CW+.010, y-.030),
                     arrowstyle="->", mutation_scale=8, lw=lw, color=F1_GREY))
        ax.text(gapc, y-.044, loss, ha="center", va="top",
                fontsize=fs, color=F1_GREY, linespacing=1.35)

    _bracket_v(ax, CR+CW+.018, cy["b"]-CH/2, cy["t"]+CH/2, color=F1_MICROBE, lw=1.0)
    ax.text(CR+CW+.055, (cy["b"]+cy["t"])/2, r"$p_m$", color=F1_MICROBE,
            fontsize=9.5, ha="center", va="center")
    _bracket_h(ax, CL, CR+CW, cy["b"]-CH/2-.042, color=F1_ALLELE, lw=1.0)
    ax.text((CL+CR+CW)/2, cy["b"]-CH/2-.090, r"$p_g$", color=F1_ALLELE,
            fontsize=9.5, ha="center", va="center")
    ax.text(.50, .075, r"association   $D=p_{gm}-p_gp_m$", ha="center",
            fontsize=8.6, color=F1_ASSOC)

    # =====================================================================
    # B. At fixed p_g, show the actual fast variables approaching their QME
    # values. Two small stacked axes avoid normalization and twin y-axes.
    # =====================================================================
    p_fast = f1_params(beta=10.0)
    pgfix = 0.50
    t, z = solve_fast_fixed_pg(p_fast, pg_fixed=pgfix, pm0=.55, D0=0.0,
                               t_end=1.25, n=600)
    pm_eq = float(pm_qme(pgfix, p_fast))
    D_eq = float(D_qme(pgfix, p_fast))

    sub = gs[0, 1].subgridspec(2, 1, hspace=.10)
    ax_pm = fig.add_subplot(sub[0, 0])
    ax_D = fig.add_subplot(sub[1, 0], sharex=ax_pm)
    panel(ax_pm, "B", dx=-10, dy=6)
    ax_pm.set_title("Fast microbial variables relax to QME", loc="left", pad=3, fontsize=7.8)

    ax_pm.plot(t, z[0], color=F1_MICROBE, lw=2.0)
    ax_pm.axhline(pm_eq, color=F1_GREY, lw=.9, ls="--")
    ax_pm.axvline(1.0, color=F1_GREY, lw=.8, ls=":")
    ax_pm.text(.98, pm_eq, r"$\tilde p_m$", transform=ax_pm.get_yaxis_transform(),
               ha="right", va="bottom", fontsize=5.8, color=F1_GREY)
    ax_pm.set_ylabel(r"$p_m$")
    ax_pm.set_ylim(.545, .610)
    ax_pm.set_yticks([.55, .60])
    ax_pm.tick_params(axis="x", labelbottom=False)

    ax_D.plot(t, z[1], color=F1_ASSOC, lw=2.0)
    ax_D.axhline(D_eq, color=F1_GREY, lw=.9, ls="--")
    ax_D.axvline(1.0, color=F1_GREY, lw=.8, ls=":")
    ax_D.text(.98, D_eq, r"$\tilde D$", transform=ax_D.get_yaxis_transform(),
              ha="right", va="bottom", fontsize=5.8, color=F1_GREY)
    ax_D.set_ylabel(r"$D$")
    ax_D.set_xlabel("time (host generations)")
    ax_D.set_xlim(0, 1.25)
    ax_D.set_xticks([0, .5, 1.0])
    ax_D.set_ylim(-.00004, .00066)
    ax_D.set_yticks([0.0, D_eq])
    ax_D.set_yticklabels(["0.0000", f"{D_eq:.4f}"])

    # =====================================================================
    # C. The QME manifold parameterized by rescue-allele frequency only.
    # Ticks are placed at the coordinates corresponding to p_g = 0, 0.5, 1.
    # =====================================================================
    p = f1_params(beta=10.0)
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "C", dx=-10, dy=6)
    pg = np.linspace(0, 1, 500)
    pm_vals = pm_qme(pg, p)
    D_vals = D_qme(pg, p)
    ax.plot(pm_vals, D_vals, color="#1a1a1a", lw=2.4, solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
    for k in (165, 300, 430):
        ax.add_patch(FancyArrowPatch((pm_vals[k], D_vals[k]), (pm_vals[k+16], D_vals[k+16]),
                     arrowstyle="-|>", mutation_scale=14, lw=0, color="#1a1a1a", zorder=5))

    pg_marks = [0.0, 0.5, 1.0]
    pm_marks = [float(pm_qme(q, p)) for q in pg_marks]
    D_marks = [float(D_qme(q, p)) for q in pg_marks]
    labels = [r"$p_g=0$", r"$p_g=0.5$", r"$p_g=1$"]
    offsets = [(16, 6), (0, 5), (16, 6)]
    for xx, yy, lab, (dxo, dyo) in zip(pm_marks, D_marks, labels, offsets):
        ax.plot(xx, yy, "o", ms=4.6, mfc="white", mec="#1a1a1a", mew=1.1, zorder=6)
        ax.annotate(lab, (xx, yy), xytext=(dxo, dyo), textcoords="offset points",
                    ha="center", fontsize=6.2)

    xspan = pm_marks[-1] - pm_marks[0]
    ytop = D_marks[1]
    ax.set_xlim(pm_marks[0] - .10*xspan, pm_marks[-1] + .10*xspan)
    ax.set_ylim(-.10*ytop, 1.22*ytop)
    ax.set_xticks(pm_marks)
    ax.set_xticklabels([f"{v:.3f}" for v in pm_marks])
    ax.set_yticks([0.0, ytop])
    ax.set_yticklabels(["0.0000", f"{ytop:.4f}"])
    ax.set_xlabel(r"QME prevalence $\tilde{p}_m$")
    ax.set_ylabel(r"QME association $\tilde{D}$")
    ax.tick_params(axis="x", colors=F1_MICROBE)
    ax.tick_params(axis="y", colors=F1_ASSOC)
    ax.set_title("QME manifold parameterized by rescue-allele frequency", loc="left", pad=3, fontsize=7.8)

    fig.subplots_adjust(left=.050, right=.975, top=.915, bottom=.120)
    save(fig, "fig1_framework_qme")


QX = 0.20        # quasi-extinction threshold, as a fraction of N(0)
R_DECLINE = 0.04  # magnitude of the environmental decline
PHAT = 0.60       # selection-free endemic prevalence
TOTAL_MICROBE = 0.05  # s_m + e, held equal across all three scenarios


def scenario_params(sg, sm, e, r=R_DECLINE, phi=0.0, c=0.05, beta=1.0):
    return Params(beta=beta, ell0=beta*(1.0-PHAT), phi=phi,
                  sg=sg, sm=sm, e=e, r=r, c=c)


def rescue_scenarios():
    """Three scenarios with the same combined microbial coefficient s_m + e.

    The first two scenarios hold both s_g and s_m + e fixed, isolating
    whether the microbial contribution acts through buffering or non-additivity.
    The third holds s_m + e fixed but lowers s_g to create a microbe-required
    regime. Rescue requires s_g + (s_m+e) p_hat > r; the host allele
    alone suffices only if s_g > r.
    """
    T = TOTAL_MICROBE
    return [
        (r"Buffering only ($s_m$)",
         scenario_params(0.05, T, 0.0), MICROBE),
        (r"Acceleration only ($e$)",
         scenario_params(0.05, 0.0, T), BOTH),
        ("Both required",
         scenario_params(0.02, T/2, T/2), COMBINED),
    ]


def _threshold_metrics(t, N, N0):
    """Return bottleneck depth and the full first below-threshold excursion.

    Crossing times are linearly interpolated between sampled points. If a
    trajectory crosses down but has not crossed back up in the supplied time
    series, recross_time is None.
    """
    thr = QX * N0
    i = int(np.argmin(N))
    below = N < thr
    changes = np.flatnonzero(np.diff(below.astype(int)) != 0)
    down = None
    up = None
    for j in changes:
        n0_, n1_ = N[j], N[j+1]
        frac = 0.0 if n1_ == n0_ else (thr - n0_) / (n1_ - n0_)
        tcross = t[j] + frac * (t[j+1] - t[j])
        if (not below[j]) and below[j+1] and down is None:
            down = float(tcross)
        elif below[j] and (not below[j+1]) and down is not None:
            up = float(tcross)
            break
    duration = 0.0 if down is None else (None if up is None else up - down)
    return N[i]/N0, t[i], duration, thr, below, down, up


def _long_threshold_metrics(solver, p, N0=.8, t_end=1200.0):
    """Compute threshold duration beyond the plotting horizon."""
    t, z = solver(p, pg0=.02, N0=N0, t_end=t_end, n=max(6000, int(t_end*10)))
    return _threshold_metrics(t, z[1], N0)


def _fmt_threshold_duration(duration):
    if duration is None:
        return "never recrosses"
    return rf"$T_<={duration:.0f}$"


def figure_rescue_mechanisms():
    fig,axes=plt.subplots(2,3,figsize=(7.15,4.75),sharex="col")
    for j,(title,p,col) in enumerate(rescue_scenarios()):
        t,z=solve_qme(p,pg0=.02,N0=.8,t_end=480,n=3000)
        th,zh=solve_host_only(p,pg0=.02,N0=.8,t_end=480,n=3000)

        # Top row: abundance / rescue-U outcome.
        ax=axes[0,j]
        depth,tmin,_,thr,below,_,_=_threshold_metrics(t,z[1],.8)
        _,_,tbel,_,_,_,_=_long_threshold_metrics(solve_qme,p,N0=.8)
        _,_,tbel0,_,_,_,_=_long_threshold_metrics(solve_host_only,p,N0=.8)
        ax.plot(th,zh[1],color=NEUTRAL,lw=1.6,ls="--")
        ax.plot(t,z[1],color=col,lw=2)
        ax.axhline(thr,color=NEUTRAL,lw=.8,ls=":")
        if below.any():
            ax.fill_between(t,1e-4,z[1],where=below,color=col,alpha=.20,lw=0)
        ax.plot(tmin,z[1].min(),"v",ms=4.5,color=col,mec="none")
        # Place the bottleneck annotation at the visual center of the log-scaled panel:
        # horizontally at the midpoint of the time axis and vertically at the
        # geometric midpoint of the plotted y-range.
        xmid = 0.5 * (t[0] + t[-1])
        ymid = np.sqrt(1e-4 * 1.3)
        ax.text(xmid, ymid, rf"$N_{{\min}}/N_0={depth:.2f}$",
                fontsize=6.2, color=col, ha="center", va="center",
                bbox=dict(fc="white", ec="none", alpha=.84, pad=.8))
        ax.text(.97,.145,_fmt_threshold_duration(tbel),transform=ax.transAxes,
                ha="right",va="bottom",fontsize=6.6,color=col)
        ref = "never recrosses" if tbel0 is None else rf"$T_<={tbel0:.0f}$"
        ax.text(.97,.055,"no microbe: " + ref,transform=ax.transAxes,
                ha="right",va="bottom",fontsize=6.0,color=NEUTRAL)
        ax.set_yscale("log"); ax.set_ylim(1e-4,1.3)
        num=lambda v: f"{v:g}"
        scenario_title(ax, f"Scenario {j+1}: {title}",
                       rf"$s_g={num(p.sg)}$,  $s_m={num(p.sm)}$,  $e={num(p.e)}$")
        if j==0:
            ax.set_ylabel(r"host abundance $N$")
            ax.text(475,thr*1.2,"quasi-extinction threshold",ha="right",
                    va="bottom",fontsize=6.1,color=NEUTRAL)

        # Bottom row: rescue-allele sweep mechanism.
        ax=axes[1,j]
        if j == 0:
            ax.plot(t,z[0],color=col,lw=2,label="host + microbe")
            ax.plot(th,zh[0],color=NEUTRAL,lw=1.6,ls="--",label="microbe removed")
        else:
            ax.plot(th,zh[0],color=NEUTRAL,lw=1.6,ls="--",label="microbe removed")
            ax.plot(t,z[0],color=col,lw=2,label="host + microbe")
        ax.set_ylim(0,1.02)
        ax.set_xlabel("time after environmental change")
        if j==0:
            ax.set_ylabel(r"rescue allele $p_g$")
            ax.legend(frameon=False,loc="lower right",handlelength=1.6)
            ax.text(.52,.46,"curves coincide",transform=ax.transAxes,
                    fontsize=6.1,color=NEUTRAL,ha="center",rotation=54)

    # Label by scenario (column) so each pair reads (A,B), (C,D), (E,F).
    for ax,l in zip(axes.flat,["A","C","E","B","D","F"]):
        panel(ax,l)
    fig.tight_layout(pad=.55,w_pad=.9,h_pad=1.0)
    save(fig,"fig2_rescue_mechanisms")


def _qme_grid(sg, smmax=.06, emax=.06, ngrid=61, dt=.25,
              r=R_DECLINE, ph=PHAT, beta=1, c=.05, phi=0.0, pg0=.02, N0=.8):
    """Sweep the QME system and compute complete threshold recrossing times.

    The integration continues until p_g has effectively fixed and every
    trajectory whose long-run density-regulated equilibrium is above the
    threshold has either recrossed the threshold or remained above it. Cases
    whose long-run equilibrium lies below the threshold are marked as never
    recrossing rather than assigned a censored finite duration.
    """
    smv=np.linspace(0,smmax,ngrid); ev=np.linspace(0,emax,ngrid)
    sm,e=np.meshgrid(smv,ev,indexing="ij")
    sg=np.full_like(sm,sg)
    pg=np.full_like(sm,pg0,dtype=float); N=np.full_like(sm,N0,dtype=float)
    minN=N.copy(); thr=QX*N0
    entry=np.full_like(sm,np.nan,dtype=float)
    exit_=np.full_like(sm,np.nan,dtype=float)

    # Long-run equilibrium after p_g -> 1.  If it lies below the illustrative
    # threshold, any eventual down-crossing can never be followed by recrossing.
    pm1=pm_qme_core(1.0,phat=ph,beta=beta,sm=sm,sg=sg,e=e,phi=phi)
    gfinal=growth_core(1.0,pm1,0.0,r=r,sg=sg,sm=sm,e=e)
    Ninf=np.maximum(gfinal/c,0.0)
    never_recross=Ninf < (thr-1e-10)

    def rhs(pg,N):
        pm=pm_qme_core(pg,phat=ph,beta=beta,sm=sm,sg=sg,e=e,phi=phi)
        D=D_qme_core(pg,phat=ph,beta=beta,sg=sg,e=e,phi=phi)
        dpg=pg_dot_core(pg,pm,D,sg=sg,sm=sm,e=e)
        growth=growth_core(pg,pm,D,r=r,sg=sg,sm=sm,e=e)
        return dpg,N*(growth-c*N)

    prevN=N.copy(); prev_below=N<thr
    max_steps=int(5000/dt)
    for k in range(max_steps):
        t=k*dt
        k1p,k1n=rhs(pg,N)
        k2p,k2n=rhs(pg+.5*dt*k1p,N+.5*dt*k1n)
        k3p,k3n=rhs(pg+.5*dt*k2p,N+.5*dt*k2n)
        k4p,k4n=rhs(pg+dt*k3p,N+dt*k3n)
        newpg=np.clip(pg+dt*(k1p+2*k2p+2*k3p+k4p)/6,0,1)
        newN=np.maximum(N+dt*(k1n+2*k2n+2*k3n+k4n)/6,1e-300)
        minN=np.minimum(minN,newN)
        below=newN<thr

        enter=(~prev_below)&below&np.isnan(entry)
        leave=prev_below&(~below)&np.isnan(exit_)
        # Linear interpolation gives smoother recrossing times than the grid dt.
        den=newN-prevN
        frac=np.divide(thr-prevN,den,out=np.zeros_like(den),where=np.abs(den)>1e-300)
        tcross=t+dt*np.clip(frac,0,1)
        entry[enter]=tcross[enter]
        exit_[leave]=tcross[leave]

        pg,N=newpg,newN
        prevN=newN.copy(); prev_below=below

        finite_needed=(~never_recross)&(~np.isnan(entry))
        all_finite_done=np.all((~finite_needed)|(~np.isnan(exit_)))
        # Once the slow sweep has effectively fixed, any trajectory still above
        # threshold with N_inf > threshold cannot develop a later down-crossing.
        if np.all(pg>0.999999) and all_finite_done:
            break
    else:
        raise RuntimeError("QME grid did not resolve all finite threshold recrossings")

    tbelow=np.where(np.isnan(entry),0.0,exit_-entry)
    unresolved=(~never_recross)&(~np.isnan(entry))&np.isnan(exit_)
    if np.any(unresolved):
        raise RuntimeError("Finite threshold recrossing remained unresolved")
    # Points whose asymptotic equilibrium is below threshold are explicitly
    # classified rather than given a censored duration.
    tbelow=np.where(never_recross,np.nan,tbelow)
    return smv,ev,minN/N0,tbelow,never_recross


def figure_rescue_landscape():
    """Fig. 3: rescue-landscape maps over the two microbial coefficients.

    A  bottleneck depth over (e, s_m).
    B  time spent below the quasi-extinction threshold over (e, s_m).

    The explanatory decomposition and matched-trajectories panels formerly in
    this figure are omitted because those points are already shown in Fig. 2.
    """
    SG_MAP = 0.02      # rescue allele insufficient on its own: s_g < r
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))

    smv, ev, minrel, tbel, never_recross = _qme_grid(SG_MAP)
    depth = np.ma.masked_where(never_recross, np.log10(minrel))
    tb = np.ma.masked_invalid(tbel)
    depth_cmap = plt.get_cmap("viridis").copy(); depth_cmap.set_bad(NOFLOW)
    time_cmap = plt.get_cmap("cividis_r").copy(); time_cmap.set_bad(NOFLOW)
    bnd = (R_DECLINE-SG_MAP)/PHAT   # rescue boundary: s_m + e

    for col, (ax, field, cmap, lab, title) in enumerate(zip(
            axes,
            [depth, tb],
            [depth_cmap, time_cmap],
            [r"$\log_{10}(N_{\min}/N_0)$", r"time below threshold, $T_{<}$"],
            ["rescue-U depth", "time below quasi-extinction threshold"])):
        panel(ax, "AB"[col])
        im = ax.pcolormesh(ev, smv, field, shading="auto", cmap=cmap)
        ax.contourf(ev, smv, never_recross.astype(float), levels=[.5, 1.5],
                    colors="none", hatches=["////"])

        xx = np.linspace(0, .06, 300)

        # Dashed: leading-order deterministic rescue boundary.
        yy = bnd - xx
        m = (yy >= 0) & (yy <= .06)
        ax.plot(xx[m], yy[m], "--", color="#111", lw=1.1)
        ax.text(.0065, bnd-.0065+.0024, "rescue boundary", fontsize=5.8,
                rotation=-42, color="#111", ha="left", va="center",
                bbox=dict(fc="white", ec="none", alpha=.72, pad=.6))

        # Dotted: the combined microbial coefficient used for Scenario 3.
        ref = TOTAL_MICROBE - xx
        mr = (ref >= 0) & (ref <= .06)
        ax.plot(xx[mr], ref[mr], ":", color="#111", lw=.9, alpha=.80)
        ax.text(.033, TOTAL_MICROBE-.033+.0015,
                rf"$s_m+e={TOTAL_MICROBE:.2f}$", fontsize=5.8,
                rotation=-42, color="#111", alpha=.90, ha="left", va="center",
                bbox=dict(fc="white", ec="none", alpha=.68, pad=.5))

        ax.text(.004, .0032, "never recrosses", color="#111", fontsize=6.3,
                bbox=dict(fc="white", ec="none", alpha=.85, pad=1.4))
        ax.set_xlabel(r"gene–microbe interaction $e$")
        if col == 0:
            ax.set_ylabel(r"microbial selection $s_m$")
        scenario_title(ax, title, pad=4)

        # Scenario 3 from Fig. 2: s_m=e=0.025.
        ax.plot(TOTAL_MICROBE/2, TOTAL_MICROBE/2, "o", ms=5, mfc="white",
                mec="#111", mew=.9, zorder=5)
        ax.annotate("Scenario 3", (TOTAL_MICROBE/2, TOTAL_MICROBE/2),
                    xytext=(5, 4), textcoords="offset points", fontsize=6.2,
                    fontweight="bold", zorder=5)

        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
        cb.set_label(lab, labelpad=1)
        ax.text(.975, .06, rf"$s_g={SG_MAP:.2f}$, $r={R_DECLINE:.2f}$",
                transform=ax.transAxes, ha="right", fontsize=6.2,
                bbox=dict(fc="white", ec="none", alpha=.85, pad=1.4))

    fig.subplots_adjust(left=.095, right=.965, top=.88, bottom=.18, wspace=.34)
    save(fig, "fig3_rescue_landscape")


def sad_base():
    p=Params(beta=1,ell0=.4,sg=.05,sm=.04,e=.05,r=.100,c=.05,phi=0)
    return p,phi_critical(p)


def figure_sad():
    base,pc=sad_base()
    fig=plt.figure(figsize=(7.15,5.25))
    gs=gridspec.GridSpec(2,2,figure=fig,width_ratios=[1,1],height_ratios=[1,1],
                         wspace=.42,hspace=.48)

    mults=[.5,1,3]
    cols=[MICROBE,NEUTRAL,DILUTION]
    styles=["-","--","-."]
    N0=.7
    t_end=600
    n=1800

    traj=[]
    for m in mults:
        p=Params(**{**base.__dict__,"phi":m*pc})
        t,z=solve_qme(p,pg0=.02,N0=N0,t_end=t_end,n=n)
        traj.append((m,p,t,z))

    # A. rescue allele trajectories
    ax=fig.add_subplot(gs[0,0]); panel(ax,"A")
    for (m,p,t,z),col,ls in zip(traj,cols,styles):
        ax.plot(t,z[0],color=col,lw=2.0,ls=ls,label=rf"$\Lambda={m:g}$")
    ax.set_xlim(0,t_end); ax.set_ylim(0,1.02)
    ax.set_xlabel("time after environmental change")
    ax.set_ylabel(r"rescue allele $p_g$")
    scenario_title(ax,"Rescue allele sweeps despite dilution",pad=4)
    ax.legend(frameon=False,loc="lower right",handlelength=2.0)

    # B. microbial prevalence trajectories implied by the QME.
    ax=fig.add_subplot(gs[0,1]); panel(ax,"B")
    for (m,p,t,z),col,ls in zip(traj,cols,styles):
        pm_t = pm_qme(z[0],p)
        ax.plot(t,pm_t,color=col,lw=2.0,ls=ls,label=rf"$\Lambda={m:g}$")
    ax.axhline(base.phat_m,color=F1_GREY,lw=.8,ls=":")
    ax.text(.015,.665, r"$\hat p_m$", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5, color=F1_GREY)
    ax.set_xlim(0,t_end)
    ax.set_xlabel("time after environmental change")
    ax.set_ylabel(r"QME prevalence $\tilde{p}_m$")
    scenario_title(ax,r"Direction of prevalence change switches at $\Lambda=1$",pad=4)
    ax.legend(frameon=False,loc="center right",handlelength=2.0)

    # C. post-sweep low-density growth over (Lambda, Q).
    #
    # The colour scale is linear and symmetric about zero, so equal colour
    # steps are equal growth steps everywhere on the bar. That requires the
    # data to straddle zero, which is why Q runs below 1.
    #
    # Q>1 is the LEADING-ORDER criterion, evaluated at the baseline prevalence
    # p_hat. The field plotted here is the full QME post-sweep growth, in which
    # the microbe's own advantage lifts prevalence above p_hat. Rescue is
    # therefore still possible slightly below Q=1 when dilution is weak: at
    # Lambda=0 the boundary sits at Q=0.970, and it crosses Q=1 at Lambda=1.29.
    # Do not draw a Q=1 line here; it is not a boundary of this field.
    ax=fig.add_subplot(gs[1,0]); panel(ax,"C")
    lam=np.linspace(0,3.6,181)
    Q=np.linspace(0.90,1.30,201)
    L,Qm=np.meshgrid(lam,Q)
    ph=base.phat_m
    S0=base.sg+(base.sm+base.e)*ph
    rgrid=S0/Qm
    phigrid=L*pc
    pm1=pm_qme_core(1.0,phat=ph,beta=base.beta,sm=base.sm,sg=base.sg,
                    e=base.e,phi=phigrid)
    val=growth_core(1.0,pm1,0.0,r=rgrid,sg=base.sg,sm=base.sm,e=base.e)/rgrid
    M=float(np.abs(val).max())
    im=ax.pcolormesh(lam,Q,val,shading="auto",cmap="coolwarm",
                     norm=Normalize(vmin=-M,vmax=M))
    ax.axvline(1,color="#111",lw=1,ls="--")
    ax.text(0.95,1.21,r"$\Lambda=1$",fontsize=6.2,rotation=90,
            ha="right",va="center",color="#111")
    ax.contour(lam,Q,val,levels=[0],colors="#111",linewidths=1.4)
    ax.annotate("demographic rescue boundary",xy=(2.80,1.0381),
                xytext=(2.55,1.125),ha="center",fontsize=6.0,color="#111",
                arrowprops=dict(arrowstyle="-",lw=.7,color="#111",
                                shrinkA=0,shrinkB=2))
    ax.set_xlabel(r"relative dilution $\Lambda=\phi/\phi_c$")
    ax.set_ylabel(r"leading-order rescue margin $Q$")
    scenario_title(ax,"Dilution can eliminate positive post-sweep growth",pad=4)
    ax.text(2.30,1.272,r"$\tilde D<0$; prevalence falls",fontsize=6.2,
            ha="center",va="top",color="#111",
            bbox=dict(fc="white",ec="none",alpha=.8,pad=1.4))
    ax.text(0.12,1.045,"rescue",fontsize=6.4,ha="left",va="bottom",color="#111")
    ax.text(0.12,0.932,"no rescue",fontsize=6.4,ha="left",va="bottom",color="#111")
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
    cb.set_label(r"post-sweep low-density growth $/r$",labelpad=2,fontsize=6.0)
    cb.ax.tick_params(labelsize=6.0)
    cb.locator=MultipleLocator(0.1); cb.update_ticks()

    # D. abundance trajectories.
    ax=fig.add_subplot(gs[1,1]); panel(ax,"D")
    for (m,p,t,z),col,ls in zip(traj,cols,styles):
        ax.plot(t,z[1],color=col,lw=2.0,ls=ls,label=rf"$\Lambda={m:g}$")
    ax.axhline(.01,color=NEUTRAL,lw=.8,ls=":")
    ax.text(t_end*.985,.0118,"quasi-extinction threshold",
            ha="right",va="bottom",fontsize=6.0,color=NEUTRAL)
    ax.set_yscale("log"); ax.set_ylim(2e-3,1); ax.set_xlim(0,t_end)
    ax.set_xlabel("time after environmental change")
    ax.set_ylabel(r"host abundance $N$")
    scenario_title(ax,"Dilution can erase rescue",pad=4)
    ax.legend(frameon=False,loc="upper right",handlelength=2.0)

    fig.subplots_adjust(left=.095,right=.965,top=.915,bottom=.085)
    save(fig,"fig4_selection_amplified_dilution")


def figure_validation_rescue():
    fig,axes=plt.subplots(2,3,figsize=(7.15,4.35),sharex="col")
    for j,(title,p,col) in enumerate(rescue_scenarios()):
        tq,zq=solve_qme(p,pg0=.02,N0=.8,t_end=350,n=1600); tf,zf=solve_full(p,pg0=.02,N0=.8,t_end=350,n=1600)
        axes[0,j].plot(tf,zf[1],color=NEUTRAL,lw=2.3,alpha=.55,label="full"); axes[0,j].plot(tq,zq[0],color=col,lw=1.4,ls="--",label="QME")
        axes[0,j].set_title(title,loc="left",pad=4); axes[0,j].set_ylim(0,1.02)
        if j==0: axes[0,j].set_ylabel(r"$p_g$"); axes[0,j].legend(frameon=False)
        axes[1,j].plot(tf,zf[3],color=NEUTRAL,lw=2.3,alpha=.55); axes[1,j].plot(tq,zq[1],color=col,lw=1.4,ls="--")
        axes[1,j].set_yscale("log"); axes[1,j].set_ylim(1e-4,1.3); axes[1,j].set_xlabel("time")
        if j==0: axes[1,j].set_ylabel(r"$N$")
    for ax,l in zip(axes.flat,"ABCDEF"): panel(ax,l)
    fig.tight_layout(pad=.6,w_pad=.9); save(fig,"figS1_rescue_validation")


def figure_validation_sad():
    base,pc=sad_base(); mults=[.5,1,3]; cols=[MICROBE,NEUTRAL,DILUTION]
    fig,axes=plt.subplots(2,2,figsize=(7.15,4.8),sharex=True)
    for m,col in zip(mults,cols):
        p=Params(**{**base.__dict__,"phi":m*pc}); tq,zq=solve_qme(p,pg0=.02,N0=.7,t_end=600,n=1800); tf,zf=solve_full(p,pg0=.02,N0=.7,t_end=600,n=1800)
        axes[0,0].plot(tf,zf[0],color=col,lw=1.8,label=rf"$\Lambda={m:g}$ full"); axes[0,0].plot(tq,pm_qme(zq[0],p),color=col,lw=1.1,ls="--")
        axes[0,1].plot(tf,zf[2],color=col,lw=1.8); axes[0,1].plot(tq,D_qme(zq[0],p),color=col,lw=1.1,ls="--")
        axes[1,0].plot(tf,zf[1],color=col,lw=1.8); axes[1,0].plot(tq,zq[0],color=col,lw=1.1,ls="--")
        axes[1,1].plot(tf,zf[3],color=col,lw=1.8); axes[1,1].plot(tq,zq[1],color=col,lw=1.1,ls="--")
    axes[0,0].set_ylabel(r"$p_m$"); axes[0,1].set_ylabel(r"$D$"); axes[1,0].set_ylabel(r"$p_g$"); axes[1,1].set_ylabel(r"$N$")
    axes[1,0].set_xlabel("time"); axes[1,1].set_xlabel("time"); axes[1,1].set_yscale("log"); axes[1,1].set_ylim(2e-3,1)
    axes[0,0].legend(frameon=False,fontsize=5.9)
    for ax,l in zip(axes.flat,"ABCD"): panel(ax,l)
    fig.tight_layout(pad=.6); save(fig,"figS2_sad_validation")


# ---------------------------------------------------------------------------
# SI Fig. S3: broader qualitative robustness of QME sign predictions.
# Ported from the original validation workflow to the current manuscript model
# and notation. No second model implementation is used.
# ---------------------------------------------------------------------------

ROBUST_METRIC_NAMES = [
    "prevalence",
    "association",
    "microbial effect on adaptation",
    "microbial contribution to growth",
]


def _robust_subsystem_fun(z, pg, p):
    dz = frequency_rhs(0.0, np.array([z[0], pg, z[1]]), p)
    return np.array([dz[0], dz[2]])


def _stable_conditional_equilibrium(pg, p):
    """Stable physical equilibrium of (p_m,D) conditional on fixed p_g."""
    seeds = [
        (float(np.clip(pm_qme(pg, p), 1e-5, 1 - 1e-5)), float(D_qme(pg, p))),
        (p.phat_m, 0.0),
        (.15, 0.0),
        (.5, 0.0),
        (.85, 0.0),
    ]
    roots = []
    for seed in seeds:
        sol = root(lambda z: _robust_subsystem_fun(z, pg, p), seed, method="hybr")
        if not sol.success:
            continue
        pm, D = map(float, sol.x)
        if not is_physical_state(pm, pg, D, tol=2e-7):
            continue
        if np.linalg.norm(_robust_subsystem_fun([pm, D], pg, p), ord=np.inf) > 2e-7:
            continue

        # Finite-difference Jacobian for the conditional fast subsystem.
        h = 1e-6
        J = np.empty((2, 2))
        for j in range(2):
            zp = np.array([pm, D]); zm = zp.copy()
            zp[j] += h; zm[j] -= h
            J[:, j] = (
                _robust_subsystem_fun(zp, pg, p)
                - _robust_subsystem_fun(zm, pg, p)
            ) / (2 * h)
        if np.max(np.real(np.linalg.eigvals(J))) < 1e-6:
            if not any(np.linalg.norm(np.array([pm, D]) - np.array(r)) < 1e-5
                       for r in roots):
                roots.append((pm, D))

    if not roots:
        return None
    q = np.array([float(pm_qme(pg, p)), float(D_qme(pg, p))])
    return min(roots, key=lambda z: np.linalg.norm(np.array(z) - q))


def _robust_natural_scales(pg, p, smax):
    return np.array([
        max(smax / p.beta, 1e-12),
        max(smax * p.phat_m * pg * (1 - pg) / p.beta, 1e-12),
        max(smax, 1e-12),
        max(smax, 1e-12),
    ])


def _robust_qme_metrics(pg, p):
    return np.array([
        float(pm_qme(pg, p) - p.phat_m),
        float(D_qme(pg, p)),
        float(p.e * p.phat_m),
        float(p.phat_m * (p.sm + p.e * pg)),
    ])


def _robust_full_metrics(pm, pg, D, p):
    return np.array([
        pm - p.phat_m,
        D,
        float(microbial_adaptation_effect(pm, pg, D, p)),
        float(microbial_growth_contribution(pm, pg, D, p)),
    ])


def run_sobol_robustness(n=1024, seed=20260728):
    """Run the retained 2^10-point Sobol robustness analysis."""
    m = int(round(np.log2(n)))
    if 2**m != n:
        raise ValueError("n must be a power of two")
    U = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(m=m)

    traj_ok = [[] for _ in range(4)]
    traj_smax = []
    point_dist = [[] for _ in range(4)]
    point_match = [[] for _ in range(4)]
    total_ok = []
    plausible = 0

    for u in U:
        smax = 10**(np.log10(.005) + (np.log10(.5) - np.log10(.005)) * u[0])
        raw = 2 * u[1:4] - 1
        raw = raw / max(np.max(np.abs(raw)), 1e-8)
        sm, sg, e = smax * raw
        phat = .05 + .90 * u[4]
        phi = .80 * u[5]
        p = Params(beta=1.0, ell0=1.0-phat, phi=phi,
                   sm=sm, sg=sg, e=e, r=.04, c=.05)

        eq = _stable_conditional_equilibrium(.1, p)
        if eq is None:
            continue
        pm0, D0 = eq
        leading = p.sg + p.e * p.phat_m
        target = .9 if leading >= 0 else .01
        T = 8.0 / max(abs(leading), .20 * smax, 1e-3)
        try:
            t, pm, pg, D = solve_frequency_trajectory(
                p, pg0=.1, pm0=pm0, D0=D0,
                t_end=T, pg_target=target, n=150,
            )
        except Exception:
            continue

        if (np.any(~np.isfinite(pm)) or
            any(not is_physical_state(float(a), float(b), float(c), tol=5e-5)
                for a, b, c in zip(pm, pg, D))):
            continue

        plausible += 1
        traj_smax.append(smax)
        for k in range(4):
            matches = []
            for pm_i, pg_i, D_i in zip(pm, pg, D):
                q = _robust_qme_metrics(float(pg_i), p)
                f = _robust_full_metrics(float(pm_i), float(pg_i), float(D_i), p)
                scale = _robust_natural_scales(float(pg_i), p, smax)
                dist = abs(q[k]) / scale[k]
                point_dist[k].append(dist)
                point_match[k].append(int(np.sign(q[k]) == np.sign(f[k])))
                if dist > .05:
                    matches.append(np.sign(q[k]) == np.sign(f[k]))
            traj_ok[k].append(bool(matches) and all(matches) if matches else np.nan)

        # Direction of total host-allele selection relative to leading QME sign.
        full_s = []
        for pm_i, pg_i, D_i in zip(pm, pg, D):
            dpg = frequency_rhs(0.0, np.array([pm_i, pg_i, D_i]), p)[1]
            full_s.append(dpg / (pg_i * (1 - pg_i)))
        if abs(leading) > .05 * smax:
            total_ok.append(bool(np.all(np.sign(full_s) == np.sign(leading))))
        else:
            total_ok.append(np.nan)

    return (
        plausible,
        np.asarray(traj_smax),
        [np.asarray(x, float) for x in traj_ok],
        [np.asarray(x, float) for x in point_dist],
        [np.asarray(x, int) for x in point_match],
        np.asarray(total_ok, float),
    )


def _conditional_metric_grid(kind, xaxis, yaxis, pg, pbuilder):
    Q = np.full((len(yaxis), len(xaxis)), np.nan)
    F = np.full_like(Q, np.nan)
    physical = np.zeros_like(Q, dtype=bool)
    for i, yv in enumerate(yaxis):
        for j, xv in enumerate(xaxis):
            p = pbuilder(float(xv), float(yv))
            eq = _stable_conditional_equilibrium(pg, p)
            if eq is None:
                continue
            pm, D = eq
            physical[i, j] = True
            Q[i, j] = _robust_qme_metrics(pg, p)[kind]
            F[i, j] = _robust_full_metrics(pm, pg, D, p)[kind]
    return Q, F, physical


def _robust_classification(Q, F, physical, scale=.5):
    # 0 agreement, 1 QME+ full-, 2 QME- full+, 3 boundary, 4 no equilibrium.
    C = np.full(Q.shape, 4, dtype=int)
    m = physical
    C[m] = 0
    boundary = m & (np.abs(Q) <= .05 * scale)
    C[boundary] = 3
    use = m & ~boundary
    C[use & (Q > 0) & (F < 0)] = 1
    C[use & (Q < 0) & (F > 0)] = 2
    return C


def figure_qme_qualitative_robustness(n_sobol=1024, grid=45):
    """SI Fig. S3: robustness of first-order QME qualitative predictions."""
    plausible, svals, traj_ok, pdist, pmatch, total_ok = run_sobol_robustness(n_sobol)

    bins = np.logspace(np.log10(.005), np.log10(.5), 7)
    centers = np.sqrt(bins[:-1] * bins[1:])
    frac = np.full((4, len(centers)), np.nan)
    for k in range(4):
        arr = traj_ok[k]
        for b in range(len(centers)):
            m = ((svals >= bins[b]) &
                 ((svals <= bins[b+1]) if b == len(centers)-1 else (svals < bins[b+1])))
            vals = arr[m]
            vals = vals[np.isfinite(vals)]
            frac[k, b] = 100 * np.mean(vals) if len(vals) else np.nan

    dbins = np.logspace(-2, .15, 7)
    dcent = np.sqrt(dbins[:-1] * dbins[1:])
    pfrac = np.full((4, len(dcent)), np.nan)
    for k in range(4):
        d = pdist[k]; mm = pmatch[k]
        for b in range(len(dcent)):
            hi = dbins[b+1] if b < len(dcent)-1 else np.inf
            sel = (d >= dbins[b]) & (d < hi)
            pfrac[k, b] = 100 * np.mean(mm[sel]) if np.any(sel) else np.nan

    # Strong-selection slices retained from the original robustness analysis,
    # now evaluated through qme_rescue_model.py.
    g = grid
    xc = np.linspace(-.5, .5, g); yc = np.linspace(0, .8, g)
    Qc, Fc, Pc = _conditional_metric_grid(
        0, xc, yc, .5,
        lambda sm, phi: Params(beta=1, ell0=.3, phi=phi,
                               sm=sm, sg=.15, e=.25, r=.04, c=.05))

    xd = np.linspace(-.5, .5, g); yd = np.linspace(-.5, .5, g)
    Qd, Fd, Pd = _conditional_metric_grid(
        1, xd, yd, .5,
        lambda e, sm: Params(beta=1, ell0=.3, phi=.4,
                             sm=sm, sg=.15, e=e, r=.04, c=.05))

    xe = np.linspace(-.5, .5, g); ye = np.linspace(-.5, .5, g)
    builder = lambda e, sm: Params(beta=1, ell0=.6, phi=.7,
                                   sm=sm, sg=.3, e=e, r=.04, c=.05)
    Qe, Fe, Pe = _conditional_metric_grid(2, xe, ye, .5, builder)
    Qf, Ff, Pf = _conditional_metric_grid(3, xe, ye, .5, builder)

    Cs = [
        _robust_classification(Qc, Fc, Pc, .5),
        _robust_classification(Qd, Fd, Pd, .5*.7*.5*.5),
        _robust_classification(Qe, Fe, Pe, .5),
        _robust_classification(Qf, Ff, Pf, .5),
    ]
    grids = [(xc, yc, Qc, Fc), (xd, yd, Qd, Fd),
             (xe, ye, Qe, Fe), (xe, ye, Qf, Ff)]

    # Keep the original SI layout so only the model implementation changes.
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 10.5,
        "mathtext.fontset": "stix",
    })
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.4))
    fig.subplots_adjust(left=.07, right=.985, top=.95, bottom=.19,
                        hspace=.34, wspace=.22)

    labels = ["prevalence", "host--microbe association",
              "microbial effect on adaptation", "microbial contribution to growth"]
    for k in range(4):
        axs[0, 0].plot(centers, frac[k], marker="o", lw=1.6, label=labels[k])
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_ylim(92, 100.4)
    axs[0, 0].set(
        xlabel=r"largest selective effect, $\max(|s_m|,|s_g|,|e|)/\beta$",
        ylabel="trajectories retaining QME sign (%)",
    )
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower left")

    for k in range(4):
        axs[0, 1].plot(dcent, pfrac[k], marker="o", lw=1.6)
    axs[0, 1].set_xscale("log")
    axs[0, 1].set_ylim(84, 100.4)
    axs[0, 1].set(
        xlabel="distance from QME zero boundary\n(relative to first-order scale)",
        ylabel="pointwise sign agreement (%)",
    )

    plot_axes = [axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]
    titles = ["Microbial prevalence", "Host-microbe association",
              "Microbial effect on host adaptation", "Microbial contribution to growth"]
    cmap = ListedColormap(["#d9d9d9", "#d95f5f", "#5d82c1", "#ffffff", "#8e8e8e"])
    for ax, C, (xx, yy, Qv, Fv), title in zip(plot_axes, Cs, grids, titles):
        ax.pcolormesh(xx, yy, C, cmap=cmap, vmin=-.5, vmax=4.5, shading="auto")
        X, Y = np.meshgrid(xx, yy)
        try:
            ax.contour(X, Y, Fv, levels=[0], colors="#1f77b4", linewidths=1.5)
        except Exception:
            pass
        try:
            ax.contour(X, Y, Qv, levels=[0], colors="#6a1b9a",
                       linewidths=1.5, linestyles="--")
        except Exception:
            pass
        ax.set_title(title)

    plot_axes[0].set(xlabel=r"microbial selection coefficient, $s_m$",
                     ylabel=r"$f(1-v)$")
    for ax in plot_axes[1:]:
        ax.set(xlabel=r"non-additive interaction coefficient, $e$",
               ylabel=r"microbial selection coefficient, $s_m$")

    for lab, ax in zip("abcdef", axs.flat):
        ax.text(.025, .965, lab, transform=ax.transAxes,
                ha="left", va="top", fontsize=17, fontweight="bold")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(fc="#d9d9d9", label="same qualitative prediction"),
        Patch(fc="#d95f5f", label="QME positive; full model negative"),
        Patch(fc="#5d82c1", label="QME negative; full model positive"),
        Patch(fc="#ffffff", ec="#777", label="near QME zero boundary"),
        Patch(fc="#8e8e8e", label="no stable physical equilibrium"),
        Line2D([], [], color="#1f77b4", label="full nonlinear zero boundary"),
        Line2D([], [], color="#6a1b9a", ls="--", label="QME zero boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.53, .035), fontsize=10.5)

    save(fig, "figS3_qme_qualitative_robustness")

    overall = []
    for arr in traj_ok:
        v = arr[np.isfinite(arr)]
        overall.append(100 * np.mean(v))
    tv = total_ok[np.isfinite(total_ok)]
    strongest = []
    lo = bins[-2]
    m = svals >= lo
    for arr in traj_ok:
        v = arr[m]
        v = v[np.isfinite(v)]
        strongest.append(100 * np.mean(v) if len(v) else np.nan)

    print("Sobol robustness: plausible trajectories:", plausible, "of", n_sobol)
    print("Sobol robustness: trajectory sign retention (%):",
          dict(zip(ROBUST_METRIC_NAMES, overall)))
    print("Sobol robustness: total rescue-allele direction retention (%):",
          100 * np.mean(tv) if len(tv) else np.nan)
    print("Sobol robustness: strongest-bin retention (%):", strongest,
          "bin lower edge", lo)



if __name__ == "__main__":
    # Complete retained manuscript figure set: main-text Figs. 1--4 and
    # SI Figs. S1--S3. Every calculation uses qme_rescue_model.py.
    figure_framework()
    figure_rescue_mechanisms()
    figure_rescue_landscape()
    figure_sad()
    figure_validation_rescue()
    figure_validation_sad()
    figure_qme_qualitative_robustness()
    print("Wrote main-text Figs. 1--4 and SI Figs. S1--S3 to", FIG)
