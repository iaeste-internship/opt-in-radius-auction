"""Schematic for Proposition 1 (passenger incentive compatibility).

One passenger i, two drivers. d1 is inside her true pickup limit r, d2 is
outside it. Three panels show which hyperedges exist under a truthful report,
an over-report and an under-report of the pickup radius, and why neither
misreport can raise her true utility.

Output: figures/prop1_misreport_schematic.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from figure_style import apply_ieee_style

OUT = Path("figures") / "prop1_misreport_schematic.png"

TRUE_R = 1.0
D1 = (0.62, 0.42)    # inside the true limit
D2 = (1.48, -0.30)   # outside the true limit
PASSENGER = (0.0, 0.0)

ORA_BLUE = "#1f5f99"
BAD_RED = "#b23a3a"
GREY = "#7a7a7a"

PANELS = [
    {
        "title": "(a) Truthful report  $\\hat r_i = r_i$",
        "reported": TRUE_R,
        "edges": [("d1", "ok")],
        "note": "Edges: $\\{d_1, i\\}$\nMechanism picks the best\nacceptable outcome",
        "utility": "$u_i = u_i^{*}$",
    },
    {
        "title": "(b) Over-report  $\\hat r_i > r_i$",
        "reported": 1.75,
        "edges": [("d1", "ok"), ("d2", "bad")],
        "note": "Adds $\\{d_2, i\\}$, beyond the true limit\nIf selected: unacceptable\nIf not: outcome was already\navailable under (a)",
        "utility": "$u_i = -\\infty$  or  $u_i \\leq u_i^{*}$",
    },
    {
        "title": "(c) Under-report  $\\hat r_i < r_i$",
        "reported": 0.55,
        "edges": [],
        "note": "Removes $\\{d_1, i\\}$, an acceptable edge\nOnly acceptable outcomes\nare lost",
        "utility": "$u_i \\leq u_i^{*}$",
    },
]


def draw_panel(ax, spec):
    ax.set_aspect("equal")
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.95, 1.95)
    ax.axis("off")

    ax.add_patch(Circle(PASSENGER, TRUE_R, fill=False, lw=1.3,
                        color=ORA_BLUE))
    if abs(spec["reported"] - TRUE_R) > 1e-9:
        ax.add_patch(Circle(PASSENGER, spec["reported"], fill=False, lw=1.2,
                            ls=(0, (4, 3)), color=GREY))

    for name, status in spec["edges"]:
        xy = D1 if name == "d1" else D2
        color = ORA_BLUE if status == "ok" else BAD_RED
        ax.plot([PASSENGER[0], xy[0]], [PASSENGER[1], xy[1]], color=color,
                lw=2.0, solid_capstyle="round", zorder=2)

    ax.plot(*PASSENGER, "o", ms=7, color="black", zorder=3)
    ax.annotate("$i$", PASSENGER, xytext=(-11, -11),
                textcoords="offset points")
    for label, xy in (("$d_1$", D1), ("$d_2$", D2)):
        ax.plot(*xy, "s", ms=6.5, color="white", mec="black", mew=1.1,
                zorder=3)
        ax.annotate(label, xy, xytext=(6, 5), textcoords="offset points")

    ax.set_title(spec["title"], fontsize=9, pad=4)
    ax.text(0.0, -2.05, spec["utility"], ha="center", va="top", fontsize=8.5)
    ax.text(0.0, -2.55, spec["note"], ha="center", va="top", fontsize=7.5,
            linespacing=1.25)


def main():
    apply_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.7))
    for ax, spec in zip(axes, PANELS):
        draw_panel(ax, spec)

    handles = [
        plt.Line2D([], [], color=ORA_BLUE, lw=1.3,
                   label="True pickup limit $r_i$"),
        plt.Line2D([], [], color=GREY, lw=1.2, ls=(0, (4, 3)),
                   label="Reported limit $\\hat r_i$"),
        plt.Line2D([], [], color=ORA_BLUE, lw=2.0,
                   label="Hyperedge within $r_i$"),
        plt.Line2D([], [], color=BAD_RED, lw=2.0,
                   label="Hyperedge beyond $r_i$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.42,
                        wspace=0.05)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
