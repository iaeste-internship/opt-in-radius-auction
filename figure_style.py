"""Shared matplotlib styling for IEEE Access two-column display.

Figures in the manuscript are placed at widths between 3.3 in and 6.3 in.
Matplotlib defaults (10 pt labels, 8 pt annotations) on an 8 in canvas are
scaled down by Word to 40-80 percent, so on the printed page the axis text
falls to 4-8 pt. This module fixes two things at once:

1. figures are drawn at their final printed width (no downscaling in Word);
2. font sizes are set so that the printed size is what is set here.

FIGSIZE holds the target (width, height) in inches for each paper figure.
The width equals the width the image is inserted at in the manuscript.
"""

from __future__ import annotations

import matplotlib as mpl

# Printed point sizes. IEEE asks for legible figure text; 8 pt is the floor.
BASE_FONT_PT = 9
AXIS_LABEL_PT = 10
TICK_LABEL_PT = 9
LEGEND_PT = 8
ANNOTATION_PT = 8
TITLE_PT = 10

# Target figure sizes in inches, keyed by output file stem.
FIGSIZE = {
    "radius_welfare_match_rate": (4.25, 4.80),
    "radius_violation_rate": (6.30, 3.90),
    "real_radius_welfare_match_rate": (3.50, 3.95),
    "real_radius_violation_rate": (5.80, 3.60),
    "scalability_runtime_breakdown": (6.20, 3.65),
    "welfare_vs_radius_violations": (5.50, 4.10),
    "passenger_utility_vs_violations": (5.50, 4.10),
    "real_welfare_vs_radius_violations": (5.50, 4.10),
    "real_passenger_utility_vs_violations": (5.50, 4.10),
}


def apply_ieee_style() -> None:
    """Set global rcParams so figure text is legible at printed size."""
    mpl.rcParams.update(
        {
            "font.size": BASE_FONT_PT,
            "axes.labelsize": AXIS_LABEL_PT,
            "axes.titlesize": TITLE_PT,
            "xtick.labelsize": TICK_LABEL_PT,
            "ytick.labelsize": TICK_LABEL_PT,
            "legend.fontsize": LEGEND_PT,
            "legend.title_fontsize": LEGEND_PT,
            "figure.titlesize": TITLE_PT,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def figsize_for(stem: str, fallback: tuple[float, float]) -> tuple[float, float]:
    """Return the target size for an output file stem, or a fallback."""
    return FIGSIZE.get(stem, fallback)
