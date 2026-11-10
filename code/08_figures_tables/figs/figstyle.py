"""
Shared figure style for the submission (Elsevier / ScienceDirect).

* SciencePlots 'nature' look, no LaTeX
* fonts embedded as TrueType (pdf.fonttype 42) -- Elsevier's checks dislike Type 3
* 7 pt minimum text at final print size
* ticks only on the bottom / left axes, no minor ticks (cleaner multi-panel time series)

Usage in every figure script:
    from figstyle import apply_style, FS
    apply_style()
"""
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' / 'nature' styles)

FS = 7          # base font size (pt) for all figure text
FS_SMALL = 7    # tick labels / annotations -- kept at the 7 pt floor


def apply_style() -> None:
    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType, not Type 3
        "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
        "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": FS,
        "xtick.top": False, "ytick.right": False,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
    })
