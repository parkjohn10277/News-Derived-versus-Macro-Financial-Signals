"""
Rebuild every results-dependent figure and table of the manuscript from the GDP-corrected run, and copy them
into final/ under the file names main.tex / Supplementary.tex use.

    python build_all.py            # after XAI/run_all.py has finished for E7

Steps (each script is a copy of the original, repointed to ../GDP_corrected; see README.md)
  1. tables/make_tbl_*.py                     tbl_performance, tbl_block_tests_{auc,dm}, tbl_within_model_{auc,dm}
  2. figs/make_fig_aux_inputs.py              fig_aux_inputs (release-dated macro inputs)
  3. forecasting/make_fig_forecasting.py      auc_by_set ("Test AUC by feature set"), delta_auc
  4. XAI/make_xai_figures.py --exp E7         XAI tables, then the four XAI figures of Section 5.3
  5. XAI/insights/run_insights.py --exp E7    flow-size, global-equity, volatility and model-agreement analyses
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
EXP = "E7"
FINAL = HERE / "final"


def run(script: Path, *args):
    print(f"\n>>> {script.relative_to(HERE)} {' '.join(args)}", flush=True)
    r = subprocess.run([PY, "-W", "ignore", script.name, *args], cwd=script.parent)
    if r.returncode:
        raise SystemExit(f"failed: {script}")


def main():
    for s in ("make_tbl_performance.py", "make_tbl_block_tests.py", "make_tbl_within_model.py"):
        run(HERE / "tables" / s)
    run(HERE / "figs" / "make_fig_aux_inputs.py")
    run(HERE / "forecasting" / "make_fig_forecasting.py")
    X = HERE / "XAI"
    run(X / "make_xai_figures.py", "--exp", EXP)
    for s in ("make_fig_shap_summary.py", "make_fig_network.py", "make_fig_temporal.py", "make_fig_lags.py"):
        run(X / s, "--exp", EXP)
    run(X / "insights" / "run_insights.py", "--exp", EXP)
    for s in ("make_fig_q1.py", "make_fig_q2.py", "make_fig_q4.py", "make_fig_q6.py"):
        run(X / "insights" / s, "--exp", EXP)

    # ---- collect under the manuscript's names (paths relative to the Overleaf project)
    F, I = X / "figures" / EXP, X / "insights" / EXP
    copies = {
        "figs/xai/xai_shap_summary.pdf": F / "xai_shap_summary.pdf",
        "figs/xai/xai_network.pdf": F / "xai_network.pdf",
        "figs/xai/xai_temporal.pdf": F / "xai_temporal.pdf",
        "figs/xai/xai_lags.pdf": F / "xai_lags.pdf",
        "figs/xai/magnitude.pdf": I / "q1_figure.pdf",
        "figs/xai/global_equity.pdf": I / "q2_figure.pdf",
        "figs/xai/volatility.pdf": I / "q4_figure.pdf",
        "figs/forecasting/model_agreement.pdf": I / "q6_agreement_figure.pdf",
        "figs/fig_aux_inputs.pdf": HERE / "figs" / "fig_aux_inputs.pdf",
        "Test AUC by feature set.png": HERE / "forecasting" / "auc_by_set.png",
        "Change in test AUC from adding each information block.png": HERE / "forecasting" / "delta_auc.png",
        "figs/forecasting/auc_by_set.pdf": HERE / "forecasting" / "auc_by_set.pdf",
        "figs/forecasting/delta_auc.pdf": HERE / "forecasting" / "delta_auc.pdf",
    }
    copies.update({f"tables/{p.name}": p for p in sorted((HERE / "tables").glob("*.tex"))})
    for dst, src in copies.items():
        out = FINAL / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    print(f"\n{len(copies)} files copied to {FINAL}")


if __name__ == "__main__":
    main()
