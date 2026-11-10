# Expanding-origin check with all ten models

## Purpose

`../Expanding_origin` repeated the paper's protocol at three earlier forecast origins, but only for the logistic regression and the three tree models. This folder adds the **seven neural networks** for those origins. The robustness check then covers the same ten models as the main results.

`../Expanding_origin` and `../GDP_corrected` are not modified.

## Windows

The windows are identical to `../Expanding_origin`:

| Window | Phase-1 training | Validation (epoch selection) | Phase-2 refit | Test (scored once) | Test days |
|---|---|---|---|---|---|
| W1 | 2015 – 2020 | 2021 | 2015 – 2021 | 2022 | 246 |
| W2 | 2015 – 2021 | 2022 | 2015 – 2022 | 2023 | 245 |
| W3 | 2015 – 2022 | 2023 | 2015 – 2023 | 2024 | 244 |
| W4 | the paper's split | | | 8 May 2025 – 26 Jun 2026 | 278 |

W4 is not rerun. It is `../GDP_corrected`, the main results.

## What is run here

7 networks × E0–E7 × seeds 42–46 × W1–W3 = **840 runs**:
- LSTM, CNN-LSTM, GRU-LSTM, BiGRU-LSTM, BiLSTM, iTransformer, DA-RNN.

**What is identical to the paper's run:**
- the code;
- the hyperparameters: lookback 30, horizon 1, batch 64, at most 100 epochs, patience 10 (20 for DA-RNN), learning rate 1e-3 (1e-4 for iTransformer);
- class weights, the AUX clip, the arcsinh flow scaling and the seeds.

**What differs, per window:**
- the split dates;
- the end of the input, which is truncated at the end of the test year.

`setup_windows.py` makes one code copy per window in `windows/<W>/models`. It checks that every window input is a **byte-identical prefix** of `../GDP_corrected/data/input_gdp_corrected.csv` (md5 `52cc0eb4…`). `run_windows.py` checks this again before running.

The **logistic regression and tree models** for W1–W4 are already in `../Expanding_origin/results`. Their code is the same deterministic tree code as `../GDP_corrected`, and W4 reproduces it exactly. They are not rerun; the combined analysis reads them from there.

## Run

```
cd .\Expanding_origin_all_models
python setup_windows.py          # once (already done)
python run_windows.py            # 840 runs, resumable; fast networks first, DA-RNN last (log: run_log.txt)
python run_windows.py --collect  # results_nn_all.csv
python analyze_all_models.py     # analysis/ (ten models + logit, W1-W4)
python make_si_tables.py         # analysis/tbl_S14_block_by_window.tex, tbl_S16_auc_by_window.tex
```

## Status

- **Finished:** 21:19 on 9 Nov 2026, after 217 min on the GPU (started after the E7 XAI).
- **Runs:** 840 OK, 0 failures (`run_log.txt`).
- **Check:** the W4 AUCs in `analysis/auc_by_window.csv` equal the main run (`../GDP_corrected/results_all.csv`, mean over seeds) to three decimals for all 80 model × set cells.

## Results (ten models; change in AUC × 100, mean over 4 matched pairs; models improved out of 10; Wilcoxon p across models)

| Block | 2022 | 2023 | 2024 | 2025–26 (paper) | Pooled (40) |
|---|---|---|---|---|---|
| AUX | +1.11 (7), .160 | +2.99 (8), .010 | +2.19 (9), .084 | +4.69 (9), .004 | +2.75 (33), < .001 |
| LDA | −2.04 (4), .084 | −1.63 (1), .004 | +0.48 (8), .049 | −1.16 (1), .004 | −1.09 (14), .003 |
| LLM | −0.22 (4), .695 | −1.10 (0), .002 | −0.30 (1), .065 | −0.22 (4), .492 | −0.46 (9), .002 |

- **AUX:** it helps on average in every year but less before 2025. DA-RNN loses from AUX in every window, and iTransformer is flat before 2025.
- **Text blocks:** LDA and LLM lower the AUC on average.
- **More:** `analysis/summary.txt` and Section 0.1 of `../Manuscript_changes/MANUSCRIPT_CHANGES.md`.

## Outputs

- `windows/<W>/models/<Model>/results/<name>/E<k>/seed_<s>/`: `metrics.json`, `predictions.csv`, and the model checkpoint.
- `results_nn_all.csv`: one row per window × network × set × seed.
- `analysis/`: the combined ten-model + logit analysis per window (`summary.txt`, `auc_by_window.csv`, `block_effects_by_model.csv`, `block_by_window.csv`, `block_by_family.csv`) and the SI tables (`tbl_S14_block_by_window.tex`, `tbl_S16_auc_by_window.tex`).
  - AUC by window, model and set.
  - Block effects (AUX, LDA, LLM) per window: the mean over the 4 matched pairs and 5 seeds per model, and a Wilcoxon test across models.
  - A pooled test over window × model.

## Caveats

These are the same as in `../Expanding_origin`:
- The LDA topic models were fitted on the full 2015–2026 corpus.
- EXAONE may have seen 2022–24 text in pretraining.
- Both can only favour the text features in W1–W3, so they work against the finding that the text adds nothing.
