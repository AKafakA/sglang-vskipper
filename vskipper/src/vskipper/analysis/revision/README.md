# Analysis scripts of the revision

Read-only scripts over measured cells. Each takes the campaign folder that holds the cells and writes JSON or prints a table.

| script | result |
|---|---|
| `mb_metrics.py CAMPAIGN_ROOT 4900 gsm8k_g7a OUT.json` | post-trained checkpoint under natural stopping: mean and p99 E2E, TTFT and TPOT changes against upstream, output tokens, limit hits |
| `served_loops_skip.py CAMPAIGN_ROOT 4900 gsm8k_g7a ../../../../training/llama3-flexidepth-grpo/fdfix_lib.py OUT.json` | repeated generations and decode skip fractions per cell |
| `e3_matched_skip.py EXTRACT_ROOT OUT.json` | decode skip fractions of the released checkpoint under matched work |
| `tps_audit.py` | `pair(upstream_record, treatment_record)`: TPS and RPS in the injection window at 1.25 x Q* for one repetition (the post-trained checkpoint's cells included) |
| `pen_audit.py CAMPAIGN_ROOT SCORES_DIR tps_audit.py OUT_DIR` | repetition penalty 1.15, repetitions 1152 to 1157: latency means and p99, TPS, RPS, quality |
| `pd_audit.py CAMPAIGN_ROOT OUT_DIR` | prefill-decode disaggregation: p99 E2E, TTFT, TPOT and makespan over six paired repetitions |
| `analyze_natural.py` | natural-stopping summary of one experiment: E2E, TPOT, output tokens, limit hits per arm |

Quality is scored by `vskipper/src/vskipper/experiments/score_natural_lane_lmeval.py` on the evaluation-split rows of each cell. Intervals are t-based 95 %
half-widths over the paired per-repetition changes.
