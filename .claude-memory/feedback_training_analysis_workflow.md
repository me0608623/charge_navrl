---
name: Training Analysis Workflow
description: User wants 3-tier analysis for every training run — live monitoring, post-run report, and scripted analysis
type: feedback
---

When user asks about training status, results, or "how's the training", use this 3-tier workflow:

1. **Live monitoring** — read the log file (path varies per run), check for divergence, NaN, crash, stuck metrics
2. **Post-run report** — pull WandB history via API or CSV, compare groups, summarize which is best and why
3. **Scripted analysis** — use `scripts/reinforcement_learning/skrl/analyze_wandb_run.py` for repeatable analysis

**Why:** User doesn't want to manually parse WandB dashboards. They want structured, actionable summaries.

**How to apply:** When user says "訓練的如何" / "分析某次訓練" / "compare runs":
- First check if training is still running (ps aux / log tail)
- If running: do live monitoring (tier 1)
- If finished: do post-run report (tier 2) using the analysis script (tier 3)
- Log file paths vary per run — check `logs/` directory or ask user
- WandB project: `charge_skrl`, entity: `me0608623-none`
