---
name: feedback_ask_yaml_before_train
description: Every training launch must ask user which experiment_config YAML to use
type: feedback
---

Every time before launching a training run, always ask the user which `--experiment_config` YAML file to use.

**Why:** The YAML experiment config controls WandB tags, hyperparameters, and profiles. Without it, tags don't get written to WandB and parameters may be wrong. Previous runs were launched without `--experiment_config` causing missing tags.

**How to apply:** Before any `train_rnn_car_wdclip.py` launch, ask: "要使用哪個 experiment_config YAML？" and wait for the user's answer. Never assume or skip this step.
