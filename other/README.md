# Other Archived Work

This folder contains non-mainline files copied from the local workspace for handoff.

These files are not part of the WDClean / WDClip reproducible training path. They are kept here so another machine or agent can inspect them without relying on this machine's untracked files.

## Layout

```text
other/local_agent_state/
```

Local agent/tooling state copied for reference only. Do not use these files as runtime dependencies.

```text
other/experimental/
```

Experimental scripts and packages copied from their original paths. They are archived here, not wired into the package import path.

If one of these features is needed, move or copy it back to its original repo path and test it in a dedicated commit.

## Original Paths

```text
other/local_agent_state/CLAUDE.md
  <- CLAUDE.md

other/local_agent_state/.claude/scheduled_tasks.lock
  <- .claude/scheduled_tasks.lock

other/local_agent_state/.director-mode/
  <- .director-mode/

other/experimental/scripts/pedestrian/
  <- scripts/pedestrian/

other/experimental/scripts/reinforcement_learning/skrl/train_marl_rnn.py
  <- scripts/reinforcement_learning/skrl/train_marl_rnn.py

other/experimental/scripts/reinforcement_learning/skrl/train_rnn_car_ppo_mb.py
  <- scripts/reinforcement_learning/skrl/train_rnn_car_ppo_mb.py

other/experimental/scripts/reinforcement_learning/skrl/virtual_spot_utils.py
  <- scripts/reinforcement_learning/skrl/virtual_spot_utils.py

other/experimental/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/direct_marl/
  <- source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/direct_marl/
```
