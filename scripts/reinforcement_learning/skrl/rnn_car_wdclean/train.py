#!/usr/bin/env python3
"""
rnn_car_wdclean entrypoint -- behaviorally equivalent to train_rnn_car_wdclip.py.

Usage:
  PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/rnn_car_wdclean/train.py \
    --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
    --num_envs 4096 --headless --seed 1 --use_a2c \
    --run_name wdclean_test
"""

import sys
from pathlib import Path

# Ensure this package dir and parent (skrl/) are importable
_pkg_dir = str(Path(__file__).resolve().parent)
_parent_dir = str(Path(__file__).resolve().parent.parent)
for p in [_pkg_dir, _parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- 1. CLI + AppLauncher (must be before any Isaac/torch imports) ---
from args import create_parser
from isaaclab.app import AppLauncher

parser = create_parser()
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

headless_mode = getattr(args_cli, "headless", False) or "--headless" in sys.argv
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. Post-launcher: run training ---
from trainer import run

run(args_cli, headless_mode)
