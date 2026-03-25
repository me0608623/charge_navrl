#!/usr/bin/env python3
"""測試 argparse 參數解析"""

import argparse

parser = argparse.ArgumentParser(description="Test argparse")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="sb3_cfg_entry_point", help="Agent config.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")

args, hydra_args = parser.parse_known_args()

print(f"task: {args.task}")
print(f"agent: {args.agent}")
print(f"num_envs: {args.num_envs}")
print(f"hydra_args: {hydra_args}")
