#!/usr/bin/env python3
"""詳細測試 argparse 參數解析"""

import argparse
import sys

# 列出所有原始參數
print("=== sys.argv ===")
for i, arg in enumerate(sys.argv):
    print(f"  [{i}]: {repr(arg)}")

# 解析參數
parser = argparse.ArgumentParser(description="Test argparse")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="sb3_cfg_entry_point", help="Agent config.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")

args, hydra_args = parser.parse_known_args()

print(f"\n=== Parsed Arguments ===")
print(f"task: {repr(args.task)}")
print(f"agent: {repr(args.agent)}")
print(f"num_envs: {repr(args.num_envs)}")
print(f"hydra_args: {hydra_args}")

if args.task is None:
    print("\n❌ ERROR: --task is None!")
    print("   This explains the error you're seeing.")
else:
    print(f"\n✅ SUCCESS: task = {args.task}")
