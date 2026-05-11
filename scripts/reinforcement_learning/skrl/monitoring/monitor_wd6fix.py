#!/usr/bin/env python3
"""Training monitor for wd6fix run — reads WandB CSV + output log."""
import csv, os, sys, json

RUN_DIR = "/home/aa/IsaacLab/wandb/run-20260504_184508-d7cdkms7"
LOG_FILE = os.path.join(RUN_DIR, "files", "output.log")
WANDB_SUMMARY = os.path.join(RUN_DIR, "files", "wandb-summary.json")

def get_latest_console():
    """Extract last 5 [N/300] CHARGE lines from output.log."""
    if not os.path.exists(LOG_FILE):
        return "No output.log yet"
    lines = []
    with open(LOG_FILE) as f:
        for line in f:
            if "[CHARGE" in line and "/300]" in line:
                lines.append(line.strip())
    if not lines:
        # fallback: any [N/300] line
        with open(LOG_FILE) as f:
            for line in f:
                if "/300]" in line:
                    lines.append(line.strip())
    return "\n".join(lines[-5:]) if lines else "No CHARGE summary lines yet"

def get_wandb_summary():
    """Read wandb-summary.json for latest metrics."""
    if not os.path.exists(WANDB_SUMMARY):
        return {}
    with open(WANDB_SUMMARY) as f:
        return json.load(f)

def check_abort_conditions(summary):
    """Check for abort-worthy conditions."""
    alerts = []
    # rnn feature drift
    fm = summary.get("aux/rnn_feature_mean", 0)
    if fm > 5:
        alerts.append(f"⚠️ rnn_feature_mean = {fm:.3f} (>5, approaching abort threshold)")
    if fm > 10:
        alerts.append(f"🚨 rnn_feature_mean = {fm:.3f} (>10, SHOULD ABORT)")
    
    # VE collapse
    ve = summary.get("perf/explained_variance", None)
    if ve is not None and ve < -0.5:
        alerts.append(f"🚨 VE = {ve:.3f} (<-0.5, critic collapsed)")
    
    # NaN
    for k, v in summary.items():
        if isinstance(v, float) and (v != v):  # NaN check
            alerts.append(f"🚨 NaN detected in {k}")
    
    return alerts

if __name__ == "__main__":
    console = get_latest_console()
    summary = get_wandb_summary()
    alerts = check_abort_conditions(summary)
    
    # Extract key metrics for quick view
    keys_to_show = {
        "episode/success_rate": "SR",
        "episode/collision_rate": "CR",
        "perf/explained_variance": "VE",
        "aux/rnn_feature_mean": "fm",
        "aux/rnn_feature_delta": "fd",
        "aux/aux_loss": "aux_loss",
        "rl/entropy": "entropy",
        "charge/goal_velocity": "gV",
        "charge/heading_error_deg": "hErr",
        "_step": "step",
        "_runtime": "runtime_s",
    }
    
    metrics_str = ""
    for k, label in keys_to_show.items():
        v = summary.get(k)
        if v is not None:
            if isinstance(v, float):
                metrics_str += f"  {label}={v:.4f}\n"
            else:
                metrics_str += f"  {label}={v}\n"
    
    print("=" * 60)
    print("WD6FIX TRAINING MONITOR — wd6fix_a2c_sparse_stage1_s1")
    print(f"WandB: jzld0rsy")
    print("=" * 60)
    print(f"\n📊 Key Metrics (WandB summary):")
    print(metrics_str if metrics_str else "  No metrics yet")
    
    if alerts:
        print(f"\n{'🚨' * 3} ALERTS:")
        for a in alerts:
            print(f"  {a}")
    else:
        print(f"\n✅ No abort conditions triggered")
    
    print(f"\n📋 Console (last 5 CHARGE summaries):")
    print(console)
    print("=" * 60)
