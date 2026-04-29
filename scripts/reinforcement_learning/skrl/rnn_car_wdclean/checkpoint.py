"""Checkpoint save/load -- extracted from train_rnn_car_wdclip.py."""

import torch


def save_checkpoint(path, *, extractor, preprocess_rnn, policy_head, value_head,
                    obs_policy, obs_value, charge_opt_rl, charge_opt_aux,
                    obs_optimizer, obs_normalizer, iteration, total_steps,
                    args_cli, use_extractor):
    """Save training checkpoint (same keys as train_rnn_car_wdclip.py)."""
    ckpt = {
        "preprocess_rnn": preprocess_rnn.state_dict(),
        "policy_head": policy_head.state_dict(),
        "value_head": value_head.state_dict(),
        "obs_policy": obs_policy.state_dict(),
        "obs_value": obs_value.state_dict(),
        "charge_opt_rl": charge_opt_rl.state_dict(),
        "charge_opt_aux": charge_opt_aux.state_dict(),
        "obs_optimizer": obs_optimizer.state_dict(),
        "obs_normalizer": {
            "mean": obs_normalizer.mean,
            "var": obs_normalizer.var,
            "count": obs_normalizer.count,
        },
        "iteration": iteration,
        "total_steps": total_steps,
        "args": vars(args_cli),
    }
    if use_extractor:
        ckpt["extractor"] = extractor.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, *, extractor, preprocess_rnn, policy_head, value_head,
                    obs_policy, obs_value, obs_normalizer, device, use_extractor):
    """Load training checkpoint (compatible with train_rnn_car_wdclip.py checkpoints)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if use_extractor and "extractor" in ckpt:
        extractor.load_state_dict(ckpt["extractor"])
    preprocess_rnn.load_state_dict(ckpt["preprocess_rnn"])
    policy_head.load_state_dict(ckpt["policy_head"])
    value_head.load_state_dict(ckpt["value_head"])
    if "obs_policy" in ckpt:
        obs_policy.load_state_dict(ckpt["obs_policy"])
        obs_value.load_state_dict(ckpt["obs_value"])
    if "obs_normalizer" in ckpt:
        obs_normalizer.mean = ckpt["obs_normalizer"]["mean"]
        obs_normalizer.var = ckpt["obs_normalizer"]["var"]
        obs_normalizer.count = ckpt["obs_normalizer"]["count"]
    return ckpt
