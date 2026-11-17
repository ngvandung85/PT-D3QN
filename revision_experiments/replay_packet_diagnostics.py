#!/usr/bin/env python3
"""Packet-level service and delay diagnostics by replaying archived actions.

The slot-level archive records the action selected in every evaluation slot.
Exogenous randomness in the simulator is indexed by scenario seed, episode and
slot and does not depend on the actions taken, so replaying the archived action
sequence through the unmodified simulator reconstructs the complete packet
state of every evaluation episode of every policy, including learned policies,
without retraining. The replay is verified slot by slot against the archived
delivered bits and completed packets.

  python replay_packet_diagnostics.py --simulator simulator --results results --user-count 60 --out packet_diagnostics

Outputs, one row per policy and replicate seed, in <out>/:
  service_diagnostics_seed.csv   episode-level and persistent non-service,
                                 distribution of per-user completed packets,
                                 delay sample counts, unfinished-packet ages
  service_diagnostics_summary.csv  means and 95% intervals over seeds
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_scale_control as sc

COLUMNS = ["method", "replicate_seed", "evaluation_episode", "slot", "is_idle",
           "selected_uav", "selected_channel", "selected_user", "selected_mode",
           "delivered_bits", "packets_delivered"]


def replay_group(v7, cfg, group):
    """Replay all evaluation episodes of one (policy, replicate)."""
    n_user, horizon, dt = cfg.n_user, cfg.horizon, cfg.slot_duration_s
    probe = v7.UAVEnv(cfg, 0)
    index = {tuple(int(x) for x in row): i for i, row in
             enumerate(probe.actions[: probe.n_transmission_actions])}
    idle = int(probe.idle_action)

    completed = []                    # per episode: packets completed per user
    delays, unfinished_ages = [], []
    episodes_with_completion = 0
    mismatches = 0
    for episode, rows in group.groupby("evaluation_episode", sort=True):
        rows = rows.sort_values("slot")
        env = v7.UAVEnv(cfg, v7.evaluation_episode_seed(cfg, int(episode)))
        env.reset()
        before = env.packet_delay_sum_s.copy()
        done_before = env.delivered_packets.copy()
        for r in rows.itertuples(index=False):
            if r.is_idle:
                action = idle
            else:
                action = index[(int(r.selected_uav), int(r.selected_channel),
                                int(r.selected_user), int(r.selected_mode))]
            pre_delay = env.packet_delay_sum_s.sum()
            pre_done = int(env.delivered_packets.sum())
            info = env.step(action, "eut")[-1]
            if (abs(float(info["delivered_bits"]) - float(r.delivered_bits)) > 1e-6
                    or int(info["packets_delivered"]) != int(r.packets_delivered)):
                mismatches += 1
            newly = int(env.delivered_packets.sum()) - pre_done
            if newly:
                # delay of the packets completed in this slot, in seconds
                delays.extend([(env.packet_delay_sum_s.sum() - pre_delay) / newly]
                              * newly)
        per_user = env.delivered_packets - done_before
        completed.append(per_user.copy())
        episodes_with_completion += int(per_user.sum() > 0)
        for queue in env.packet_queues:
            for _, arrival_slot in queue:
                unfinished_ages.append((horizon - arrival_slot) * dt)
    completed = np.array(completed)                 # episodes x users
    ages = np.array(unfinished_ages)
    delays = np.array(delays)
    n_episodes = len(completed)
    out = {
        "replay_mismatched_slots": mismatches,
        "episodes": n_episodes,
        # non-service
        "episode_no_completion_pct": 100 * float((completed == 0).mean()),
        "persistent_no_completion_pct":
            100 * float((completed.sum(axis=0) == 0).mean()),
        "users_completing_per_episode": float((completed > 0).sum(axis=1).mean()),
        # distribution of per-user completed packets within an episode
        "share_0_packets_pct": 100 * float((completed == 0).mean()),
        "share_1_packet_pct": 100 * float((completed == 1).mean()),
        "share_2_packets_pct": 100 * float((completed == 2).mean()),
        "share_3plus_packets_pct": 100 * float((completed >= 3).mean()),
        "max_packets_one_user": int(completed.max()),
        "completed_packets_per_episode": float(completed.sum(axis=1).mean()),
        # delay sample
        "delay_packets_total": int(len(delays)),
        "delay_episodes_contributing": int(episodes_with_completion),
        "mean_packet_delay_s": float(delays.mean()) if len(delays) else np.nan,
        "median_packet_delay_s": float(np.median(delays)) if len(delays) else np.nan,
        # unfinished packets at the end of the episode
        "unfinished_packets_per_episode": float(len(ages) / n_episodes),
        "unfinished_mean_age_s": float(ages.mean()) if len(ages) else np.nan,
        "unfinished_median_age_s": float(np.median(ages)) if len(ages) else np.nan,
        "unfinished_from_episode_start_pct":
            100 * float((ages >= horizon * dt - 1e-9).mean()) if len(ages) else np.nan,
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--simulator", default=".")
    parser.add_argument("--results", required=True)
    parser.add_argument("--user-count", type=int, default=60)
    parser.add_argument("--step-records",
                        help="slot-level archive; default "
                             "<results>/users_NNN/evaluation_step_raw.csv")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--out", default="packet_diagnostics")
    args = parser.parse_args()

    sc.load_simulator(args.simulator)
    v7 = sc.v7
    configs = {c.seed: c for c in sc.load_configs(args.results, args.user_count)}
    path = Path(args.step_records or Path(args.results)
                / f"users_{args.user_count:03d}" / "evaluation_step_raw.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(path, usecols=COLUMNS)
    if args.methods:
        frame = frame[frame.method.isin(args.methods)]
    rows = []
    for (method, seed), group in frame.groupby(["method", "replicate_seed"],
                                               sort=True):
        t0 = time.time()
        row = replay_group(v7, configs[int(seed)], group)
        row.update(n_user=args.user_count, method=method,
                   replicate_seed=int(seed))
        rows.append(row)
        print(f"{method} seed={seed}: {time.time() - t0:.0f}s, "
              f"mismatched slots {row['replay_mismatched_slots']}", flush=True)
        pd.DataFrame(rows).to_csv(out / "service_diagnostics_seed.csv",
                                  index=False)
    seed_level = pd.DataFrame(rows)
    from scipy import stats as st
    numeric = [c for c in seed_level.columns
               if c not in ("method", "replicate_seed", "n_user")]
    grouped = seed_level.groupby("method")[numeric]
    mean, sd, n = grouped.mean(), grouped.std(ddof=1), grouped.count()
    half = st.t.ppf(0.975, (n - 1).clip(lower=1)) * sd / np.sqrt(n)
    summary = mean.add_suffix("_mean").join(half.add_suffix("_ci95_half"))
    summary.to_csv(out / "service_diagnostics_summary.csv")
    print(f"\ntotal mismatched slots: {int(seed_level.replay_mismatched_slots.sum())}")
    print(f"written: {out}/service_diagnostics_*.csv")


if __name__ == "__main__":
    main()
