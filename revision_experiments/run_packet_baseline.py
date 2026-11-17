#!/usr/bin/env python3
"""Completion-aware scheduling baseline with a fairness constraint.

Evaluation only: no training. The rule is evaluated by the unmodified
evaluate() function of pt_d3qn_uav.py on the archived evaluation scenarios,
under the same joint feasibility mask as every other policy.

Rule (Completion_First_Fair), applied in every slot:
  1. Feasible set: transmission actions allowed by the environment's mask.
     If it is empty, select Idle.
  2. Fairness constraint: keep only actions whose user has the smallest
     number of completed packets among users with a feasible action.
  3. Completion criterion: for each remaining action, expected progress on
     the user's head-of-line packet is
         success probability * min(deliverable bits, residual packet bits).
     Select the action with the fewest expected slots to complete that
     packet, residual bits / expected progress.
  4. Ties: older head-of-line packet first, then lower transmit energy.

Information used. Shared with the myopic oracles: per-action success
probability, transmission rate and energy. Additional packet state, not in
the observation vector of the learned policies: per-user completed-packet
count, residual bits of the head-of-line packet, and its arrival slot.

  python run_packet_baseline.py selftest --simulator .
  python run_packet_baseline.py run --simulator . --results results --user-counts 5 15 30 45 60 --out packet_baseline
  python run_packet_baseline.py analyze --seed-metrics results/user_count_evaluation_seed_summary.csv --user-seed-metrics results/user_count_user_seed_summary.csv --out packet_baseline
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_scale_control as sc

NAME = "Completion_First_Fair"
COMPARE = ["PT_D3QN", "EUT_D3QN", "PT_Myopic_Oracle",
           "Greedy_Physical_Utility", "Random"]
METRICS = ["utility_shortfall", "delivered_packets", "offered_pdr",
           "throughput_mbps", "final_jain_fairness",
           "total_energy_consumed_j", "no_completed_packet_pct",
           "zero_delivered_bits_pct", "discounted_transformed_return",
           "discounted_linear_return"]
v7 = None


def completion_first_fair(env):
    n_tx = env.n_transmission_actions
    feasible = np.flatnonzero(env.valid_mask()[:n_tx])
    if not len(feasible):
        return int(env.idle_action)
    outcomes = env.outcomes()
    users = env.actions[feasible, 2]

    completed = env.delivered_packets[users]
    keep = completed == completed.min()                     # fairness constraint
    feasible, users = feasible[keep], users[keep]

    residual = np.array([env.packet_queues[u][0][0] for u in users])
    arrival = np.array([env.packet_queues[u][0][1] for u in users])
    bits = outcomes["tx_bps"][feasible] * env.c.slot_duration_s
    progress = outcomes["success"][feasible] * np.minimum(bits, residual)
    slots = residual / np.maximum(progress, 1e-12)          # completion criterion
    energy = outcomes["energy_j"][feasible]
    order = np.lexsort((energy, arrival, slots))            # last key is primary
    return int(feasible[order[0]])


class ReturnSink:
    """Collects realised discounted returns per evaluation episode."""

    def __init__(self, gamma, scale):
        self.gamma, self.scale = gamma, scale
        self.transformed, self.linear = {}, {}

    def write(self, row):
        if not row.get("service_opportunity"):
            return
        key = row["evaluation_episode"]
        weight = self.gamma ** int(row["slot"])
        self.transformed[key] = self.transformed.get(key, 0.0) + weight * (
            float(row["pt_subjective_utility"]) / self.scale)
        self.linear[key] = self.linear.get(key, 0.0) + weight * float(
            row["centered_eut_utility"])

    append = write


@contextlib.contextmanager
def baseline_registered():
    original = v7.heuristic_action

    def dispatch(env, name, rng):
        if name == NAME:
            return completion_first_fair(env)
        return original(env, name, rng)

    v7.heuristic_action = dispatch
    try:
        yield
    finally:
        v7.heuristic_action = original


def evaluate_policy(name, cfg):
    users = []
    sink = ReturnSink(cfg.gamma, cfg.pt_reward_scale_constant)
    with baseline_registered():
        rows = v7.evaluate(name, cfg, None, user_detail_rows=users,
                           step_sink=sink)
    episodes = pd.DataFrame(rows)
    per_user = pd.DataFrame(users).groupby("user_index")[
        ["delivered_packets", "delivered_mbits"]].mean()
    row = {m: float(episodes[m].mean()) for m in METRICS if m in episodes}
    row["no_completed_packet_pct"] = 100 * float(
        (per_user.delivered_packets == 0).mean())
    row["zero_delivered_bits_pct"] = 100 * float(
        (per_user.delivered_mbits == 0).mean())
    row["discounted_transformed_return"] = float(
        np.mean(list(sink.transformed.values())))
    row["discounted_linear_return"] = float(
        np.mean(list(sink.linear.values())))
    return row


def cmd_run(args):
    global v7
    module_path = sc.load_simulator(args.simulator)
    v7 = sc.v7
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "packet_baseline_seed_metrics.csv"
    rows = pd.read_csv(target).to_dict("records") if target.exists() else []
    seen = {(r["n_user"], r["method"], r["replicate_seed"]) for r in rows}
    methods = [NAME, *(args.also or [])]
    for n_user in args.user_counts:
        for cfg in sc.load_configs(args.results, n_user):
            for method in methods:
                if (n_user, method, cfg.seed) in seen:
                    continue
                t0 = time.time()
                row = evaluate_policy(method, cfg)
                row.update(n_user=n_user, method=method,
                           replicate_seed=cfg.seed)
                rows.append(row)
                pd.DataFrame(rows).to_csv(target, index=False)
                print(f"U={n_user} seed={cfg.seed} {method}: "
                      f"{time.time() - t0:.0f}s", flush=True)
    (out / "packet_baseline_manifest.json").write_text(json.dumps({
        "rule": NAME, "simulator_sha256": sc.sha256(module_path),
        "driver_sha256": sc.sha256(__file__), "numpy": np.__version__,
        "python": sys.version.split()[0]}, indent=2))
    print(f"written: {target}")


def cmd_analyze(args):
    out = Path(args.out)
    new = pd.read_csv(out / "packet_baseline_seed_metrics.csv")
    archived = pd.read_csv(args.seed_metrics)
    key = ["n_user", "method", "replicate_seed"]
    if args.user_seed_metrics:
        users = pd.read_csv(args.user_seed_metrics)
        extra = users.groupby(key).apply(lambda x: pd.Series({
            "delivered_packets": x.delivered_packets.sum(),
            "no_completed_packet_pct": 100 * (x.delivered_packets == 0).mean(),
            "zero_delivered_bits_pct": 100 * (x.delivered_mbits == 0).mean(),
        }), include_groups=False).reset_index()
        archived = archived.drop(columns=[c for c in extra.columns
                                          if c in archived.columns
                                          and c not in key]).merge(
            extra, on=key, how="left")
    if args.discounted_returns:
        archived = archived.merge(pd.read_csv(args.discounted_returns),
                                  on=key, how="left")
    both = pd.concat([archived, new[new.method == NAME]], ignore_index=True)
    show = [m for m in METRICS if m in both.columns]
    means = both[both.method.isin([NAME, *COMPARE])].groupby(
        ["n_user", "method"])[show].mean()
    means.to_csv(out / "packet_baseline_policy_means.csv")

    rows = []
    for n_user in sorted(new.n_user.unique()):
        sub = both[both.n_user == n_user]
        base = sub[sub.method == NAME].set_index("replicate_seed")
        for other in COMPARE:
            b = sub[sub.method == other].set_index("replicate_seed")
            common = base.index.intersection(b.index)
            for metric in show:
                if b[metric].notna().sum() < 2 or base[metric].notna().sum() < 2:
                    continue
                rows.append({"n_user": n_user,
                             "contrast": f"{other} - {NAME}", "metric": metric,
                             "mean_first": float(b.loc[common, metric].mean()),
                             "mean_second": float(base.loc[common, metric].mean()),
                             **sc.paired(b.loc[common, metric],
                                         base.loc[common, metric])})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "packet_baseline_contrasts.csv", index=False)
    pd.set_option("display.width", 220)
    print(means.round(4).to_string())
    print(f"\nwritten: {out}/packet_baseline_*.csv")


def cmd_selftest(args):
    global v7
    sc.load_simulator(args.simulator)
    v7 = sc.v7
    cfg = v7.build_replicate_config(
        v7.Config(n_user=12, train_episodes=5, eval_episodes=20,
                  randomize_pt_profiles=True), 7)
    plain = pd.DataFrame(v7.evaluate("Random", cfg, None))
    with baseline_registered():
        again = pd.DataFrame(v7.evaluate("Random", cfg, None))
    untouched = bool(np.array_equal(plain.utility_shortfall,
                                    again.utility_shortfall))
    print(f"1. registering the rule leaves existing baselines unchanged: {untouched}")

    env = v7.UAVEnv(cfg, v7.evaluation_episode_seed(cfg, 0))
    env.reset()
    legal = True
    for _ in range(cfg.horizon):
        action = completion_first_fair(env)
        legal &= bool(env.valid_mask()[action])
        env.step(action, "eut")
    print(f"2. every selected action satisfies the feasibility mask: {legal}")
    rule = evaluate_policy(NAME, cfg)
    greedy = evaluate_policy("Greedy_Physical_Utility", cfg)
    print(f"3. completed packets per episode: rule {rule['delivered_packets']:.2f}, "
          f"Greedy {greedy['delivered_packets']:.2f}; users with no completed "
          f"packet: rule {rule['no_completed_packet_pct']:.1f}%, "
          f"Greedy {greedy['no_completed_packet_pct']:.1f}%")
    restored = v7.heuristic_action.__name__ == "heuristic_action"
    print(f"4. dispatcher restored: {restored}")
    ok = untouched and legal and restored
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--simulator", default=".")
    p.add_argument("--results", required=True)
    p.add_argument("--user-counts", type=int, nargs="+", required=True)
    p.add_argument("--also", nargs="*",
                   help="existing baselines to re-evaluate with the same "
                        "return accounting, e.g. PT_Myopic_Oracle")
    p.add_argument("--out", default="packet_baseline")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("analyze")
    p.add_argument("--seed-metrics", required=True)
    p.add_argument("--user-seed-metrics")
    p.add_argument("--discounted-returns",
                   help="CSV with n_user, method, replicate_seed and the two "
                        "discounted-return columns for archived policies")
    p.add_argument("--out", default="packet_baseline")
    p.set_defaults(func=cmd_analyze)
    p = sub.add_parser("selftest")
    p.add_argument("--simulator", default=".")
    p.set_defaults(func=cmd_selftest)
    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
