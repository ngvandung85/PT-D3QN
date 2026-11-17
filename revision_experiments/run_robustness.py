#!/usr/bin/env python3
"""Demand and packet-size robustness conditions for the PT-D3QN comparison.

Two operating conditions, each trained and evaluated with the unmodified
simulator through a subclass of UAVEnv that changes TRAFFIC ONLY. Geometry,
shadowing, user profiles, calibrated references, energy allowance, horizon,
architecture and training budget are those of the archived run.

  matched       Aggregate demand of a smaller population is offered to a
                larger one. With --demand-users 15, the aggregate initial
                backlog is the sum of 15 draws of the original per-user rule
                and the aggregate arrival rate is 15 times the original
                per-user rate. Allocation rule: every initial packet is
                assigned to a user chosen uniformly at random among users with
                free queue capacity; arrivals are Poisson with the aggregate
                rate split equally across users.

  small_packet  Packet size is divided by --packet-ratio (default 4, i.e.
                1 Mbit) at fixed offered bit volume. Every user starts with
                the same number of BITS as in the archived scenario (the
                original draw times the ratio, in small packets), and the
                per-user packet arrival rate is multiplied by the ratio, so
                the expected offered bit volume is unchanged.

Both learned arms (EUT_D3QN, PT_D3QN) are retrained under each condition; the
four non-learning baselines are evaluated without training.

  python run_robustness.py selftest --simulator .
  python run_robustness.py run --simulator . --results results --condition matched --demand-users 15 --user-counts 60 --out robustness
  python run_robustness.py run --simulator . --results results --condition small_packet --user-counts 60 --out robustness
  python run_robustness.py analyze --seed-metrics results/user_count_evaluation_seed_summary.csv --user-seed-metrics results/user_count_user_seed_summary.csv --out robustness

`run` skips finished units and can be split with --seeds and --methods.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import run_scale_control as sc

LEARNED = ("EUT_D3QN", "PT_D3QN")
BASELINES = ("Random", "Greedy_Physical_Utility",
             "EUT_Myopic_Oracle", "PT_Myopic_Oracle")
METRICS = ["utility_shortfall", "throughput_mbps", "offered_pdr",
           "delivered_packets", "offered_packets", "final_jain_fairness",
           "idle_with_backlog_fraction", "total_energy_consumed_j",
           "delivered_total_mbits"]
v7 = None
TrafficEnv = None


def load_simulator(directory):
    global v7, TrafficEnv
    path = sc.load_simulator(directory)
    v7 = sc.v7

    class _TrafficEnv(v7.UAVEnv):
        """UAVEnv with a modified initial backlog. Arrival rate and packet
        size are changed through the configuration, not here."""
        MODE = None             # "matched" | "small_packet"
        DEMAND_USERS = None     # matched: population whose demand is offered
        BASE_PACKET_BITS = None  # small_packet: packet size of the archive

        def reset(self):
            super().reset()
            cls, c = type(self), self.c
            if cls.MODE == "matched":
                counts = self._matched_initial_packets(cls.DEMAND_USERS)
            elif cls.MODE == "small_packet":
                counts = self._same_bits_initial_packets(cls.BASE_PACKET_BITS)
            else:
                return self._state()
            self.packet_queues = [deque() for _ in range(c.n_user)]
            for user, count in enumerate(counts):
                for _ in range(int(count)):
                    self.packet_queues[user].append(
                        [float(c.packet_size_bits), 0])
            self.queue = counts.astype(float) * c.packet_size_bits
            self.initial_packet_count = counts.copy()
            self.offered_packets = counts.copy()
            self.admitted_packets = counts.copy()
            return self._state()

        @staticmethod
        def _rule_bounds(max_queue_bits, packet_bits):
            capacity = int(np.floor(max_queue_bits / packet_bits))
            lower = max(0, int(np.ceil(0.25 * capacity)))
            upper = max(lower, int(np.floor(0.75 * capacity)))
            return capacity, lower, upper

        def _matched_initial_packets(self, demand_users):
            c = self.c
            capacity, lower, upper = self._rule_bounds(
                c.max_queue_bits, c.packet_size_bits)
            rng = np.random.default_rng(v7._derived_seed(
                620_000, self.environment_seed, self.episode_index, 0))
            total = int(rng.integers(lower, upper + 1, size=demand_users,
                                     dtype=np.int64).sum())
            counts = np.zeros(c.n_user, dtype=np.int64)
            for _ in range(total):
                free = np.flatnonzero(counts < capacity)
                if not len(free):
                    break
                counts[int(rng.choice(free))] += 1
            return counts

        def _same_bits_initial_packets(self, base_packet_bits):
            c = self.c
            ratio = int(round(base_packet_bits / c.packet_size_bits))
            _, lower, upper = self._rule_bounds(
                c.max_queue_bits, base_packet_bits)
            # Same generator and draw as the archived scenario.
            rng = np.random.default_rng(v7._derived_seed(
                610_000, self.environment_seed, self.episode_index, 1))
            original = rng.integers(lower, upper + 1, size=c.n_user,
                                    dtype=np.int64)
            return original * ratio

    TrafficEnv = _TrafficEnv
    return path


@contextlib.contextmanager
def traffic_env(mode, demand_users=None, base_packet_bits=None):
    original = v7.UAVEnv
    TrafficEnv.MODE = mode
    TrafficEnv.DEMAND_USERS = demand_users
    TrafficEnv.BASE_PACKET_BITS = base_packet_bits
    v7.UAVEnv = TrafficEnv
    try:
        yield
    finally:
        v7.UAVEnv = original
        TrafficEnv.MODE = None


def condition_config(cfg, condition, demand_users, packet_ratio):
    """Configuration for one condition; profiles and references unchanged."""
    if condition == "matched":
        if demand_users >= cfg.n_user:
            raise SystemExit("--demand-users must be smaller than the "
                             "population it is offered to")
        return replace(cfg, arrival_rate_pps=cfg.arrival_rate_pps
                       * demand_users / cfg.n_user), cfg.packet_size_bits
    if condition == "small_packet":
        small = cfg.packet_size_bits // packet_ratio
        if small * packet_ratio != cfg.packet_size_bits:
            raise SystemExit("packet size must be divisible by --packet-ratio")
        return replace(cfg, packet_size_bits=small,
                       arrival_rate_pps=cfg.arrival_rate_pps * packet_ratio
                       ), cfg.packet_size_bits
    raise SystemExit(f"unknown condition {condition}")


def run_unit(method, cfg, condition, demand_users, base_bits):
    users = []
    with traffic_env(condition, demand_users, base_bits):
        if method in LEARNED:
            bank = v7.build_training_profile_bank(cfg)
            agent, history = v7.train_agent(method, cfg, bank)
        else:
            agent, history = None, []
        rows = v7.evaluate(method, cfg, agent, user_detail_rows=users)
    return rows, users, history


def tag_for(args):
    if args.condition == "matched":
        return f"matched_d{args.demand_users:03d}"
    return f"small_packet_r{args.packet_ratio}"


def cmd_run(args):
    module_path = load_simulator(args.simulator)
    tag = tag_for(args)
    units = Path(args.out) / tag
    units.mkdir(parents=True, exist_ok=True)
    methods = args.methods or [*LEARNED, *BASELINES]
    for n_user in args.user_counts:
        for archived in sc.load_configs(args.results, n_user):
            if args.seeds and archived.seed not in args.seeds:
                continue
            cfg, base_bits = condition_config(
                archived, args.condition, args.demand_users, args.packet_ratio)
            for method in methods:
                stem = f"u{n_user:03d}_seed{archived.seed:03d}_{method}"
                done = units / f"{stem}.done"
                if done.exists():
                    continue
                t0 = time.time()
                rows, users, history = run_unit(
                    method, cfg, args.condition, args.demand_users, base_bits)
                pd.DataFrame(rows).to_csv(units / f"{stem}_episodes.csv",
                                          index=False)
                pd.DataFrame(users).to_csv(units / f"{stem}_users.csv",
                                           index=False)
                if history:
                    pd.DataFrame(history).to_csv(
                        units / f"{stem}_training.csv", index=False)
                done.write_text(json.dumps({
                    "condition": tag, "n_user": n_user, "method": method,
                    "replicate_seed": archived.seed,
                    "arrival_rate_pps": cfg.arrival_rate_pps,
                    "packet_size_bits": cfg.packet_size_bits,
                    "demand_users": args.demand_users
                    if args.condition == "matched" else None,
                    "simulator_sha256": sc.sha256(module_path),
                    "seconds": time.time() - t0}, indent=2))
                print(f"[{tag}] U={n_user} seed={archived.seed} {method}: "
                      f"{time.time() - t0:.0f}s", flush=True)


def coverage(users):
    """Per-seed coverage from per-episode, per-user rows."""
    per_user = users.groupby("user_index")[
        ["delivered_packets", "delivered_mbits"]].mean()
    bottom = max(1, int(np.ceil(0.2 * len(per_user))))
    return {
        "no_completed_packet_pct":
            100.0 * float((per_user.delivered_packets == 0).mean()),
        "zero_delivered_bits_pct":
            100.0 * float((per_user.delivered_mbits == 0).mean()),
        "bottom20_delivered_mbits":
            float(per_user.delivered_mbits.nsmallest(bottom).mean()),
    }


def collect(out):
    rows = []
    for done in sorted(Path(out).glob("*/*.done")):
        meta = json.loads(done.read_text())
        episodes = pd.read_csv(done.with_name(done.stem + "_episodes.csv"))
        users = pd.read_csv(done.with_name(done.stem + "_users.csv"))
        row = {m: float(episodes[m].mean()) for m in METRICS
               if m in episodes.columns}
        row.update(coverage(users))
        row.update(condition=meta["condition"], n_user=meta["n_user"],
                   method=meta["method"],
                   replicate_seed=meta["replicate_seed"])
        rows.append(row)
    return pd.DataFrame(rows)


def cmd_analyze(args):
    out = Path(args.out)
    new = collect(out)
    if new.empty:
        raise SystemExit("No finished units.")
    new.to_csv(out / "robustness_seed_metrics.csv", index=False)

    archived = pd.read_csv(args.seed_metrics)
    if args.user_seed_metrics:
        users = pd.read_csv(args.user_seed_metrics)
        key = ["n_user", "method", "replicate_seed"]
        extra = users.groupby(key).apply(lambda x: pd.Series({
            "delivered_packets": x.delivered_packets.sum(),
            "offered_packets": x.offered_packets.sum()
            if "offered_packets" in x else np.nan,
            "no_completed_packet_pct": 100 * (x.delivered_packets == 0).mean(),
            "zero_delivered_bits_pct": 100 * (x.delivered_mbits == 0).mean(),
        }), include_groups=False).reset_index()
        archived = archived.drop(columns=[c for c in extra.columns
                                          if c in archived.columns
                                          and c not in key]).merge(
            extra, on=key, how="left")
    archived = archived.assign(condition="original")
    both = pd.concat([archived, new], ignore_index=True)

    show = ["utility_shortfall", "throughput_mbps", "offered_pdr",
            "delivered_packets", "final_jain_fairness",
            "no_completed_packet_pct", "zero_delivered_bits_pct"]
    show = [m for m in show if m in both.columns]
    conditions = [c for c in both.condition.unique() if c != "original"]
    counts = sorted(new.n_user.unique())

    means = (both[both.n_user.isin(counts) | (both.condition == "original")]
             .groupby(["condition", "n_user", "method"])[show].mean())
    means.to_csv(out / "robustness_policy_means.csv")

    rows = []
    for condition in ["original", *conditions]:
        for n_user in counts:
            sub = both[(both.condition == condition) & (both.n_user == n_user)]
            pt = sub[sub.method == "PT_D3QN"].set_index("replicate_seed")
            eut = sub[sub.method == "EUT_D3QN"].set_index("replicate_seed")
            common = pt.index.intersection(eut.index)
            if len(common) < 2:
                continue
            for metric in show:
                rows.append({"condition": condition, "n_user": n_user,
                             "metric": metric,
                             "mean_PT": float(pt.loc[common, metric].mean()),
                             "mean_EUT": float(eut.loc[common, metric].mean()),
                             **sc.paired(pt.loc[common, metric],
                                         eut.loc[common, metric])})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "robustness_pt_vs_eut.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nOffered demand per episode (packets), by condition:")
    if "offered_packets" in both.columns:
        print(both[both.method == "PT_D3QN"].groupby(
            ["condition", "n_user"]).offered_packets.mean().round(2).to_string())
    print("\nPT-D3QN minus EUT-D3QN, paired over replicate seeds:")
    print(contrasts[contrasts.metric.isin(
        ["utility_shortfall", "delivered_packets", "no_completed_packet_pct"])]
        [["condition", "n_user", "metric", "mean_PT", "mean_EUT", "diff",
          "lo", "hi", "p_t", "p_sign"]].round(4).to_string(index=False))
    print("\nPolicy means in the new conditions:")
    print(means.loc[conditions].round(4).to_string())
    print(f"\nwritten: {out}/robustness_*.csv")


def cmd_selftest(args):
    load_simulator(args.simulator)
    cfg = v7.build_replicate_config(
        v7.Config(n_user=8, train_episodes=30, eval_episodes=40,
                  randomize_pt_profiles=True), 7)
    keys = ["utility_shortfall", "throughput_mbps", "offered_packets"]

    plain = pd.DataFrame(v7.evaluate("Random", cfg, None))[keys]
    same_cfg, base = condition_config(cfg, "small_packet", None, 1)
    with traffic_env("small_packet", None, base):
        again = pd.DataFrame(v7.evaluate("Random", same_cfg, None))[keys]
    identical = bool(np.array_equal(plain.to_numpy(), again.to_numpy()))
    print(f"1. packet ratio 1 reproduces the original scenarios: {identical}")

    small_cfg, base = condition_config(cfg, "small_packet", None, 4)
    seed = v7.evaluation_episode_seed(cfg, 0)
    e0 = v7.UAVEnv(cfg, seed); e0.reset()
    with traffic_env("small_packet", None, base):
        e1 = v7.UAVEnv(small_cfg, seed); e1.reset()
    same_bits = bool(np.array_equal(e0.queue, e1.queue))
    same_refs = bool(np.array_equal(e0.reference_utility_user,
                                    e1.reference_utility_user))
    same_geometry = bool(np.array_equal(e0.user_xy, e1.user_xy)
                         and np.array_equal(e0.shadow, e1.shadow))
    print(f"2. small packets: initial bits per user identical: {same_bits}; "
          f"references identical: {same_refs}; geometry and shadowing "
          f"identical: {same_geometry}")
    with traffic_env("small_packet", None, base):
        small = pd.DataFrame(v7.evaluate("Random", small_cfg, None))
    bits0 = (plain.offered_packets * cfg.packet_size_bits).mean() / 1e6
    bits1 = (small.offered_packets * small_cfg.packet_size_bits).mean() / 1e6
    print(f"3. offered volume per episode: {bits0:.1f} Mbit originally, "
          f"{bits1:.1f} Mbit with 1 Mbit packets (equal in expectation)")

    demand_users = 3
    m_cfg, base = condition_config(cfg, "matched", demand_users, 1)
    with traffic_env("matched", demand_users, base):
        matched = pd.DataFrame(v7.evaluate("Random", m_cfg, None))
        env = v7.UAVEnv(m_cfg, seed); env.reset()
    reference = v7.build_replicate_config(
        v7.Config(n_user=demand_users, train_episodes=30, eval_episodes=40,
                  randomize_pt_profiles=True), 7)
    small_pop = pd.DataFrame(v7.evaluate("Random", reference, None))
    capacity = int(cfg.max_queue_bits // cfg.packet_size_bits)
    print(f"4. matched demand: {matched.offered_packets.mean():.2f} packets "
          f"offered to {cfg.n_user} users against "
          f"{small_pop.offered_packets.mean():.2f} for {demand_users} users "
          f"originally; max initial per user {int(env.initial_packet_count.max())}"
          f" <= capacity {capacity}")
    restored = v7.UAVEnv is not TrafficEnv
    print(f"5. environment restored: {restored}")
    close = abs(matched.offered_packets.mean()
                - small_pop.offered_packets.mean()) < 1.0
    ok = identical and same_bits and same_refs and same_geometry and restored and close
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run")
    p.add_argument("--simulator", default=".")
    p.add_argument("--results", required=True)
    p.add_argument("--condition", required=True,
                   choices=("matched", "small_packet"))
    p.add_argument("--user-counts", type=int, nargs="+", required=True)
    p.add_argument("--demand-users", type=int, default=15)
    p.add_argument("--packet-ratio", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--methods", nargs="*")
    p.add_argument("--out", default="robustness")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze")
    p.add_argument("--seed-metrics", required=True)
    p.add_argument("--user-seed-metrics")
    p.add_argument("--out", default="robustness")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("selftest")
    p.add_argument("--simulator", default=".")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
