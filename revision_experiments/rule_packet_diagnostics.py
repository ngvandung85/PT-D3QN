#!/usr/bin/env python3
"""Packet-level diagnostics of the Completion_First_Fair rule.

The rule is not part of the archived slot-level records. This script evaluates
it with the unmodified simulator, records its actions, and passes them through
the same replay as replay_packet_diagnostics.py, so its diagnostics are
computed by the same code as those of the archived policies.

  python rule_packet_diagnostics.py --simulator . --results results --user-counts 5 15 30 45 60 --out packet_diagnostics_rule
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import run_scale_control as sc
import run_packet_baseline as pb
import replay_packet_diagnostics as rd


class _Sink:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append({k: row[k] for k in rd.COLUMNS if k in row})

    append = write


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--simulator", default=".")
    parser.add_argument("--results", required=True)
    parser.add_argument("--user-counts", type=int, nargs="+",
                        default=[5, 15, 30, 45, 60])
    parser.add_argument("--out", default="packet_diagnostics_rule")
    args = parser.parse_args()

    sc.load_simulator(args.simulator)
    pb.v7 = sc.v7
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_user in args.user_counts:
        for cfg in sc.load_configs(args.results, n_user):
            sink = _Sink()
            with pb.baseline_registered():
                episodes = pd.DataFrame(
                    sc.v7.evaluate(pb.NAME, cfg, None, step_sink=sink))
            row = rd.replay_group(sc.v7, cfg, pd.DataFrame(sink.rows))
            row.update(n_user=n_user, method=pb.NAME, replicate_seed=cfg.seed,
                       delay_episode_avg=float(
                           episodes.mean_packet_delay_s.mean()))
            rows.append(row)
            print(f"U={n_user} seed={cfg.seed}: mismatched slots "
                  f"{row['replay_mismatched_slots']}", flush=True)
            pd.DataFrame(rows).to_csv(
                out / "rule_service_diagnostics_seed.csv", index=False)


if __name__ == "__main__":
    main()
