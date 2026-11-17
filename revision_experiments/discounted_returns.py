#!/usr/bin/env python3
"""Realised discounted returns from the slot-level evaluation records.

For every policy and replicate seed, the mean over evaluation episodes of
    sum_t gamma^t r_t   (gamma = 0.99, slots with a service opportunity)
under the transformed reward (pt_subjective_utility / 1.6) and under the
linear reward (centered_eut_utility).

  python discounted_returns.py --results ../results --user-counts 5 15 30 45 60 --out ../results/packet_baseline/archived_discounted_returns_all.csv

Needs results/users_NNN/evaluation_step_raw.csv (release assets).
"""
import argparse
from pathlib import Path

import pandas as pd

GAMMA = 0.99
PT_REWARD_SCALE = 1.6
COLUMNS = ["method", "replicate_seed", "evaluation_episode", "slot",
           "service_opportunity", "pt_subjective_utility",
           "centered_eut_utility"]


def returns(path, n_user):
    total = None
    for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=1_000_000):
        chunk = chunk[chunk.service_opportunity > 0]
        weight = GAMMA ** chunk.slot.astype(int)
        chunk = chunk.assign(
            discounted_transformed_return=weight
            * chunk.pt_subjective_utility / PT_REWARD_SCALE,
            discounted_linear_return=weight * chunk.centered_eut_utility)
        part = chunk.groupby(
            ["method", "replicate_seed", "evaluation_episode"])[
            ["discounted_transformed_return",
             "discounted_linear_return"]].sum()
        total = part if total is None else total.add(part, fill_value=0)
    out = total.groupby(["method", "replicate_seed"]).mean().reset_index()
    out.insert(0, "n_user", n_user)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True)
    parser.add_argument("--user-counts", type=int, nargs="+",
                        default=[5, 15, 30, 45, 60])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frames = [returns(Path(args.results) / f"users_{u:03d}"
                      / "evaluation_step_raw.csv", u)
              for u in args.user_counts]
    pd.concat(frames).sort_values(
        ["n_user", "method", "replicate_seed"]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
