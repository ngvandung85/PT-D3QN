#!/usr/bin/env python3
"""Scale-matched linear reward control for the PT-D3QN / EUT-D3QN comparison.

Adds ONE training arm, EUT_SCALED_D3QN, whose reward is c_U * delta: the linear
reference-centred reward of EUT-D3QN multiplied by a frozen constant chosen so
that its standard deviation matches that of the transformed (PT) reward.

The simulator module pt_d3qn_uav.py is imported unmodified. The control is
implemented by a subclass of UAVEnv that is swapped in during TRAINING only.
Evaluation uses the original environment, the original evaluation scenarios
and the original evaluate() function, so the endpoint and the physical-service
metrics are computed exactly as for the archived arms.

Three steps, run in this order:

  1. calibrate   train the two original arms once on a calibration seed with
                 reward logging, compute c_U from TRAINING transitions only,
                 freeze it in calibration.json, and check that the rerun of
                 the original arms reproduces the archived seed metrics.
  2. run         train and evaluate EUT_SCALED_D3QN on every replicate seed.
  3. analyze     paired contrasts against the archived PT-D3QN and EUT-D3QN.

Example (Windows paths work the same way):

  python run_scale_control.py calibrate --simulator . --results results \
      --seed-metrics results/user_count_evaluation_seed_summary.csv \
      --user-counts 15 45 60 --out scale_control
  python run_scale_control.py run --simulator . --results results \
      --user-counts 15 45 60 --out scale_control
  python run_scale_control.py analyze \
      --seed-metrics results/user_count_evaluation_seed_summary.csv \
      --user-seed-metrics results/user_count_user_seed_summary.csv \
      --out scale_control

`run` can be split across terminals with --user-counts and --seeds; finished
units are skipped, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL = "EUT_SCALED_D3QN"
ORIGINAL_ARMS = ("EUT_D3QN", "PT_D3QN")
METRICS = [
    "utility_shortfall",          # designated endpoint
    "throughput_mbps",
    "offered_pdr",
    "delivered_packets",
    "final_jain_fairness",
    "idle_with_backlog_fraction",
    "total_energy_consumed_j",
]
REGRESSION_METRICS = ["utility_shortfall", "throughput_mbps", "offered_pdr"]
v7 = None            # the simulator module, set by load_simulator()
ScaledEnv = None     # subclass of v7.UAVEnv, built by load_simulator()


# ---------------------------------------------------------------- utilities
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_simulator(directory):
    """Import pt_d3qn_uav.py unmodified and build the training-only subclass."""
    global v7, ScaledEnv
    directory = Path(directory).resolve()
    module_path = directory / "pt_d3qn_uav.py"
    if not module_path.exists():
        raise SystemExit(f"pt_d3qn_uav.py not found in {directory}")
    sys.path.insert(0, str(directory))
    import pt_d3qn_uav as module
    v7 = module

    class _ScaledEnv(module.UAVEnv):
        """UAVEnv whose LINEAR reward is multiplied by SCALE.

        SCALE = 1.0 leaves every reward unchanged, so the class is then
        behaviourally identical to UAVEnv; this is the regression check.
        Rewards are logged on service-opportunity slots only.
        """
        SCALE = 1.0
        STATS = None          # RunningStats or None

        def outcome_reward(self, physical_utility, reward_model, user_index):
            reward = super().outcome_reward(
                physical_utility, reward_model, user_index)
            if reward_model == "eut":
                reward = type(self).SCALE * reward
            return reward

        def idle_reward_components(self):
            components = dict(super().idle_reward_components())
            components["eut_reward"] = (
                type(self).SCALE * components["eut_reward"])
            return components

        def step(self, action, reward_model="eut"):
            out = super().step(action, reward_model)
            stats = type(self).STATS
            info = out[-1]
            if stats is not None and info.get("service_opportunity"):
                delta = float(info["centered_eut_utility"])
                transformed = (float(info["pt_subjective_utility"])
                               / float(self.pt_reward_scale))
                stats.add(delta, transformed, float(out[1]))
            return out

    ScaledEnv = _ScaledEnv
    return module_path


class RunningStats:
    """Streaming moments of (linear delta, transformed reward, used reward)."""

    def __init__(self):
        self.n = 0
        self.s = np.zeros(3)
        self.ss = np.zeros(3)
        self.sabs = np.zeros(3)

    def add(self, delta, transformed, used):
        x = np.array([delta, transformed, used])
        self.n += 1
        self.s += x
        self.ss += x * x
        self.sabs += np.abs(x)

    def merge(self, other):
        self.n += other.n
        self.s += other.s
        self.ss += other.ss
        self.sabs += other.sabs
        return self

    def sd(self):
        if self.n < 2:
            return np.full(3, np.nan)
        var = (self.ss - self.s ** 2 / self.n) / (self.n - 1)
        return np.sqrt(np.maximum(var, 0.0))

    def summary(self):
        sd = self.sd()
        mean_abs = self.sabs / max(self.n, 1)
        return {
            "n_transitions": int(self.n),
            "sd_linear_delta": float(sd[0]),
            "sd_transformed_reward": float(sd[1]),
            "sd_reward_used": float(sd[2]),
            "mean_abs_linear_delta": float(mean_abs[0]),
            "mean_abs_transformed_reward": float(mean_abs[1]),
        }


@contextlib.contextmanager
def training_env(scale, stats):
    """Swap the scaled environment in for the duration of one training run."""
    original = v7.UAVEnv
    ScaledEnv.SCALE = float(scale)
    ScaledEnv.STATS = stats
    v7.UAVEnv = ScaledEnv
    try:
        yield
    finally:
        v7.UAVEnv = original
        ScaledEnv.SCALE = 1.0
        ScaledEnv.STATS = None


def load_configs(results_dir, n_user):
    """Resolved replicate configurations of the archived run for one count."""
    path = Path(results_dir) / f"users_{n_user:03d}" / "config.json"
    if not path.exists():
        raise SystemExit(f"Archived configuration not found: {path}")
    payload = json.loads(path.read_text())
    if payload.get("quick_run_only"):
        raise SystemExit(f"{path} is a quick run, not the archived experiment")
    keys = {f.name for f in fields(v7.Config)}
    configs = [v7.Config(**{k: v for k, v in row.items() if k in keys})
               for row in payload["resolved_replicate_configs"]]
    for cfg in configs:
        if cfg.n_user != n_user:
            raise SystemExit(f"{path}: n_user mismatch")
    return configs


def seed_means(rows):
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes("number")
    return numeric.mean(numeric_only=True)


def train_and_evaluate(arm, label, cfg, scale, stats):
    """Train with the (possibly scaled) reward; evaluate on the original env."""
    bank = v7.build_training_profile_bank(cfg)
    with training_env(scale, stats):
        agent, history = v7.train_agent(arm, cfg, bank)
    rows = v7.evaluate(label, cfg, agent)      # original UAVEnv, original scenarios
    return rows, history


# ---------------------------------------------------------------- calibrate
def cmd_calibrate(args):
    module_path = load_simulator(args.simulator)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "calibration.json"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists. The constants are frozen; "
                         "use --force only before any `run` unit has started.")
    archived = pd.read_csv(args.seed_metrics) if args.seed_metrics else None

    result = {
        "rule": ("c_U = SD(transformed reward) / SD(linear delta), both "
                 "evaluated on the same TRAINING transitions of the two "
                 "original arms on the calibration seed; service-opportunity "
                 "slots only; no evaluation transition is used"),
        "calibration_seed": args.calibration_seed,
        "simulator_sha256": sha256(module_path),
        "driver_sha256": sha256(__file__),
        "numpy": np.__version__, "pandas": pd.__version__,
        "python": sys.version.split()[0],
        "user_counts": {},
    }
    for n_user in args.user_counts:
        configs = load_configs(args.results, n_user)
        match = [c for c in configs if c.seed == args.calibration_seed]
        if not match:
            raise SystemExit(f"seed {args.calibration_seed} not in archive")
        cfg = match[0]
        pooled, per_arm, regression = RunningStats(), {}, {}
        for arm in ORIGINAL_ARMS:
            stats = RunningStats()
            t0 = time.time()
            rows, _ = train_and_evaluate(arm, arm, cfg, 1.0, stats)
            per_arm[arm] = stats.summary()
            pooled.merge(stats)
            means = seed_means(rows)
            if archived is not None:
                ref = archived[(archived.n_user == n_user)
                               & (archived.method == arm)
                               & (archived.replicate_seed == cfg.seed)]
                if len(ref) == 1:
                    regression[arm] = {
                        m: {"rerun": float(means[m]),
                            "archived": float(ref.iloc[0][m]),
                            "abs_diff": float(abs(means[m] - ref.iloc[0][m]))}
                        for m in REGRESSION_METRICS if m in ref.columns}
            print(f"[calibrate] U={n_user} {arm}: {stats.n} training "
                  f"transitions, {time.time() - t0:.0f}s", flush=True)
        summary = pooled.summary()
        scale = summary["sd_transformed_reward"] / summary["sd_linear_delta"]
        worst = max((v["abs_diff"] for arm in regression.values()
                     for v in arm.values()), default=float("nan"))
        result["user_counts"][str(n_user)] = {
            "scale_constant": float(scale),
            "pooled": summary, "per_arm": per_arm,
            "regression_vs_archive": regression,
            "regression_max_abs_diff": worst,
        }
        print(f"[calibrate] U={n_user}: c = {scale:.6f}   "
              f"(SD linear {summary['sd_linear_delta']:.4f} -> target "
              f"{summary['sd_transformed_reward']:.4f});  rerun vs archive "
              f"max |diff| = {worst:.3g}", flush=True)
        if archived is not None and not (worst <= args.regression_tolerance):
            print("  WARNING: the rerun of the original arms does not "
                  "reproduce the archive within tolerance. Do not pair the "
                  "control with archived arms until this is explained "
                  "(library versions? modified simulator?).", flush=True)
    payload = json.dumps(result, indent=2, sort_keys=True)
    target.write_text(payload)
    (out / "calibration.sha256").write_text(
        hashlib.sha256(payload.encode()).hexdigest() + "\n")
    print(f"frozen: {target}")


# ---------------------------------------------------------------------- run
def cmd_run(args):
    module_path = load_simulator(args.simulator)
    out = Path(args.out)
    cal_path = out / "calibration.json"
    if not cal_path.exists():
        raise SystemExit("Run `calibrate` first.")
    cal_text = cal_path.read_text()
    recorded = (out / "calibration.sha256").read_text().strip()
    if hashlib.sha256(cal_text.encode()).hexdigest() != recorded:
        raise SystemExit("calibration.json changed after it was frozen.")
    cal = json.loads(cal_text)
    if cal["simulator_sha256"] != sha256(module_path):
        raise SystemExit("Simulator changed after calibration.")
    units = out / "units"
    units.mkdir(exist_ok=True)

    for n_user in args.user_counts:
        if str(n_user) not in cal["user_counts"]:
            raise SystemExit(f"U={n_user} was not calibrated")
        scale = cal["user_counts"][str(n_user)]["scale_constant"]
        for cfg in load_configs(args.results, n_user):
            if args.seeds and cfg.seed not in args.seeds:
                continue
            stem = units / f"u{n_user:03d}_seed{cfg.seed:03d}"
            done = stem.with_suffix(".done")
            if done.exists():
                print(f"[run] U={n_user} seed={cfg.seed}: already done")
                continue
            stats = RunningStats()
            t0 = time.time()
            rows, history = train_and_evaluate(
                "EUT_D3QN", CONTROL, cfg, scale, stats)
            pd.DataFrame(rows).to_csv(
                stem.with_name(stem.name + "_episodes.csv"), index=False)
            pd.DataFrame(history).to_csv(
                stem.with_name(stem.name + "_training.csv"), index=False)
            meta = {"n_user": n_user, "replicate_seed": cfg.seed,
                    "method": CONTROL, "scale_constant": scale,
                    "train_episodes": cfg.train_episodes,
                    "eval_episodes": cfg.eval_episodes,
                    "training_reward": stats.summary(),
                    "seconds": time.time() - t0}
            done.write_text(json.dumps(meta, indent=2))
            print(f"[run] U={n_user} seed={cfg.seed}: {meta['seconds']:.0f}s, "
                  f"training-reward SD {meta['training_reward']['sd_reward_used']:.4f}",
                  flush=True)


# ------------------------------------------------------------------ analyze
def exact_sign_flip(diff):
    diff = np.asarray(diff, float)
    observed = abs(diff.mean())
    # Relative tolerance: the identity assignment must always be counted,
    # whatever the magnitude of the differences.
    threshold = observed * (1.0 - 1e-12) - 1e-15
    count = sum(abs(np.dot(signs, diff)) / len(diff) >= threshold
                for signs in itertools.product((1.0, -1.0), repeat=len(diff)))
    return count / 2 ** len(diff)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired(a, b):
    from scipy import stats as st
    d = (a - b).dropna().to_numpy()
    n, mean = len(d), float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return dict(n=n, diff=mean, lo=mean, hi=mean, p_t=np.nan, p_sign=np.nan)
    half = st.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return dict(n=n, diff=mean, lo=mean - half, hi=mean + half,
                p_t=float(st.ttest_1samp(d, 0.0).pvalue),
                p_sign=float(exact_sign_flip(d)))


def cmd_analyze(args):
    out = Path(args.out)
    frames, training = [], []
    for done in sorted((out / "units").glob("*.done")):
        meta = json.loads(done.read_text())
        episodes = pd.read_csv(done.with_name(done.stem + "_episodes.csv"))
        row = episodes.select_dtypes("number").mean(numeric_only=True)
        row["n_user"], row["replicate_seed"] = meta["n_user"], meta["replicate_seed"]
        row["method"] = CONTROL
        frames.append(row)
        training.append({"n_user": meta["n_user"],
                         "replicate_seed": meta["replicate_seed"],
                         "scale_constant": meta["scale_constant"],
                         **meta["training_reward"]})
    if not frames:
        raise SystemExit("No finished units.")
    control = pd.DataFrame(frames)
    control.to_csv(out / "scale_control_seed_metrics.csv", index=False)
    pd.DataFrame(training).to_csv(out / "scale_control_training_reward.csv",
                                  index=False)

    archived = pd.read_csv(args.seed_metrics)
    if args.user_seed_metrics and "delivered_packets" not in archived.columns:
        # Completed packets per episode = sum over users of the per-user means.
        users = pd.read_csv(args.user_seed_metrics)
        packets = (users.groupby(["n_user", "method", "replicate_seed"])
                   .delivered_packets.sum().reset_index())
        archived = archived.merge(
            packets, on=["n_user", "method", "replicate_seed"], how="left")
    contrasts = [("PT_D3QN", CONTROL, "shape at matched scale: PT - scaled linear"),
                 (CONTROL, "EUT_D3QN", "scale: scaled linear - linear"),
                 ("PT_D3QN", "EUT_D3QN", "bundled: PT - linear (archived)")]
    rows = []
    for n_user in sorted(control.n_user.unique()):
        table = {CONTROL: control[control.n_user == n_user]
                 .set_index("replicate_seed")}
        for arm in ORIGINAL_ARMS:
            table[arm] = archived[(archived.n_user == n_user)
                                  & (archived.method == arm)
                                  ].set_index("replicate_seed")
        seeds = table[CONTROL].index
        if len(seeds) != len(table["EUT_D3QN"].index):
            print(f"  note: U={n_user} has {len(seeds)} control seeds; "
                  "contrasts use the seeds available in all arms")
        for first, second, label in contrasts:
            for metric in METRICS:
                if any(metric not in table[k].columns for k in (first, second)):
                    continue
                a = table[first].loc[table[first].index.intersection(seeds), metric]
                b = table[second].loc[table[second].index.intersection(seeds), metric]
                rows.append({"n_user": n_user, "contrast": label,
                             "metric": metric,
                             "mean_first": float(a.mean()),
                             "mean_second": float(b.mean()),
                             **paired(a, b)})
    result = pd.DataFrame(rows)
    # Holm within each contrast, across the selected user counts, endpoint only.
    result["holm_t"], result["holm_sign"] = np.nan, np.nan
    primary = result.metric == "utility_shortfall"
    for label in result.contrast.unique():
        mask = primary & (result.contrast == label) & result.p_t.notna()
        if mask.sum():
            result.loc[mask, "holm_t"] = holm(result.loc[mask, "p_t"])
            result.loc[mask, "holm_sign"] = holm(result.loc[mask, "p_sign"])
    result.to_csv(out / "scale_control_contrasts.csv", index=False)

    pd.set_option("display.width", 200)
    show = result[primary][["n_user", "contrast", "mean_first", "mean_second",
                            "diff", "lo", "hi", "p_t", "holm_t",
                            "p_sign", "holm_sign"]]
    print("\nDesignated endpoint (utility shortfall):")
    print(show.round(4).to_string(index=False))
    print("\nTraining-reward SD achieved by the control (target = transformed SD):")
    print(pd.DataFrame(training).groupby("n_user")[
        ["scale_constant", "sd_reward_used"]].mean().round(4).to_string())
    print(f"\nwritten: {out / 'scale_control_contrasts.csv'}")


# --------------------------------------------------------------- self-test
def cmd_selftest(args):
    """Small end-to-end check that needs no archived results."""
    load_simulator(args.simulator)
    base = v7.Config(n_user=5, train_episodes=60, eval_episodes=3)
    cfg = v7.build_replicate_config(base, 7)
    keys = ["utility_shortfall", "throughput_mbps", "offered_pdr",
            "delivered_packets"]

    bank = v7.build_training_profile_bank(cfg)
    agent, _ = v7.train_agent("EUT_D3QN", cfg, bank)
    plain = pd.DataFrame(v7.evaluate("EUT_D3QN", cfg, agent))[keys]

    stats = RunningStats()
    rows, _ = train_and_evaluate("EUT_D3QN", CONTROL, cfg, 1.0, stats)
    unit = pd.DataFrame(rows)[keys]
    identical = bool(np.array_equal(plain.to_numpy(), unit.to_numpy()))
    print(f"1. scale 1.0 reproduces EUT_D3QN exactly: {identical}")

    s = stats.summary()
    scale = s["sd_transformed_reward"] / s["sd_linear_delta"]
    stats2 = RunningStats()
    rows2, _ = train_and_evaluate("EUT_D3QN", CONTROL, cfg, scale, stats2)
    achieved = stats2.summary()
    ratio = achieved["sd_reward_used"] / (scale * achieved["sd_linear_delta"])
    print(f"2. scale {scale:.4f}: reward used = c * delta on every logged "
          f"slot (SD ratio {ratio:.6f})")
    changed = not np.array_equal(unit.to_numpy(), pd.DataFrame(rows2)[keys].to_numpy())
    print(f"3. a non-unit scale changes the learned policy: {changed}")
    print(f"4. environment restored after training: {v7.UAVEnv is not ScaledEnv}")
    d = np.array([-0.0814, 0.0019, 0.0005, -0.1162, 0.0032,
                  -0.1009, -0.1173, -0.1123, 0.0005, -0.0025])
    print(f"5. exact sign-flip on the published U=45 differences: "
          f"{exact_sign_flip(d):.4f} (published 0.0488)")
    ok = identical and abs(ratio - 1) < 1e-9 and v7.UAVEnv is not ScaledEnv
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, results=True):
        p.add_argument("--simulator", default=".",
                       help="folder containing the unmodified pt_d3qn_uav.py")
        p.add_argument("--out", default="scale_control")
        if results:
            p.add_argument("--results", required=True,
                           help="archived results folder with users_NNN/config.json")
            p.add_argument("--user-counts", type=int, nargs="+", required=True)

    p = sub.add_parser("calibrate"); common(p)
    p.add_argument("--seed-metrics", help="archived seed-level metrics CSV "
                   "(n_user, method, replicate_seed, ...) for the regression check")
    p.add_argument("--calibration-seed", type=int, default=7)
    p.add_argument("--regression-tolerance", type=float, default=1e-9)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("run"); common(p)
    p.add_argument("--seeds", type=int, nargs="*")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze")
    p.add_argument("--seed-metrics", required=True)
    p.add_argument("--user-seed-metrics",
                   help="archived user-level seed summary; supplies completed "
                        "packets when the seed-level file has no such column")
    p.add_argument("--out", default="scale_control")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("selftest"); common(p, results=False)
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
