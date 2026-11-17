#!/usr/bin/env python3
"""Validate a completed single run or user-count sweep before analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from pt_d3qn_uav import (
    ALL_METHODS,
    Config,
    OUTPUT_SCHEMA_VERSION,
    PRIMARY_ENDPOINT,
    RELEASE_VERSION,
    UAVEnv,
    evaluation_episode_seed,
)


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_jitter_in_bounds(training, evaluation, jitter, atol=1e-12):
    """Validate multiplicative jitter without dividing by a zero reference."""
    training = np.asarray(training, dtype=float)
    evaluation = np.asarray(evaluation, dtype=float)
    return bool(
        np.all(
            np.abs(training - evaluation)
            <= float(jitter) * np.abs(evaluation) + float(atol)
        )
    )


def validate_folder(folder, code_hash):
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    seeds = cfg["run_seeds"]
    n_user = int(cfg["n_user"])
    eval_episodes = int(cfg["eval_episodes"])
    horizon = int(cfg["horizon"])
    expected_episodes = len(ALL_METHODS) * len(seeds) * eval_episodes

    episode = pd.read_csv(folder / "evaluation_raw.csv")
    user = pd.read_csv(folder / "evaluation_user_raw.csv")
    selection = pd.read_csv(folder / "evaluation_selection_raw.csv")
    profiles = pd.read_csv(folder / "pt_user_profiles.csv")
    training_profiles = pd.read_csv(folder / "training_pt_profiles.csv")
    paired = pd.read_csv(folder / "paired_pt_vs_eut_d3qn.csv")
    complexity = pd.read_csv(folder / "algorithm_complexity_raw.csv")
    reproducibility = json.loads(
        (folder / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )

    require(
        cfg["output_schema_version"] == OUTPUT_SCHEMA_VERSION,
        f"{folder}: wrong schema",
    )
    require(cfg["release_version"] == RELEASE_VERSION, f"{folder}: wrong release")
    require(len(episode) == expected_episodes, f"{folder}: episode row count")
    require(
        len(user) == expected_episodes * n_user,
        f"{folder}: per-user row count",
    )
    require(
        len(profiles) == len(seeds) * n_user,
        f"{folder}: evaluation-profile row count",
    )
    require(
        len(training_profiles)
        == len(seeds) * int(cfg["training_profile_count"]) * n_user,
        f"{folder}: training-profile row count",
    )
    require(
        len(complexity) == len(ALL_METHODS) * len(seeds),
        f"{folder}: complexity row count",
    )
    require(
        set(episode.method) == set(ALL_METHODS),
        f"{folder}: method set",
    )
    require(
        reproducibility["code_sha256"] == code_hash,
        f"{folder}: code hash differs from validator package",
    )

    require(
        np.allclose(
            episode.offered_pdr,
            episode.delivered_packets / episode.offered_packets,
        ),
        f"{folder}: offered PDR reconstruction",
    )
    require(
        np.allclose(episode.pdr, episode.offered_pdr),
        f"{folder}: pdr alias",
    )
    require(
        (
            episode.admitted_packets + episode.dropped_packets
            == episode.offered_packets
        ).all(),
        f"{folder}: episode packet conservation",
    )
    require(
        (
            user.admitted_packets + user.dropped_packets
            == user.offered_packets
        ).all(),
        f"{folder}: user packet conservation",
    )
    require(
        episode.final_jain_fairness.dropna().between(0, 1).all(),
        f"{folder}: Jain bounds",
    )
    require(
        selection.groupby(
            ["replicate_seed", "method", "evaluation_episode", "dimension"]
        ).selection_count.sum().eq(horizon).all(),
        f"{folder}: selection marginals do not sum to horizon",
    )
    require(
        paired.is_prespecified_primary_endpoint.sum() == 1
        and paired.loc[
            paired.is_prespecified_primary_endpoint, "metric"
        ].iloc[0]
        == PRIMARY_ENDPOINT,
        f"{folder}: primary endpoint flag",
    )
    require(
        {
            "exact_sign_flip_p_value_two_sided",
            "holm_adjusted_paired_t_p_value",
            "holm_adjusted_sign_flip_p_value",
            "holm_secondary_family_member",
            "exact_sign_flip_assumption",
        }
        <= set(paired.columns),
        f"{folder}: paired inference columns",
    )
    if int(cfg["training_profile_count"]) > 1:
        require(
            training_profiles.evaluation_profile_held_out.all(),
            f"{folder}: evaluation profile is not held out",
        )
        joined = training_profiles.merge(
            profiles[
                ["replicate_seed", "user_index", "reference_utility_u"]
            ],
            on=["replicate_seed", "user_index"],
            suffixes=("_training", "_evaluation"),
            validate="many_to_one",
        )
        jitter = float(cfg["training_reference_jitter_fraction"])
        require(
            reference_jitter_in_bounds(
                joined.reference_utility_u_training,
                joined.reference_utility_u_evaluation,
                jitter,
            ),
            f"{folder}: training U_ref perturbation is out of bounds",
        )

    if bool(cfg["save_step_details"]):
        step = pd.read_csv(folder / "evaluation_step_raw.csv")
        expected_steps = expected_episodes * horizon
        require(len(step) == expected_steps, f"{folder}: slot row count")
        common_keys = ["replicate_seed", "evaluation_episode", "slot"]
        require(
            step.groupby(common_keys).outcome_uniform.nunique().eq(1).all(),
            f"{folder}: common outcome draws",
        )
        require(
            step.groupby(common_keys).packets_generated.nunique().eq(1).all(),
            f"{folder}: common arrival counts",
        )
        arrival_signatures = step.assign(
            arrival_events_by_user=step.arrival_events_by_user.fillna("")
        )
        require(
            arrival_signatures.groupby(common_keys)
            .arrival_events_by_user.nunique(dropna=False)
            .eq(1)
            .all(),
            f"{folder}: common per-user arrival vectors",
        )
        attempted = step[step.attempted == 1]
        require(
            (
                (attempted.outcome_uniform < attempted.success_probability)
                .astype(float)
                .to_numpy()
                == attempted.transmission_success.to_numpy()
            ).all(),
            f"{folder}: link-outcome mapping",
        )
        idle = step[step.is_idle == 1]
        require(
            (idle[["selected_uav", "selected_channel", "selected_user", "selected_mode"]] == -1)
            .all(axis=None),
            f"{folder}: idle sentinel",
        )
        idle_with_backlog = idle[idle.service_opportunity == 1]
        require(
            idle_with_backlog[
                [
                    "centered_eut_utility",
                    "pt_subjective_utility",
                    "reference_satisfaction",
                    "loss_event",
                    "utility_shortfall",
                ]
            ]
            .notna()
            .all(axis=None),
            f"{folder}: idle-with-backlog downside metrics are missing",
        )
        require(
            idle[
                [
                    "attempt_reference_satisfaction",
                    "attempt_loss_event",
                    "attempt_utility_shortfall",
                ]
            ]
            .isna()
            .all(axis=None),
            f"{folder}: attempt-conditional idle metrics must be undefined",
        )
        no_opportunity = step[step.service_opportunity == 0]
        require(
            no_opportunity[
                ["reference_satisfaction", "loss_event", "utility_shortfall"]
            ]
            .isna()
            .all(axis=None),
            f"{folder}: no-demand slots must not be classified as losses",
        )
        opportunities = step[step.service_opportunity == 1]
        require(
            np.allclose(
                opportunities.reference_satisfaction
                + opportunities.loss_event,
                1.0,
            )
            and np.allclose(
                opportunities.utility_shortfall,
                np.maximum(-opportunities.centered_eut_utility, 0.0),
            ),
            f"{folder}: opportunity-level downside identities",
        )
        group_keys = ["replicate_seed", "method", "evaluation_episode"]
        episode_indexed = episode.set_index(group_keys).sort_index()
        step_grouped = step.groupby(group_keys).agg(
            generated=("packets_generated", "sum"),
            delivered=("packets_delivered", "sum"),
            energy=("energy_j", "sum"),
            bits=("delivered_bits", "sum"),
        ).sort_index()
        require(
            np.array_equal(
                step_grouped.generated.to_numpy(),
                episode_indexed.generated_arrival_packets.to_numpy(),
            ),
            f"{folder}: generated-packet reconstruction",
        )
        require(
            np.array_equal(
                step_grouped.delivered.to_numpy(),
                episode_indexed.delivered_packets.to_numpy(),
            ),
            f"{folder}: delivered-packet reconstruction",
        )
        require(
            np.allclose(
                step_grouped.energy,
                episode_indexed.total_energy_consumed_j,
            ),
            f"{folder}: energy reconstruction",
        )
        require(
            np.allclose(
                step_grouped.bits.to_numpy()
                / (horizon * float(cfg["slot_duration_s"]) * 1e6),
                episode_indexed.throughput_mbps.to_numpy(),
            ),
            f"{folder}: throughput reconstruction",
        )

    return n_user


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Single-run folder or sweep root")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    require(root.is_dir(), f"Output folder does not exist: {root}")
    code_hash = sha256_file(Path(__file__).with_name("pt_d3qn_uav.py"))
    folders = sorted(root.glob("users_[0-9][0-9][0-9]"))
    if not folders:
        folders = [root]
    counts = [validate_folder(folder, code_hash) for folder in folders]

    if len(folders) > 1:
        config_fields = {field.name for field in fields(Config)}
        configs_by_count = {}
        for folder in folders:
            payload = json.loads((folder / "config.json").read_text())
            configs_by_count[int(payload["n_user"])] = {
                int(item["seed"]): Config(
                    **{
                        key: value
                        for key, value in item.items()
                        if key in config_fields
                    }
                )
                for item in payload["resolved_replicate_configs"]
            }
        smallest_count = min(configs_by_count)
        for seed, small_cfg in configs_by_count[smallest_count].items():
            for episode in range(min(small_cfg.eval_episodes, 3)):
                scenario_seed = evaluation_episode_seed(small_cfg, episode)
                for count, by_seed in configs_by_count.items():
                    small_env = UAVEnv(small_cfg, scenario_seed)
                    small_env.reset()
                    large_cfg = by_seed[seed]
                    large_env = UAVEnv(large_cfg, scenario_seed)
                    large_env.reset()
                    require(
                        np.array_equal(
                            small_env.user_xy,
                            large_env.user_xy[:smallest_count],
                        )
                        and np.array_equal(
                            small_env.initial_packet_count,
                            large_env.initial_packet_count[:smallest_count],
                        )
                        and np.array_equal(
                            small_env.shadow,
                            large_env.shadow[:, :, :smallest_count],
                        ),
                        "Cross-count evaluation scenarios are not nested: "
                        f"seed={seed}, episode={episode}, n_user={count}",
                    )
                    require(
                        np.array_equal(
                            small_env.base_path_loss_db,
                            large_env.base_path_loss_db[:, :smallest_count],
                        ),
                        "Cross-count base link states are not nested: "
                        f"seed={seed}, episode={episode}, n_user={count}",
                    )
                    for slot in range(min(small_cfg.horizon, 5)):
                        _, _, _, small_metrics = small_env.step(
                            small_env.idle_action, "eut"
                        )
                        _, _, _, large_metrics = large_env.step(
                            large_env.idle_action, "eut"
                        )
                        require(
                            small_metrics["outcome_uniform"]
                            == large_metrics["outcome_uniform"]
                            and np.array_equal(
                                small_env.generated_arrival_packets,
                                large_env.generated_arrival_packets[
                                    :smallest_count
                                ],
                            )
                            and np.array_equal(
                                small_env.shadow,
                                large_env.shadow[:, :, :smallest_count],
                            ),
                            "Cross-count dynamic streams are not nested: "
                            f"seed={seed}, episode={episode}, slot={slot}, "
                            f"n_user={count}",
                        )

    nesting_path = root / "user_count_profile_nesting_check.csv"
    if nesting_path.is_file():
        nesting = pd.read_csv(nesting_path)
        require(
            nesting.all_sampled_pt_traits_nested_match.all()
            and nesting.reference_utility_nested_match.all()
            and
            nesting.all_behavioral_parameters_nested_match.all(),
            "Cross-count behavioral profiles are not nested",
        )
    training_nesting_path = (
        root / "user_count_training_profile_nesting_check.csv"
    )
    if training_nesting_path.is_file():
        training_nesting = pd.read_csv(training_nesting_path)
        require(
            training_nesting.all_training_behavioral_parameters_nested_match.all(),
            "Cross-count training behavioral profiles are not nested",
        )
    print(
        f"PASS: validated {len(folders)} output folder(s); "
        f"user counts={counts}"
    )


if __name__ == "__main__":
    main()
