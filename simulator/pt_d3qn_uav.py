#!/usr/bin/env python3
"""Fair comparison of nine UAV resource-selection policies.

The decision is a discrete tuple ``(UAV, channel, user, transmission mode)``.
All policies share one physical environment, feasible-action mask, scenario
stream, training budget, and evaluation protocol.

The key controlled comparison is EUT-D3QN versus PT-D3QN. Both agents observe
the same realized success/failure outcome and the same per-user behavioral
profile. EUT-D3QN receives utility centered on the selected user's reference,
while PT-D3QN applies that user's prospect-theory value function to the same
outcome. No probability weighting is used: this follows the PT model and
reference implementation supplied with Mitsis et al., Sensors 2020.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_rel
from scipy.stats import truncnorm


RELEASE_VERSION = "6.1.0"
OUTPUT_SCHEMA_VERSION = 4


def _resolve_user_profile(values, fallback, n_user, name):
    """Return one finite floating-point PT parameter per user."""
    if values is None:
        result = np.full(n_user, fallback, dtype=float)
    else:
        result = np.asarray(values, dtype=float)
        if result.shape != (n_user,):
            raise ValueError(f"{name} must contain exactly n_user={n_user} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _derived_seed(base_seed: int, *parts: int) -> int:
    """Derive a reproducible uint32 seed from independent namespace parts."""
    values = (base_seed, *parts)
    if any(int(value) < 0 for value in values):
        raise ValueError("Seed components must be non-negative integers")
    sequence = np.random.SeedSequence([int(value) for value in values])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _truncated_normal_sample(rng, center, std, lower, upper, size, name):
    """Sample without clipping atoms at the user-declared bounds."""
    values = (center, std, lower, upper)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} distribution values must be finite")
    if std <= 0 or lower >= upper or not lower <= center <= upper:
        raise ValueError(
            f"{name} requires std>0, lower<upper and center within bounds"
        )
    standardized_lower = (lower - center) / std
    standardized_upper = (upper - center) / std
    return truncnorm.rvs(
        standardized_lower,
        standardized_upper,
        loc=center,
        scale=std,
        size=size,
        random_state=rng,
    )


def _sparse_count_signature(values):
    """Compact, lossless per-user integer vector for slot-level CSV output."""
    return ";".join(
        f"{index}:{int(value)}"
        for index, value in enumerate(np.asarray(values))
        if int(value) != 0
    )


@dataclass
class Config:
    # Discrete decision space.
    n_uav: int = 3
    n_channel: int = 4
    n_user: int = 6
    n_mode: int = 3

    # Physical scenario (SI units unless explicitly stated).
    area_side_m: float = 500.0
    uav_altitude_m: float = 100.0
    user_height_m: float = 1.5
    carrier_frequency_hz: float = 2.0e9
    system_bandwidth_hz: float = 10.0e6
    noise_psd_dbm_hz: float = -174.0
    receiver_noise_figure_db: float = 9.0
    los_a: float = 9.61
    los_b: float = 0.16
    eta_los_db: float = 1.0
    eta_nlos_db: float = 20.0
    shadow_sigma_db: float = 4.0
    shadow_rho: float = 0.95
    slot_duration_s: float = 0.10
    packet_size_bits: int = 4_000_000
    arrival_rate_pps: float = 0.25
    max_queue_bits: float = 20.0e6
    # Per-UAV transmit-radio budget. At 2 J the highest-power mode cannot be
    # used for all 80 slots, so the energy constraint is experimentally active.
    initial_radio_energy_j: float = 2.0
    min_tx_power_dbm: float = 10.0
    max_tx_power_dbm: float = 30.0
    min_spectral_eff_bps_hz: float = 0.5
    max_spectral_eff_bps_hz: float = 4.0
    qos_rate_mbps: float = 1.0
    qos_reliability: float = 0.99

    # Dimensionless physical utility. Weights must sum to one. The historical
    # field name ``w_fairness`` is retained for config compatibility, but this
    # is a custom service-equity shaping term, not Jain's fairness index.
    w_rate: float = 0.45
    w_reliability: float = 0.25
    w_fairness: float = 0.20
    w_energy: float = 0.10

    # RL protocol shared by every learning agent.
    horizon: int = 80
    train_episodes: int = 1500
    eval_episodes: int = 100
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50_000
    warmup: int = 1000
    update_frequency: int = 4
    target_update: int = 500
    hidden: int = 128
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995

    # Prospect Theory defaults used in the supplied UAV-MEC paper/code.
    # Optional tuples override the corresponding default for each user. Their
    # length must equal n_user. None broadcasts the scalar/default reference.
    alpha_gain: float = 0.20
    beta_loss: float = 0.20
    loss_aversion: float = 1.20
    reference_utility: float | None = None
    alpha_gain_by_user: tuple[float, ...] | None = None
    beta_loss_by_user: tuple[float, ...] | None = None
    loss_aversion_by_user: tuple[float, ...] | None = None
    reference_utility_by_user: tuple[float, ...] | None = None

    # Optional heterogeneous-population sensitivity design. The centers
    # alpha=beta=0.2 and k=1.2 are source-backed; every dispersion/bound below
    # is an explicit study assumption, not an empirically fitted population.
    randomize_pt_profiles: bool = False
    pt_profile_seed_base: int = 300_000
    alpha_gain_std: float = 0.03
    alpha_gain_lower: float = 0.10
    alpha_gain_upper: float = 0.30
    beta_loss_std: float = 0.03
    beta_loss_lower: float = 0.10
    beta_loss_upper: float = 0.30
    loss_aversion_std: float = 0.15
    loss_aversion_lower: float = 1.00
    loss_aversion_upper: float = 1.60

    # A fixed, declared scale keeps PT rewards comparable across user counts.
    # It must not depend on the largest sampled k_u in a particular population.
    pt_reward_scale_constant: float = 1.60

    # Detailed evaluation rows are streamed to CSV rather than retained in RAM.
    save_step_details: bool = True

    # Optional behavioral-domain randomization. With count > 1, RL agents train
    # on a separate profile bank and are evaluated on the held-out replicate
    # profile generated from pt_profile_seed_base.
    training_profile_count: int = 1
    training_profile_seed_base: int = 350_000
    training_reference_jitter_fraction: float = 0.20

    # Reference choices: qos_aspiration, random_policy_calibration, explicit.
    # Calibration uses independent pre-training scenarios and never evaluation
    # data. The resolved per-user values are then frozen for the full replicate.
    reference_utility_mode: str = "qos_aspiration"
    reference_calibration_episodes: int = 50
    reference_calibration_seed_base: int = 200_000
    # Resolved metadata: population size used to calibrate U_ref. In a
    # user-count sweep this is the maximum count so the same user's reference
    # remains fixed across every smaller prefix experiment.
    reference_calibration_population_size: int | None = None
    evaluation_seed_base: int = 100_000
    seed: int = 7


class UAVEnv:
    """Unit-consistent downlink UAV-BS resource-selection environment."""

    def __init__(self, cfg: Config, seed: int | None = None):
        self.c = cfg
        count_values = (cfg.n_uav, cfg.n_channel, cfg.n_user, cfg.n_mode)
        if min(count_values) < 1:
            raise ValueError("UAV, channel, user and mode counts must be >= 1")
        if min(
            cfg.area_side_m,
            cfg.uav_altitude_m,
            cfg.carrier_frequency_hz,
            cfg.system_bandwidth_hz,
            cfg.slot_duration_s,
            cfg.packet_size_bits,
            cfg.max_queue_bits,
            cfg.initial_radio_energy_j,
            cfg.horizon,
        ) <= 0:
            raise ValueError(
                "Physical dimensions, capacities and horizon must be positive"
            )
        if cfg.user_height_m < 0 or cfg.uav_altitude_m <= cfg.user_height_m:
            raise ValueError(
                "UAV altitude must exceed the non-negative user height"
            )
        if cfg.arrival_rate_pps < 0:
            raise ValueError("arrival_rate_pps must be non-negative")
        if cfg.max_queue_bits + 1e-12 < cfg.packet_size_bits:
            raise ValueError(
                "max_queue_bits must hold at least one whole packet"
            )
        if not 0 <= cfg.shadow_rho < 1:
            raise ValueError("shadow_rho must be in [0, 1)")
        if cfg.shadow_sigma_db < 0:
            raise ValueError("shadow_sigma_db must be non-negative")
        if cfg.min_tx_power_dbm > cfg.max_tx_power_dbm:
            raise ValueError(
                "min_tx_power_dbm must not exceed max_tx_power_dbm"
            )
        if (
            cfg.min_spectral_eff_bps_hz <= 0
            or cfg.min_spectral_eff_bps_hz > cfg.max_spectral_eff_bps_hz
        ):
            raise ValueError(
                "Spectral-efficiency bounds must be positive and ordered"
            )
        if cfg.qos_rate_mbps < 0 or not 0 <= cfg.qos_reliability <= 1:
            raise ValueError(
                "QoS rate/reliability values are outside their domains"
            )
        weights = (
            cfg.w_rate,
            cfg.w_reliability,
            cfg.w_fairness,
            cfg.w_energy,
        )
        if min(weights) < 0 or not np.isclose(sum(weights), 1.0):
            raise ValueError(
                "Utility weights must be non-negative and sum to one"
            )
        if not (0 < cfg.alpha_gain <= 1 and 0 < cfg.beta_loss <= 1):
            raise ValueError("PT curvature parameters must be in (0, 1]")
        if cfg.loss_aversion <= 0:
            raise ValueError("loss_aversion must be positive")
        if (
            not np.isfinite(cfg.pt_reward_scale_constant)
            or cfg.pt_reward_scale_constant <= 0
        ):
            raise ValueError("pt_reward_scale_constant must be positive")
        if cfg.reference_utility is not None and not np.isfinite(
            cfg.reference_utility
        ):
            raise ValueError("reference_utility must be finite when specified")
        if cfg.reference_utility_mode not in {
            "qos_aspiration",
            "random_policy_calibration",
            "explicit",
        }:
            raise ValueError(
                "reference_utility_mode must be qos_aspiration, "
                "random_policy_calibration or explicit"
            )
        if cfg.reference_calibration_episodes < 1:
            raise ValueError("reference_calibration_episodes must be >= 1")
        if (
            cfg.reference_calibration_population_size is not None
            and cfg.reference_calibration_population_size < cfg.n_user
        ):
            raise ValueError(
                "reference_calibration_population_size must be >= n_user"
            )
        if min(
            cfg.seed,
            cfg.evaluation_seed_base,
            cfg.pt_profile_seed_base,
            cfg.training_profile_seed_base,
            cfg.reference_calibration_seed_base,
        ) < 0:
            raise ValueError("All seed values must be non-negative")
        if cfg.training_profile_count < 1:
            raise ValueError("training_profile_count must be >= 1")
        if not 0 <= cfg.training_reference_jitter_fraction < 1:
            raise ValueError(
                "training_reference_jitter_fraction must be in [0, 1)"
            )
        if cfg.train_episodes < 1 or cfg.eval_episodes < 1:
            raise ValueError("Training and evaluation episode counts must be >= 1")
        if not 0 <= cfg.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not np.isfinite(cfg.lr) or cfg.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if cfg.batch_size < 1 or cfg.replay_size < 1 or cfg.warmup < 0:
            raise ValueError(
                "batch_size/replay_size must be positive and warmup non-negative"
            )
        if cfg.hidden < 1 or cfg.target_update < 1:
            raise ValueError("hidden and target_update must be >= 1")
        epsilon_values = (cfg.eps_start, cfg.eps_end, cfg.eps_decay)
        if not np.all(np.isfinite(epsilon_values)) or not (
            0 <= cfg.eps_end <= cfg.eps_start <= 1 and 0 < cfg.eps_decay <= 1
        ):
            raise ValueError(
                "epsilon requires 0 <= eps_end <= eps_start <= 1 and "
                "0 < eps_decay <= 1"
            )
        if cfg.replay_size < max(cfg.warmup, cfg.batch_size):
            raise ValueError(
                "replay_size must be >= max(warmup, batch_size) so learning "
                "updates are reachable"
            )
        if cfg.update_frequency < 1:
            raise ValueError("update_frequency must be >= 1")

        self.alpha_gain_user = _resolve_user_profile(
            cfg.alpha_gain_by_user,
            cfg.alpha_gain,
            cfg.n_user,
            "alpha_gain_by_user",
        )
        self.beta_loss_user = _resolve_user_profile(
            cfg.beta_loss_by_user,
            cfg.beta_loss,
            cfg.n_user,
            "beta_loss_by_user",
        )
        self.loss_aversion_user = _resolve_user_profile(
            cfg.loss_aversion_by_user,
            cfg.loss_aversion,
            cfg.n_user,
            "loss_aversion_by_user",
        )
        if not np.all(
            (self.alpha_gain_user > 0) & (self.alpha_gain_user <= 1)
        ) or not np.all(
            (self.beta_loss_user > 0) & (self.beta_loss_user <= 1)
        ):
            raise ValueError(
                "Every user-specific PT curvature must be in (0, 1]"
            )
        if not np.all(self.loss_aversion_user > 0):
            raise ValueError(
                "Every user-specific loss-aversion value must be positive"
            )
        self.environment_seed = int(cfg.seed if seed is None else seed)
        # Retained only as a policy/test convenience stream. Exogenous scenario
        # values use per-episode/per-slot namespaces below, so they are nested
        # across different user counts and cannot be shifted by action choices.
        self.rng = np.random.default_rng(self.environment_seed)
        self.episode_index = -1
        transmission_actions = np.array(
            np.unravel_index(
                np.arange(cfg.n_uav * cfg.n_channel * cfg.n_user * cfg.n_mode),
                (cfg.n_uav, cfg.n_channel, cfg.n_user, cfg.n_mode),
            )
        ).T
        self.n_transmission_actions = len(transmission_actions)
        # The final sentinel is an explicit idle/no-transmission decision.
        self.idle_action = self.n_transmission_actions
        self.actions = np.vstack(
            [transmission_actions, np.array([[-1, -1, -1, -1]], dtype=int)]
        )
        self.n_actions = len(self.actions)
        # Each RL agent observes the same four behavioral descriptors per user.
        # This prevents PT-D3QN from receiving privileged profile information.
        self.behavior_state_dim = 4 * cfg.n_user
        self.base_link_start = (
            cfg.n_uav + 2 * cfg.n_user + self.behavior_state_dim
        )
        # Base/mean link loss is observed separately from instantaneous CSI.
        # This is required because shadowing is AR(1): total CSI alone does not
        # identify the conditional distribution of the next total CSI.
        self.csi_start = self.base_link_start + cfg.n_uav * cfg.n_user
        self.state_dim = (
            self.csi_start
            + cfg.n_uav * cfg.n_channel * cfg.n_user
            + 1
        )
        self.tx_power_dbm = np.linspace(
            cfg.min_tx_power_dbm, cfg.max_tx_power_dbm, cfg.n_mode
        )
        self.tx_power_w = 10 ** ((self.tx_power_dbm - 30) / 10)
        self.mode_se = np.linspace(
            cfg.min_spectral_eff_bps_hz,
            cfg.max_spectral_eff_bps_hz,
            cfg.n_mode,
        )
        self.channel_bw_hz = cfg.system_bandwidth_hz / cfg.n_channel
        self.noise_dbm = (
            cfg.noise_psd_dbm_hz
            + 10 * np.log10(self.channel_bw_hz)
            + cfg.receiver_noise_figure_db
        )

        # A transparent QoS aspiration/reference. It is a study design choice,
        # not claimed to be a 3GPP constant. It is always retained as a fixed
        # secondary endpoint, even when the learned reward uses a calibrated or
        # explicitly supplied behavioral reference.
        rate_ref = min(
            cfg.qos_rate_mbps
            * 1e6
            / (self.channel_bw_hz * self.mode_se.max()),
            1.0,
        )
        energy_ref = self.tx_power_w.min() / self.tx_power_w.max()
        self.qos_reference_utility = float(
            cfg.w_rate * rate_ref
            + cfg.w_reliability * cfg.qos_reliability
            + cfg.w_fairness
            - cfg.w_energy * energy_ref
        )
        self.qos_reference_utility_user = np.full(
            cfg.n_user, self.qos_reference_utility, dtype=float
        )
        default_reference = (
            self.qos_reference_utility
            if cfg.reference_utility is None
            else float(cfg.reference_utility)
        )
        self.reference_utility_user = _resolve_user_profile(
            cfg.reference_utility_by_user,
            default_reference,
            cfg.n_user,
            "reference_utility_by_user",
        )
        # Backward-compatible scalar summary for manifests and legacy analysis.
        # All action scoring and rewards use reference_utility_user instead.
        self.reference_utility = float(self.reference_utility_user.mean())

        # Fixed UAV horizontal coordinates; users are redrawn per episode.
        if cfg.n_uav == 1:
            self.uav_xy = np.array(
                [[cfg.area_side_m / 2, cfg.area_side_m / 2]], dtype=float
            )
        else:
            ang = np.linspace(0, 2 * np.pi, cfg.n_uav, endpoint=False)
            radius = 0.28 * cfg.area_side_m
            self.uav_xy = np.c_[
                cfg.area_side_m / 2 + radius * np.cos(ang),
                cfg.area_side_m / 2 + radius * np.sin(ang),
            ]

    @property
    def pt_reward_scale(self) -> float:
        """Fixed global scale; independent of sampled users and user count."""
        return float(self.c.pt_reward_scale_constant)

    def set_behavior_profile(
        self, alpha, beta, loss_aversion, reference_utility=None
    ):
        """Set an already sampled training profile without changing references."""
        alpha = _resolve_user_profile(alpha, self.c.alpha_gain, self.c.n_user, "alpha")
        beta = _resolve_user_profile(beta, self.c.beta_loss, self.c.n_user, "beta")
        loss_aversion = _resolve_user_profile(
            loss_aversion,
            self.c.loss_aversion,
            self.c.n_user,
            "loss_aversion",
        )
        if not np.all((alpha > 0) & (alpha <= 1)) or not np.all(
            (beta > 0) & (beta <= 1)
        ):
            raise ValueError("Every training PT curvature must be in (0, 1]")
        if not np.all(loss_aversion > 0):
            raise ValueError("Every training loss-aversion value must be positive")
        self.alpha_gain_user = alpha
        self.beta_loss_user = beta
        self.loss_aversion_user = loss_aversion
        if reference_utility is not None:
            reference = _resolve_user_profile(
                reference_utility,
                self.reference_utility,
                self.c.n_user,
                "reference_utility",
            )
            self.reference_utility_user = reference
            self.reference_utility = float(reference.mean())

    def reset(self):
        c = self.c
        self.episode_index += 1
        self.t = 0
        self.energy = np.full(c.n_uav, c.initial_radio_energy_j, dtype=float)
        geometry_rng = np.random.default_rng(
            _derived_seed(
                610_000, self.environment_seed, self.episode_index, 0
            )
        )
        queue_rng = np.random.default_rng(
            _derived_seed(
                610_000, self.environment_seed, self.episode_index, 1
            )
        )
        shadow_rng = np.random.default_rng(
            _derived_seed(
                610_000, self.environment_seed, self.episode_index, 2
            )
        )
        self.user_xy = geometry_rng.uniform(
            0, c.area_side_m, (c.n_user, 2)
        )
        packet_capacity = int(np.floor(c.max_queue_bits / c.packet_size_bits))
        lower = max(0, int(np.ceil(0.25 * packet_capacity)))
        upper = max(lower, int(np.floor(0.75 * packet_capacity)))
        initial_packets = queue_rng.integers(
            lower, upper + 1, size=c.n_user, dtype=np.int64
        )
        self.packet_queues = [deque() for _ in range(c.n_user)]
        for user, count in enumerate(initial_packets):
            for _ in range(int(count)):
                self.packet_queues[user].append(
                    [float(c.packet_size_bits), 0]
                )
        self.queue = initial_packets.astype(float) * c.packet_size_bits
        self.initial_packet_count = initial_packets.copy()
        self.offered_packets = initial_packets.copy()
        self.admitted_packets = initial_packets.copy()
        self.generated_arrival_packets = np.zeros(c.n_user, dtype=np.int64)
        self.dropped_packets = np.zeros(c.n_user, dtype=np.int64)
        self.delivered_packets = np.zeros(c.n_user, dtype=np.int64)
        self.packet_delay_sum_s = np.zeros(c.n_user, dtype=float)
        self.transmission_attempts = np.zeros(c.n_user, dtype=np.int64)
        self.transmission_successes = np.zeros(c.n_user, dtype=np.int64)
        self.delivered = np.zeros(c.n_user)
        self.shadow = shadow_rng.normal(
            0,
            c.shadow_sigma_db,
            (c.n_user, c.n_uav, c.n_channel),
        ).transpose(1, 2, 0)
        self.base_path_loss_db = self._path_loss_db()
        return self._state()

    def _state(self):
        # Expose both mean link state and instantaneous CSI. The reference SNR
        # uses maximum configured power and is mapped smoothly to [-1, 1].
        base_reference_snr_db = (
            self.c.max_tx_power_dbm
            - self.base_path_loss_db
            - self.noise_dbm
        )
        reference_snr_db = (
            self.c.max_tx_power_dbm
            - self._instantaneous_path_loss_db()
            - self.noise_dbm
        )
        normalized_base_link = np.tanh(base_reference_snr_db / 20.0)
        normalized_csi = np.tanh(reference_snr_db / 20.0)
        normalized_loss_aversion = self.loss_aversion_user / (
            1.0 + self.loss_aversion_user
        )
        normalized_reference = np.tanh(self.reference_utility_user)
        return np.r_[
            self.energy / self.c.initial_radio_energy_j,
            self.queue / self.c.max_queue_bits,
            np.clip(
                self.delivered
                / (
                    self.c.horizon
                    * self.channel_bw_hz
                    * self.mode_se.max()
                    * self.c.slot_duration_s
                ),
                0,
                1,
            ),
            self.alpha_gain_user,
            self.beta_loss_user,
            normalized_loss_aversion,
            normalized_reference,
            normalized_base_link.ravel(),
            normalized_csi.ravel(),
            self.t / self.c.horizon,
        ].astype(np.float32)

    def _instantaneous_path_loss_db(self):
        return self.base_path_loss_db[:, None, :] + self.shadow

    def _path_loss_db(self):
        c = self.c
        horizontal = np.linalg.norm(
            self.uav_xy[:, None, :] - self.user_xy[None, :, :], axis=2
        )
        dz = c.uav_altitude_m - c.user_height_m
        distance = np.sqrt(horizontal**2 + dz**2)
        theta = np.degrees(np.arctan2(dz, np.maximum(horizontal, 1e-9)))
        p_los = 1 / (1 + c.los_a * np.exp(-c.los_b * (theta - c.los_a)))
        fspl = 20 * np.log10(
            4 * np.pi * c.carrier_frequency_hz * distance / 299_792_458.0
        )
        return fspl + p_los * c.eta_los_db + (1 - p_los) * c.eta_nlos_db

    def transmission_energy_mask(self):
        actions = self.actions[: self.n_transmission_actions]
        required = self.tx_power_w[actions[:, 3]] * self.c.slot_duration_s
        return (
            self.energy[actions[:, 0]] + 1e-12 >= required
        )

    def valid_mask(self):
        actions = self.actions[: self.n_transmission_actions]
        transmission_mask = self.transmission_energy_mask() & (
            self.queue[actions[:, 2]] > 1e-9
        )
        # Idle is always a legitimate energy-conservation decision. It is not
        # reward-free when backlog exists; step() assigns a declared
        # reference-dependent no-service opportunity cost.
        mask = np.r_[transmission_mask, True]
        return mask

    def outcomes(self):
        """Return physical outcomes for every feasible or infeasible action."""
        actions = self.actions[: self.n_transmission_actions]
        uav, channel, user, mode = actions.T
        path_loss = self._instantaneous_path_loss_db()[uav, channel, user]
        rx_dbm = self.tx_power_dbm[mode] - path_loss
        snr_db = rx_dbm - self.noise_dbm
        avg_snr = 10 ** (np.clip(snr_db, -80, 80) / 10)
        threshold = 2 ** self.mode_se[mode] - 1

        # Rayleigh block-fading success probability.
        success = np.exp(-threshold / np.maximum(avg_snr, 1e-12))
        gross_bps = self.channel_bw_hz * self.mode_se[mode]
        available_bps = self.queue[user] / self.c.slot_duration_s
        tx_bps = np.minimum(gross_bps, available_bps)
        expected_goodput_bps = success * tx_bps
        energy = self.tx_power_w[mode] * self.c.slot_duration_s

        mean_delivered = self.delivered.mean() + self.c.packet_size_bits
        service_equity = (
            np.clip(
                mean_delivered
                / (self.delivered[user] + self.c.packet_size_bits),
                0,
                2,
            )
            / 2
        )
        rate_norm = np.clip(
            tx_bps / (self.channel_bw_hz * self.mode_se.max()), 0, 1
        )
        energy_norm = energy / (
            self.tx_power_w.max() * self.c.slot_duration_s
        )

        # The same physical utility is shared by all nine policies.
        u_success = (
            self.c.w_rate * rate_norm
            + self.c.w_reliability
            + self.c.w_fairness * service_equity
            - self.c.w_energy * energy_norm
        )
        u_failure = (
            self.c.w_fairness * service_equity
            - self.c.w_energy * energy_norm
        )
        expected_utility = success * u_success + (1 - success) * u_failure
        # The last element is the idle sentinel. Its physical quantities are
        # explicit and are ignored unless no transmission is feasible.
        append_zero = lambda values: np.r_[values, 0.0]
        append_nan = lambda values: np.r_[values, np.nan]
        return {
            "expected_utility": append_zero(expected_utility),
            "u_success": append_zero(u_success),
            "u_failure": append_zero(u_failure),
            "success": append_zero(success),
            "energy_j": append_zero(energy),
            "tx_bps": append_zero(tx_bps),
            "expected_goodput_bps": append_zero(expected_goodput_bps),
            "service_equity": append_nan(service_equity),
            "snr_db": append_nan(snr_db),
            "path_loss_db": append_nan(path_loss),
        }

    def pt_value(self, delta_utility, user_index):
        """User-specific PT value function without probability weighting."""
        value = np.asarray(delta_utility, dtype=float)
        users = np.asarray(user_index, dtype=int)
        if np.any((users < 0) | (users >= self.c.n_user)):
            raise IndexError("user_index is outside the configured user set")
        alpha = self.alpha_gain_user[users]
        beta = self.beta_loss_user[users]
        loss_aversion = self.loss_aversion_user[users]
        return np.where(
            value >= 0,
            np.maximum(value, 0) ** alpha,
            -loss_aversion * np.maximum(-value, 0) ** beta,
        )

    def outcome_reward(
        self,
        physical_utility: float,
        reward_model: str,
        user_index: int,
    ) -> float:
        """Transform one realized physical outcome into a learning reward."""
        reference = self.reference_utility_user[user_index]
        delta = float(physical_utility - reference)
        if reward_model == "eut":
            return delta
        if reward_model == "pt":
            return float(
                self.pt_value(delta, user_index) / self.pt_reward_scale
            )
        raise ValueError(f"Unknown reward model: {reward_model}")

    def idle_reward_components(self):
        """Return the declared no-service opportunity cost for current backlog.

        A slot with no queued demand has no service opportunity and therefore
        zero reward. If one or more users are backlogged, their reference
        utilities define the cost of deliberately or forcibly idling. Equal
        weighting avoids making the reward depend on packet-size scaling.
        """
        active_users = np.flatnonzero(self.queue > 1e-9)
        if not len(active_users):
            return {
                "active_users": active_users,
                "reference": np.nan,
                "centered_eut_utility": np.nan,
                "pt_subjective_utility": np.nan,
                "qos_reference": np.nan,
                "qos_centered_utility": np.nan,
                "eut_reward": 0.0,
                "pt_reward": 0.0,
            }
        references = self.reference_utility_user[active_users]
        qos_references = self.qos_reference_utility_user[active_users]
        centered = -references
        pt_values = self.pt_value(centered, active_users)
        return {
            "active_users": active_users,
            "reference": float(references.mean()),
            "centered_eut_utility": float(centered.mean()),
            "pt_subjective_utility": float(pt_values.mean()),
            "qos_reference": float(qos_references.mean()),
            "qos_centered_utility": float((-qos_references).mean()),
            "eut_reward": float(centered.mean()),
            "pt_reward": float(pt_values.mean() / self.pt_reward_scale),
        }

    def idle_objective_reward(self, reward_model: str) -> float:
        components = self.idle_reward_components()
        if reward_model == "eut":
            return components["eut_reward"]
        if reward_model == "pt":
            return components["pt_reward"]
        raise ValueError(reward_model)

    def eut_scores(self):
        """Expected linear reward centered on the selected user's reference."""
        action_users = self.actions[: self.n_transmission_actions, 2]
        transmission_scores = (
            self.outcomes()["expected_utility"][: self.n_transmission_actions]
            - self.reference_utility_user[action_users]
        )
        return np.r_[transmission_scores, self.idle_objective_reward("eut")]

    def pt_scores(self):
        """Expected PT reward for all actions using objective probabilities."""
        outcomes = self.outcomes()
        action_users = self.actions[: self.n_transmission_actions, 2]
        references = self.reference_utility_user[action_users]
        gain_success = (
            outcomes["u_success"][: self.n_transmission_actions] - references
        )
        gain_failure = (
            outcomes["u_failure"][: self.n_transmission_actions] - references
        )
        transmission_scores = (
            outcomes["success"][: self.n_transmission_actions]
            * self.pt_value(gain_success, action_users)
            + (1 - outcomes["success"][: self.n_transmission_actions])
            * self.pt_value(gain_failure, action_users)
        ) / self.pt_reward_scale
        return np.r_[transmission_scores, self.idle_objective_reward("pt")]

    def objective_scores(self, reward_model: str):
        if reward_model == "eut":
            return self.eut_scores()
        if reward_model == "pt":
            return self.pt_scores()
        raise ValueError(reward_model)

    def step(self, action: int, reward_model: str = "eut"):
        if action < 0 or action >= self.n_actions:
            raise IndexError("Action index is outside the action space")
        if reward_model not in {"eut", "pt"}:
            raise ValueError(f"Unknown reward model: {reward_model}")
        valid = self.valid_mask()
        if not valid[action]:
            raise ValueError("Invalid action passed to step")

        outcomes = self.outcomes()
        queue_before = self.queue.copy()
        energy_before = self.energy.copy()
        active_users_before = np.flatnonzero(queue_before > 1e-9)
        service_opportunity = float(len(active_users_before) > 0)
        uniform_draw = float(
            np.random.default_rng(
                _derived_seed(
                    610_000,
                    self.environment_seed,
                    self.episode_index,
                    3,
                    self.t,
                )
            ).random()
        )
        is_idle = action == self.idle_action
        attempted = float(not is_idle)
        success = 0.0
        bits = 0.0
        completed_packets = 0
        selected_uav = selected_channel = selected_user = selected_mode = -1
        reference = np.nan
        delta_utility = np.nan
        pt_subjective_utility = np.nan
        qos_delta_utility = np.nan
        physical_utility = 0.0
        reward = 0.0

        if not is_idle:
            selected_uav, selected_channel, selected_user, selected_mode = (
                int(value) for value in self.actions[action]
            )
            success = float(uniform_draw < outcomes["success"][action])
            self.transmission_attempts[selected_user] += 1
            self.transmission_successes[selected_user] += int(success)
            self.energy[selected_uav] = max(
                0.0,
                self.energy[selected_uav] - outcomes["energy_j"][action],
            )
            if success:
                remaining_capacity = min(
                    self.queue[selected_user],
                    outcomes["tx_bps"][action] * self.c.slot_duration_s,
                )
                packet_queue = self.packet_queues[selected_user]
                while remaining_capacity > 1e-9 and packet_queue:
                    packet = packet_queue[0]
                    served = min(float(packet[0]), remaining_capacity)
                    packet[0] -= served
                    remaining_capacity -= served
                    bits += served
                    if packet[0] <= 1e-9:
                        _remaining, arrival_slot = packet_queue.popleft()
                        completed_packets += 1
                        self.delivered_packets[selected_user] += 1
                        self.packet_delay_sum_s[selected_user] += (
                            (self.t + 1 - int(arrival_slot))
                            * self.c.slot_duration_s
                        )
                self.queue[selected_user] = max(
                    0.0, self.queue[selected_user] - bits
                )
                self.delivered[selected_user] += bits

            physical_utility = float(
                outcomes["u_success"][action]
                if success
                else outcomes["u_failure"][action]
            )
            reference = float(self.reference_utility_user[selected_user])
            delta_utility = physical_utility - reference
            reward = self.outcome_reward(
                physical_utility,
                reward_model,
                selected_user,
            )
            pt_subjective_utility = float(
                self.pt_value(delta_utility, selected_user)
            )
            qos_delta_utility = (
                physical_utility
                - float(self.qos_reference_utility_user[selected_user])
            )
        elif service_opportunity:
            idle_components = self.idle_reward_components()
            reference = idle_components["reference"]
            delta_utility = idle_components["centered_eut_utility"]
            pt_subjective_utility = idle_components[
                "pt_subjective_utility"
            ]
            qos_delta_utility = idle_components["qos_centered_utility"]
            reward = idle_components[f"{reward_model}_reward"]

        arrival_packets = np.random.default_rng(
            _derived_seed(
                610_000,
                self.environment_seed,
                self.episode_index,
                4,
                self.t,
            )
        ).poisson(
            self.c.arrival_rate_pps * self.c.slot_duration_s,
            self.c.n_user,
        ).astype(np.int64)
        admitted_now = np.zeros(self.c.n_user, dtype=np.int64)
        dropped_now = np.zeros(self.c.n_user, dtype=np.int64)
        for user, generated in enumerate(arrival_packets):
            available_packet_slots = int(
                np.floor(
                    (self.c.max_queue_bits - self.queue[user] + 1e-9)
                    / self.c.packet_size_bits
                )
            )
            admitted = min(int(generated), max(available_packet_slots, 0))
            dropped = int(generated) - admitted
            admitted_now[user] = admitted
            dropped_now[user] = dropped
            for _ in range(admitted):
                self.packet_queues[user].append(
                    [float(self.c.packet_size_bits), self.t + 1]
                )
            self.queue[user] += admitted * self.c.packet_size_bits
        self.generated_arrival_packets += arrival_packets
        self.offered_packets += arrival_packets
        self.admitted_packets += admitted_now
        self.dropped_packets += dropped_now
        innovation = np.random.default_rng(
            _derived_seed(
                610_000,
                self.environment_seed,
                self.episode_index,
                5,
                self.t,
            )
        ).normal(
            0,
            self.c.shadow_sigma_db,
            (self.c.n_user, self.c.n_uav, self.c.n_channel),
        ).transpose(1, 2, 0)
        self.shadow = (
            self.c.shadow_rho * self.shadow
            + np.sqrt(1 - self.c.shadow_rho**2) * innovation
        )
        self.t += 1

        metrics = {
            "utility": physical_utility,
            "centered_eut_utility": delta_utility,
            "pt_subjective_utility": pt_subjective_utility,
            "reference_utility_user": reference,
            "reference_satisfaction": (
                float(delta_utility >= 0) if service_opportunity else np.nan
            ),
            "loss_event": (
                float(delta_utility < 0) if service_opportunity else np.nan
            ),
            "utility_shortfall": (
                max(-delta_utility, 0.0) if service_opportunity else np.nan
            ),
            "qos_reference_satisfaction": (
                float(qos_delta_utility >= 0)
                if service_opportunity
                else np.nan
            ),
            "qos_loss_event": (
                float(qos_delta_utility < 0)
                if service_opportunity
                else np.nan
            ),
            "qos_utility_shortfall": (
                max(-qos_delta_utility, 0.0)
                if service_opportunity
                else np.nan
            ),
            # Attempt-conditional metrics remain available as explicit
            # diagnostics; the canonical metrics above include idle slots with
            # backlog and are the valid downside endpoints.
            "attempt_centered_eut_utility": (
                delta_utility if not is_idle else np.nan
            ),
            "attempt_pt_subjective_utility": (
                pt_subjective_utility if not is_idle else np.nan
            ),
            "attempt_reference_satisfaction": (
                float(delta_utility >= 0) if not is_idle else np.nan
            ),
            "attempt_loss_event": (
                float(delta_utility < 0) if not is_idle else np.nan
            ),
            "attempt_utility_shortfall": (
                max(-delta_utility, 0.0) if not is_idle else np.nan
            ),
            "attempt_qos_reference_satisfaction": (
                float(qos_delta_utility >= 0) if not is_idle else np.nan
            ),
            "attempt_qos_loss_event": (
                float(qos_delta_utility < 0) if not is_idle else np.nan
            ),
            "attempt_qos_utility_shortfall": (
                max(-qos_delta_utility, 0.0) if not is_idle else np.nan
            ),
            "selected_alpha": (
                float(self.alpha_gain_user[selected_user])
                if not is_idle
                else np.nan
            ),
            "selected_beta": (
                float(self.beta_loss_user[selected_user])
                if not is_idle
                else np.nan
            ),
            "selected_loss_aversion": (
                float(self.loss_aversion_user[selected_user])
                if not is_idle
                else np.nan
            ),
            "throughput_mbps": float(bits / self.c.slot_duration_s / 1e6),
            "transmission_success": success if not is_idle else np.nan,
            "transmission_outage": 1.0 - success if not is_idle else np.nan,
            "attempted": attempted,
            "is_idle": float(is_idle),
            "service_opportunity": service_opportunity,
            "idle_with_backlog": float(is_idle and service_opportunity),
            "idle_without_backlog": float(is_idle and not service_opportunity),
            "backlogged_user_count_before": int(len(active_users_before)),
            "success_probability": (
                float(outcomes["success"][action]) if not is_idle else np.nan
            ),
            "outcome_uniform": uniform_draw,
            "energy_j": float(outcomes["energy_j"][action]),
            "fairness": self.jain(),
            "snr_db": float(outcomes["snr_db"][action]),
            "path_loss_db": float(outcomes["path_loss_db"][action]),
            "delivered_bits": bits,
            "packets_generated": int(arrival_packets.sum()),
            "packets_admitted": int(admitted_now.sum()),
            "packets_dropped": int(dropped_now.sum()),
            "packets_delivered": int(completed_packets),
            "queue_total_before_mbits": float(queue_before.sum() / 1e6),
            "queue_total_after_mbits": float(self.queue.sum() / 1e6),
            "selected_energy_before_j": (
                float(energy_before[selected_uav]) if not is_idle else np.nan
            ),
            "selected_energy_after_j": (
                float(self.energy[selected_uav]) if not is_idle else np.nan
            ),
            "selected_queue_before_mbits": (
                float(queue_before[selected_user] / 1e6)
                if not is_idle
                else np.nan
            ),
            "selected_queue_after_mbits": (
                float(self.queue[selected_user] / 1e6)
                if not is_idle
                else np.nan
            ),
            "selected_delivered_cumulative_mbits": (
                float(self.delivered[selected_user] / 1e6)
                if not is_idle
                else np.nan
            ),
            "arrival_events_by_user": _sparse_count_signature(
                arrival_packets
            ),
            "admitted_events_by_user": _sparse_count_signature(
                admitted_now
            ),
            "dropped_events_by_user": _sparse_count_signature(dropped_now),
            "selected_uav": selected_uav,
            "selected_channel": selected_channel,
            "selected_user": selected_user,
            "selected_mode": selected_mode,
        }
        return self._state(), reward, self.t >= self.c.horizon, metrics

    def jain(self):
        values = np.asarray(self.delivered, dtype=float)
        denominator = len(values) * float(np.square(values).sum())
        if denominator <= 0:
            return np.nan
        value = float(values.sum() ** 2 / denominator)
        return float(np.clip(value, 0.0, 1.0))


def _sample_behavior_profile_arrays(
    cfg: Config, replicate_seed: int, seed_base: int, profile_index: int
):
    """Draw one nested PT profile from independent parameter/user streams."""

    def draw_user_parameter(namespace, center, std, lower, upper, name):
        return np.array(
            [
                float(
                    _truncated_normal_sample(
                        np.random.default_rng(
                            _derived_seed(
                                seed_base,
                                replicate_seed,
                                profile_index,
                                namespace,
                                user,
                            )
                        ),
                        center,
                        std,
                        lower,
                        upper,
                        1,
                        name,
                    )[0]
                )
                for user in range(cfg.n_user)
            ],
            dtype=float,
        )

    return (
        draw_user_parameter(
            0,
            cfg.alpha_gain,
            cfg.alpha_gain_std,
            cfg.alpha_gain_lower,
            cfg.alpha_gain_upper,
            "alpha_gain",
        ),
        draw_user_parameter(
            1,
            cfg.beta_loss,
            cfg.beta_loss_std,
            cfg.beta_loss_lower,
            cfg.beta_loss_upper,
            "beta_loss",
        ),
        draw_user_parameter(
            2,
            cfg.loss_aversion,
            cfg.loss_aversion_std,
            cfg.loss_aversion_lower,
            cfg.loss_aversion_upper,
            "loss_aversion",
        ),
    )


def sample_pt_user_profiles(cfg: Config, replicate_seed: int) -> Config:
    """Resolve one fixed behavioral profile for a Monte Carlo replicate.

    The source-backed scalar settings are distribution centers only. The
    truncated-normal dispersions and bounds are declared study assumptions.
    """
    resolved = replace(cfg, seed=int(replicate_seed))
    if not cfg.randomize_pt_profiles:
        return resolved
    if any(
        profile is not None
        for profile in (
            cfg.alpha_gain_by_user,
            cfg.beta_loss_by_user,
            cfg.loss_aversion_by_user,
        )
    ):
        raise ValueError(
            "Random PT profiles cannot be combined with explicit "
            "alpha/beta/loss-aversion user profiles"
        )

    alpha, beta, loss_aversion = _sample_behavior_profile_arrays(
        cfg,
        replicate_seed,
        cfg.pt_profile_seed_base,
        0,
    )
    return replace(
        resolved,
        alpha_gain_by_user=tuple(float(value) for value in alpha),
        beta_loss_by_user=tuple(float(value) for value in beta),
        loss_aversion_by_user=tuple(
            float(value) for value in loss_aversion
        ),
    )


def calibrate_random_policy_reference(
    cfg: Config, replicate_seed: int
) -> tuple[float, ...]:
    """Calibrate U_ref,u on independent pre-training random-policy states.

    At a calibration state where user u has at least one feasible transmission,
    its observation is the mean expected physical utility over that user's
    feasible UAV/channel/mode actions. Empty-queue or energy-infeasible users
    are omitted at that state. A uniformly random valid action, including the
    explicitly scored idle decision, advances the environment. Training and
    evaluation data are not
    used.
    """
    if cfg.reference_utility is not None or cfg.reference_utility_by_user is not None:
        raise ValueError(
            "Random-policy reference calibration cannot be combined with an "
            "explicit reference utility"
        )
    calibration_cfg = replace(
        cfg,
        reference_utility_mode="qos_aspiration",
        reference_utility=None,
        reference_utility_by_user=None,
    )
    totals = np.zeros(cfg.n_user, dtype=float)
    counts = np.zeros(cfg.n_user, dtype=np.int64)
    for episode in range(cfg.reference_calibration_episodes):
        environment_seed = _derived_seed(
            cfg.reference_calibration_seed_base,
            replicate_seed,
            episode,
            0,
        )
        policy_seed = _derived_seed(
            cfg.reference_calibration_seed_base,
            replicate_seed,
            episode,
            1,
        )
        env = UAVEnv(calibration_cfg, environment_seed)
        env.reset()
        policy_rng = np.random.default_rng(policy_seed)
        for _ in range(cfg.horizon):
            valid = env.valid_mask()
            outcomes = env.outcomes()
            for user in range(cfg.n_user):
                user_actions = valid & (env.actions[:, 2] == user)
                if user_actions.any():
                    totals[user] += float(
                        outcomes["expected_utility"][user_actions].mean()
                    )
                    counts[user] += 1
            action = int(policy_rng.choice(np.flatnonzero(valid)))
            env.step(action, "eut")
    if not np.all(counts > 0):
        raise RuntimeError("Reference calibration produced an empty user sample")
    reference = totals / counts
    if not np.all(np.isfinite(reference)):
        raise RuntimeError("Reference calibration produced non-finite values")
    return tuple(float(value) for value in reference)


def _effective_reference_mode(cfg: Config) -> str:
    """Infer explicit mode for backward-compatible programmatic configs."""
    if cfg.reference_utility_mode == "random_policy_calibration":
        return cfg.reference_utility_mode
    if cfg.reference_utility is not None or cfg.reference_utility_by_user is not None:
        return "explicit"
    return cfg.reference_utility_mode


def build_replicate_config(cfg: Config, replicate_seed: int) -> Config:
    """Freeze all user traits and references for one fair replicate."""
    if int(replicate_seed) < 0:
        raise ValueError("replicate_seed must be non-negative")
    resolved = sample_pt_user_profiles(cfg, int(replicate_seed))
    reference_mode = _effective_reference_mode(resolved)
    resolved = replace(resolved, reference_utility_mode=reference_mode)
    if reference_mode == "random_policy_calibration":
        reference = calibrate_random_policy_reference(
            resolved, int(replicate_seed)
        )
        resolved = replace(
            resolved,
            reference_utility_by_user=reference,
            reference_calibration_population_size=resolved.n_user,
        )
    elif reference_mode == "explicit" and (
        resolved.reference_utility is None
        and resolved.reference_utility_by_user is None
    ):
        raise ValueError(
            "Explicit reference mode requires reference_utility or "
            "reference_utility_by_user"
        )
    UAVEnv(resolved, resolved.seed)
    return resolved


def build_training_profile_bank(cfg: Config):
    """Return the shared training profile schedule for all RL ablations."""
    env = UAVEnv(cfg, cfg.seed)
    if not cfg.randomize_pt_profiles or cfg.training_profile_count == 1:
        return [
            {
                "profile_index": 0,
                "profile_seed": _derived_seed(
                    cfg.pt_profile_seed_base, cfg.seed, 0
                )
                if cfg.randomize_pt_profiles
                else np.nan,
                "alpha": env.alpha_gain_user.copy(),
                "beta": env.beta_loss_user.copy(),
                "loss_aversion": env.loss_aversion_user.copy(),
                "reference_utility": env.reference_utility_user.copy(),
                "evaluation_profile_held_out": False,
            }
        ]

    bank = []
    for profile_index in range(cfg.training_profile_count):
        alpha, beta, loss_aversion = _sample_behavior_profile_arrays(
            cfg,
            cfg.seed,
            cfg.training_profile_seed_base,
            profile_index,
        )
        reference_factors = np.array(
            [
                1.0
                + np.random.default_rng(
                    _derived_seed(
                        cfg.training_profile_seed_base,
                        cfg.seed,
                        profile_index,
                        3,
                        user,
                    )
                ).uniform(
                    -cfg.training_reference_jitter_fraction,
                    cfg.training_reference_jitter_fraction,
                )
                for user in range(cfg.n_user)
            ],
            dtype=float,
        )
        reference_utility = env.reference_utility_user * reference_factors
        bank.append(
            {
                "profile_index": profile_index,
                "profile_seed": _derived_seed(
                    cfg.training_profile_seed_base,
                    cfg.seed,
                    profile_index,
                ),
                "alpha": alpha,
                "beta": beta,
                "loss_aversion": loss_aversion,
                "reference_utility": reference_utility,
                "evaluation_profile_held_out": True,
            }
        )
    return bank


def write_training_pt_profiles(configs, out):
    """Persist every profile in the common RL training-domain bank."""
    rows = []
    for cfg in configs:
        for profile in build_training_profile_bank(cfg):
            for user in range(cfg.n_user):
                rows.append(
                    {
                        "replicate_seed": cfg.seed,
                        "training_profile_index": profile["profile_index"],
                        "training_profile_seed": profile["profile_seed"],
                        "alpha_profile_seed": (
                            _derived_seed(
                                cfg.training_profile_seed_base,
                                cfg.seed,
                                profile["profile_index"],
                                0,
                                user,
                            )
                            if profile["evaluation_profile_held_out"]
                            else np.nan
                        ),
                        "beta_profile_seed": (
                            _derived_seed(
                                cfg.training_profile_seed_base,
                                cfg.seed,
                                profile["profile_index"],
                                1,
                                user,
                            )
                            if profile["evaluation_profile_held_out"]
                            else np.nan
                        ),
                        "loss_aversion_profile_seed": (
                            _derived_seed(
                                cfg.training_profile_seed_base,
                                cfg.seed,
                                profile["profile_index"],
                                2,
                                user,
                            )
                            if profile["evaluation_profile_held_out"]
                            else np.nan
                        ),
                        "reference_profile_seed": (
                            _derived_seed(
                                cfg.training_profile_seed_base,
                                cfg.seed,
                                profile["profile_index"],
                                3,
                                user,
                            )
                            if profile["evaluation_profile_held_out"]
                            else np.nan
                        ),
                        "user_index": user,
                        "user_label": f"U{user + 1}",
                        "alpha_u": float(profile["alpha"][user]),
                        "beta_u": float(profile["beta"][user]),
                        "k_u": float(profile["loss_aversion"][user]),
                        "reference_utility_u": float(
                            profile["reference_utility"][user]
                        ),
                        "evaluation_profile_held_out": profile[
                            "evaluation_profile_held_out"
                        ],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "training_pt_profiles.csv", index=False)
    return frame


def evaluation_episode_seed(cfg: Config, episode: int) -> int:
    """Held-out scenario seed shared by all methods within one replicate."""
    return _derived_seed(cfg.evaluation_seed_base, cfg.seed, episode, 0)


def evaluation_policy_seed(cfg: Config, episode: int) -> int:
    """Independent evaluation-policy random stream for one replicate."""
    return _derived_seed(cfg.evaluation_seed_base, cfg.seed, episode, 1)


def write_seed_manifest(configs, out):
    """Record every top-level random stream used by the experiment."""
    if isinstance(configs, Config):
        configs = [configs]
    rows = []
    for cfg in configs:
        agent_seed = _derived_seed(400_000, cfg.seed)
        rows.extend(
            [
                {
                    "replicate_seed": cfg.seed,
                    "stream": "training_environment",
                    "episode": -1,
                    "resolved_seed": cfg.seed,
                },
                {
                    "replicate_seed": cfg.seed,
                    "stream": "agent_online_initialization",
                    "episode": -1,
                    "resolved_seed": agent_seed,
                },
                {
                    "replicate_seed": cfg.seed,
                    "stream": "agent_exploration_and_replay",
                    "episode": -1,
                    "resolved_seed": agent_seed + 2_000_003,
                },
            ]
        )
        if cfg.randomize_pt_profiles:
            rows.append(
                {
                    "replicate_seed": cfg.seed,
                    "stream": "pt_profile",
                    "episode": -1,
                    "resolved_seed": _derived_seed(
                        cfg.pt_profile_seed_base, cfg.seed, 0
                    ),
                }
            )
            for user in range(cfg.n_user):
                for namespace, parameter in enumerate(
                    ("alpha", "beta", "loss_aversion")
                ):
                    rows.append(
                        {
                            "replicate_seed": cfg.seed,
                            "stream": f"pt_profile_{parameter}",
                            "episode": -1,
                            "user_index": user,
                            "resolved_seed": _derived_seed(
                                cfg.pt_profile_seed_base,
                                cfg.seed,
                                0,
                                namespace,
                                user,
                            ),
                        }
                    )
            if cfg.training_profile_count > 1:
                for profile_index in range(cfg.training_profile_count):
                    rows.append(
                        {
                            "replicate_seed": cfg.seed,
                            "stream": "training_pt_profile",
                            "episode": -1,
                            "profile_index": profile_index,
                            "resolved_seed": _derived_seed(
                                cfg.training_profile_seed_base,
                                cfg.seed,
                                profile_index,
                            ),
                        }
                    )
        if _effective_reference_mode(cfg) == "random_policy_calibration":
            calibration_population = (
                cfg.reference_calibration_population_size
                if cfg.reference_calibration_population_size is not None
                else cfg.n_user
            )
            for episode in range(cfg.reference_calibration_episodes):
                rows.extend(
                    [
                        {
                            "replicate_seed": cfg.seed,
                            "stream": "reference_calibration_environment",
                            "episode": episode,
                            "calibration_population_size": (
                                calibration_population
                            ),
                            "resolved_seed": _derived_seed(
                                cfg.reference_calibration_seed_base,
                                cfg.seed,
                                episode,
                                0,
                            ),
                        },
                        {
                            "replicate_seed": cfg.seed,
                            "stream": "reference_calibration_policy",
                            "episode": episode,
                            "calibration_population_size": (
                                calibration_population
                            ),
                            "resolved_seed": _derived_seed(
                                cfg.reference_calibration_seed_base,
                                cfg.seed,
                                episode,
                                1,
                            ),
                        },
                    ]
                )
        for episode in range(cfg.eval_episodes):
            rows.extend(
                [
                    {
                        "replicate_seed": cfg.seed,
                        "stream": "evaluation_environment",
                        "episode": episode,
                        "resolved_seed": evaluation_episode_seed(cfg, episode),
                    },
                    {
                        "replicate_seed": cfg.seed,
                        "stream": "evaluation_policy",
                        "episode": episode,
                        "resolved_seed": evaluation_policy_seed(cfg, episode),
                    },
                ]
            )
    pd.DataFrame(rows).to_csv(out / "seed_manifest.csv", index=False)


class Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.m = [np.zeros_like(value) for value in params]
        self.v = [np.zeros_like(value) for value in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for index, (param, grad) in enumerate(zip(params, grads)):
            grad = np.clip(grad, -5, 5)
            self.m[index] = 0.9 * self.m[index] + 0.1 * grad
            self.v[index] = 0.999 * self.v[index] + 0.001 * grad * grad
            param -= self.lr * (
                self.m[index] / (1 - 0.9**self.t)
            ) / (np.sqrt(self.v[index] / (1 - 0.999**self.t)) + 1e-8)


class QNet:
    """Two-layer NumPy MLP with an optional dueling head."""

    def __init__(
        self, n_input, n_output, hidden, dueling, rng, lr, with_optimizer=True
    ):
        self.dueling = dueling
        self.W1 = rng.normal(0, np.sqrt(2 / n_input), (n_input, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        if dueling:
            self.Wv = rng.normal(0, 0.05, (hidden, 1))
            self.bv = np.zeros(1)
            self.Wa = rng.normal(0, 0.05, (hidden, n_output))
            self.ba = np.zeros(n_output)
            self.params = [
                self.W1,
                self.b1,
                self.W2,
                self.b2,
                self.Wv,
                self.bv,
                self.Wa,
                self.ba,
            ]
        else:
            self.Wq = rng.normal(0, 0.05, (hidden, n_output))
            self.bq = np.zeros(n_output)
            self.params = [
                self.W1,
                self.b1,
                self.W2,
                self.b2,
                self.Wq,
                self.bq,
            ]
        self.opt = Adam(self.params, lr) if with_optimizer else None

    def forward(self, values, cache=False):
        z1 = values @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0)
        if self.dueling:
            state_value = h2 @ self.Wv + self.bv
            advantage = h2 @ self.Wa + self.ba
            q_values = state_value + advantage - advantage.mean(1, keepdims=True)
        else:
            q_values = h2 @ self.Wq + self.bq
        return (q_values, (values, z1, h1, z2, h2)) if cache else q_values

    def train(self, values, actions, target):
        if self.opt is None:
            raise RuntimeError("Cannot train a target-only network")
        q_values, cache = self.forward(values, True)
        batch_size = len(values)
        prediction = q_values[np.arange(batch_size), actions]
        td_error = prediction - target
        absolute_error = np.abs(td_error)
        loss = np.where(
            absolute_error <= 1,
            0.5 * td_error * td_error,
            absolute_error - 0.5,
        )
        grad = np.where(
            absolute_error <= 1, td_error, np.sign(td_error)
        ) / batch_size
        dq = np.zeros_like(q_values)
        dq[np.arange(batch_size), actions] = grad
        values, z1, h1, z2, h2 = cache

        if self.dueling:
            dv = dq.sum(1, keepdims=True)
            da = dq - dq.mean(1, keepdims=True)
            grads = [None] * 8
            grads[4] = h2.T @ dv
            grads[5] = dv.sum(0)
            grads[6] = h2.T @ da
            grads[7] = da.sum(0)
            dh = dv @ self.Wv.T + da @ self.Wa.T
        else:
            grads = [None] * 6
            grads[4] = h2.T @ dq
            grads[5] = dq.sum(0)
            dh = dq @ self.Wq.T

        dz2 = dh * (z2 > 0)
        grads[2] = h1.T @ dz2
        grads[3] = dz2.sum(0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0)
        grads[0] = values.T @ dz1
        grads[1] = dz1.sum(0)
        self.opt.step(self.params, grads)
        return float(np.mean(loss))

    def copy_from(self, other):
        for param, source in zip(self.params, other.params):
            param[:] = source


class Agent:
    def __init__(
        self,
        env,
        cfg,
        double=False,
        dueling=False,
        reward_model="eut",
        seed=0,
    ):
        self.c = cfg
        self.double = double
        self.reward_model = reward_model
        # Separate streams prevent a larger/smaller architecture from shifting
        # the exploration and replay-sampling random sequence.
        network_rng = np.random.default_rng(seed)
        target_rng = np.random.default_rng(seed + 1_000_003)
        self.rng = np.random.default_rng(seed + 2_000_003)
        self.net = QNet(
            env.state_dim,
            env.n_actions,
            cfg.hidden,
            dueling,
            network_rng,
            cfg.lr,
        )
        self.target = QNet(
            env.state_dim,
            env.n_actions,
            cfg.hidden,
            dueling,
            target_rng,
            cfg.lr,
            with_optimizer=False,
        )
        self.target.copy_from(self.net)
        self.buf = deque(maxlen=cfg.replay_size)
        self.steps = 0
        self.n_actions = env.n_actions
        self.parameter_count = int(sum(param.size for param in self.net.params))

    def act(self, state, mask, epsilon=0):
        ids = np.flatnonzero(mask)
        if len(ids) == 0:
            raise RuntimeError("The valid action set is empty")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(ids))
        q_values = self.net.forward(state[None])[0]
        q_values[~mask] = -np.inf
        return int(q_values.argmax())

    def update(self):
        if len(self.buf) < max(self.c.warmup, self.c.batch_size):
            return np.nan
        ids = self.rng.choice(
            len(self.buf), self.c.batch_size, replace=False
        )
        batch = [self.buf[index] for index in ids]
        state = np.array([item[0] for item in batch])
        action = np.array([item[1] for item in batch])
        reward = np.array([item[2] for item in batch])
        next_state = np.array([item[3] for item in batch])
        done = np.array([item[4] for item in batch])
        stored_masks = np.array([item[5] for item in batch])
        if stored_masks.dtype == bool and stored_masks.shape[1] == self.n_actions:
            masks = stored_masks
        else:
            masks = np.unpackbits(
                stored_masks.astype(np.uint8),
                axis=1,
                count=self.n_actions,
                bitorder="little",
            ).astype(bool)

        target_q = self.target.forward(next_state)
        target_q[~masks] = -np.inf
        if self.double:
            online_q = self.net.forward(next_state)
            online_q[~masks] = -np.inf
            next_action = online_q.argmax(1)
            next_value = target_q[np.arange(len(batch)), next_action]
        else:
            next_value = target_q.max(1)

        target = reward + self.c.gamma * (1 - done) * next_value
        loss = self.net.train(state, action, target)
        self.steps += 1
        if self.steps % self.c.target_update == 0:
            self.target.copy_from(self.net)
        return loss


BASELINES = [
    "Random",
    "Greedy_Physical_Utility",
    "EUT_Myopic_Oracle",
    "PT_Myopic_Oracle",
]

RL_SPECS = {
    "EUT_DQN": (False, False, "eut"),
    "EUT_DDQN": (True, False, "eut"),
    "EUT_Dueling_DQN": (False, True, "eut"),
    "EUT_D3QN": (True, True, "eut"),
    "PT_D3QN": (True, True, "pt"),
}

ALL_METHODS = [*BASELINES, *RL_SPECS]

EVALUATION_METRICS = [
    "utility",
    "throughput_mbps",
    "offered_pdr",
    "admitted_pdr",
    "packet_drop_rate",
    "mean_packet_delay_s",
    "transmission_success_rate",
    "outage",
    "energy_j",
    "energy_per_attempt_j",
    "total_energy_consumed_j",
    "idle_fraction",
    "service_opportunity_fraction",
    "idle_with_backlog_fraction",
    "idle_without_backlog_fraction",
    "valid_transmission_action_fraction",
    "energy_masked_action_fraction",
    "final_jain_fairness",
    "delivered_total_mbits",
    "queue_remaining_mbits",
    "snr_db",
    "path_loss_db",
    "reference_satisfaction",
    "loss_event",
    "utility_shortfall",
    "pt_subjective_utility",
    "attempt_centered_eut_utility",
    "attempt_pt_subjective_utility",
    "attempt_reference_satisfaction",
    "attempt_loss_event",
    "attempt_utility_shortfall",
    "qos_reference_satisfaction",
    "qos_loss_event",
    "qos_utility_shortfall",
    "attempt_qos_reference_satisfaction",
    "attempt_qos_loss_event",
    "attempt_qos_utility_shortfall",
]

PRIMARY_ENDPOINT = "utility_shortfall"
# Non-redundant secondary family. Complementary aliases such as loss-event vs
# satisfaction and success vs outage are excluded from Holm multiplicity.
SECONDARY_INFERENCE_FAMILY = {
    "utility",
    "throughput_mbps",
    "offered_pdr",
    "packet_drop_rate",
    "mean_packet_delay_s",
    "transmission_success_rate",
    "energy_per_attempt_j",
    "final_jain_fairness",
    "reference_satisfaction",
    "pt_subjective_utility",
    "qos_utility_shortfall",
    "idle_with_backlog_fraction",
}


def heuristic_action(env, name, rng):
    mask = env.valid_mask()
    ids = np.flatnonzero(mask)
    if name == "Random":
        return int(rng.choice(ids))
    outcomes = env.outcomes()
    if name == "Greedy_Physical_Utility":
        # Assumes success and ignores outage risk. The EUT oracle below
        # explicitly averages the success and failure outcomes.
        score = outcomes["u_success"]
    elif name == "EUT_Myopic_Oracle":
        score = env.eut_scores()
    elif name == "PT_Myopic_Oracle":
        score = env.pt_scores()
    else:
        raise ValueError(name)
    score = np.where(mask, score, -np.inf)
    return int(score.argmax())


def train_agent(name, cfg, training_profiles=None):
    env = UAVEnv(cfg, cfg.seed)
    if training_profiles is None:
        training_profiles = build_training_profile_bank(cfg)
    agent_seed = _derived_seed(400_000, cfg.seed)
    agent = Agent(env, cfg, *RL_SPECS[name], seed=agent_seed)
    epsilon = cfg.eps_start
    rows = []
    environment_steps = 0
    for episode in range(cfg.train_episodes):
        profile = training_profiles[episode % len(training_profiles)]
        env.set_behavior_profile(
            profile["alpha"],
            profile["beta"],
            profile["loss_aversion"],
            profile["reference_utility"],
        )
        state = env.reset()
        objective_return = 0.0
        physical_return = 0.0
        losses = []
        for _ in range(cfg.horizon):
            mask = env.valid_mask()
            action = agent.act(state, mask, epsilon)
            next_state, reward, done, metrics = env.step(
                action, agent.reward_model
            )
            agent.buf.append(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    float(done),
                    np.packbits(env.valid_mask(), bitorder="little"),
                )
            )
            environment_steps += 1
            loss = (
                agent.update()
                if environment_steps % cfg.update_frequency == 0
                else np.nan
            )
            if np.isfinite(loss):
                losses.append(loss)
            state = next_state
            objective_return += reward
            physical_return += metrics["utility"]
        rows.append(
            {
                "n_user": cfg.n_user,
                "method": name,
                "replicate_seed": cfg.seed,
                "training_seed": cfg.seed,
                "episode": episode,
                "training_profile_index": profile["profile_index"],
                "training_profile_seed": profile["profile_seed"],
                "evaluation_profile_held_out": profile[
                    "evaluation_profile_held_out"
                ],
                "objective_return": objective_return,
                "physical_utility_return": physical_return,
                "loss": np.mean(losses) if losses else np.nan,
                "epsilon": epsilon,
                "environment_steps": environment_steps,
                "optimizer_updates": agent.steps,
            }
        )
        epsilon = max(cfg.eps_end, epsilon * cfg.eps_decay)
    return agent, rows


class CsvRowSink:
    """Stream homogeneous dictionaries to one CSV without retaining rows."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = None
        self.writer = None
        self.row_count = 0

    def write(self, row):
        if self.handle is None:
            self.handle = self.path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.row_count += 1

    def append(self, row):
        """List-compatible alias used by the evaluation detail collectors."""
        self.write(row)

    def close(self):
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.handle = None


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def _close_and_verify_csv_sink(sink, expected_rows, label):
    """Flush one streamed CSV and verify both logical and on-disk row counts."""
    sink.close()
    with sink.path.open("r", encoding="utf-8", newline="") as handle:
        written_rows = max(sum(1 for _ in handle) - 1, 0)
    if sink.row_count != expected_rows or written_rows != expected_rows:
        raise RuntimeError(
            f"Incomplete {sink.path.name}: expected {expected_rows}, "
            f"recorded {sink.row_count}, read back {written_rows} ({label})"
        )


def evaluate(
    name,
    cfg,
    agent=None,
    user_detail_rows=None,
    selection_detail_rows=None,
    step_sink=None,
):
    """Evaluate a policy and record episode, user, action and optional slot data."""
    rows = []
    for episode in range(cfg.eval_episodes):
        episode_seed = evaluation_episode_seed(cfg, episode)
        env = UAVEnv(cfg, episode_seed)
        state = env.reset()
        initial_queue = env.queue.copy()
        initial_packets = env.initial_packet_count.copy()
        step_metrics = []
        user_selection_count = np.zeros(cfg.n_user, dtype=np.int64)
        user_delivered_bits = np.zeros(cfg.n_user, dtype=float)
        user_metric_names = (
            "utility",
            "attempt_centered_eut_utility",
            "attempt_pt_subjective_utility",
            "attempt_reference_satisfaction",
            "attempt_loss_event",
            "attempt_utility_shortfall",
            "attempt_qos_reference_satisfaction",
            "attempt_qos_loss_event",
            "attempt_qos_utility_shortfall",
            "throughput_mbps",
            "transmission_success",
            "transmission_outage",
            "energy_j",
        )
        user_metric_sums = {
            metric: np.zeros(cfg.n_user, dtype=float)
            for metric in user_metric_names
        }
        dimension_sizes = {
            "uav": cfg.n_uav,
            "channel": cfg.n_channel,
            "user": cfg.n_user,
            "mode": cfg.n_mode,
        }
        # Each marginal has one additional Idle category, so its counts sum to
        # exactly the episode horizon even when no transmission is feasible.
        dimension_counts = {
            name_: np.zeros(size + 1, dtype=np.int64)
            for name_, size in dimension_sizes.items()
        }
        rng = np.random.default_rng(evaluation_policy_seed(cfg, episode))
        for slot in range(cfg.horizon):
            mask = env.valid_mask()
            energy_feasible = env.transmission_energy_mask()
            valid_transmission_count = int(
                mask[: env.n_transmission_actions].sum()
            )
            energy_feasible_count = int(energy_feasible.sum())
            if agent is not None:
                action = agent.act(state, mask, 0)
            else:
                action = heuristic_action(env, name, rng)
            selected = env.actions[action]
            is_idle = action == env.idle_action
            delivered_before = env.delivered.copy()
            state, _reward, _done, metrics = env.step(action, "eut")
            metrics.update(
                {
                    "valid_transmission_action_count": valid_transmission_count,
                    "valid_transmission_action_fraction": (
                        valid_transmission_count / env.n_transmission_actions
                    ),
                    "energy_feasible_transmission_action_count": (
                        energy_feasible_count
                    ),
                    "energy_masked_transmission_action_count": (
                        env.n_transmission_actions - energy_feasible_count
                    ),
                    "energy_masked_action_fraction": (
                        (env.n_transmission_actions - energy_feasible_count)
                        / env.n_transmission_actions
                    ),
                    "nonempty_queue_user_count_after": int(
                        np.count_nonzero(env.queue > 1e-9)
                    ),
                }
            )
            step_metrics.append(metrics)

            if not is_idle:
                selected_uav, selected_channel, selected_user, selected_mode = (
                    int(value) for value in selected
                )
                user_selection_count[selected_user] += 1
                user_delivered_bits[selected_user] += (
                    env.delivered[selected_user]
                    - delivered_before[selected_user]
                )
                for metric, values in user_metric_sums.items():
                    values[selected_user] += float(metrics[metric])
                indices = {
                    "uav": selected_uav,
                    "channel": selected_channel,
                    "user": selected_user,
                    "mode": selected_mode,
                }
                for dimension, category in indices.items():
                    dimension_counts[dimension][category] += 1
            else:
                for dimension, size in dimension_sizes.items():
                    dimension_counts[dimension][size] += 1

            if step_sink is not None:
                step_sink.write(
                    {
                        "n_user": cfg.n_user,
                        "method": name,
                        "replicate_seed": cfg.seed,
                        "training_seed": -1 if agent is None else cfg.seed,
                        "evaluation_episode": episode,
                        "episode_seed": episode_seed,
                        "slot": slot,
                        "action_index": action,
                        **metrics,
                    }
                )

        averaged = pd.DataFrame(step_metrics).mean(numeric_only=True).to_dict()
        averaged["service_opportunity_fraction"] = averaged.pop(
            "service_opportunity"
        )
        averaged["idle_with_backlog_fraction"] = averaged.pop(
            "idle_with_backlog"
        )
        averaged["idle_without_backlog_fraction"] = averaged.pop(
            "idle_without_backlog"
        )
        offered_packets = int(env.offered_packets.sum())
        admitted_packets = int(env.admitted_packets.sum())
        generated_packets = int(env.generated_arrival_packets.sum())
        dropped_packets = int(env.dropped_packets.sum())
        delivered_packets = int(env.delivered_packets.sum())
        attempts = int(env.transmission_attempts.sum())
        successes = int(env.transmission_successes.sum())
        offered_pdr = _safe_ratio(delivered_packets, offered_packets)
        admitted_pdr = _safe_ratio(delivered_packets, admitted_packets)
        transmission_success_rate = _safe_ratio(successes, attempts)
        packet_drop_rate = _safe_ratio(dropped_packets, generated_packets)
        mean_packet_delay_s = _safe_ratio(
            env.packet_delay_sum_s.sum(), delivered_packets
        )
        averaged.update(
            {
                "n_user": cfg.n_user,
                "method": name,
                "replicate_seed": cfg.seed,
                "training_seed": -1 if agent is None else cfg.seed,
                "evaluation_episode": episode,
                "episode_seed": episode_seed,
                "final_jain_fairness": env.jain(),
                "delivered_total_mbits": env.delivered.sum() / 1e6,
                "queue_remaining_mbits": env.queue.sum() / 1e6,
                "offered_packets": offered_packets,
                "admitted_packets": admitted_packets,
                "generated_arrival_packets": generated_packets,
                "dropped_packets": dropped_packets,
                "delivered_packets": delivered_packets,
                "offered_pdr": offered_pdr,
                "admitted_pdr": admitted_pdr,
                # Backward-compatible name, now correctly defined at packet
                # level rather than as one Bernoulli link outcome.
                "pdr": offered_pdr,
                "packet_drop_rate": packet_drop_rate,
                "mean_packet_delay_s": mean_packet_delay_s,
                "transmission_attempts": attempts,
                "transmission_successes": successes,
                "transmission_success_rate": transmission_success_rate,
                "outage": (
                    1.0 - transmission_success_rate
                    if np.isfinite(transmission_success_rate)
                    else np.nan
                ),
                "idle_fraction": float(
                    dimension_counts["user"][-1] / cfg.horizon
                ),
                "energy_per_attempt_j": _safe_ratio(
                    sum(metric["energy_j"] for metric in step_metrics), attempts
                ),
                "total_energy_consumed_j": float(
                    cfg.n_uav * cfg.initial_radio_energy_j - env.energy.sum()
                ),
            }
        )
        rows.append(averaged)

        if user_detail_rows is not None:
            delivered_total = float(user_delivered_bits.sum())
            for user in range(cfg.n_user):
                count = int(user_selection_count[user])

                def selected_mean(metric):
                    return (
                        float(user_metric_sums[metric][user] / count)
                        if count
                        else np.nan
                    )

                user_delivered_packets = int(env.delivered_packets[user])
                user_offered_packets = int(env.offered_packets[user])
                user_admitted_packets = int(env.admitted_packets[user])
                user_generated_packets = int(env.generated_arrival_packets[user])
                user_dropped_packets = int(env.dropped_packets[user])
                user_attempts = int(env.transmission_attempts[user])
                user_successes = int(env.transmission_successes[user])
                user_detail_rows.append(
                    {
                        "n_user": cfg.n_user,
                        "method": name,
                        "replicate_seed": cfg.seed,
                        "training_seed": -1 if agent is None else cfg.seed,
                        "evaluation_episode": episode,
                        "episode_seed": episode_seed,
                        "user_index": user,
                        "user_label": f"U{user + 1}",
                        "selection_count": count,
                        "selection_fraction": count / cfg.horizon,
                        "initial_packet_count": int(initial_packets[user]),
                        "initial_queue_mbits": initial_queue[user] / 1e6,
                        "delivered_mbits": user_delivered_bits[user] / 1e6,
                        "delivered_share": (
                            user_delivered_bits[user] / delivered_total
                            if delivered_total > 0
                            else 0.0
                        ),
                        "final_queue_mbits": env.queue[user] / 1e6,
                        "offered_packets": user_offered_packets,
                        "admitted_packets": user_admitted_packets,
                        "generated_arrival_packets": user_generated_packets,
                        "dropped_packets": user_dropped_packets,
                        "delivered_packets": user_delivered_packets,
                        "offered_pdr": _safe_ratio(
                            user_delivered_packets, user_offered_packets
                        ),
                        "admitted_pdr": _safe_ratio(
                            user_delivered_packets, user_admitted_packets
                        ),
                        "packet_drop_rate": _safe_ratio(
                            user_dropped_packets, user_generated_packets
                        ),
                        "mean_packet_delay_s": _safe_ratio(
                            env.packet_delay_sum_s[user], user_delivered_packets
                        ),
                        "transmission_attempts": user_attempts,
                        "transmission_successes": user_successes,
                        "transmission_success_rate": _safe_ratio(
                            user_successes, user_attempts
                        ),
                        "selected_utility_mean": selected_mean("utility"),
                        "selected_centered_eut_utility_mean": selected_mean(
                            "attempt_centered_eut_utility"
                        ),
                        "selected_pt_subjective_utility_mean": selected_mean(
                            "attempt_pt_subjective_utility"
                        ),
                        "selected_reference_satisfaction": selected_mean(
                            "attempt_reference_satisfaction"
                        ),
                        "selected_loss_event": selected_mean(
                            "attempt_loss_event"
                        ),
                        "selected_utility_shortfall": selected_mean(
                            "attempt_utility_shortfall"
                        ),
                        "selected_qos_reference_satisfaction": selected_mean(
                            "attempt_qos_reference_satisfaction"
                        ),
                        "selected_qos_loss_event": selected_mean(
                            "attempt_qos_loss_event"
                        ),
                        "selected_qos_utility_shortfall": selected_mean(
                            "attempt_qos_utility_shortfall"
                        ),
                        "selected_throughput_mbps": selected_mean(
                            "throughput_mbps"
                        ),
                        "selected_transmission_success_rate": selected_mean(
                            "transmission_success"
                        ),
                        "selected_transmission_outage": selected_mean(
                            "transmission_outage"
                        ),
                        "selected_energy_j": selected_mean("energy_j"),
                        "alpha_u": float(env.alpha_gain_user[user]),
                        "beta_u": float(env.beta_loss_user[user]),
                        "k_u": float(env.loss_aversion_user[user]),
                        "reference_utility_u": float(
                            env.reference_utility_user[user]
                        ),
                        "qos_reference_utility_u": float(
                            env.qos_reference_utility_user[user]
                        ),
                    }
                )

        if selection_detail_rows is not None:
            for dimension, counts in dimension_counts.items():
                non_idle_size = dimension_sizes[dimension]
                for category, count in enumerate(counts):
                    is_idle_category = category == non_idle_size
                    selection_detail_rows.append(
                        {
                            "n_user": cfg.n_user,
                            "method": name,
                            "replicate_seed": cfg.seed,
                            "training_seed": -1 if agent is None else cfg.seed,
                            "evaluation_episode": episode,
                            "episode_seed": episode_seed,
                            "dimension": dimension,
                            "category_index": -1 if is_idle_category else category,
                            "category_label": (
                                "Idle"
                                if is_idle_category
                                else (
                                    f"U{category + 1}"
                                    if dimension == "user"
                                    else f"{dimension}_{category + 1}"
                                )
                            ),
                            "selection_count": int(count),
                            "selection_fraction": float(count / cfg.horizon),
                        }
                    )
    return rows


def write_parameter_sources(cfg, out):
    env = UAVEnv(cfg, cfg.seed)
    resolved_reference = env.reference_utility
    reference_mode = _effective_reference_mode(cfg)
    reference_is_derived = reference_mode in {
        "qos_aspiration",
        "random_policy_calibration",
    }
    if reference_mode == "random_policy_calibration":
        reference_basis = (
            "Mean of resolved per-user references calibrated on independent "
            "pre-training random-policy states"
        )
    elif reference_mode == "qos_aspiration":
        reference_basis = (
            "Mean of resolved per-user references derived from the declared "
            "QoS aspiration"
        )
    else:
        reference_basis = "Mean of the explicitly supplied per-user references"
    if cfg.randomize_pt_profiles:
        profile_status = "Derived sensitivity draw"
        profile_basis = (
            "Seeded truncated-normal draw from declared experimental bounds "
            "and dispersion"
        )
        profile_source_title = ""
        profile_source_locator = "See pt_user_profiles.csv and seed_manifest.csv"
        profile_source_url = ""
    elif any(
        value is not None
        for value in (
            cfg.alpha_gain_by_user,
            cfg.beta_loss_by_user,
            cfg.loss_aversion_by_user,
        )
    ):
        profile_status = "Experimental assumption"
        profile_basis = "Explicit user-specific experimental input"
        profile_source_title = ""
        profile_source_locator = ""
        profile_source_url = ""
    else:
        profile_status = "Derived from source-backed scalar"
        profile_basis = "Scalar source-backed default broadcast to every user"
        profile_source_title = "Mitsis et al., Sensors 2020"
        profile_source_locator = "Eq. (4), Table 1"
        profile_source_url = "https://www.mdpi.com/1424-8220/20/8/2434"
    reference_vector_status = (
        "Derived study design"
        if reference_is_derived
        else "Experimental assumption"
    )
    rows = [
        (
            "carrier_frequency_hz",
            cfg.carrier_frequency_hz,
            "LTE macro evaluation assumption; a compatibility anchor, not a mandate",
            "Compatible choice",
            "3GPP TR 36.931 v15.0.0",
            "Table 5.3.3-1 (Macro system assumptions)",
            "https://www.etsi.org/deliver/etsi_tr/136900_136999/136931/15.00.00_60/tr_136931v150000p.pdf",
        ),
        (
            "system_bandwidth_hz",
            cfg.system_bandwidth_hz,
            "LTE macro evaluation assumption; a compatibility anchor, not a mandate",
            "Compatible choice",
            "3GPP TR 36.931 v15.0.0",
            "Table 5.3.3-1 (Macro system assumptions)",
            "https://www.etsi.org/deliver/etsi_tr/136900_136999/136931/15.00.00_60/tr_136931v150000p.pdf",
        ),
        (
            "noise_psd_dbm_hz",
            cfg.noise_psd_dbm_hz,
            "Rounded room-temperature thermal-noise density kT0",
            "Source-backed",
            "ITU-R Report BT.2383-2",
            "Thermal-noise-density entries (kT0 = -173.98 dBm/Hz)",
            "https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2383-2-2019-PDF-E.pdf",
        ),
        (
            "receiver_noise_figure_db",
            cfg.receiver_noise_figure_db,
            "LTE macro UE receiver assumption; a compatibility anchor",
            "Compatible choice",
            "3GPP TR 36.931 v15.0.0",
            "Table 5.3.3-1 (UE noise figure)",
            "https://www.etsi.org/deliver/etsi_tr/136900_136999/136931/15.00.00_60/tr_136931v150000p.pdf",
        ),
        (
            "user_height_m",
            cfg.user_height_m,
            "Terrestrial user-terminal height; a compatibility anchor",
            "Compatible choice",
            "3GPP TR 38.901 v18.1.0",
            "Calibration parameter table (UT height = 1.5 m)",
            "https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.01.00_60/tr_138901v180100p.pdf",
        ),
        (
            "los_a",
            cfg.los_a,
            "Urban A2G LoS parameter",
            "Source-backed",
            "Al-Hourani et al., IEEE WCL 2014",
            "Table I (Urban environment)",
            "https://doi.org/10.1109/LWC.2014.2342736",
        ),
        (
            "los_b",
            cfg.los_b,
            "Urban A2G LoS parameter",
            "Source-backed",
            "Al-Hourani et al., IEEE WCL 2014",
            "Table I (Urban environment)",
            "https://doi.org/10.1109/LWC.2014.2342736",
        ),
        (
            "eta_los_db",
            cfg.eta_los_db,
            "Urban excessive LoS loss",
            "Source-backed",
            "Al-Hourani et al., IEEE WCL 2014",
            "Table I (Urban environment)",
            "https://doi.org/10.1109/LWC.2014.2342736",
        ),
        (
            "eta_nlos_db",
            cfg.eta_nlos_db,
            "Urban excessive NLoS loss",
            "Source-backed",
            "Al-Hourani et al., IEEE WCL 2014",
            "Table I (Urban environment)",
            "https://doi.org/10.1109/LWC.2014.2342736",
        ),
        (
            "alpha_gain",
            cfg.alpha_gain,
            "Sensitivity parameter used by the supplied UAV-MEC paper/code",
            "Source-backed",
            "Mitsis et al., Sensors 2020",
            "Eq. (4), Table 1; supplied parameters.py lines 72-73",
            "https://www.mdpi.com/1424-8220/20/8/2434",
        ),
        (
            "beta_loss",
            cfg.beta_loss,
            "Set equal to alpha as in the supplied UAV-MEC paper",
            "Source-backed",
            "Mitsis et al., Sensors 2020",
            "Text below Eq. (4); supplied code uses one exponent a_n",
            "https://www.mdpi.com/1424-8220/20/8/2434",
        ),
        (
            "loss_aversion",
            cfg.loss_aversion,
            "Loss-aversion parameter used by the supplied UAV-MEC paper/code",
            "Source-backed",
            "Mitsis et al., Sensors 2020",
            "Eq. (4), Table 1; supplied parameters.py lines 72-73",
            "https://www.mdpi.com/1424-8220/20/8/2434",
        ),
        (
            "reference_utility",
            resolved_reference,
            reference_basis,
            (
                "Derived study design"
                if reference_is_derived
                else "Experimental assumption"
            ),
            "This implementation",
            "Per-user values are documented in PT_USER_PROFILES.md",
            "",
        ),
        (
            "alpha_gain_by_user",
            json.dumps(env.alpha_gain_user.tolist()),
            profile_basis,
            profile_status,
            profile_source_title,
            profile_source_locator,
            profile_source_url,
        ),
        (
            "beta_loss_by_user",
            json.dumps(env.beta_loss_user.tolist()),
            profile_basis,
            profile_status,
            profile_source_title,
            profile_source_locator,
            profile_source_url,
        ),
        (
            "loss_aversion_by_user",
            json.dumps(env.loss_aversion_user.tolist()),
            profile_basis,
            profile_status,
            profile_source_title,
            profile_source_locator,
            profile_source_url,
        ),
        (
            "reference_utility_by_user",
            json.dumps(env.reference_utility_user.tolist()),
            reference_basis,
            reference_vector_status,
            "This implementation",
            "See pt_user_profiles.csv and seed_manifest.csv",
            "",
        ),
    ]
    sourced = {row[0] for row in rows}
    for key, value in asdict(cfg).items():
        if key not in sourced:
            rows.append(
                (
                    key,
                    value,
                    "Scenario/training design; sensitivity analysis required",
                    "Experimental assumption",
                    "",
                    "",
                    "",
                )
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "parameter",
            "value",
            "basis",
            "status",
            "source_title",
            "source_locator",
            "source_url",
        ],
    )
    frame.to_csv(out / "parameter_sources.csv", index=False)
    lines = [
        "# Parameter provenance",
        "",
        "| Parameter | Value | Status | Basis/source |",
        "|---|---:|---|---|",
    ]
    for _, row in frame.iterrows():
        source = row.source_title
        if row.source_locator:
            source += f", {row.source_locator}" if source else row.source_locator
        if row.source_url:
            source += f" ([link]({row.source_url}))"
        basis = row.basis + (f"; {source}" if source else "")
        lines.append(
            f"| `{row.parameter}` | {row.value} | {row.status} | {basis} |"
        )
    (out / "PARAMETER_SOURCES.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_user_pt_profiles(configs, out):
    """Write every resolved profile actually used by the experiment."""
    if isinstance(configs, Config):
        configs = [configs]
    frames = []
    for cfg in configs:
        env = UAVEnv(cfg, cfg.seed)
        if cfg.randomize_pt_profiles:
            behavioral_origin = (
                "seeded truncated-normal sensitivity draw; distribution is "
                "a study assumption"
            )
            profile_seed = _derived_seed(
                cfg.pt_profile_seed_base, cfg.seed, 0
            )
            alpha_seeds = [
                _derived_seed(cfg.pt_profile_seed_base, cfg.seed, 0, 0, user)
                for user in range(cfg.n_user)
            ]
            beta_seeds = [
                _derived_seed(cfg.pt_profile_seed_base, cfg.seed, 0, 1, user)
                for user in range(cfg.n_user)
            ]
            loss_seeds = [
                _derived_seed(cfg.pt_profile_seed_base, cfg.seed, 0, 2, user)
                for user in range(cfg.n_user)
            ]
        elif cfg.alpha_gain_by_user is None:
            behavioral_origin = "broadcast source-backed scalar default"
            profile_seed = np.nan
            alpha_seeds = beta_seeds = loss_seeds = [np.nan] * cfg.n_user
        else:
            behavioral_origin = "explicit experimental user profile"
            profile_seed = np.nan
            alpha_seeds = beta_seeds = loss_seeds = [np.nan] * cfg.n_user
        reference_mode = _effective_reference_mode(cfg)
        if reference_mode == "random_policy_calibration":
            calibration_population = (
                cfg.reference_calibration_population_size
                if cfg.reference_calibration_population_size is not None
                else cfg.n_user
            )
            reference_origin = (
                "independent pre-training random-policy calibration at "
                f"n_user={calibration_population}; exact prefix frozen"
            )
        elif reference_mode == "qos_aspiration":
            reference_origin = "derived QoS aspiration"
        else:
            reference_origin = "explicit experimental reference"
        frames.append(
            pd.DataFrame(
                {
                    "replicate_seed": [cfg.seed] * cfg.n_user,
                    "profile_seed": [profile_seed] * cfg.n_user,
                    "alpha_profile_seed": alpha_seeds,
                    "beta_profile_seed": beta_seeds,
                    "loss_aversion_profile_seed": loss_seeds,
                    "user_index": np.arange(cfg.n_user),
                    "user_label": [
                        f"U{index + 1}" for index in range(cfg.n_user)
                    ],
                    "alpha_u": env.alpha_gain_user,
                    "beta_u": env.beta_loss_user,
                    "k_u": env.loss_aversion_user,
                    "reference_utility_u": env.reference_utility_user,
                    "behavioral_origin": [behavioral_origin] * cfg.n_user,
                    "reference_origin": [reference_origin] * cfg.n_user,
                    "reference_calibration_episodes": [
                        cfg.reference_calibration_episodes
                    ]
                    * cfg.n_user,
                    "reference_calibration_seed_base": [
                        cfg.reference_calibration_seed_base
                    ]
                    * cfg.n_user,
                    "reference_calibration_population_size": [
                        (
                            cfg.reference_calibration_population_size
                            if cfg.reference_calibration_population_size
                            is not None
                            else cfg.n_user
                        )
                    ]
                    * cfg.n_user,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(out / "pt_user_profiles.csv", index=False)
    lines = [
        "# Resolved user-specific Prospect Theory profiles",
        "",
        "These exact profiles are exposed to every RL agent. EUT uses each",
        "user's reference linearly; PT also uses alpha, beta and k. A sampled",
        "profile is frozen for the whole Monte Carlo replicate and shared by",
        "all nine methods. Sampling spreads and bounds are sensitivity-design",
        "assumptions, not standard or empirically fitted population values.",
        "",
        "| Replicate | User | $\\alpha_u$ | $\\beta_u$ | $k_u$ | $U_{ref,u}$ | Origin |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in frame.iterrows():
        origin = (
            f"behavior: {row.behavioral_origin}; "
            f"reference: {row.reference_origin}"
        )
        lines.append(
            f"| {int(row.replicate_seed)} | {row.user_label} | "
            f"{row.alpha_u:.6g} | {row.beta_u:.6g} | {row.k_u:.6g} | "
            f"{row.reference_utility_u:.6g} | {origin} |"
        )
    (out / "PT_USER_PROFILES.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return frame


def write_algorithm_sources(out):
    rows = [
        (
            "Random",
            "Uniform sample over feasible actions",
            "Custom sanity baseline",
            "No privileged model information",
            "",
            "",
        ),
        (
            "Greedy_Physical_Utility",
            "Maximizes success-outcome physical utility and ignores outage risk",
            "Custom myopic baseline",
            "Distinct from the EUT expected-utility oracle",
            "",
            "",
        ),
        (
            "EUT_Myopic_Oracle",
            "Enumerates feasible actions and maximizes one-step expected utility centered on the selected user's reference",
            "von Neumann-Morgenstern expected utility",
            "Exact one-step enumeration is this study's oracle; not a learned policy",
            "von Neumann and Morgenstern, Theory of Games and Economic Behavior",
            "https://www.jstor.org/stable/j.ctt1r2gkx",
        ),
        (
            "PT_Myopic_Oracle",
            "Enumerates actions and maximizes the selected user's expected PT value",
            "Mitsis et al. UAV-MEC PT model",
            "The PT function is sourced; exhaustive one-step enumeration is a custom oracle",
            "Mitsis et al., Sensors 2020",
            "https://www.mdpi.com/1424-8220/20/8/2434",
        ),
        (
            "EUT_DQN",
            "Replay, target network, epsilon-greedy and Huber TD loss",
            "Mnih et al. 2015",
            "Algorithmic adaptation with a task-specific NumPy MLP and Adam",
            "Mnih et al., DQN",
            "https://www.nature.com/articles/nature14236",
        ),
        (
            "EUT_DDQN",
            "Online action selection and target-network evaluation",
            "van Hasselt et al. 2016",
            "Same EUT reward and training budget as EUT-DQN",
            "Mnih et al., DQN; van Hasselt et al., Double DQN",
            "https://www.nature.com/articles/nature14236;https://ojs.aaai.org/index.php/AAAI/article/view/10295",
        ),
        (
            "EUT_Dueling_DQN",
            "Separate V(s) and A(s,a) heads with mean aggregation",
            "Wang et al. 2016",
            "Uses the vanilla DQN target and the same EUT reward",
            "Mnih et al., DQN; Wang et al., Dueling Network Architectures",
            "https://www.nature.com/articles/nature14236;https://proceedings.mlr.press/v48/wangf16.html",
        ),
        (
            "EUT_D3QN",
            "Double-DQN target plus dueling architecture",
            "DDQN + Dueling",
            "Direct architectural control for PT-D3QN",
            "van Hasselt et al., Double DQN; Wang et al., Dueling Network Architectures",
            "https://ojs.aaai.org/index.php/AAAI/article/view/10295;https://proceedings.mlr.press/v48/wangf16.html",
        ),
        (
            "PT_D3QN",
            "EUT-D3QN architecture with each realized outcome transformed by the selected user's PT value function",
            "Proposed composition: DQN + DDQN + Dueling + PT",
            "Sensors 2020 supplies PT only, not D3QN; the controlled composition is proposed here",
            "Mnih et al., DQN; van Hasselt et al., Double DQN; Wang et al., Dueling Network Architectures; Mitsis et al., Sensors 2020",
            "https://www.nature.com/articles/nature14236;https://ojs.aaai.org/index.php/AAAI/article/view/10295;https://proceedings.mlr.press/v48/wangf16.html;https://www.mdpi.com/1424-8220/20/8/2434",
        ),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "method",
            "implemented_definition",
            "basis",
            "audit_note",
            "source_titles",
            "source_urls",
        ],
    )
    frame.to_csv(out / "algorithm_sources.csv", index=False)
    lines = [
        "# Algorithm provenance and audit",
        "",
        "| Method | Implemented definition | Audit note | Source |",
        "|---|---|---|---|",
    ]
    for _, row in frame.iterrows():
        titles = row.source_titles.split(";") if row.source_titles else []
        urls = row.source_urls.split(";") if row.source_urls else []
        links = [
            f"[{title.strip()}]({url.strip()})"
            for title, url in zip(titles, urls)
        ]
        source = row.basis + (" — " + "; ".join(links) if links else "")
        lines.append(
            f"| `{row.method}` | {row.implemented_definition} | "
            f"{row.audit_note} | {source} |"
        )
    (out / "ALGORITHM_SOURCES.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_fairness_protocol(out):
    text = """# Fair comparison protocol

## Common controls

- One physical UAV environment and action mask for every method.
- One transmission tuple: `(UAV, channel, user, transmission mode)`, plus an
  always-valid idle/energy-conservation decision.
- One global transmission per slot; simultaneous multi-UAV scheduling is not
  represented by this action model.
- For each Monte Carlo replicate, identical episode horizon, behavioral profile
  and held-out evaluation scenario seeds for all nine methods.
- Independent replicate seeds generate independent training scenarios, PT
  profile draws and held-out evaluation scenarios through separate seed spaces.
- Identical seeded exogenous random-number streams for user placement, shadowing,
  arrivals and the uniform link-outcome draw. Different selected links can still
  produce different binary outcomes because their success probabilities differ.
- Identical state, replay size, batch size, optimizer, learning rate, exploration
  schedule, target-update interval and environment interactions for all RL agents.
- The same per-user `(alpha_u, beta_u, k_u, U_ref,u)` profile is visible to all
  RL agents. EUT uses the selected user's reference linearly; PT additionally
  applies that user's curvature and loss aversion.
- A randomized profile is sampled once per user and replicate, then frozen. It
  is never redrawn per action, slot, episode, method or network architecture.
- Per-user/per-parameter seed namespaces make behavioral profiles nested across
  the 5/15/30/45/60-user sweep. Evaluation references are calibrated once at
  the maximum population and exact prefixes are reused at smaller counts.
- When `training_profile_count > 1`, all RL agents cycle through the same
  separately seeded `(alpha,beta,k,U_ref)` training bank; the evaluation profile
  is held out.
- A calibrated `U_ref,u` uses independent pre-training random-policy scenarios;
  neither training trajectories nor held-out evaluation outcomes are used.
- EUT-D3QN and PT-D3QN have identical initial parameters for each replicate.
- PT rewards use one fixed declared divisor across seeds and user counts.
- FIFO packet arrivals, admissions, drops, partial service, completion and delay
  use the same definitions for all methods. Link success is not mislabeled PDR.
- Energy and non-empty-queue constraints are active in the shared transmission
  mask. Idle with backlog receives an explicit no-service opportunity cost;
  idle without demand receives zero reward and is not classified as a loss.
- The state contains both mean/base link loss and instantaneous CSI, making the
  AR(1) shadow transition observable to a feed-forward agent.

## Controlled differences

- DQN/DDQN/Dueling/D3QN change only the documented architecture or TD target.
- EUT-D3QN and PT-D3QN change only the outcome-reward transformation.
- PT Myopic Oracle and PT-D3QN call the same user-indexed PT value function.
- PT uses objective success/failure probabilities, matching the supplied Sensors
  2020 model; no probability-weighting function is silently added.

## Interpretation

Random, Greedy and the two Myopic Oracles are non-learning references. They are
reported separately from the RL architectural ablation. All final comparisons
use common physical metrics; EUT and PT training returns are not compared as if
they shared the same scale. Truncated-normal spreads and bounds are declared
sensitivity assumptions, not source-backed behavioral population estimates.
The Monte Carlo replicate is the inferential unit. Exact sign-flip and paired
Student-t tests are both reported. The sign-flip test states its exchangeability
assumption, and Holm correction is applied only to a declared non-redundant
secondary family. Utility shortfall is the prespecified primary endpoint.
"""
    (out / "FAIRNESS_PROTOCOL.md").write_text(text, encoding="utf-8")


def write_model_scope(cfg, out):
    env = UAVEnv(cfg, cfg.seed)
    resolved_reference = env.reference_utility
    reference_vector = ", ".join(
        f"{value:.6g}" for value in env.reference_utility_user
    )
    reference_mode = _effective_reference_mode(cfg)
    if reference_mode == "random_policy_calibration":
        calibration_population = (
            cfg.reference_calibration_population_size
            if cfg.reference_calibration_population_size is not None
            else cfg.n_user
        )
        reference_description = (
            "`U_ref,u` is the average, over independent pre-training "
            "calibration states, of user `u`'s expected physical utility "
            "under a uniform choice among that user's feasible "
            "UAV/channel/mode actions. A separate uniform random feasible "
            "action advances each calibration trajectory. No training or "
            "held-out evaluation outcome enters this estimate. The recorded "
            f"calibration population size is `{calibration_population}`; "
            "user-count sweeps reuse exact reference prefixes."
        )
    elif reference_mode == "explicit":
        reference_description = (
            "`U_ref,u` is supplied explicitly by the experimenter and is an "
            "experimental input unless estimated from behavioral data."
        )
    else:
        reference_description = (
            "`U_ref,u` is the declared QoS-aspiration expression: "
            "`w_rate*rate_ref + w_reliability*qos_reliability + "
            "w_fairness - w_energy*energy_ref`."
        )
    if cfg.randomize_pt_profiles:
        profile_description = (
            "For each Monte Carlo replicate, alpha, beta and k are sampled "
            "once per user from the declared truncated-normal sensitivity "
            "distributions (alpha and beta independently) and then frozen "
            "for all methods and episodes. "
            "Only the distribution centers are source-backed; standard "
            "deviations and bounds are study assumptions."
        )
    else:
        profile_description = (
            "When no explicit user profile is supplied, the scalar defaults "
            "are broadcast to all users. An explicit heterogeneous profile "
            "is an experimental input unless estimated from behavioral data."
        )
    text = f"""# Physical-model and provenance scope

## What the model is

This is a unit-consistent, abstract **downlink UAV-BS** scheduling model. A
static UAV-BS selects one orthogonal channel, one ground user and one discrete
transmission mode per slot. This is one global transmission per slot, not
simultaneous multi-UAV scheduling. The mean air-to-ground loss follows Al-Hourani et
al.; custom correlated log-normal shadowing and an analytical Rayleigh
block-fading success probability are added by this implementation.

The model is **not a full 3GPP channel or system-level simulator**. In
particular, [3GPP TR 36.777](https://www.3gpp.org/dynareport/36777.htm) studies
enhanced LTE support for aerial *UEs*, whereas this code treats the UAV as the
base station and the receiver as a terrestrial user. The 2 GHz carrier, 10 MHz
bandwidth, 9 dB terrestrial-UE noise figure and 1.5 m user height are only
3GPP-compatible anchors. They do not make the combined channel model 3GPP
compliant.

## Explicit study choices

- UAV/user counts, area, altitude, traffic, queues, power limits, spectral
  efficiencies, QoS targets, shadowing dynamics, utility weights and all RL
  hyperparameters are experimental assumptions unless individually marked in
  `parameter_sources.csv`.
- The linearly spaced power/spectral-efficiency modes are abstract modes, not a
  3GPP MCS table.
- Energy is transmit-radio energy only: `P_tx * slot_duration`. Propulsion,
  hovering, circuitry and computation energy are excluded.
- FIFO packet queues support partial head-packet service. Offered/admitted PDR,
  admission drops, transmission success and packet delay are separate metrics.
- Idle is always valid. With backlog, zero service is evaluated against the
  mean reference of currently backlogged users and receives a corresponding
  EUT/PT opportunity cost. Without backlog, idle has zero reward and no risk
  event is defined. Attempt-conditional risk metrics are saved separately.
- Both static mean/base link loss and instantaneous CSI are observed. This
  preserves the Markov description under the declared AR(1) shadow process.
- Inter-UAV co-channel interference, antenna patterns, mobility and trajectory
  control are excluded. Channels are orthogonal in the current abstraction.
- `w_fairness` is retained as a backward-compatible field name, but its reward
  component is a custom per-user **service-equity shaping term**, not Jain's
  fairness index. Jain's index is computed separately only as an evaluation
  metric.

## Prospect-Theory adaptation

The value function and scalar centers `alpha=beta=0.2`, `k=1.2` follow Eq. (4),
the simulation table and supplied code for [Mitsis et al., Sensors
2020](https://www.mdpi.com/1424-8220/20/8/2434). The paper's reference point is
tied to local-processing satisfaction; that meaning cannot be copied directly
to this UAV scheduling utility. {reference_description}

{profile_description} For the written first replicate, the resolved reference
vector is `[{reference_vector}]` and its mean is `{resolved_reference:.12g}`.
The complete replicate-specific table is in `pt_user_profiles.csv`.

All RL agents observe the same per-user profile in their state. EUT methods use
`U - U_ref,u`; PT methods additionally apply the selected user's
`alpha_u`, `beta_u` and `k_u`. Thus EUT-D3QN and PT-D3QN still differ only by
the nonlinear reward transformation. PT uses the fixed declared reward divisor
`{cfg.pt_reward_scale_constant:.6g}` across every population and seed. The supplied implementation uses
objective probabilities and no probability weighting, so this code does the
same. PT is applied to each one-step realized utility outcome, not to a learned
distribution of whole-episode returns.

When `training_profile_count={cfg.training_profile_count}` exceeds one, all RL
methods train on the same separately seeded behavioral profile bank, while the
written evaluation profile is held out. The training bank also applies an
independent uniform relative perturbation of up to
`{100 * cfg.training_reference_jitter_fraction:.6g}%` to each `U_ref,u`; this is
a declared robustness assumption, not a fitted human distribution. A value of one is a fixed-profile
control and does not establish behavioral generalization.

## Interpretation of runs

`--quick` changes training to 8 episodes, evaluation to 3 episodes, horizon to
25 slots, batch size to 32, warm-up to 64 transitions and target update to 30
updates. It is a software smoke test only. Scientific conclusions require a
declared training budget, convergence checks, multiple independent Monte Carlo
replicates, uncertainty intervals and sensitivity analyses.
"""
    (out / "MODEL_SCOPE.md").write_text(text, encoding="utf-8")


def plot_common_scenario(cfg, path):
    env = UAVEnv(cfg, cfg.seed)
    env.reset()
    labels = [name.replace("_", " ") for name in ALL_METHODS]
    fig, (left, ax) = plt.subplots(
        1, 2, figsize=(13, 6.7), gridspec_kw={"width_ratios": [0.85, 1.5]}
    )
    left.axis("off")
    left.set_title("Common decision problem", weight="bold")
    left.text(0.02, 0.96, "Policies", weight="bold", transform=left.transAxes)
    left.text(
        0.04,
        0.91,
        "\n".join("• " + label for label in labels),
        va="top",
        transform=left.transAxes,
        linespacing=1.35,
    )
    details = (
        "Tx action: (UAV, channel, user, mode)\n"
        "Idle: always valid; backlog opportunity cost\n"
        f"|A| = {env.n_actions:,} including idle\n\n"
        f"{cfg.n_uav} UAV-BSs at {cfg.uav_altitude_m:g} m\n"
        f"{cfg.n_user} ground users at {cfg.user_height_m:g} m\n"
        f"{cfg.n_channel} orthogonal channels\n"
        f"{cfg.n_mode} power/SE modes\n\n"
        f"$f_c$ = {cfg.carrier_frequency_hz / 1e9:g} GHz\n"
        f"$B$ = {cfg.system_bandwidth_hz / 1e6:g} MHz"
    )
    left.text(
        0.02,
        0.40,
        details,
        va="top",
        transform=left.transAxes,
        linespacing=1.45,
    )
    ax.set_title(f"Physical map for seed {cfg.seed}", weight="bold")
    ax.set_xlim(0, cfg.area_side_m)
    ax.set_ylim(0, cfg.area_side_m)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)
    ax.scatter(
        env.user_xy[:, 0],
        env.user_xy[:, 1],
        s=38,
        c="#70ad47",
        edgecolor="#385723",
        label="Ground user",
        zorder=3,
    )
    ax.scatter(
        env.uav_xy[:, 0],
        env.uav_xy[:, 1],
        s=160,
        marker="^",
        c="#4472c4",
        edgecolor="#203864",
        label="UAV-BS projection",
        zorder=4,
    )
    # User labels become unreadable in the 30--60 user scaling scenarios.
    # Every point and nearest-UAV connector is still drawn; labels are kept for
    # smaller scenarios where they remain useful.
    if cfg.n_user <= 20:
        for index, (x_value, y_value) in enumerate(env.user_xy):
            ax.annotate(
                f"U{index + 1}",
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    for index, (x_value, y_value) in enumerate(env.uav_xy):
        ax.annotate(
            f"UAV{index + 1}",
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
        )
    nearest = np.argmin(
        np.linalg.norm(
            env.uav_xy[:, None, :] - env.user_xy[None, :, :], axis=2
        ),
        axis=0,
    )
    cmap = plt.get_cmap("tab20", max(1, min(cfg.n_channel, 20)))
    line_styles = ["-", "--", ":", "-."]
    for user, uav in enumerate(nearest):
        channel = user % cfg.n_channel
        ax.plot(
            [env.uav_xy[uav, 0], env.user_xy[user, 0]],
            [env.uav_xy[uav, 1], env.user_xy[user, 1]],
            color=cmap(channel % 20),
            alpha=min(0.45, 12.0 / cfg.n_user),
            lw=1 if cfg.n_user <= 20 else 0.6,
            linestyle=line_styles[channel % len(line_styles)],
        )
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.01,
        0.01,
        "Lines show illustrative nearest-UAV geometry only; they are not policy actions.",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.suptitle(
        "Common Scenario for All Compared Policies", fontsize=14, weight="bold"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def summarize_evaluation(evaluation, out):
    metric_columns = EVALUATION_METRICS
    grouping = ["method", "replicate_seed"]
    if "n_user" in evaluation.columns:
        grouping.insert(0, "n_user")
    seed_level = (
        evaluation.groupby(grouping)[metric_columns]
        .mean()
        .reset_index()
    )
    seed_level.to_csv(out / "evaluation_seed_summary.csv", index=False)

    rows = []
    for method in ALL_METHODS:
        subset = seed_level[seed_level.method == method]
        for metric in metric_columns:
            values = subset[metric].dropna().to_numpy()
            mean = float(values.mean()) if len(values) else np.nan
            std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            ci95 = (
                float(student_t.ppf(0.975, len(values) - 1))
                * std
                / np.sqrt(len(values))
                if len(values) > 1
                else np.nan
            )
            rows.append(
                {
                    "n_user": (
                        int(subset.n_user.iloc[0])
                        if "n_user" in subset.columns and len(subset)
                        else np.nan
                    ),
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "std_across_replicates": std,
                    "n_replicates": len(values),
                    "ci95_half_width": ci95,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    return seed_level, summary


def _student_t_summary(frame, grouping, metrics):
    """Summarize replicate-level values without treating episodes as n."""
    rows = []
    for keys, group in frame.groupby(grouping, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(grouping, keys))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            count = len(values)
            mean = float(values.mean()) if count else np.nan
            std = float(values.std(ddof=1)) if count > 1 else np.nan
            ci95 = (
                float(student_t.ppf(0.975, count - 1))
                * std
                / np.sqrt(count)
                if count > 1
                else np.nan
            )
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "mean": mean,
                    "std_across_replicates": std,
                    "n_replicates": count,
                    "ci95_half_width": ci95,
                }
            )
    return pd.DataFrame(rows)


def summarize_user_evaluation(user_evaluation, out, write_raw=True):
    """Save episode, replicate and across-replicate per-user plot data."""
    if write_raw:
        user_evaluation.to_csv(out / "evaluation_user_raw.csv", index=False)
    metrics = [
        "selection_count",
        "selection_fraction",
        "initial_packet_count",
        "initial_queue_mbits",
        "delivered_mbits",
        "delivered_share",
        "final_queue_mbits",
        "offered_packets",
        "admitted_packets",
        "generated_arrival_packets",
        "dropped_packets",
        "delivered_packets",
        "offered_pdr",
        "admitted_pdr",
        "packet_drop_rate",
        "mean_packet_delay_s",
        "transmission_attempts",
        "transmission_successes",
        "transmission_success_rate",
        "selected_utility_mean",
        "selected_centered_eut_utility_mean",
        "selected_pt_subjective_utility_mean",
        "selected_reference_satisfaction",
        "selected_loss_event",
        "selected_utility_shortfall",
        "selected_qos_reference_satisfaction",
        "selected_qos_loss_event",
        "selected_qos_utility_shortfall",
        "selected_throughput_mbps",
        "selected_transmission_success_rate",
        "selected_transmission_outage",
        "selected_energy_j",
        "alpha_u",
        "beta_u",
        "k_u",
        "reference_utility_u",
        "qos_reference_utility_u",
    ]
    seed_level = (
        user_evaluation.groupby(
            [
                "n_user",
                "method",
                "replicate_seed",
                "user_index",
                "user_label",
            ],
            as_index=False,
        )[metrics]
        .mean()
    )
    seed_level.to_csv(out / "evaluation_user_seed_summary.csv", index=False)
    summary = _student_t_summary(
        seed_level,
        ["n_user", "method", "user_index", "user_label"],
        metrics,
    )
    summary.to_csv(out / "evaluation_user_summary.csv", index=False)
    return seed_level, summary


def summarize_selection_evaluation(selection_evaluation, out, write_raw=True):
    """Save marginal UAV/channel/user/mode selection frequencies."""
    if write_raw:
        selection_evaluation.to_csv(
            out / "evaluation_selection_raw.csv", index=False
        )
    metrics = ["selection_count", "selection_fraction"]
    seed_level = (
        selection_evaluation.groupby(
            [
                "n_user",
                "method",
                "replicate_seed",
                "dimension",
                "category_index",
                "category_label",
            ],
            as_index=False,
        )[metrics]
        .mean()
    )
    seed_level.to_csv(
        out / "evaluation_selection_seed_summary.csv", index=False
    )
    summary = _student_t_summary(
        seed_level,
        [
            "n_user",
            "method",
            "dimension",
            "category_index",
            "category_label",
        ],
        metrics,
    )
    summary.to_csv(out / "evaluation_selection_summary.csv", index=False)
    return seed_level, summary


def write_plot_ready_tables(
    training,
    summary,
    user_seed_level,
    selection_seed_level,
    out,
):
    """Write compact numeric inputs for later publication figures."""
    physical_metrics = {
        "utility",
        "throughput_mbps",
        "offered_pdr",
        "packet_drop_rate",
        "mean_packet_delay_s",
        "transmission_success_rate",
        "energy_per_attempt_j",
        "final_jain_fairness",
    }
    risk_metrics = {
        "reference_satisfaction",
        "loss_event",
        "utility_shortfall",
        "pt_subjective_utility",
        "qos_reference_satisfaction",
        "qos_loss_event",
        "qos_utility_shortfall",
    }
    physical = summary[summary.metric.isin(physical_metrics)].copy()
    risk = summary[summary.metric.isin(risk_metrics)].copy()
    physical.to_csv(out / "plot_physical_metrics.csv", index=False)
    risk.to_csv(out / "plot_pt_risk_metrics.csv", index=False)

    curve = (
        training.groupby(["n_user", "method", "episode"])
        .agg(
            objective_return_mean=("objective_return", "mean"),
            objective_return_std=("objective_return", "std"),
            physical_utility_return_mean=("physical_utility_return", "mean"),
            physical_utility_return_std=("physical_utility_return", "std"),
            n_replicates=("replicate_seed", "nunique"),
        )
        .reset_index()
    )
    critical = curve.n_replicates.map(
        lambda count: (
            float(student_t.ppf(0.975, count - 1)) if count > 1 else np.nan
        )
    )
    curve["objective_return_ci95_half_width"] = (
        critical
        * curve.objective_return_std
        / np.sqrt(curve.n_replicates)
    )
    curve["physical_utility_return_ci95_half_width"] = (
        critical
        * curve.physical_utility_return_std
        / np.sqrt(curve.n_replicates)
    )
    curve.to_csv(out / "plot_training_curves.csv", index=False)
    user_seed_level.to_csv(out / "plot_user_metrics.csv", index=False)
    selection_seed_level.to_csv(
        out / "plot_selection_distribution.csv", index=False
    )
    return {
        "physical": physical,
        "risk": risk,
        "training_curves": curve,
    }


def write_paired_pt_comparison(seed_level, out):
    """Paired PT-D3QN minus EUT-D3QN comparison across replicates."""
    metric_columns = [
        metric for metric in EVALUATION_METRICS if metric in seed_level.columns
    ]
    seed_column = (
        "replicate_seed"
        if "replicate_seed" in seed_level.columns
        else "training_seed"
    )
    eut = seed_level[seed_level.method == "EUT_D3QN"].set_index(seed_column)
    prospect = seed_level[seed_level.method == "PT_D3QN"].set_index(
        seed_column
    )
    common_seeds = eut.index.intersection(prospect.index)
    n_user = (
        int(seed_level.n_user.iloc[0])
        if "n_user" in seed_level.columns and len(seed_level)
        else np.nan
    )
    rows = []
    for metric in metric_columns:
        eut_values = eut.loc[common_seeds, metric].to_numpy(dtype=float)
        pt_values = prospect.loc[common_seeds, metric].to_numpy(dtype=float)
        pairwise_finite = np.isfinite(eut_values) & np.isfinite(pt_values)
        eut_values = eut_values[pairwise_finite]
        pt_values = pt_values[pairwise_finite]
        differences = pt_values - eut_values
        count = len(differences)
        mean_difference = (
            float(differences.mean()) if count else np.nan
        )
        std_difference = (
            float(differences.std(ddof=1)) if count > 1 else np.nan
        )
        ci95 = (
            float(student_t.ppf(0.975, count - 1))
            * std_difference
            / np.sqrt(count)
            if count > 1
            else np.nan
        )
        if count > 1 and not np.allclose(differences, differences[0]):
            test = ttest_rel(pt_values, eut_values)
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        elif count > 1 and np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        elif count > 1:
            statistic = float(np.copysign(np.inf, differences[0]))
            p_value = 0.0
        else:
            statistic, p_value = np.nan, np.nan
        rows.append(
            {
                "n_user": n_user,
                "metric": metric,
                "difference_definition": "PT_D3QN - EUT_D3QN",
                "n_paired_replicates": count,
                "mean_paired_difference": mean_difference,
                "std_paired_difference": std_difference,
                "student_t_ci95_half_width": ci95,
                "paired_t_statistic": statistic,
                "paired_t_p_value_two_sided": p_value,
                "exact_sign_flip_p_value_two_sided": _exact_sign_flip_p_value(
                    differences
                ),
                "is_prespecified_primary_endpoint": (
                    metric == PRIMARY_ENDPOINT
                ),
                "holm_secondary_family_member": (
                    metric in SECONDARY_INFERENCE_FAMILY
                ),
                "exact_sign_flip_assumption": (
                    "paired differences are exchangeable under sign reversal"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    family = frame["holm_secondary_family_member"].to_numpy(dtype=bool)
    frame["holm_adjusted_paired_t_p_value"] = np.nan
    frame["holm_adjusted_sign_flip_p_value"] = np.nan
    frame.loc[family, "holm_adjusted_paired_t_p_value"] = _holm_adjust(
        frame.loc[family, "paired_t_p_value_two_sided"].to_numpy(dtype=float)
    )
    frame.loc[family, "holm_adjusted_sign_flip_p_value"] = _holm_adjust(
        frame.loc[
            family, "exact_sign_flip_p_value_two_sided"
        ].to_numpy(dtype=float)
    )
    frame.to_csv(out / "paired_pt_vs_eut_d3qn.csv", index=False)
    return frame


def _exact_sign_flip_p_value(differences):
    """Exact paired two-sided randomization test for at most 20 nonzero pairs."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = np.abs(values[np.abs(values) > 1e-15])
    if len(values) == 0:
        return np.nan
    if len(nonzero) == 0:
        return 1.0
    if len(nonzero) > 20:
        return np.nan
    sums = np.array([0.0])
    for value in nonzero:
        sums = np.r_[sums + value, sums - value]
    observed = abs(float(values.sum()))
    tolerance = 1e-12 * max(1.0, observed)
    return float(np.mean(np.abs(sums) + tolerance >= observed))


def _holm_adjust(p_values):
    """Holm family-wise-error adjustment, preserving NaN entries."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted
    ordered = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _plot_ci95(values):
    """Student-t half-width using only finite observations."""
    clean = pd.Series(values).dropna().to_numpy(dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) <= 1:
        return 0.0
    return float(
        student_t.ppf(0.975, len(clean) - 1)
        * clean.std(ddof=1)
        / np.sqrt(len(clean))
    )


def plot_results(training, seed_level, out):
    metrics = [
        ("utility", "Physical utility"),
        ("throughput_mbps", "Throughput (Mbit/s)"),
        ("offered_pdr", "Offered packet delivery ratio"),
        ("energy_per_attempt_j", "Energy/attempt (J)"),
        ("final_jain_fairness", "Final Jain fairness"),
    ]
    colors = ["#999999"] * 4 + ["#4472c4"] * 4 + ["#c00000"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        grouped = seed_level.groupby("method")[metric]
        means = grouped.mean().reindex(ALL_METHODS)
        errors = grouped.apply(_plot_ci95).reindex(ALL_METHODS)
        ax.bar(
            np.arange(len(means)),
            means.values,
            yerr=errors.values,
            color=colors,
            capsize=3,
        )
        ax.set_xticks(np.arange(len(means)))
        ax.set_xticklabels(
            [name.replace("_", " ") for name in means.index],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric in {"offered_pdr", "final_jain_fairness"}:
            ax.set_ylim(0, 1.05)
        elif metric in {"throughput_mbps", "energy_per_attempt_j"}:
            ax.set_ylim(bottom=0)

    # Plot common physical utility because EUT/PT objective returns do not
    # share the same semantic scale.
    curves = training.groupby(["method", "episode"])[
        "physical_utility_return"
    ].agg(["mean", "std", "count"]).reset_index()
    for name, group in curves.groupby("method"):
        group = group.sort_values("episode")
        mean_curve = group["mean"].rolling(10, min_periods=1).mean()
        axes[5].plot(
            group.episode,
            mean_curve,
            label=name.replace("_", " "),
        )
        if int(group["count"].max()) > 1:
            critical = student_t.ppf(0.975, group["count"] - 1)
            half_width = (
                critical * group["std"] / np.sqrt(group["count"])
            ).rolling(10, min_periods=1).mean()
            axes[5].fill_between(
                group.episode,
                mean_curve - half_width,
                mean_curve + half_width,
                alpha=0.12,
            )
    axes[5].set_title("Training physical-utility return")
    axes[5].set_xlabel("Episode")
    axes[5].legend(fontsize=7)
    axes[5].grid(alpha=0.25)
    fig.suptitle(
        "Common physical metrics (95% Student-t CI; bounded axes clipped)",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pt_risk_results(seed_level, out):
    """Plot secondary downside/reference metrics under the common PT profile."""
    metrics = [
        ("reference_satisfaction", "Reference satisfaction (higher is better)"),
        ("loss_event", "Loss-event rate (lower is better)"),
        ("utility_shortfall", "Mean utility shortfall (lower is better)"),
        ("pt_subjective_utility", "Mean PT subjective utility"),
    ]
    colors = ["#999999"] * 4 + ["#4472c4"] * 4 + ["#c00000"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        grouped = seed_level.groupby("method")[metric]
        means = grouped.mean().reindex(ALL_METHODS)
        errors = grouped.apply(_plot_ci95).reindex(ALL_METHODS)
        ax.bar(
            np.arange(len(means)),
            means.values,
            yerr=errors.values,
            color=colors,
            capsize=3,
        )
        ax.set_xticks(np.arange(len(means)))
        ax.set_xticklabels(
            [name.replace("_", " ") for name in means.index],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric in {"reference_satisfaction", "loss_event"}:
            ax.set_ylim(0, 1.05)
        elif metric == "utility_shortfall":
            ax.set_ylim(bottom=0)
    fig.suptitle(
        "Reference-dependent and downside-risk metrics "
        "(95% Student-t CI; bounded axes clipped)",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(out / "pt_risk_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_seeds(seed, seeds_text):
    if seeds_text is None:
        seeds = [seed]
    else:
        try:
            seeds = [
                int(item.strip())
                for item in seeds_text.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--seeds must be comma-separated integers"
            ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds cannot be empty")
    if min(seeds) < 0:
        raise argparse.ArgumentTypeError("--seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    return seeds


def parse_user_counts(text):
    """Parse a positive, unique user-count scaling design."""
    if text is None:
        return None
    try:
        counts = [
            int(item.strip()) for item in text.split(",") if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--user-counts must be comma-separated integers"
        ) from exc
    if not counts:
        raise argparse.ArgumentTypeError("--user-counts cannot be empty")
    if min(counts) < 1:
        raise argparse.ArgumentTypeError(
            "--user-counts values must all be positive"
        )
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError(
            "--user-counts must not contain duplicates"
        )
    return counts


def parse_user_profile(text, n_user, option_name):
    """Parse a comma-separated user profile without inventing heterogeneity."""
    if text is None:
        return None
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must contain comma-separated numbers"
        ) from exc
    if len(values) != n_user:
        raise argparse.ArgumentTypeError(
            f"{option_name} requires exactly {n_user} values, one per user"
        )
    if not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(
            f"{option_name} must contain only finite values"
        )
    return values


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_reproducibility_manifest(out, cfg, run_seeds):
    """Record the executable environment without exposing private paths."""
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release_version": RELEASE_VERSION,
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": __import__("scipy").__version__,
        "matplotlib_version": __import__("matplotlib").__version__,
        "code_sha256": _sha256_file(Path(__file__)),
        "n_user": cfg.n_user,
        "n_actions": cfg.n_uav * cfg.n_channel * cfg.n_user * cfg.n_mode + 1,
        "run_seeds": list(run_seeds),
        "profile_sampling_scope": (
            "one evaluation-profile draw per user and Monte Carlo replicate; "
            "frozen across methods and evaluation episodes. When "
            "training_profile_count > 1, all RL methods cycle through the "
            "same separate seeded training-profile bank"
            if cfg.randomize_pt_profiles
            else "homogeneous or explicit fixed profile; no random draw"
        ),
        "reference_calibration_population_size": (
            cfg.reference_calibration_population_size
            if cfg.reference_calibration_population_size is not None
            else cfg.n_user
        ),
    }
    (out / "reproducibility_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def benchmark_method_complexity(name, cfg, agent=None, repetitions=20):
    """Measure inference cost on one fixed state; timing is hardware-specific."""
    env = UAVEnv(cfg, _derived_seed(900_000, cfg.seed))
    state = env.reset()
    policy_rng = np.random.default_rng(_derived_seed(900_001, cfg.seed))

    def decide():
        current_mask = env.valid_mask()
        if agent is not None:
            q_values = agent.net.forward(state[None])[0]
            return int(np.argmax(np.where(current_mask, q_values, -np.inf)))
        ids = np.flatnonzero(current_mask)
        if name == "Random":
            return int(policy_rng.choice(ids))
        outcomes = env.outcomes()
        if name == "Greedy_Physical_Utility":
            scores = outcomes["u_success"]
        elif name == "EUT_Myopic_Oracle":
            scores = env.eut_scores()
        elif name == "PT_Myopic_Oracle":
            scores = env.pt_scores()
        else:
            raise ValueError(name)
        return int(np.argmax(np.where(current_mask, scores, -np.inf)))

    for _ in range(3):
        decide()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        decide()
        samples.append((time.perf_counter_ns() - started) / 1e6)

    packed_mask_bytes = int(np.ceil(env.n_actions / 8))
    # Estimate assuming unshared state/next-state arrays. Python containers and
    # possible object sharing are implementation-specific, so this is not a
    # rigorous lower bound on resident memory.
    replay_payload_bytes = (
        2 * env.state_dim * np.dtype(np.float32).itemsize
        + packed_mask_bytes
        + 3 * np.dtype(np.float64).itemsize
    )
    parameter_count = 0 if agent is None else agent.parameter_count
    return {
        "n_user": cfg.n_user,
        "method": name,
        "replicate_seed": cfg.seed,
        "state_dimension": env.state_dim,
        "transmission_action_count": env.n_transmission_actions,
        "total_action_count_including_idle": env.n_actions,
        "network_parameter_count": parameter_count,
        "network_parameter_bytes_float64": parameter_count * 8,
        "packed_mask_bytes_per_transition": packed_mask_bytes,
        "unpacked_bool_mask_bytes_per_transition": env.n_actions,
        "unshared_replay_array_payload_estimate_bytes_per_transition": (
            replay_payload_bytes
        ),
        "unshared_replay_array_payload_estimate_mib_at_capacity": (
            replay_payload_bytes * cfg.replay_size / 1024**2
        ),
        "decision_time_ms_median": float(np.median(samples)),
        "decision_time_ms_mean": float(np.mean(samples)),
        "timing_repetitions": repetitions,
        "timing_note": "hardware-dependent fixed-state microbenchmark",
    }


def write_algorithm_complexity(rows, out):
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "algorithm_complexity_raw.csv", index=False)
    identifiers = [
        "n_user",
        "method",
        "state_dimension",
        "transmission_action_count",
        "total_action_count_including_idle",
        "network_parameter_count",
        "network_parameter_bytes_float64",
        "packed_mask_bytes_per_transition",
        "unpacked_bool_mask_bytes_per_transition",
        "unshared_replay_array_payload_estimate_bytes_per_transition",
        "unshared_replay_array_payload_estimate_mib_at_capacity",
        "timing_note",
    ]
    summary = (
        raw.groupby(identifiers, as_index=False, dropna=False)
        .agg(
            n_replicates=("replicate_seed", "nunique"),
            decision_time_ms_median_across_replicates=(
                "decision_time_ms_median",
                "median",
            ),
            decision_time_ms_mean_across_replicates=(
                "decision_time_ms_mean",
                "mean",
            ),
        )
    )
    summary.to_csv(out / "algorithm_complexity.csv", index=False)
    return raw, summary


def build_nested_sweep_replicate_configs(base_cfg, user_counts, run_seeds):
    """Resolve one max-population behavioral profile per replicate.

    All four evaluation descriptors (alpha, beta, k and U_ref) are resolved at
    the largest user count. Smaller experiments receive exact prefixes, which
    prevents population-size-dependent recalibration from confounding the
    user-count sensitivity analysis.
    """
    counts = tuple(int(value) for value in user_counts)
    if not counts or min(counts) < 1 or len(set(counts)) != len(counts):
        raise ValueError("user_counts must be unique positive integers")
    maximum = max(counts)
    maximum_cfg = replace(
        base_cfg,
        n_user=maximum,
        alpha_gain_by_user=None,
        beta_loss_by_user=None,
        loss_aversion_by_user=None,
        reference_utility_by_user=None,
        randomize_pt_profiles=True,
        reference_utility_mode="random_policy_calibration",
        reference_calibration_population_size=None,
    )
    maximum_resolved = {
        int(seed): build_replicate_config(maximum_cfg, int(seed))
        for seed in run_seeds
    }
    by_count = {}
    for count in counts:
        count_base = replace(
            base_cfg,
            n_user=count,
            alpha_gain_by_user=None,
            beta_loss_by_user=None,
            loss_aversion_by_user=None,
            reference_utility_by_user=None,
            randomize_pt_profiles=True,
            reference_utility_mode="random_policy_calibration",
            reference_calibration_population_size=maximum,
        )
        resolved = []
        for seed in run_seeds:
            source = maximum_resolved[int(seed)]
            resolved_cfg = replace(
                count_base,
                seed=int(seed),
                alpha_gain_by_user=tuple(source.alpha_gain_by_user[:count]),
                beta_loss_by_user=tuple(source.beta_loss_by_user[:count]),
                loss_aversion_by_user=tuple(
                    source.loss_aversion_by_user[:count]
                ),
                reference_utility_by_user=tuple(
                    source.reference_utility_by_user[:count]
                ),
            )
            UAVEnv(resolved_cfg, resolved_cfg.seed)
            resolved.append(resolved_cfg)
        by_count[count] = (count_base, resolved)
    return by_count


def run_experiment(
    base_cfg,
    run_seeds,
    out,
    quick_run_only=False,
    compact_return=False,
    resolved_replicate_configs=None,
):
    """Run one fixed user-count experiment and persist all raw/plot data."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {out}. Use a new folder so "
            "stale files cannot be mixed with this run."
        )
    out.mkdir(parents=True, exist_ok=True)
    if resolved_replicate_configs is None:
        replicate_configs = [
            build_replicate_config(base_cfg, seed) for seed in run_seeds
        ]
    else:
        replicate_configs = list(resolved_replicate_configs)
        if [cfg.seed for cfg in replicate_configs] != list(run_seeds):
            raise ValueError(
                "resolved_replicate_configs must follow run_seeds exactly"
            )
        if any(cfg.n_user != base_cfg.n_user for cfg in replicate_configs):
            raise ValueError(
                "resolved_replicate_configs have inconsistent user counts"
            )

    random.seed(run_seeds[0])
    np.random.seed(run_seeds[0])
    first_cfg = replicate_configs[0]
    plot_common_scenario(first_cfg, out / "common_scenario.png")
    write_parameter_sources(first_cfg, out)
    profile_frame = write_user_pt_profiles(replicate_configs, out)
    training_profile_frame = write_training_pt_profiles(
        replicate_configs, out
    )
    write_seed_manifest(replicate_configs, out)
    write_algorithm_sources(out)
    write_fairness_protocol(out)
    write_model_scope(first_cfg, out)

    all_training = []
    all_evaluation = []
    all_complexity = []
    user_sink = CsvRowSink(out / "evaluation_user_raw.csv")
    selection_sink = CsvRowSink(out / "evaluation_selection_raw.csv")
    step_sink = (
        CsvRowSink(out / "evaluation_step_raw.csv")
        if base_cfg.save_step_details
        else None
    )

    for cfg in replicate_configs:
        seed = cfg.seed
        training_profiles = build_training_profile_bank(cfg)
        for name in BASELINES:
            print(
                f"Evaluating {name} (n_user={cfg.n_user}, "
                f"replicate_seed={seed})"
            )
            all_evaluation.extend(
                evaluate(
                    name,
                    cfg,
                    user_detail_rows=user_sink,
                    selection_detail_rows=selection_sink,
                    step_sink=step_sink,
                )
            )
            all_complexity.append(benchmark_method_complexity(name, cfg))

        # Train and evaluate one agent at a time. This keeps only one replay
        # buffer resident, which is essential for the 30--60 user scenarios.
        for name in RL_SPECS:
            print(f"Training {name} (n_user={cfg.n_user}, seed={seed})")
            agent, history = train_agent(name, cfg, training_profiles)
            all_training.extend(history)
            print(
                f"Evaluating {name} (n_user={cfg.n_user}, "
                f"replicate_seed={seed})"
            )
            all_evaluation.extend(
                evaluate(
                    name,
                    cfg,
                    agent,
                    user_detail_rows=user_sink,
                    selection_detail_rows=selection_sink,
                    step_sink=step_sink,
                )
            )
            all_complexity.append(
                benchmark_method_complexity(name, cfg, agent)
            )
            del agent
            gc.collect()

    expected_episodes = (
        len(ALL_METHODS) * len(replicate_configs) * base_cfg.eval_episodes
    )
    _close_and_verify_csv_sink(
        user_sink,
        expected_episodes * base_cfg.n_user,
        "per-user evaluation rows",
    )
    selection_categories_per_episode = (
        base_cfg.n_uav
        + base_cfg.n_channel
        + base_cfg.n_user
        + base_cfg.n_mode
        + 4
    )
    _close_and_verify_csv_sink(
        selection_sink,
        expected_episodes * selection_categories_per_episode,
        "selection-marginal rows",
    )
    if step_sink is not None:
        expected_step_rows = (
            len(ALL_METHODS)
            * len(replicate_configs)
            * base_cfg.eval_episodes
            * base_cfg.horizon
        )
        _close_and_verify_csv_sink(
            step_sink, expected_step_rows, "slot-level evaluation rows"
        )

    training = pd.DataFrame(all_training)
    evaluation = pd.DataFrame(all_evaluation)
    complexity_raw, complexity_summary = write_algorithm_complexity(
        all_complexity, out
    )
    training.to_csv(out / "training_history.csv", index=False)
    evaluation.to_csv(out / "evaluation_raw.csv", index=False)
    seed_level, summary = summarize_evaluation(evaluation, out)
    paired = write_paired_pt_comparison(seed_level, out)
    user_evaluation = pd.read_csv(out / "evaluation_user_raw.csv")
    user_seed_level, user_summary = summarize_user_evaluation(
        user_evaluation, out, write_raw=False
    )
    del user_evaluation
    gc.collect()
    selection_evaluation = pd.read_csv(out / "evaluation_selection_raw.csv")
    selection_seed_level, selection_summary = (
        summarize_selection_evaluation(
            selection_evaluation, out, write_raw=False
        )
    )
    del selection_evaluation
    gc.collect()
    plot_tables = write_plot_ready_tables(
        training,
        summary,
        user_seed_level,
        selection_seed_level,
        out,
    )

    config_payload = asdict(base_cfg)
    config_payload["release_version"] = RELEASE_VERSION
    config_payload["run_seeds"] = list(run_seeds)
    config_payload["resolved_replicate_configs"] = [
        asdict(cfg) for cfg in replicate_configs
    ]
    config_payload["comparison_methods"] = ALL_METHODS
    config_payload["quick_run_only"] = bool(quick_run_only)
    config_payload["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    config_payload["profile_sampling_scope"] = (
        "once_per_user_per_monte_carlo_replicate_frozen_for_all_methods;"
        "sweep_references_resolved_at_maximum_population_and_prefix_frozen"
        if base_cfg.randomize_pt_profiles
        else "fixed_nonrandom_profile"
    )
    (out / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    write_reproducibility_manifest(out, first_cfg, run_seeds)
    plot_results(training, seed_level, out)
    plot_pt_risk_results(seed_level, out)

    physical = summary[
        summary.metric.isin(
            [
                "utility",
                "throughput_mbps",
                "offered_pdr",
                "energy_per_attempt_j",
                "final_jain_fairness",
            ]
        )
    ].pivot(index="method", columns="metric", values="mean")
    physical = physical.reindex(ALL_METHODS)
    print(f"\nCommon physical metrics for n_user={base_cfg.n_user}")
    print(physical.round(4))

    result = {
        "n_user": base_cfg.n_user,
        "output": out,
        "config": first_cfg,
        "replicate_configs": replicate_configs,
        "profiles": profile_frame,
        "training_profiles": training_profile_frame,
        "training": training,
        "evaluation": evaluation,
        "seed_level": seed_level,
        "summary": summary,
        "paired": paired,
        "user_seed_level": user_seed_level,
        "user_summary": user_summary,
        "selection_seed_level": selection_seed_level,
        "selection_summary": selection_summary,
        "plot_physical": plot_tables["physical"],
        "plot_risk": plot_tables["risk"],
        "plot_training": plot_tables["training_curves"],
        "complexity_raw": complexity_raw,
        "complexity_summary": complexity_summary,
    }
    if compact_return:
        for key in ("training", "evaluation", "user_summary", "selection_summary"):
            result.pop(key, None)
    return result


def write_user_count_sweep_outputs(results, out, user_counts, run_seeds, quick):
    """Write compact cross-user-count tables without duplicating raw files."""
    out = Path(out)

    def combine(key):
        return pd.concat([result[key] for result in results], ignore_index=True)

    combined = {
        "user_count_evaluation_seed_summary.csv": combine("seed_level"),
        "user_count_summary.csv": combine("summary"),
        "user_count_physical_metrics.csv": combine("plot_physical"),
        "user_count_pt_risk_metrics.csv": combine("plot_risk"),
        "user_count_paired_pt_vs_eut_d3qn.csv": combine("paired"),
        "user_count_training_curves.csv": combine("plot_training"),
        "user_count_user_seed_summary.csv": combine("user_seed_level"),
        "user_count_selection_seed_summary.csv": combine(
            "selection_seed_level"
        ),
        "user_count_algorithm_complexity.csv": combine(
            "complexity_summary"
        ),
    }
    combined["user_count_pt_profiles.csv"] = pd.concat(
        [
            result["profiles"].assign(n_user=result["n_user"])
            for result in results
        ],
        ignore_index=True,
    )
    combined["user_count_training_pt_profiles.csv"] = pd.concat(
        [
            result["training_profiles"].assign(n_user=result["n_user"])
            for result in results
        ],
        ignore_index=True,
    )
    profile_checks = []
    for (replicate_seed, user_index), group in combined[
        "user_count_pt_profiles.csv"
    ].groupby(["replicate_seed", "user_index"]):
        row = {
            "replicate_seed": replicate_seed,
            "user_index": user_index,
            "user_label": f"U{int(user_index) + 1}",
            "observed_user_counts": ",".join(
                str(value) for value in sorted(group.n_user.unique())
            ),
        }
        sampled_traits_match = True
        for metric in ("alpha_u", "beta_u", "k_u"):
            values = group[metric].to_numpy(dtype=float)
            match = bool(np.all(values == values[0]))
            row[f"{metric}_nested_match"] = match
            row[f"{metric}_value"] = float(values[0])
            sampled_traits_match &= match
        reference_values = group["reference_utility_u"].to_numpy(dtype=float)
        reference_match = bool(np.all(reference_values == reference_values[0]))
        row["all_sampled_pt_traits_nested_match"] = sampled_traits_match
        row["reference_utility_nested_match"] = reference_match
        row["reference_utility_value"] = float(reference_values[0])
        row["reference_utility_spread"] = float(
            reference_values.max() - reference_values.min()
        )
        row["all_behavioral_parameters_nested_match"] = bool(
            sampled_traits_match and reference_match
        )
        profile_checks.append(row)
    nesting = pd.DataFrame(profile_checks)
    if not nesting.all_behavioral_parameters_nested_match.all():
        raise RuntimeError(
            "PT traits or reference utilities are not nested across "
            "user-count runs"
        )
    combined["user_count_profile_nesting_check.csv"] = nesting
    training_checks = []
    for keys, group in combined[
        "user_count_training_pt_profiles.csv"
    ].groupby(
        ["replicate_seed", "training_profile_index", "user_index"]
    ):
        replicate_seed, profile_index, user_index = keys
        row = {
            "replicate_seed": replicate_seed,
            "training_profile_index": profile_index,
            "user_index": user_index,
            "user_label": f"U{int(user_index) + 1}",
            "observed_user_counts": ",".join(
                str(value) for value in sorted(group.n_user.unique())
            ),
        }
        all_match = True
        for metric in ("alpha_u", "beta_u", "k_u", "reference_utility_u"):
            values = group[metric].to_numpy(dtype=float)
            match = bool(np.all(values == values[0]))
            row[f"{metric}_nested_match"] = match
            row[f"{metric}_value"] = float(values[0])
            all_match &= match
        row["all_training_behavioral_parameters_nested_match"] = all_match
        training_checks.append(row)
    training_nesting = pd.DataFrame(training_checks)
    if not training_nesting.all_training_behavioral_parameters_nested_match.all():
        raise RuntimeError(
            "Training PT profiles are not nested across user-count runs"
        )
    combined["user_count_training_profile_nesting_check.csv"] = (
        training_nesting
    )
    for filename, frame in combined.items():
        frame.to_csv(out / filename, index=False)

    manifest_rows = []
    for result in results:
        cfg = result["config"]
        env = UAVEnv(cfg, cfg.seed)
        profile_counts = result["profiles"].groupby("replicate_seed").size()
        manifest_rows.append(
            {
                "n_user": cfg.n_user,
                "folder": result["output"].name,
                "state_dimension": env.state_dim,
                "action_count": env.n_actions,
                "transmission_action_count": env.n_transmission_actions,
                "n_replicates": len(run_seeds),
                "train_episodes": cfg.train_episodes,
                "eval_episodes": cfg.eval_episodes,
                "horizon": cfg.horizon,
                "aggregate_arrival_rate_pps": (
                    cfg.n_user * cfg.arrival_rate_pps
                ),
                "aggregate_initial_packet_capacity": (
                    cfg.n_user
                    * int(cfg.max_queue_bits // cfg.packet_size_bits)
                ),
                "randomize_pt_profiles": cfg.randomize_pt_profiles,
                "reference_mode": _effective_reference_mode(cfg),
                "reference_calibration_population_size": (
                    cfg.reference_calibration_population_size
                ),
                "profile_rows_per_replicate_min": int(profile_counts.min()),
                "profile_rows_per_replicate_max": int(profile_counts.max()),
                "profile_sampling_scope": (
                    "once_per_user_per_replicate_frozen_all_methods"
                ),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        out / "user_count_sweep_manifest.csv", index=False
    )
    sweep_config = {
        "user_counts": list(user_counts),
        "folder_pattern": "users_NNN",
        "run_seeds": list(run_seeds),
        "quick_run_only": bool(quick),
        "random_pt_profiles_required": True,
        "reference_mode": "random_policy_calibration",
        "reference_calibration_scope": (
            "once_per_replicate_at_maximum_user_count_then_exact_prefixes"
        ),
        "training_profile_count": int(results[0]["config"].training_profile_count),
        "evaluation_profile_held_out": bool(
            results[0]["config"].training_profile_count > 1
        ),
        "slot_level_output_enabled": bool(
            results[0]["config"].save_step_details
        ),
        "raw_data_location": (
            "Each users_NNN folder contains training, episode, slot, user and "
            "selection raw CSV files; root files are compact plot tables."
        ),
        "interpretation": (
            "This sweep jointly scales population, aggregate offered load, "
            "state dimension and action dimension; it is not an isolated "
            "action-space-only intervention."
        ),
        "cross_count_nesting": (
            "For a common seed and episode, existing users retain identical "
            "alpha, beta, k, U_ref, geometry, initial packet counts, per-link "
            "shadowing, arrivals and shadow innovations; additional users "
            "are appended."
        ),
    }
    (out / "user_count_sweep_config.json").write_text(
        json.dumps(sweep_config, indent=2), encoding="utf-8"
    )
    readme = """# User-count scaling outputs

Each `users_NNN` directory is a complete, independent experiment for one user
count. Random PT traits and evaluation references are resolved once per user
and Monte Carlo replicate at the maximum population, then exact prefixes are
frozen for all nine methods and all episodes. The root `user_count_*.csv`
files concatenate replicate-level or plot-ready values across user counts; raw
episode data remain in the corresponding subdirectory to avoid duplication.

For statistical inference, use replicate-level rows, not evaluation episodes or
users as independent samples.
"""
    (out / "README_USER_COUNT_SWEEP.md").write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_pt_d3qn")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seeds",
        help=(
            "Comma-separated Monte Carlo replicate seeds; final inference "
            "should use at least 10 independent replicates"
        ),
    )
    parser.add_argument("--n-uav", type=int, default=3)
    parser.add_argument("--n-channel", type=int, default=4)
    parser.add_argument("--n-user", type=int, default=6)
    parser.add_argument(
        "--user-counts",
        help=(
            "Run a user-count sweep into users_NNN subfolders, e.g. "
            "5,15,30,45,60. Sweep mode requires randomized PT profiles."
        ),
    )
    parser.add_argument("--n-mode", type=int, default=3)
    parser.add_argument("--pt-alpha", type=float, default=0.20)
    parser.add_argument("--pt-beta", type=float, default=0.20)
    parser.add_argument("--loss-aversion", type=float, default=1.20)
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--random-pt-profiles",
        dest="pt_profile_mode",
        action="store_const",
        const="random",
        help=(
            "Sample alpha_u, beta_u and k_u once per user/replicate from "
            "declared truncated-normal sensitivity distributions (default "
            "when no explicit user vectors are supplied)"
        ),
    )
    profile_group.add_argument(
        "--homogeneous-pt-profiles",
        dest="pt_profile_mode",
        action="store_const",
        const="homogeneous",
        help=(
            "Broadcast scalar alpha, beta and k to every user. This is an "
            "explicit control mode and is rejected by --user-counts."
        ),
    )
    parser.add_argument("--pt-profile-seed-base", type=int, default=300_000)
    parser.add_argument("--pt-alpha-std", type=float, default=0.03)
    parser.add_argument("--pt-alpha-lower", type=float, default=0.10)
    parser.add_argument("--pt-alpha-upper", type=float, default=0.30)
    parser.add_argument("--pt-beta-std", type=float, default=0.03)
    parser.add_argument("--pt-beta-lower", type=float, default=0.10)
    parser.add_argument("--pt-beta-upper", type=float, default=0.30)
    parser.add_argument("--loss-aversion-std", type=float, default=0.15)
    parser.add_argument("--loss-aversion-lower", type=float, default=1.00)
    parser.add_argument("--loss-aversion-upper", type=float, default=1.60)
    parser.add_argument(
        "--pt-reward-scale",
        type=float,
        default=1.60,
        help="Fixed PT reward divisor shared by every seed and user count",
    )
    parser.add_argument(
        "--no-step-details",
        action="store_true",
        help="Disable evaluation_step_raw.csv to reduce disk use",
    )
    parser.add_argument(
        "--training-profile-count",
        type=int,
        default=1,
        help=(
            "Number of shared randomized PT profiles cycled during RL "
            "training; values >1 reserve the evaluation profile as held out"
        ),
    )
    parser.add_argument(
        "--training-profile-seed-base", type=int, default=350_000
    )
    parser.add_argument(
        "--training-reference-jitter",
        type=float,
        default=0.20,
        help=(
            "Uniform relative U_ref perturbation for held-out behavioral "
            "training profiles (default: +/-20%%)"
        ),
    )
    parser.add_argument(
        "--arrival-rate-pps", type=float, default=0.25
    )
    parser.add_argument(
        "--initial-radio-energy-j", type=float, default=2.0
    )
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--update-frequency", type=int, default=4)
    parser.add_argument(
        "--pt-alpha-users",
        help="Comma-separated alpha_u values; exactly one per user",
    )
    parser.add_argument(
        "--pt-beta-users",
        help="Comma-separated beta_u values; exactly one per user",
    )
    parser.add_argument(
        "--loss-aversion-users",
        help="Comma-separated k_u values; exactly one per user",
    )
    parser.add_argument(
        "--reference-utility-users",
        help="Comma-separated U_ref,u values; exactly one per user",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("qos_aspiration", "random_policy_calibration", "explicit"),
        help=(
            "Reference construction. Default: random-policy calibration with "
            "--random-pt-profiles, otherwise QoS aspiration; explicit is "
            "inferred when --reference-utility-users is provided"
        ),
    )
    parser.add_argument("--reference-calibration-episodes", type=int)
    parser.add_argument(
        "--reference-calibration-seed-base", type=int, default=200_000
    )
    parser.add_argument("--evaluation-seed-base", type=int, default=100_000)
    args = parser.parse_args()

    if min(args.n_uav, args.n_channel, args.n_user, args.n_mode) < 1:
        parser.error("All scenario counts must be >= 1")
    try:
        run_seeds = parse_seeds(args.seed, args.seeds)
        user_counts = parse_user_counts(args.user_counts)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    explicit_profile_text = any(
        value is not None
        for value in (
            args.pt_alpha_users,
            args.pt_beta_users,
            args.loss_aversion_users,
            args.reference_utility_users,
        )
    )
    if user_counts is not None and explicit_profile_text:
        parser.error(
            "--user-counts cannot be combined with explicit per-user PT "
            "vectors because their required length changes across folders"
        )

    try:
        alpha_by_user = parse_user_profile(
            args.pt_alpha_users,
            args.n_user,
            "--pt-alpha-users",
        )
        beta_by_user = parse_user_profile(
            args.pt_beta_users,
            args.n_user,
            "--pt-beta-users",
        )
        loss_aversion_by_user = parse_user_profile(
            args.loss_aversion_users,
            args.n_user,
            "--loss-aversion-users",
        )
        reference_by_user = parse_user_profile(
            args.reference_utility_users,
            args.n_user,
            "--reference-utility-users",
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    explicit_behavioral_profiles = any(
        profile is not None
        for profile in (alpha_by_user, beta_by_user, loss_aversion_by_user)
    )
    if args.pt_profile_mode == "random" and explicit_behavioral_profiles:
        parser.error(
            "--random-pt-profiles cannot be combined with explicit "
            "alpha/beta/loss-aversion user profiles"
        )
    if args.pt_profile_mode == "homogeneous" and explicit_behavioral_profiles:
        parser.error(
            "--homogeneous-pt-profiles cannot be combined with explicit "
            "alpha/beta/loss-aversion user profiles"
        )
    if args.pt_profile_mode == "random":
        random_pt_profiles = True
    elif args.pt_profile_mode == "homogeneous":
        random_pt_profiles = False
    elif explicit_behavioral_profiles:
        random_pt_profiles = False
    else:
        # Safe research default: do not silently broadcast one profile to all
        # users when the proposed method is user-conditioned.
        random_pt_profiles = True
    if user_counts is not None and not random_pt_profiles:
        parser.error(
            "--user-counts requires randomized PT profiles; remove "
            "--homogeneous-pt-profiles"
        )
    if args.training_profile_count < 1:
        parser.error("--training-profile-count must be >= 1")
    if args.training_profile_count > 1 and not random_pt_profiles:
        parser.error(
            "--training-profile-count > 1 requires randomized PT profiles"
        )
    if args.reference_mode is None:
        if reference_by_user is not None:
            reference_mode = "explicit"
        elif random_pt_profiles:
            reference_mode = "random_policy_calibration"
        else:
            reference_mode = "qos_aspiration"
    else:
        reference_mode = args.reference_mode
    if reference_mode == "explicit" and reference_by_user is None:
        parser.error(
            "--reference-mode explicit requires --reference-utility-users"
        )
    if (
        reference_mode == "random_policy_calibration"
        and reference_by_user is not None
    ):
        parser.error(
            "Random-policy calibration cannot be combined with explicit "
            "--reference-utility-users"
        )
    if reference_mode == "qos_aspiration" and reference_by_user is not None:
        parser.error(
            "QoS-aspiration mode cannot be combined with explicit "
            "--reference-utility-users"
        )
    if (
        user_counts is not None
        and reference_mode != "random_policy_calibration"
    ):
        parser.error(
            "--user-counts requires --reference-mode "
            "random_policy_calibration so U_ref,u is resolved once per user"
        )

    base_cfg = Config(
        seed=run_seeds[0],
        n_uav=args.n_uav,
        n_channel=args.n_channel,
        n_user=args.n_user,
        n_mode=args.n_mode,
        alpha_gain=args.pt_alpha,
        beta_loss=args.pt_beta,
        loss_aversion=args.loss_aversion,
        alpha_gain_by_user=alpha_by_user,
        beta_loss_by_user=beta_by_user,
        loss_aversion_by_user=loss_aversion_by_user,
        reference_utility_by_user=reference_by_user,
        randomize_pt_profiles=random_pt_profiles,
        pt_profile_seed_base=args.pt_profile_seed_base,
        alpha_gain_std=args.pt_alpha_std,
        alpha_gain_lower=args.pt_alpha_lower,
        alpha_gain_upper=args.pt_alpha_upper,
        beta_loss_std=args.pt_beta_std,
        beta_loss_lower=args.pt_beta_lower,
        beta_loss_upper=args.pt_beta_upper,
        loss_aversion_std=args.loss_aversion_std,
        loss_aversion_lower=args.loss_aversion_lower,
        loss_aversion_upper=args.loss_aversion_upper,
        pt_reward_scale_constant=args.pt_reward_scale,
        save_step_details=not args.no_step_details,
        training_profile_count=args.training_profile_count,
        training_profile_seed_base=args.training_profile_seed_base,
        training_reference_jitter_fraction=args.training_reference_jitter,
        arrival_rate_pps=args.arrival_rate_pps,
        initial_radio_energy_j=args.initial_radio_energy_j,
        replay_size=args.replay_size,
        update_frequency=args.update_frequency,
        reference_utility_mode=reference_mode,
        reference_calibration_seed_base=args.reference_calibration_seed_base,
        evaluation_seed_base=args.evaluation_seed_base,
    )
    if args.quick:
        base_cfg.train_episodes = 8
        base_cfg.eval_episodes = 3
        base_cfg.horizon = 25
        base_cfg.warmup = 64
        base_cfg.batch_size = 32
        base_cfg.target_update = 30
        if args.reference_calibration_episodes is None:
            base_cfg.reference_calibration_episodes = 5
    if args.train_episodes is not None:
        base_cfg.train_episodes = args.train_episodes
    if args.eval_episodes is not None:
        base_cfg.eval_episodes = args.eval_episodes
    if args.reference_calibration_episodes is not None:
        base_cfg.reference_calibration_episodes = (
            args.reference_calibration_episodes
        )
    if base_cfg.train_episodes < 1 or base_cfg.eval_episodes < 1:
        parser.error("Episode counts must be >= 1")
    try:
        UAVEnv(base_cfg, base_cfg.seed)
    except ValueError as exc:
        parser.error(str(exc))
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error(
            f"Output directory is not empty: {out}. Choose a new --output "
            "folder to prevent stale-result mixing."
        )
    out.mkdir(parents=True, exist_ok=True)
    try:
        if user_counts is None:
            run_experiment(
                base_cfg,
                run_seeds,
                out,
                quick_run_only=args.quick,
            )
        else:
            results = []
            nested_configs = build_nested_sweep_replicate_configs(
                base_cfg, user_counts, run_seeds
            )
            for count in user_counts:
                count_cfg, resolved_configs = nested_configs[count]
                count_out = out / f"users_{count:03d}"
                results.append(
                    run_experiment(
                        count_cfg,
                        run_seeds,
                        count_out,
                        quick_run_only=args.quick,
                        compact_return=True,
                        resolved_replicate_configs=resolved_configs,
                    )
                )
            write_user_count_sweep_outputs(
                results,
                out,
                user_counts,
                run_seeds,
                args.quick,
            )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
