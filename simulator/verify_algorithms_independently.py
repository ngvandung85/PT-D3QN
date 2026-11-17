#!/usr/bin/env python3
"""Independent numerical audit for the PT-D3QN UAV implementation.

This script recomputes the key equations without calling the corresponding
scoring formula, and checks neural-network gradients by finite differences.
It is intentionally separate from the functional unit-test suite.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pt_d3qn_uav import (
    Adam,
    Agent,
    Config,
    QNet,
    RL_SPECS,
    UAVEnv,
    _exact_sign_flip_p_value,
    build_nested_sweep_replicate_configs,
    build_replicate_config,
    evaluation_episode_seed,
    heuristic_action,
)


PASSED = []


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    PASSED.append(name)


def check_close(name, actual, expected, rtol=1e-10, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    PASSED.append(name)


def physical_equation_checks():
    cfg = Config(seed=19, n_uav=2, n_channel=2, n_user=3, n_mode=3)
    env = UAVEnv(cfg, 19)
    env.reset()
    action = 7
    uav, channel, user, mode = env.actions[action]
    outcomes = env.outcomes()

    dx = env.uav_xy[uav, 0] - env.user_xy[user, 0]
    dy = env.uav_xy[uav, 1] - env.user_xy[user, 1]
    horizontal = math.hypot(dx, dy)
    vertical = cfg.uav_altitude_m - cfg.user_height_m
    distance = math.hypot(horizontal, vertical)
    elevation = math.degrees(math.atan2(vertical, max(horizontal, 1e-9)))
    p_los = 1.0 / (
        1.0
        + cfg.los_a * math.exp(-cfg.los_b * (elevation - cfg.los_a))
    )
    fspl = 20.0 * math.log10(
        4.0
        * math.pi
        * cfg.carrier_frequency_hz
        * distance
        / 299_792_458.0
    )
    base_loss = (
        fspl + p_los * cfg.eta_los_db + (1.0 - p_los) * cfg.eta_nlos_db
    )
    instantaneous_loss = base_loss + env.shadow[uav, channel, user]
    check_close(
        "A2G path-loss equation",
        outcomes["path_loss_db"][action],
        instantaneous_loss,
    )

    channel_bw = cfg.system_bandwidth_hz / cfg.n_channel
    noise_dbm = (
        cfg.noise_psd_dbm_hz
        + 10.0 * math.log10(channel_bw)
        + cfg.receiver_noise_figure_db
    )
    snr_db = env.tx_power_dbm[mode] - instantaneous_loss - noise_dbm
    threshold = 2.0 ** env.mode_se[mode] - 1.0
    average_snr = 10.0 ** (np.clip(snr_db, -80.0, 80.0) / 10.0)
    success_probability = math.exp(-threshold / max(average_snr, 1e-12))
    check_close("noise-power equation", env.noise_dbm, noise_dbm)
    check_close("SNR equation", outcomes["snr_db"][action], snr_db)
    check_close(
        "Rayleigh success equation",
        outcomes["success"][action],
        success_probability,
    )

    tx_bps = min(channel_bw * env.mode_se[mode], env.queue[user] / cfg.slot_duration_s)
    rate_norm = tx_bps / (channel_bw * env.mode_se.max())
    mean_delivered = env.delivered.mean() + cfg.packet_size_bits
    service_equity = np.clip(
        mean_delivered / (env.delivered[user] + cfg.packet_size_bits), 0, 2
    ) / 2
    energy = env.tx_power_w[mode] * cfg.slot_duration_s
    energy_norm = energy / (env.tx_power_w.max() * cfg.slot_duration_s)
    u_success = (
        cfg.w_rate * rate_norm
        + cfg.w_reliability
        + cfg.w_fairness * service_equity
        - cfg.w_energy * energy_norm
    )
    u_failure = cfg.w_fairness * service_equity - cfg.w_energy * energy_norm
    expected = success_probability * u_success + (1 - success_probability) * u_failure
    check_close("successful-outcome utility", outcomes["u_success"][action], u_success)
    check_close("failure-outcome utility", outcomes["u_failure"][action], u_failure)
    check_close("expected physical utility", outcomes["expected_utility"][action], expected)


def prospect_theory_checks():
    cfg = Config(
        n_user=2,
        alpha_gain_by_user=(0.2, 0.7),
        beta_loss_by_user=(0.3, 0.8),
        loss_aversion_by_user=(1.1, 1.9),
        reference_utility_by_user=(0.4, 0.6),
        pt_reward_scale_constant=2.0,
    )
    env = UAVEnv(cfg, 31)
    env.reset()
    deltas = np.array([0.25, -0.25])
    users = np.array([0, 1])
    manual = np.array([0.25**0.2, -1.9 * 0.25**0.8])
    check_close("user-indexed PT value", env.pt_value(deltas, users), manual)

    outcomes = env.outcomes()
    action_users = env.actions[: env.n_transmission_actions, 2]
    references = env.reference_utility_user[action_users]
    success_delta = outcomes["u_success"][: env.n_transmission_actions] - references
    failure_delta = outcomes["u_failure"][: env.n_transmission_actions] - references
    alpha = env.alpha_gain_user[action_users]
    beta = env.beta_loss_user[action_users]
    k_value = env.loss_aversion_user[action_users]
    success_value = np.where(
        success_delta >= 0,
        np.maximum(success_delta, 0) ** alpha,
        -k_value * np.maximum(-success_delta, 0) ** beta,
    )
    failure_value = np.where(
        failure_delta >= 0,
        np.maximum(failure_delta, 0) ** alpha,
        -k_value * np.maximum(-failure_delta, 0) ** beta,
    )
    probability = outcomes["success"][: env.n_transmission_actions]
    expected_pt = (
        probability * success_value + (1.0 - probability) * failure_value
    ) / cfg.pt_reward_scale_constant
    check_close(
        "PT oracle equals expected realized PT reward",
        env.pt_scores()[: env.n_transmission_actions],
        expected_pt,
    )
    check("fixed PT scale", env.pt_reward_scale == 2.0)
    manual_idle_pt = np.mean(
        [
            -1.1 * 0.4**0.3,
            -1.9 * 0.6**0.8,
        ]
    ) / cfg.pt_reward_scale_constant
    check_close(
        "idle PT opportunity-cost score",
        env.pt_scores()[env.idle_action],
        manual_idle_pt,
    )


def qnet_loss(net, values, actions, targets):
    prediction = net.forward(values)[np.arange(len(values)), actions]
    error = prediction - targets
    return float(np.mean(np.where(np.abs(error) <= 1, 0.5 * error**2, np.abs(error) - 0.5)))


class GradientCapture:
    def __init__(self):
        self.grads = None

    def step(self, _params, grads):
        self.grads = [value.copy() for value in grads]


def finite_difference_gradient_check(dueling):
    rng = np.random.default_rng(123)
    net = QNet(3, 3, 4, dueling, rng, 1e-3)
    values = np.array([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.7]])
    actions = np.array([0, 2])
    targets = np.array([0.15, -0.25])
    capture = GradientCapture()
    net.opt = capture
    net.train(values, actions, targets)
    analytic = capture.grads
    epsilon = 1e-6
    worst = 0.0
    for parameter, gradient in zip(net.params, analytic):
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = qnet_loss(net, values, actions, targets)
            parameter[index] = original - epsilon
            minus = qnet_loss(net, values, actions, targets)
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        worst = max(worst, float(np.max(np.abs(numerical - gradient))))
    check(
        f"{'dueling' if dueling else 'plain'} Q-network finite-difference gradient",
        worst < 2e-8,
    )


def optimizer_and_algorithm_checks():
    parameter = np.array([1.0, -2.0])
    gradient = np.array([0.2, -0.4])
    optimizer = Adam([parameter], 1e-3)
    optimizer.step([parameter], [gradient])
    expected = np.array([1.0, -2.0]) - 1e-3 * gradient / (np.abs(gradient) + 1e-8)
    check_close("Adam first-step equation", parameter, expected, rtol=1e-12)

    check("DQN flags", RL_SPECS["EUT_DQN"] == (False, False, "eut"))
    check("Double-DQN flags", RL_SPECS["EUT_DDQN"] == (True, False, "eut"))
    check("Dueling-DQN flags", RL_SPECS["EUT_Dueling_DQN"] == (False, True, "eut"))
    check("EUT-D3QN flags", RL_SPECS["EUT_D3QN"] == (True, True, "eut"))
    check("PT-D3QN controlled reward ablation", RL_SPECS["PT_D3QN"] == (True, True, "pt"))


def bellman_target_checks():
    cfg = Config(
        n_uav=1,
        n_channel=1,
        n_user=1,
        n_mode=2,
        horizon=1,
        batch_size=1,
        warmup=1,
        replay_size=2,
        gamma=0.5,
        target_update=100,
    )
    env = UAVEnv(cfg, 51)
    state = env.reset()

    class FakeOnline:
        def __init__(self):
            self.params = []
            self.target_seen = None

        def forward(self, values):
            return np.tile([5.0, 1.0, -1.0], (len(values), 1))

        def train(self, _values, _actions, target):
            self.target_seen = target.copy()
            return 0.0

        def copy_from(self, _other):
            pass

    class FakeTarget:
        params = []

        def forward(self, values):
            return np.tile([2.0, 10.0, -1.0], (len(values), 1))

        def copy_from(self, _other):
            pass

    for method, (double, dueling, reward_model) in RL_SPECS.items():
        agent = Agent(
            env,
            cfg,
            double=double,
            dueling=dueling,
            reward_model=reward_model,
            seed=73,
        )
        agent.net = FakeOnline()
        agent.target = FakeTarget()
        agent.buf.append(
            (
                state,
                0,
                1.0,
                state.copy(),
                0.0,
                np.array([True, True, False]),
            )
        )
        agent.update()
        expected = 1.0 + 0.5 * (2.0 if double else 10.0)
        check_close(
            f"{method} Bellman target",
            agent.net.target_seen[0],
            expected,
        )

    terminal = Agent(env, cfg, double=True, seed=74)
    terminal.net = FakeOnline()
    terminal.target = FakeTarget()
    terminal.buf.append(
        (
            state,
            0,
            1.25,
            state.copy(),
            1.0,
            np.array([True, True, False]),
        )
    )
    terminal.update()
    check_close(
        "terminal transition suppresses bootstrap",
        terminal.net.target_seen[0],
        1.25,
    )


def controlled_reward_ablation_trajectory_check():
    cfg = Config(
        n_uav=1,
        n_channel=1,
        n_user=2,
        n_mode=2,
        horizon=12,
        reference_utility_by_user=(0.4, 0.6),
    )
    eut_env = UAVEnv(cfg, 888)
    pt_env = UAVEnv(cfg, 888)
    eut_state = eut_env.reset()
    pt_state = pt_env.reset()
    check_close("EUT/PT initial observation identity", eut_state, pt_state, atol=0)
    reward_differences = 0
    for _slot in range(cfg.horizon):
        valid = np.flatnonzero(eut_env.valid_mask()[: eut_env.n_transmission_actions])
        action = int(valid[0]) if len(valid) else eut_env.idle_action
        eut_state, eut_reward, _done, eut_metrics = eut_env.step(action, "eut")
        pt_state, pt_reward, _done, pt_metrics = pt_env.step(action, "pt")
        np.testing.assert_array_equal(eut_state, pt_state)
        for metric in (
            "utility",
            "throughput_mbps",
            "transmission_success",
            "energy_j",
            "delivered_bits",
            "queue_total_after_mbits",
        ):
            np.testing.assert_allclose(
                eut_metrics[metric],
                pt_metrics[metric],
                rtol=0,
                atol=0,
            )
        reward_differences += int(not np.isclose(eut_reward, pt_reward))
    check(
        "EUT/PT physical trajectory identity",
        True,
    )
    check(
        "EUT/PT trajectory differs only in reward transformation",
        reward_differences > 0,
    )


def simulation_protocol_checks():
    small = build_replicate_config(
        Config(n_user=5, randomize_pt_profiles=True), 7
    )
    large = build_replicate_config(
        Config(n_user=60, randomize_pt_profiles=True), 7
    )
    for field in (
        "alpha_gain_by_user",
        "beta_loss_by_user",
        "loss_aversion_by_user",
    ):
        check(
            f"nested {field}",
            np.array_equal(getattr(small, field), np.asarray(getattr(large, field))[:5]),
        )

    nested = build_nested_sweep_replicate_configs(
        Config(
            n_uav=1,
            n_channel=1,
            n_mode=1,
            n_user=5,
            horizon=2,
            reference_calibration_episodes=1,
            randomize_pt_profiles=True,
            reference_utility_mode="random_policy_calibration",
        ),
        [5, 15],
        [7],
    )
    nested_small = nested[5][1][0]
    nested_large = nested[15][1][0]
    check(
        "nested evaluation reference utility",
        np.array_equal(
            nested_small.reference_utility_by_user,
            np.asarray(nested_large.reference_utility_by_user)[:5],
        ),
    )

    small_env = UAVEnv(Config(n_user=5), 2024)
    large_env = UAVEnv(Config(n_user=60), 2024)
    small_env.reset()
    large_env.reset()
    check_close("nested user geometry", small_env.user_xy, large_env.user_xy[:5], atol=0)
    check_close("nested initial packet queues", small_env.queue, large_env.queue[:5], atol=0)
    check_close(
        "nested initial shadowing",
        small_env.shadow,
        large_env.shadow[:, :, :5],
        atol=0,
    )
    state = small_env._state()
    base_start = small_env.base_link_start
    base_stop = base_start + small_env.c.n_uav * small_env.c.n_user
    expected_base = np.tanh(
        (
            small_env.c.max_tx_power_dbm
            - small_env.base_path_loss_db
            - small_env.noise_dbm
        )
        / 20.0
    ).ravel()
    check_close(
        "Markov base-link feature",
        state[base_start:base_stop],
        expected_base,
        rtol=1e-6,
    )

    cfg = Config(seed=7, n_user=3, horizon=12)
    traces = []
    for method in ("Random", "Greedy_Physical_Utility"):
        env = UAVEnv(cfg, evaluation_episode_seed(cfg, 0))
        env.reset()
        rng = np.random.default_rng(998)
        draws = []
        for _ in range(cfg.horizon):
            action = heuristic_action(env, method, rng)
            _state, _reward, _done, metrics = env.step(action, "eut")
            draws.append(metrics["outcome_uniform"])
        traces.append(draws)
    check_close("common outcome-uniform random numbers", traces[0], traces[1], atol=0)

    low_energy = UAVEnv(
        Config(
            n_uav=1,
            n_channel=1,
            n_user=1,
            n_mode=2,
            initial_radio_energy_j=0.0015,
        ),
        5,
    )
    low_energy.reset()
    mask = low_energy.valid_mask()
    check(
        "energy mask rejects unaffordable mode and retains idle",
        mask[0] and not mask[1] and mask[low_energy.idle_action],
    )
    low_energy.step(0, "eut")
    _state, idle_reward, _done, idle_metrics = low_energy.step(
        low_energy.idle_action, "eut"
    )
    check(
        "idle backlog is penalized after energy depletion",
        idle_reward < 0
        and idle_metrics["loss_event"] == 1
        and idle_metrics["utility_shortfall"] > 0,
    )

    env = UAVEnv(Config(n_uav=1, n_channel=1, n_user=1, n_mode=1), 22)
    env.reset()
    initial_offered = int(env.offered_packets.sum())
    for _ in range(10):
        action = int(np.flatnonzero(env.valid_mask())[0])
        env.step(action, "eut")
    check(
        "packet conservation",
        int(env.offered_packets.sum())
        == initial_offered + int(env.generated_arrival_packets.sum()),
    )
    check(
        "admission/drop conservation",
        int(env.offered_packets.sum())
        == int(env.admitted_packets.sum()) + int(env.dropped_packets.sum()),
    )
    check("Jain bound", np.isnan(env.jain()) or 0.0 <= env.jain() <= 1.0)

    mask = env.valid_mask()
    packed = np.packbits(mask, bitorder="little")
    unpacked = np.unpackbits(
        packed, count=env.n_actions, bitorder="little"
    ).astype(bool)
    check("packed replay-mask round trip", np.array_equal(mask, unpacked))

    check_close(
        "exact sign-flip p-value",
        _exact_sign_flip_p_value(np.array([1.0, 1.0, 1.0])),
        0.25,
    )

    wrapper = Path(__file__).with_name("run_user_count_sweep.ps1").read_text(
        encoding="utf-8"
    )
    quick_guards = all(
        f'-not $Quick -or $PSBoundParameters.ContainsKey("{parameter}")'
        in wrapper
        for parameter in (
            "TrainEpisodes",
            "EvalEpisodes",
            "ReferenceCalibrationEpisodes",
        )
    )
    check("PowerShell quick-mode override guards", quick_guards)


def main():
    physical_equation_checks()
    prospect_theory_checks()
    finite_difference_gradient_check(False)
    finite_difference_gradient_check(True)
    optimizer_and_algorithm_checks()
    bellman_target_checks()
    controlled_reward_ablation_trajectory_check()
    simulation_protocol_checks()
    print(f"PASS: {len(PASSED)} independent numerical checks")
    for name in PASSED:
        print(f"  [OK] {name}")


if __name__ == "__main__":
    main()
