import tempfile
import unittest
from pathlib import Path

import numpy as np

from validate_results import reference_jitter_in_bounds

from pt_d3qn_uav import (
    ALL_METHODS,
    Agent,
    Config,
    EVALUATION_METRICS,
    RL_SPECS,
    UAVEnv,
    _plot_ci95,
    build_nested_sweep_replicate_configs,
    build_training_profile_bank,
    build_replicate_config,
    calibrate_random_policy_reference,
    evaluate,
    evaluation_episode_seed,
    heuristic_action,
    parse_seeds,
    parse_user_counts,
    parse_user_profile,
    run_experiment,
    write_algorithm_sources,
    write_fairness_protocol,
    write_model_scope,
    write_paired_pt_comparison,
    write_parameter_sources,
    write_seed_manifest,
    write_user_pt_profiles,
)


class PhysicalEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(seed=11)
        self.env = UAVEnv(self.cfg, 11)
        self.state = self.env.reset()

    def test_dimensions_and_units(self):
        cfg = self.cfg
        env = self.env
        self.assertEqual(
            env.n_actions,
            cfg.n_uav * cfg.n_channel * cfg.n_user * cfg.n_mode + 1,
        )
        self.assertEqual(env.idle_action, env.n_actions - 1)
        self.assertEqual(len(self.state), env.state_dim)
        self.assertEqual(
            env.state_dim,
            cfg.n_uav
            + 6 * cfg.n_user
            + cfg.n_uav * cfg.n_user
            + cfg.n_uav * cfg.n_channel * cfg.n_user
            + 1,
        )
        expected_noise = (
            cfg.noise_psd_dbm_hz
            + 10 * np.log10(cfg.system_bandwidth_hz / cfg.n_channel)
            + cfg.receiver_noise_figure_db
        )
        self.assertAlmostEqual(env.noise_dbm, expected_noise, 10)
        self.assertTrue(
            np.all(
                (env.base_path_loss_db > 60)
                & (env.base_path_loss_db < 160)
            )
        )

    def test_invalid_physical_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "counts must be"):
            UAVEnv(Config(n_channel=0), 1)
        with self.assertRaisesRegex(ValueError, "altitude must exceed"):
            UAVEnv(Config(uav_altitude_m=1.0, user_height_m=1.5), 1)
        with self.assertRaisesRegex(ValueError, "shadow_rho"):
            UAVEnv(Config(shadow_rho=1.01), 1)
        with self.assertRaisesRegex(ValueError, "learning updates are reachable"):
            UAVEnv(Config(replay_size=63, warmup=64, batch_size=32), 1)
        with self.assertRaisesRegex(ValueError, "target_update"):
            UAVEnv(Config(target_update=0), 1)
        with self.assertRaisesRegex(ValueError, "gamma"):
            UAVEnv(Config(gamma=1.01), 1)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            UAVEnv(Config(eps_start=0.1, eps_end=0.2), 1)
        with self.assertRaisesRegex(ValueError, "episode counts"):
            UAVEnv(Config(train_episodes=0), 1)

    def test_reference_jitter_validation_handles_zero_reference(self):
        self.assertTrue(
            reference_jitter_in_bounds([0.0, 0.9], [0.0, 1.0], 0.2)
        )
        self.assertFalse(
            reference_jitter_in_bounds([0.01, 0.9], [0.0, 1.0], 0.2)
        )

    def test_derived_link_budget_and_reference_values(self):
        env = UAVEnv(Config(), 7)
        expected_channel_bw_hz = 10.0e6 / 4
        expected_noise_dbm = -174 + 10 * np.log10(expected_channel_bw_hz) + 9
        expected_reference = 0.45 * 0.1 + 0.25 * 0.99 + 0.20 - 0.10 * 0.01
        self.assertAlmostEqual(env.channel_bw_hz, expected_channel_bw_hz, 10)
        self.assertAlmostEqual(env.noise_dbm, expected_noise_dbm, 10)
        self.assertAlmostEqual(env.reference_utility, expected_reference, 12)
        np.testing.assert_allclose(
            env.reference_utility_user,
            np.full(env.c.n_user, expected_reference),
        )
        np.testing.assert_allclose(env.tx_power_w, [0.01, 0.1, 1.0])

    def test_single_uav_is_centered(self):
        env = UAVEnv(Config(n_uav=1, area_side_m=500.0), 7)
        np.testing.assert_array_equal(env.uav_xy, [[250.0, 250.0]])

    def test_energy_mask_is_active_and_idle_has_backlog_cost(self):
        cfg = Config(
            n_uav=1,
            n_channel=1,
            n_user=1,
            n_mode=2,
            initial_radio_energy_j=0.0015,
            horizon=3,
        )
        env = UAVEnv(cfg, 5)
        env.reset()
        mask = env.valid_mask()
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[env.idle_action])
        env.step(0, "eut")
        mask = env.valid_mask()
        self.assertFalse(mask[: env.n_transmission_actions].any())
        self.assertTrue(mask[env.idle_action])
        energy_before = env.energy.copy()
        _state, reward, _done, metrics = env.step(env.idle_action, "eut")
        np.testing.assert_array_equal(env.energy, energy_before)
        self.assertLess(reward, 0.0)
        self.assertEqual(metrics["attempted"], 0.0)
        self.assertEqual(metrics["service_opportunity"], 1.0)
        self.assertEqual(metrics["idle_with_backlog"], 1.0)
        self.assertEqual(metrics["loss_event"], 1.0)
        self.assertGreater(metrics["utility_shortfall"], 0.0)
        self.assertTrue(np.isnan(metrics["attempt_utility_shortfall"]))

    def test_empty_queue_cannot_create_false_delivery_success(self):
        env = UAVEnv(Config(n_uav=1, n_channel=1, n_user=1, n_mode=1), 3)
        env.reset()
        env.queue[:] = 0.0
        env.packet_queues[0].clear()
        mask = env.valid_mask()
        self.assertTrue(mask[env.idle_action])
        self.assertFalse(mask[: env.n_transmission_actions].any())
        _state, reward, _done, metrics = env.step(env.idle_action, "eut")
        self.assertEqual(metrics["delivered_bits"], 0.0)
        self.assertEqual(metrics["packets_delivered"], 0)
        self.assertTrue(np.isnan(metrics["transmission_success"]))
        self.assertEqual(reward, 0.0)
        self.assertEqual(metrics["service_opportunity"], 0.0)
        self.assertEqual(metrics["idle_without_backlog"], 1.0)
        self.assertTrue(np.isnan(metrics["utility_shortfall"]))

    def test_jain_is_undefined_without_service_and_clipped_when_defined(self):
        env = UAVEnv(Config(n_user=2), 4)
        env.reset()
        self.assertTrue(np.isnan(env.jain()))
        env.delivered[:] = [1.0, 1.0]
        self.assertEqual(env.jain(), 1.0)
        env.delivered[:] = [1.0, 0.0]
        self.assertGreaterEqual(env.jain(), 0.0)
        self.assertLessEqual(env.jain(), 1.0)

    def test_link_budget_is_monotone_in_power(self):
        outcomes = self.env.outcomes()
        shape = (
            self.cfg.n_uav,
            self.cfg.n_channel,
            self.cfg.n_user,
            self.cfg.n_mode,
        )
        snr = outcomes["snr_db"][: self.env.n_transmission_actions].reshape(shape)
        energy = outcomes["energy_j"][: self.env.n_transmission_actions].reshape(shape)
        self.assertTrue(np.all(np.diff(snr, axis=3) > 0))
        self.assertTrue(np.all(np.diff(energy, axis=3) > 0))

    def test_state_contains_total_link_csi(self):
        cfg = self.cfg
        csi_start = self.env.csi_start
        csi_stop = csi_start + cfg.n_uav * cfg.n_channel * cfg.n_user
        before = self.env._state()[csi_start:csi_stop].copy()
        self.env.base_path_loss_db += 15.0
        after = self.env._state()[csi_start:csi_stop]
        self.assertTrue(np.all(after < before))

    def test_state_separately_contains_base_link_state(self):
        cfg = self.cfg
        base_start = self.env.base_link_start
        base_stop = base_start + cfg.n_uav * cfg.n_user
        expected = np.tanh(
            (
                cfg.max_tx_power_dbm
                - self.env.base_path_loss_db
                - self.env.noise_dbm
            )
            / 20.0
        ).ravel()
        np.testing.assert_allclose(
            self.env._state()[base_start:base_stop], expected, rtol=1e-6
        )

        # Keep total instantaneous loss unchanged while changing its mean and
        # shadow decomposition. A Markov observation must distinguish them.
        before = self.env._state().copy()
        self.env.base_path_loss_db += 5.0
        self.env.shadow -= 5.0
        after = self.env._state()
        np.testing.assert_allclose(
            before[self.env.csi_start :],
            after[self.env.csi_start :],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertFalse(np.array_equal(before[base_start:base_stop], after[base_start:base_stop]))

    def test_probability_and_expected_utility(self):
        outcomes = self.env.outcomes()
        self.assertIn("service_equity", outcomes)
        self.assertNotIn("fairness", outcomes)
        probability = outcomes["success"]
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
        reconstructed = (
            probability * outcomes["u_success"]
            + (1 - probability) * outcomes["u_failure"]
        )
        np.testing.assert_allclose(
            outcomes["expected_utility"], reconstructed, rtol=1e-12, atol=1e-12
        )

    def test_reference_dependent_metrics_are_internally_consistent(self):
        action = int(np.flatnonzero(self.env.valid_mask())[0])
        _state, _reward, _done, metrics = self.env.step(action, "pt")
        self.assertEqual(
            metrics["reference_satisfaction"] + metrics["loss_event"],
            1.0,
        )
        self.assertAlmostEqual(
            metrics["utility_shortfall"],
            max(-metrics["centered_eut_utility"], 0.0),
        )

    def test_pt_score_is_expected_value_of_same_outcome_reward(self):
        outcomes = self.env.outcomes()
        action_users = self.env.actions[: self.env.n_transmission_actions, 2]
        reward_success = np.array(
            [
                self.env.outcome_reward(value, "pt", int(user))
                for value, user in zip(
                    outcomes["u_success"][: self.env.n_transmission_actions], action_users
                )
            ]
        )
        reward_failure = np.array(
            [
                self.env.outcome_reward(value, "pt", int(user))
                for value, user in zip(
                    outcomes["u_failure"][: self.env.n_transmission_actions], action_users
                )
            ]
        )
        reconstructed = (
            outcomes["success"][: self.env.n_transmission_actions] * reward_success
            + (1 - outcomes["success"][: self.env.n_transmission_actions]) * reward_failure
        )
        np.testing.assert_allclose(
            self.env.pt_scores()[: self.env.n_transmission_actions],
            reconstructed,
            rtol=1e-12,
            atol=1e-12,
        )
        expected_idle = self.env.idle_reward_components()["pt_reward"]
        self.assertAlmostEqual(
            self.env.pt_scores()[self.env.idle_action], expected_idle
        )

    def test_eut_score_is_expected_value_of_same_outcome_reward(self):
        outcomes = self.env.outcomes()
        action_users = self.env.actions[: self.env.n_transmission_actions, 2]
        references = self.env.reference_utility_user[action_users]
        reward_success = outcomes["u_success"][: self.env.n_transmission_actions] - references
        reward_failure = outcomes["u_failure"][: self.env.n_transmission_actions] - references
        reconstructed = (
            outcomes["success"][: self.env.n_transmission_actions] * reward_success
            + (1 - outcomes["success"][: self.env.n_transmission_actions]) * reward_failure
        )
        np.testing.assert_allclose(
            self.env.eut_scores()[: self.env.n_transmission_actions],
            reconstructed,
            rtol=1e-12,
            atol=1e-12,
        )
        expected_idle = self.env.idle_reward_components()["eut_reward"]
        self.assertAlmostEqual(
            self.env.eut_scores()[self.env.idle_action], expected_idle
        )

    def test_idle_with_backlog_matches_declared_reward_and_metrics(self):
        env = UAVEnv(
            Config(
                n_uav=1,
                n_channel=1,
                n_user=2,
                n_mode=1,
                reference_utility_by_user=(0.4, 0.6),
            ),
            8,
        )
        env.reset()
        expected_eut = -0.5
        expected_pt_raw = np.mean(
            env.pt_value(np.array([-0.4, -0.6]), np.array([0, 1]))
        )
        self.assertAlmostEqual(env.eut_scores()[env.idle_action], expected_eut)
        self.assertAlmostEqual(
            env.pt_scores()[env.idle_action], expected_pt_raw / env.pt_reward_scale
        )
        _state, reward, _done, metrics = env.step(env.idle_action, "pt")
        self.assertAlmostEqual(reward, expected_pt_raw / env.pt_reward_scale)
        self.assertAlmostEqual(metrics["centered_eut_utility"], expected_eut)
        self.assertAlmostEqual(metrics["utility_shortfall"], 0.5)
        self.assertEqual(metrics["reference_satisfaction"], 0.0)
        self.assertEqual(metrics["loss_event"], 1.0)

    def test_pt_defaults_match_supplied_uav_mec_code(self):
        self.assertEqual(self.cfg.alpha_gain, 0.2)
        self.assertEqual(self.cfg.beta_loss, 0.2)
        self.assertEqual(self.cfg.loss_aversion, 1.2)
        np.testing.assert_allclose(self.env.alpha_gain_user, 0.2)
        np.testing.assert_allclose(self.env.beta_loss_user, 0.2)
        np.testing.assert_allclose(self.env.loss_aversion_user, 1.2)
        self.assertFalse(hasattr(self.cfg, "probability_gain"))
        self.assertFalse(hasattr(self.cfg, "probability_loss"))
        self.assertEqual(self.env.pt_reward_scale, 1.6)
        larger = UAVEnv(
            build_replicate_config(
                Config(n_user=60, randomize_pt_profiles=True), 7
            ),
            7,
        )
        self.assertEqual(larger.pt_reward_scale, self.env.pt_reward_scale)

    def test_user_specific_pt_profiles_are_resolved_and_observed(self):
        cfg = Config(
            n_user=3,
            alpha_gain_by_user=(0.2, 0.4, 0.6),
            beta_loss_by_user=(0.3, 0.5, 0.7),
            loss_aversion_by_user=(1.1, 1.4, 1.8),
            reference_utility_by_user=(0.4, 0.5, 0.6),
        )
        env = UAVEnv(cfg, 9)
        state = env.reset()
        np.testing.assert_allclose(env.alpha_gain_user, [0.2, 0.4, 0.6])
        np.testing.assert_allclose(env.beta_loss_user, [0.3, 0.5, 0.7])
        np.testing.assert_allclose(env.loss_aversion_user, [1.1, 1.4, 1.8])
        np.testing.assert_allclose(env.reference_utility_user, [0.4, 0.5, 0.6])

        start = cfg.n_uav + 2 * cfg.n_user
        np.testing.assert_allclose(
            state[start : start + cfg.n_user],
            env.alpha_gain_user,
            rtol=1e-6,
        )
        start += cfg.n_user
        np.testing.assert_allclose(
            state[start : start + cfg.n_user],
            env.beta_loss_user,
            rtol=1e-6,
        )

    def test_user_specific_pt_value_changes_with_behavioral_profile(self):
        env = UAVEnv(
            Config(
                n_user=2,
                alpha_gain_by_user=(0.2, 0.8),
                beta_loss_by_user=(0.2, 0.8),
                loss_aversion_by_user=(1.1, 2.0),
            ),
            3,
        )
        gains = env.pt_value(np.array([0.25, 0.25]), np.array([0, 1]))
        losses = env.pt_value(np.array([-0.25, -0.25]), np.array([0, 1]))
        self.assertNotEqual(float(gains[0]), float(gains[1]))
        self.assertNotEqual(float(losses[0]), float(losses[1]))

    def test_heterogeneous_pt_oracle_matches_expected_realized_rewards(self):
        env = UAVEnv(
            Config(
                n_user=3,
                alpha_gain_by_user=(0.2, 0.4, 0.6),
                beta_loss_by_user=(0.3, 0.5, 0.7),
                loss_aversion_by_user=(1.1, 1.4, 1.8),
                reference_utility_by_user=(0.4, 0.5, 0.6),
            ),
            4,
        )
        env.reset()
        outcomes = env.outcomes()
        users = env.actions[: env.n_transmission_actions, 2]
        reward_success = np.array(
            [
                env.outcome_reward(value, "pt", int(user))
                for value, user in zip(
                    outcomes["u_success"][: env.n_transmission_actions], users
                )
            ]
        )
        reward_failure = np.array(
            [
                env.outcome_reward(value, "pt", int(user))
                for value, user in zip(
                    outcomes["u_failure"][: env.n_transmission_actions], users
                )
            ]
        )
        reconstructed = (
            outcomes["success"][: env.n_transmission_actions] * reward_success
            + (1 - outcomes["success"][: env.n_transmission_actions]) * reward_failure
        )
        np.testing.assert_allclose(
            env.pt_scores()[: env.n_transmission_actions],
            reconstructed,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_invalid_user_specific_pt_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly n_user=3"):
            UAVEnv(Config(n_user=3, alpha_gain_by_user=(0.2, 0.3)), 1)
        with self.assertRaisesRegex(ValueError, "curvature"):
            UAVEnv(
                Config(n_user=2, beta_loss_by_user=(0.2, 1.2)),
                1,
            )
        with self.assertRaisesRegex(ValueError, "loss-aversion"):
            UAVEnv(
                Config(n_user=2, loss_aversion_by_user=(1.2, 0.0)),
                1,
            )

    def test_random_profiles_are_reproducible_bounded_and_seed_specific(self):
        base = Config(n_user=40, randomize_pt_profiles=True)
        first = build_replicate_config(base, 7)
        repeated = build_replicate_config(base, 7)
        different_seed = build_replicate_config(base, 17)

        for field in (
            "alpha_gain_by_user",
            "beta_loss_by_user",
            "loss_aversion_by_user",
        ):
            np.testing.assert_array_equal(
                getattr(first, field), getattr(repeated, field)
            )
            self.assertFalse(
                np.array_equal(
                    getattr(first, field), getattr(different_seed, field)
                )
            )
        self.assertTrue(
            np.all(
                (np.asarray(first.alpha_gain_by_user) >= base.alpha_gain_lower)
                & (
                    np.asarray(first.alpha_gain_by_user)
                    <= base.alpha_gain_upper
                )
            )
        )
        self.assertTrue(
            np.all(
                (
                    np.asarray(first.loss_aversion_by_user)
                    >= base.loss_aversion_lower
                )
                & (
                    np.asarray(first.loss_aversion_by_user)
                    <= base.loss_aversion_upper
                )
            )
        )

    def test_random_profiles_are_nested_across_user_counts(self):
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
            np.testing.assert_array_equal(
                getattr(small, field), np.asarray(getattr(large, field))[:5]
            )

    def test_sweep_nests_all_four_behavioral_descriptors(self):
        base = Config(
            n_uav=1,
            n_channel=1,
            n_mode=1,
            n_user=5,
            horizon=2,
            reference_calibration_episodes=1,
            randomize_pt_profiles=True,
            reference_utility_mode="random_policy_calibration",
        )
        nested = build_nested_sweep_replicate_configs(
            base, [5, 15], [7, 17]
        )
        for seed_index, _seed in enumerate((7, 17)):
            small = nested[5][1][seed_index]
            large = nested[15][1][seed_index]
            for field in (
                "alpha_gain_by_user",
                "beta_loss_by_user",
                "loss_aversion_by_user",
                "reference_utility_by_user",
            ):
                np.testing.assert_array_equal(
                    getattr(small, field), np.asarray(getattr(large, field))[:5]
                )
            self.assertEqual(small.reference_calibration_population_size, 15)
            self.assertEqual(large.reference_calibration_population_size, 15)

    def test_training_profile_bank_is_shared_and_evaluation_is_held_out(self):
        cfg = build_replicate_config(
            Config(
                n_user=4,
                randomize_pt_profiles=True,
                training_profile_count=3,
            ),
            7,
        )
        first = build_training_profile_bank(cfg)
        repeated = build_training_profile_bank(cfg)
        evaluation_reference = UAVEnv(cfg, 7).reference_utility_user
        self.assertEqual(len(first), 3)
        for profile_a, profile_b in zip(first, repeated):
            self.assertTrue(profile_a["evaluation_profile_held_out"])
            np.testing.assert_array_equal(profile_a["alpha"], profile_b["alpha"])
            np.testing.assert_array_equal(
                profile_a["reference_utility"],
                profile_b["reference_utility"],
            )
            self.assertFalse(
                np.array_equal(profile_a["alpha"], cfg.alpha_gain_by_user)
            )
            self.assertFalse(
                np.array_equal(
                    profile_a["reference_utility"],
                    evaluation_reference,
                )
            )
            relative = (
                profile_a["reference_utility"]
                / evaluation_reference
            )
            self.assertTrue(np.all((relative >= 0.8) & (relative <= 1.2)))
    def test_random_profile_is_frozen_across_envs_and_resets(self):
        resolved = build_replicate_config(
            Config(
                n_user=6,
                randomize_pt_profiles=True,
                reference_utility_mode="random_policy_calibration",
                reference_calibration_episodes=2,
                horizon=3,
            ),
            7,
        )
        first = UAVEnv(resolved, 101)
        second = UAVEnv(resolved, 202)
        expected = (
            first.alpha_gain_user.copy(),
            first.beta_loss_user.copy(),
            first.loss_aversion_user.copy(),
            first.reference_utility_user.copy(),
        )
        for _ in range(3):
            first.reset()
            second.reset()
            for actual, target in zip(
                (
                    first.alpha_gain_user,
                    first.beta_loss_user,
                    first.loss_aversion_user,
                    first.reference_utility_user,
                ),
                expected,
            ):
                np.testing.assert_array_equal(actual, target)
            for actual, target in zip(
                (
                    second.alpha_gain_user,
                    second.beta_loss_user,
                    second.loss_aversion_user,
                    second.reference_utility_user,
                ),
                expected,
            ):
                np.testing.assert_array_equal(actual, target)
    def test_random_profile_rejects_explicit_behavioral_values(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            build_replicate_config(
                Config(
                    n_user=2,
                    randomize_pt_profiles=True,
                    alpha_gain_by_user=(0.2, 0.3),
                ),
                7,
            )
        with self.assertRaisesRegex(ValueError, "std>0"):
            build_replicate_config(
                Config(randomize_pt_profiles=True, alpha_gain_std=0.0),
                7,
            )

    def test_calibrated_reference_is_reproducible_and_data_separated(self):
        base = Config(
            n_user=4,
            horizon=5,
            reference_calibration_episodes=4,
            reference_utility_mode="random_policy_calibration",
        )
        first = calibrate_random_policy_reference(base, 7)
        repeated = calibrate_random_policy_reference(base, 7)
        changed_evaluation_seed = calibrate_random_policy_reference(
            Config(
                **{
                    **base.__dict__,
                    "evaluation_seed_base": base.evaluation_seed_base + 99,
                }
            ),
            7,
        )
        different_replicate = calibrate_random_policy_reference(base, 17)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(first, changed_evaluation_seed)
        self.assertFalse(np.array_equal(first, different_replicate))
        self.assertTrue(np.all(np.isfinite(first)))

        resolved = build_replicate_config(base, 7)
        first_env = UAVEnv(resolved, 101)
        second_env = UAVEnv(resolved, 202)
        np.testing.assert_array_equal(
            first_env.reference_utility_user,
            second_env.reference_utility_user,
        )

    def test_seeded_scenario_identity(self):
        second = UAVEnv(self.cfg, 11)
        second.reset()
        np.testing.assert_allclose(self.env.uav_xy, second.uav_xy)
        np.testing.assert_allclose(self.env.user_xy, second.user_xy)
        np.testing.assert_allclose(self.env.shadow, second.shadow)
        np.testing.assert_allclose(self.env.queue, second.queue)

    def test_physical_scenario_is_nested_across_user_counts(self):
        small = UAVEnv(Config(n_user=5), 12345)
        large = UAVEnv(Config(n_user=60), 12345)
        small.reset()
        large.reset()
        np.testing.assert_array_equal(small.user_xy, large.user_xy[:5])
        np.testing.assert_array_equal(small.queue, large.queue[:5])
        np.testing.assert_array_equal(small.shadow, large.shadow[:, :, :5])
        for _ in range(3):
            action_small = int(np.flatnonzero(small.valid_mask())[0])
            action_large = int(np.flatnonzero(large.valid_mask())[0])
            _s, _r, _d, metrics_small = small.step(action_small, "eut")
            _s, _r, _d, metrics_large = large.step(action_large, "eut")
            self.assertEqual(
                metrics_small["arrival_events_by_user"],
                ";".join(
                    event
                    for event in metrics_large["arrival_events_by_user"].split(";")
                    if event and int(event.split(":")[0]) < 5
                ),
            )

    def test_profile_manifest_records_every_resolved_replicate(self):
        base = Config(
            n_user=3,
            horizon=3,
            randomize_pt_profiles=True,
            reference_utility_mode="random_policy_calibration",
            reference_calibration_episodes=2,
        )
        configs = [
            build_replicate_config(base, seed) for seed in (7, 17)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_user_pt_profiles(configs, output)
            write_seed_manifest(configs, output)
            import pandas as pd

            frame = pd.read_csv(output / "pt_user_profiles.csv")
            self.assertEqual(len(frame), 2 * base.n_user)
            self.assertEqual(set(frame.replicate_seed), {7, 17})
            self.assertTrue(frame.profile_seed.notna().all())
            self.assertTrue(
                frame.reference_origin.str.contains("calibration").all()
            )
            seeds = pd.read_csv(output / "seed_manifest.csv")
            for replicate_seed in (7, 17):
                subset = seeds[seeds.replicate_seed == replicate_seed]
                self.assertEqual(
                    len(subset[subset.stream == "pt_profile"]), 1
                )
                self.assertEqual(
                    len(
                        subset[
                            subset.stream
                            == "reference_calibration_environment"
                        ]
                    ),
                    base.reference_calibration_episodes,
                )
                self.assertEqual(
                    len(subset[subset.stream == "evaluation_environment"]),
                    base.eval_episodes,
                )

    def test_all_baselines_return_valid_actions(self):
        rng = np.random.default_rng(3)
        for name in ALL_METHODS[:4]:
            action = heuristic_action(self.env, name, rng)
            self.assertTrue(self.env.valid_mask()[action], name)

    def test_greedy_and_eut_oracle_have_documented_scores(self):
        outcomes = self.env.outcomes()
        mask = self.env.valid_mask()
        greedy = heuristic_action(
            self.env,
            "Greedy_Physical_Utility",
            np.random.default_rng(1),
        )
        eut = heuristic_action(
            self.env,
            "EUT_Myopic_Oracle",
            np.random.default_rng(1),
        )
        expected_greedy = int(
            np.argmax(np.where(mask, outcomes["u_success"], -np.inf))
        )
        expected_eut = int(
            np.argmax(np.where(mask, self.env.eut_scores(), -np.inf))
        )
        self.assertEqual(greedy, expected_greedy)
        self.assertEqual(eut, expected_eut)


class FairAlgorithmDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(seed=13)
        self.env = UAVEnv(self.cfg, 13)
        self.env.reset()

    def test_exact_nine_method_set(self):
        self.assertEqual(len(ALL_METHODS), 9)
        self.assertEqual(
            ALL_METHODS,
            [
                "Random",
                "Greedy_Physical_Utility",
                "EUT_Myopic_Oracle",
                "PT_Myopic_Oracle",
                "EUT_DQN",
                "EUT_DDQN",
                "EUT_Dueling_DQN",
                "EUT_D3QN",
                "PT_D3QN",
            ],
        )

    def test_algorithm_flags(self):
        self.assertEqual(RL_SPECS["EUT_DQN"], (False, False, "eut"))
        self.assertEqual(RL_SPECS["EUT_DDQN"], (True, False, "eut"))
        self.assertEqual(
            RL_SPECS["EUT_Dueling_DQN"], (False, True, "eut")
        )
        self.assertEqual(RL_SPECS["EUT_D3QN"], (True, True, "eut"))
        self.assertEqual(RL_SPECS["PT_D3QN"], (True, True, "pt"))

    def test_powershell_quick_preserves_python_quick_defaults(self):
        script = Path(__file__).with_name("run_user_count_sweep.ps1").read_text(
            encoding="utf-8"
        )
        for parameter in (
            "TrainEpisodes",
            "EvalEpisodes",
            "ReferenceCalibrationEpisodes",
        ):
            self.assertIn(
                f'-not $Quick -or $PSBoundParameters.ContainsKey("{parameter}")',
                script,
            )
        self.assertIn(
            "Do not pass the full-run defaults in Quick mode",
            script,
        )

    def test_double_dqn_target_decouples_selection_and_evaluation(self):
        cfg = Config(
            n_uav=1,
            n_channel=1,
            n_user=1,
            n_mode=2,
            horizon=1,
            batch_size=1,
            warmup=1,
            gamma=0.5,
            target_update=100,
        )
        env = UAVEnv(cfg, 5)
        state = env.reset()
        next_state = state.copy()

        class FakeOnline:
            def __init__(self):
                self.params = []
                self.seen_target = None

            def forward(self, values):
                return np.tile([5.0, 1.0, -1.0], (len(values), 1))

            def train(self, values, actions, target):
                self.seen_target = target.copy()
                return 0.0

            def copy_from(self, other):
                pass

        class FakeTarget:
            params = []

            def forward(self, values):
                return np.tile([2.0, 10.0, -1.0], (len(values), 1))

            def copy_from(self, other):
                pass

        def captured_target(double):
            agent = Agent(env, cfg, double=double, seed=19)
            agent.net = FakeOnline()
            agent.target = FakeTarget()
            agent.buf.append(
                (
                    state,
                    0,
                    1.0,
                    next_state,
                    0.0,
                    np.array([True, True, False]),
                )
            )
            agent.update()
            return float(agent.net.seen_target[0])

        # Online argmax is action 0; target argmax is action 1.
        self.assertEqual(captured_target(double=True), 1.0 + 0.5 * 2.0)
        self.assertEqual(captured_target(double=False), 1.0 + 0.5 * 10.0)

    def test_dueling_head_uses_identifiable_mean_advantage_aggregation(self):
        agent = Agent(
            self.env,
            self.cfg,
            double=False,
            dueling=True,
            reward_model="eut",
            seed=91,
        )
        q_values, cache = agent.net.forward(self.env._state()[None], True)
        hidden = cache[-1]
        state_value = hidden @ agent.net.Wv + agent.net.bv
        np.testing.assert_allclose(
            q_values.mean(axis=1, keepdims=True),
            state_value,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_eut_d3qn_and_pt_d3qn_differ_only_by_reward(self):
        eut = RL_SPECS["EUT_D3QN"]
        prospect = RL_SPECS["PT_D3QN"]
        self.assertEqual(eut[:2], prospect[:2])
        self.assertNotEqual(eut[2], prospect[2])

    def test_eut_d3qn_and_pt_d3qn_have_identical_initial_parameters(self):
        eut = Agent(
            self.env,
            self.cfg,
            *RL_SPECS["EUT_D3QN"],
            seed=114,
        )
        prospect = Agent(
            self.env,
            self.cfg,
            *RL_SPECS["PT_D3QN"],
            seed=114,
        )
        for eut_param, pt_param in zip(eut.net.params, prospect.net.params):
            np.testing.assert_array_equal(eut_param, pt_param)

    def test_exploration_rng_is_architecture_independent(self):
        plain = Agent(
            self.env,
            self.cfg,
            *RL_SPECS["EUT_DQN"],
            seed=914,
        )
        dueling = Agent(
            self.env,
            self.cfg,
            *RL_SPECS["EUT_Dueling_DQN"],
            seed=914,
        )
        mask = self.env.valid_mask()
        plain_actions = [plain.act(self.env._state(), mask, 1.0) for _ in range(20)]
        dueling_actions = [
            dueling.act(self.env._state(), mask, 1.0) for _ in range(20)
        ]
        self.assertEqual(plain_actions, dueling_actions)

    def test_pt_and_eut_oracles_can_make_different_risk_choices(self):
        env = UAVEnv(Config(seed=222), 222)
        different = 0
        for _episode in range(8):
            env.reset()
            for _step in range(20):
                eut = heuristic_action(
                    env, "EUT_Myopic_Oracle", np.random.default_rng(1)
                )
                prospect = heuristic_action(
                    env, "PT_Myopic_Oracle", np.random.default_rng(1)
                )
                different += int(eut != prospect)
                action = int(env.rng.choice(np.flatnonzero(env.valid_mask())))
                env.step(action, "eut")
        self.assertGreater(different, 0)

    def test_seed_parser(self):
        self.assertEqual(parse_seeds(7, None), [7])
        self.assertEqual(parse_seeds(7, "7,17,27"), [7, 17, 27])
        with self.assertRaises(Exception):
            parse_seeds(7, "7,7")
        with self.assertRaises(Exception):
            parse_seeds(-1, None)
        self.assertEqual(
            parse_user_profile("0.2,0.4", 2, "--profile"),
            (0.2, 0.4),
        )
        with self.assertRaises(Exception):
            parse_user_profile("0.2", 2, "--profile")
        self.assertEqual(
            parse_user_counts("5,15,30,45,60"),
            [5, 15, 30, 45, 60],
        )
        with self.assertRaises(Exception):
            parse_user_counts("5,5")
        with self.assertRaises(Exception):
            parse_user_counts("5,0")

    def test_detailed_evaluation_preserves_episode_counts(self):
        cfg = Config(seed=7, n_user=3, horizon=4, eval_episodes=2)
        user_rows = []
        selection_rows = []
        episode_rows = evaluate(
            "Random",
            cfg,
            user_detail_rows=user_rows,
            selection_detail_rows=selection_rows,
        )
        self.assertEqual(len(episode_rows), cfg.eval_episodes)
        self.assertEqual(len(user_rows), cfg.eval_episodes * cfg.n_user)
        expected_categories = (
            cfg.n_uav + cfg.n_channel + cfg.n_user + cfg.n_mode + 4
        )
        self.assertEqual(
            len(selection_rows), cfg.eval_episodes * expected_categories
        )
        import pandas as pd

        users = pd.DataFrame(user_rows)
        selections = pd.DataFrame(selection_rows)
        self.assertTrue(
            users.groupby("evaluation_episode")
            .selection_count.sum()
            .le(cfg.horizon)
            .all()
        )
        self.assertTrue(
            selections.groupby(["evaluation_episode", "dimension"])
            .selection_count.sum()
            .eq(cfg.horizon)
            .all()
        )

    def test_run_experiment_writes_plot_ready_and_user_tables(self):
        cfg = Config(
            seed=7,
            n_uav=1,
            n_channel=1,
            n_user=2,
            n_mode=2,
            horizon=2,
            train_episodes=2,
            eval_episodes=2,
            batch_size=1,
            warmup=1,
            replay_size=10,
            target_update=1,
            randomize_pt_profiles=True,
            reference_utility_mode="random_policy_calibration",
            reference_calibration_episodes=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = run_experiment(cfg, [7], output, quick_run_only=True)
            for filename in (
                "evaluation_user_raw.csv",
                "evaluation_selection_raw.csv",
                "plot_physical_metrics.csv",
                "plot_pt_risk_metrics.csv",
                "plot_training_curves.csv",
                "plot_user_metrics.csv",
                "plot_selection_distribution.csv",
                "reproducibility_manifest.json",
                "evaluation_step_raw.csv",
                "algorithm_complexity.csv",
                "training_pt_profiles.csv",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            self.assertEqual(len(result["profiles"]), cfg.n_user)
            self.assertTrue(result["profiles"].profile_seed.notna().all())
            import pandas as pd

            steps = pd.read_csv(output / "evaluation_step_raw.csv")
            self.assertTrue(
                steps.groupby(["evaluation_episode", "slot"])
                .outcome_uniform.nunique()
                .eq(1)
                .all()
            )
            self.assertTrue(
                steps.assign(
                    arrival_events_by_user=steps.arrival_events_by_user.fillna("")
                )
                .groupby(["evaluation_episode", "slot"])
                .arrival_events_by_user.nunique(dropna=False)
                .eq(1)
                .all()
            )
            episodes = pd.read_csv(output / "evaluation_raw.csv")
            users = pd.read_csv(output / "evaluation_user_raw.csv")
            selections = pd.read_csv(output / "evaluation_selection_raw.csv")
            expected_episodes = len(ALL_METHODS) * cfg.eval_episodes
            self.assertEqual(len(users), expected_episodes * cfg.n_user)
            categories = cfg.n_uav + cfg.n_channel + cfg.n_user + cfg.n_mode + 4
            self.assertEqual(len(selections), expected_episodes * categories)
            np.testing.assert_allclose(
                episodes.offered_pdr,
                episodes.delivered_packets / episodes.offered_packets,
            )

    def test_held_out_scenarios_are_common_within_and_independent_across_replicates(self):
        cfg_a = Config(seed=7, horizon=3, eval_episodes=2)
        cfg_b = Config(seed=99, horizon=3, eval_episodes=2)
        rows_a = evaluate("Random", cfg_a)
        rows_a_repeat = evaluate("Random", cfg_a)
        rows_a_greedy = evaluate("Greedy_Physical_Utility", cfg_a)
        rows_b = evaluate("Random", cfg_b)
        self.assertNotEqual(
            [row["episode_seed"] for row in rows_a],
            [row["episode_seed"] for row in rows_b],
        )
        self.assertEqual(rows_a, rows_a_repeat)
        self.assertEqual(
            [row["episode_seed"] for row in rows_a],
            [evaluation_episode_seed(cfg_a, episode) for episode in range(2)],
        )
        self.assertEqual(
            [row["episode_seed"] for row in rows_a],
            [row["episode_seed"] for row in rows_a_greedy],
        )
        self.assertTrue(all(row["replicate_seed"] == 7 for row in rows_a))

    def test_audit_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_parameter_sources(self.cfg, path)
            write_user_pt_profiles(self.cfg, path)
            write_algorithm_sources(path)
            write_fairness_protocol(path)
            write_model_scope(self.cfg, path)
            parameter_text = (path / "PARAMETER_SOURCES.md").read_text(
                encoding="utf-8"
            )
            algorithm_text = (path / "ALGORITHM_SOURCES.md").read_text(
                encoding="utf-8"
            )
            fairness_text = (path / "FAIRNESS_PROTOCOL.md").read_text(
                encoding="utf-8"
            )
            model_scope_text = (path / "MODEL_SCOPE.md").read_text(
                encoding="utf-8"
            )
            profile_text = (path / "PT_USER_PROFILES.md").read_text(
                encoding="utf-8"
            )
            import pandas as pd

            parameter_frame = pd.read_csv(path / "parameter_sources.csv")
            algorithm_frame = pd.read_csv(path / "algorithm_sources.csv")
            self.assertIn("3GPP", parameter_text)
            self.assertIn("0.2", parameter_text)
            self.assertIn("PT_D3QN", algorithm_text)
            self.assertIn("no probability-weighting", fairness_text)
            self.assertIn("not a full 3GPP", model_scope_text)
            self.assertIn("service-equity", model_scope_text)
            self.assertIn("alpha_u", profile_text)
            profile_frame = pd.read_csv(path / "pt_user_profiles.csv")
            self.assertEqual(len(profile_frame), self.cfg.n_user)
            np.testing.assert_allclose(profile_frame.alpha_u, 0.2)

            eut_oracle = algorithm_frame.set_index("method").loc[
                "EUT_Myopic_Oracle", "source_urls"
            ]
            self.assertIn("j.ctt1r2gkx", eut_oracle)
            self.assertNotIn("1914185", eut_oracle)
            eut_d3qn = algorithm_frame.set_index("method").loc[
                "EUT_D3QN", "source_urls"
            ]
            self.assertIn("AAAI", eut_d3qn)
            self.assertIn("wangf16", eut_d3qn)
            pt_d3qn = algorithm_frame.set_index("method").loc[
                "PT_D3QN", "source_urls"
            ]
            for expected in ("nature14236", "AAAI", "wangf16", "2434"):
                self.assertIn(expected, pt_d3qn)

            parameter_lookup = parameter_frame.set_index("parameter")
            self.assertIn(
                "itu.int",
                parameter_lookup.loc["noise_psd_dbm_hz", "source_url"],
            )
            self.assertEqual(
                parameter_lookup.loc["reference_utility", "status"],
                "Derived study design",
            )
            assumption_rows = parameter_frame[
                parameter_frame.status == "Experimental assumption"
            ]
            self.assertTrue(assumption_rows.source_url.isna().all())

    def test_paired_comparison_output(self):
        rows = []
        for method, offset in [("EUT_D3QN", 0.0), ("PT_D3QN", 0.1)]:
            for seed in [7, 17, 27]:
                row = {
                    "method": method,
                    "replicate_seed": seed,
                }
                for index, metric in enumerate(EVALUATION_METRICS):
                    row[metric] = seed / 100 + index + offset
                rows.append(row)
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_paired_pt_comparison(pd.DataFrame(rows), output)
            result = pd.read_csv(output / "paired_pt_vs_eut_d3qn.csv")
            self.assertEqual(len(result), len(EVALUATION_METRICS))
            self.assertTrue((result["n_paired_replicates"] == 3).all())
            self.assertTrue(
                np.allclose(result["mean_paired_difference"][:3], [0.1, 0.1, 0.1])
            )
            self.assertIn("exact_sign_flip_p_value_two_sided", result)
            self.assertIn("holm_adjusted_sign_flip_p_value", result)
            self.assertEqual(
                int(result.is_prespecified_primary_endpoint.sum()), 1
            )
            self.assertIn("holm_secondary_family_member", result)
            self.assertTrue(
                result.loc[
                    ~result.holm_secondary_family_member,
                    "holm_adjusted_sign_flip_p_value",
                ].isna().all()
            )

    def test_plot_ci_drops_missing_replicates(self):
        values = np.array([1.0, 2.0, np.nan])
        expected = 12.706204736432095 * np.std([1.0, 2.0], ddof=1) / np.sqrt(2)
        self.assertAlmostEqual(_plot_ci95(values), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
