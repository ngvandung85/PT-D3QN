# Parameter provenance

| Parameter | Value | Status | Basis/source |
|---|---:|---|---|
| `carrier_frequency_hz` | 2000000000.0 | Compatible choice | LTE macro evaluation assumption; a compatibility anchor, not a mandate; 3GPP TR 36.931 v15.0.0, Table 5.3.3-1 (Macro system assumptions) ([link](https://www.etsi.org/deliver/etsi_tr/136900_136999/136931/15.00.00_60/tr_136931v150000p.pdf)) |
| `system_bandwidth_hz` | 10000000.0 | Compatible choice | LTE macro evaluation assumption; a compatibility anchor, not a mandate; 3GPP TR 36.931 v15.0.0, Table 5.3.3-1 (Macro system assumptions) ([link](https://www.etsi.org/deliver/etsi_tr/136900_136999/136931/15.00.00_60/tr_136931v150000p.pdf)) |
| `noise_psd_dbm_hz` | -174.0 | Source-backed | Rounded room-temperature thermal-noise density kT0; ITU-R Report BT.2383-2, Thermal-noise-density entries (kT0 = -173.98 dBm/Hz) ([link](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2383-2-2019-PDF-E.pdf)) |
| `receiver_noise_figure_db` | 9.0 | Compatible choice | LTE macro UE receiver assumption; a compatibility anchor; 3GPP TR 36.931 v15.0.0, Table 5.3.3-1 (UE noise figure) ([link](https://www.etsi.org/deliver/etsi_tr/136900_136999/136931/15.00.00_60/tr_136931v150000p.pdf)) |
| `user_height_m` | 1.5 | Compatible choice | Terrestrial user-terminal height; a compatibility anchor; 3GPP TR 38.901 v18.1.0, Calibration parameter table (UT height = 1.5 m) ([link](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.01.00_60/tr_138901v180100p.pdf)) |
| `los_a` | 9.61 | Source-backed | Urban A2G LoS parameter; Al-Hourani et al., IEEE WCL 2014, Table I (Urban environment) ([link](https://doi.org/10.1109/LWC.2014.2342736)) |
| `los_b` | 0.16 | Source-backed | Urban A2G LoS parameter; Al-Hourani et al., IEEE WCL 2014, Table I (Urban environment) ([link](https://doi.org/10.1109/LWC.2014.2342736)) |
| `eta_los_db` | 1.0 | Source-backed | Urban excessive LoS loss; Al-Hourani et al., IEEE WCL 2014, Table I (Urban environment) ([link](https://doi.org/10.1109/LWC.2014.2342736)) |
| `eta_nlos_db` | 20.0 | Source-backed | Urban excessive NLoS loss; Al-Hourani et al., IEEE WCL 2014, Table I (Urban environment) ([link](https://doi.org/10.1109/LWC.2014.2342736)) |
| `alpha_gain` | 0.2 | Source-backed | Sensitivity parameter used by the supplied UAV-MEC paper/code; Mitsis et al., Sensors 2020, Eq. (4), Table 1; supplied parameters.py lines 72-73 ([link](https://www.mdpi.com/1424-8220/20/8/2434)) |
| `beta_loss` | 0.2 | Source-backed | Set equal to alpha as in the supplied UAV-MEC paper; Mitsis et al., Sensors 2020, Text below Eq. (4); supplied code uses one exponent a_n ([link](https://www.mdpi.com/1424-8220/20/8/2434)) |
| `loss_aversion` | 1.2 | Source-backed | Loss-aversion parameter used by the supplied UAV-MEC paper/code; Mitsis et al., Sensors 2020, Eq. (4), Table 1; supplied parameters.py lines 72-73 ([link](https://www.mdpi.com/1424-8220/20/8/2434)) |
| `reference_utility` | 0.54772459400785 | Derived study design | Mean of resolved per-user references calibrated on independent pre-training random-policy states; This implementation, Per-user values are documented in PT_USER_PROFILES.md |
| `alpha_gain_by_user` | [0.20056898647488142, 0.19513257484306887, 0.21356506926742477, 0.15754610510702952, 0.2158265207955821] | Derived sensitivity draw | Seeded truncated-normal draw from declared experimental bounds and dispersion; See pt_user_profiles.csv and seed_manifest.csv |
| `beta_loss_by_user` | [0.2178556191016298, 0.21916078559397204, 0.20058004574920926, 0.23316412944568515, 0.19142909371231887] | Derived sensitivity draw | Seeded truncated-normal draw from declared experimental bounds and dispersion; See pt_user_profiles.csv and seed_manifest.csv |
| `loss_aversion_by_user` | [1.0638620294961252, 1.2547024750287157, 1.0914324650158187, 1.4804080485230275, 1.3124267984143436] | Derived sensitivity draw | Seeded truncated-normal draw from declared experimental bounds and dispersion; See pt_user_profiles.csv and seed_manifest.csv |
| `reference_utility_by_user` | [0.5476622983074212, 0.5452555696768149, 0.5441789009188458, 0.550691720503322, 0.5508344806328463] | Derived study design | Mean of resolved per-user references calibrated on independent pre-training random-policy states; This implementation, See pt_user_profiles.csv and seed_manifest.csv |
| `n_uav` | 3 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `n_channel` | 4 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `n_user` | 5 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `n_mode` | 3 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `area_side_m` | 500.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `uav_altitude_m` | 100.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `shadow_sigma_db` | 4.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `shadow_rho` | 0.95 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `slot_duration_s` | 0.1 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `packet_size_bits` | 4000000 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `arrival_rate_pps` | 0.25 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `max_queue_bits` | 20000000.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `initial_radio_energy_j` | 2.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `min_tx_power_dbm` | 10.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `max_tx_power_dbm` | 30.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `min_spectral_eff_bps_hz` | 0.5 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `max_spectral_eff_bps_hz` | 4.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `qos_rate_mbps` | 1.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `qos_reliability` | 0.99 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `w_rate` | 0.45 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `w_reliability` | 0.25 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `w_fairness` | 0.2 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `w_energy` | 0.1 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `horizon` | 80 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `train_episodes` | 3000 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `eval_episodes` | 100 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `gamma` | 0.99 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `lr` | 0.001 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `batch_size` | 64 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `replay_size` | 50000 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `warmup` | 1000 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `update_frequency` | 4 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `target_update` | 500 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `hidden` | 128 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `eps_start` | 1.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `eps_end` | 0.05 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `eps_decay` | 0.995 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `randomize_pt_profiles` | True | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `pt_profile_seed_base` | 300000 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `alpha_gain_std` | 0.03 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `alpha_gain_lower` | 0.1 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `alpha_gain_upper` | 0.3 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `beta_loss_std` | 0.03 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `beta_loss_lower` | 0.1 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `beta_loss_upper` | 0.3 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `loss_aversion_std` | 0.15 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `loss_aversion_lower` | 1.0 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `loss_aversion_upper` | 1.6 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `pt_reward_scale_constant` | 1.6 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `save_step_details` | True | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `training_profile_count` | 20 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `training_profile_seed_base` | 350000 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `training_reference_jitter_fraction` | 0.2 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `reference_utility_mode` | random_policy_calibration | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `reference_calibration_episodes` | 50 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `reference_calibration_seed_base` | 200000 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `reference_calibration_population_size` | 60 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `evaluation_seed_base` | 100000 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `seed` | 7 | Experimental assumption | Scenario/training design; sensitivity analysis required |