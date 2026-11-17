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
| `reference_utility` | 0.5475704991715037 | Derived study design | Mean of resolved per-user references calibrated on independent pre-training random-policy states; This implementation, Per-user values are documented in PT_USER_PROFILES.md |
| `alpha_gain_by_user` | [0.20056898647488142, 0.19513257484306887, 0.21356506926742477, 0.15754610510702952, 0.2158265207955821, 0.12716729747160557, 0.213801546026448, 0.18798381372695316, 0.2624300794345116, 0.2147574805770893, 0.20773215650368898, 0.2275750824925664, 0.17469698763237065, 0.20920707043087203, 0.2214910207699285, 0.24007779204327645, 0.20148445330424058, 0.19164774002690588, 0.18945725780013636, 0.15794528095700294, 0.24637963032146531, 0.197407747749045, 0.17972381040634774, 0.16399352449739424, 0.20784882739498786, 0.16471453931012486, 0.26335018378089814, 0.19520297175029833, 0.13637349382285438, 0.19875425886389395] | Derived sensitivity draw | Seeded truncated-normal draw from declared experimental bounds and dispersion; See pt_user_profiles.csv and seed_manifest.csv |
| `beta_loss_by_user` | [0.2178556191016298, 0.21916078559397204, 0.20058004574920926, 0.23316412944568515, 0.19142909371231887, 0.2064577613020714, 0.20629715131248014, 0.20471920636122798, 0.20018514496365894, 0.21988100984350686, 0.18644823042484185, 0.17535334250938348, 0.20374923986080506, 0.23913334155570432, 0.252031949654174, 0.1981298433397948, 0.17111388860151464, 0.17554041835822742, 0.14684139292298842, 0.2518597328934943, 0.24142822536578457, 0.21374402020182556, 0.1678952050059253, 0.20740988886571296, 0.18862487483698748, 0.24386194367019384, 0.23909008412589974, 0.24587685312487653, 0.255328415889953, 0.1795625124620227] | Derived sensitivity draw | Seeded truncated-normal draw from declared experimental bounds and dispersion; See pt_user_profiles.csv and seed_manifest.csv |
| `loss_aversion_by_user` | [1.0638620294961252, 1.2547024750287157, 1.0914324650158187, 1.4804080485230275, 1.3124267984143436, 1.006327748366154, 1.3319652522329093, 1.2455813549202928, 1.3865968410165, 1.1518089982953736, 1.3118241338965224, 1.2714788571658069, 1.054665373689799, 1.277088191365503, 1.0539504998229978, 1.3189144834282427, 1.217173082273642, 1.3782627338938183, 1.3153380271155468, 1.5593536813325546, 1.0330627258178495, 1.2529617329086378, 1.318728881002359, 1.096816395836259, 1.2493764019889955, 1.5475187242786421, 1.1757290624859777, 1.1350006256921608, 1.2338742537918967, 1.2578114843432395] | Derived sensitivity draw | Seeded truncated-normal draw from declared experimental bounds and dispersion; See pt_user_profiles.csv and seed_manifest.csv |
| `reference_utility_by_user` | [0.5476622983074212, 0.5452555696768149, 0.5441789009188458, 0.550691720503322, 0.5508344806328463, 0.543138997040093, 0.5476058596744152, 0.5459316900206176, 0.5470850625246191, 0.5503795874567537, 0.549354532310165, 0.5451641282429294, 0.5469304715964087, 0.5500527912172204, 0.5491473702178026, 0.5470567816596348, 0.5486556888978409, 0.5509223658260946, 0.5454350393489061, 0.5466701411857572, 0.5499028863172734, 0.5482526280072098, 0.5473682130243825, 0.5442719865989714, 0.5487250048694082, 0.5461120587834662, 0.5508990342876843, 0.5454436588439712, 0.5491225399404979, 0.5448634872137309] | Derived study design | Mean of resolved per-user references calibrated on independent pre-training random-policy states; This implementation, See pt_user_profiles.csv and seed_manifest.csv |
| `n_uav` | 3 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `n_channel` | 4 | Experimental assumption | Scenario/training design; sensitivity analysis required |
| `n_user` | 30 | Experimental assumption | Scenario/training design; sensitivity analysis required |
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