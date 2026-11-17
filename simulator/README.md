# PT-D3QN UAV fair-comparison package — audited release V7

This package compares nine resource-selection policies in one abstract
downlink UAV-BS environment. The controlled behavioral comparison is
`EUT_D3QN` versus `PT_D3QN`.

## Run on Windows

Install the dependencies and verify the implementation:

```powershell
py -m pip install -r requirements.txt
py -m unittest -v test_pt_d3qn_uav.py
py verify_algorithms_independently.py
```

Run a short software check for all requested user counts:

```powershell
.\run_user_count_sweep.ps1 `
  -Quick `
  -Seeds "7" `
  -TrainingProfileCount 3 `
  -Output "quick_user_count_check"
```

Run the configured experiment:

```powershell
.\run_user_count_sweep.ps1 -Output "results_user_count_sweep_v7"
```

The wrapper uses the Python file beside the wrapper, runs both verification
suites before simulation, refuses to mix results into a non-empty folder, and
runs `validate_results.py` on every completed output table.

In V7, `-Quick` preserves Python's smoke-test defaults (8 training episodes,
3 evaluation episodes and 5 calibration episodes). `-TrainEpisodes`,
`-EvalEpisodes` or `-ReferenceCalibrationEpisodes` overrides a quick default
only when that parameter is explicitly written on the PowerShell command line.

The default full wrapper configuration is:

- user counts `5, 15, 30, 45, 60`;
- 20 Monte Carlo seeds `7, 17, ..., 197`;
- 3000 training episodes per RL method and seed;
- 100 held-out evaluation episodes per method and seed;
- 50 independent reference-calibration episodes;
- 20 randomized training PT profiles per seed;
- one separate held-out PT profile for evaluation;
- 2 J transmit-radio budget per UAV;
- 50,000-transition replay capacity and one optimizer update per four
  environment interactions;
- slot-level raw output enabled.

The slot-, user- and selection-level raw tables are streamed to disk and their
row counts are read back before summarization. This avoids retaining millions
of Python dictionaries in memory in the 60-user final experiment.

The wrapper rejects fewer than 10 seeds for a non-quick final run unless
`-AllowExploratorySeedCount` is supplied explicitly. Five seeds remain suitable
only for an exploratory run. The default 20 supports exact sign-flip inference
and more stable confidence intervals.

## Nine methods

1. `Random`: uniform over currently feasible actions.
2. `Greedy_Physical_Utility`: maximizes success-outcome utility and ignores
   outage risk.
3. `EUT_Myopic_Oracle`: maximizes one-step expected centered utility.
4. `PT_Myopic_Oracle`: maximizes one-step expected PT value.
5. `EUT_DQN`.
6. `EUT_DDQN`.
7. `EUT_Dueling_DQN`.
8. `EUT_D3QN`.
9. `PT_D3QN`.

The myopic methods enumerate all feasible one-step actions; they are not
learned policies. DQN uses replay, a target network and Huber TD loss. DDQN uses
online selection and target-network evaluation. Dueling DQN uses
`Q(s,a)=V(s)+A(s,a)-mean_a A(s,a)`. D3QN combines DDQN and the dueling head.
`EUT_D3QN` and `PT_D3QN` have identical architectures, initialization, profile
schedule, scenarios and training budgets; only their reward transformation
differs.

## Corrected action and energy model

A transmission action is:

```text
(UAV, orthogonal channel, ground user, transmission mode)
```

The environment permits **one global transmission per slot**. It does not model
simultaneous transmissions by several UAVs. A final idle decision is always
valid. A transmission is feasible when the
selected user's queue is non-empty and the selected UAV has enough radio energy
for that mode and slot.

The old worst-case-energy precondition has been removed, so energy can now
actually disable actions. Idle permits deliberate energy conservation and
prevents false transmissions when queues are empty or remaining energy is
insufficient. If backlog exists, idle receives an explicit no-service
opportunity cost; it is not a reward-free escape. Action counts are:

| Users | Transmission actions | Total actions |
|---:|---:|---:|
| 5 | 180 | 181 |
| 15 | 540 | 541 |
| 30 | 1080 | 1081 |
| 45 | 1620 | 1621 |
| 60 | 2160 | 2161 |

Radio energy is only `P_tx * slot_duration`. Propulsion, hovering, circuitry and
computation energy are outside this model. The default 2 J is an experimental
choice that makes the high-power constraint active over an 80-slot episode;
`energy_masked_action_fraction` records the realized restriction rate.

## Packet queue and PDR definitions

Queues contain whole FIFO packets with per-packet remaining bits and arrival
slots. A successful transmission may partially serve the head packet; a packet
is delivered only when all its bits have been served. Queue overflow drops
whole newly arriving packets and is recorded explicitly.

The output separates four concepts:

- `transmission_success_rate = successful attempts / attempts`;
- `offered_pdr = completed packets / (initial packets + generated arrivals)`;
- `admitted_pdr = completed packets / admitted packets`;
- `packet_drop_rate = dropped arrivals / generated arrivals`.

The legacy column `pdr` is an exact alias of `offered_pdr`.
`transmission_success` is never treated as PDR. The simulator also records
packet delay, attempts, generated/admitted/dropped/delivered packets, queue
values, bits and energy.

Jain fairness is undefined (`NaN`) when nobody has received service. Otherwise
it is computed from delivered bits and clipped numerically to `[0,1]`.

## Physical and behavioral model

The model uses SI units, Al-Hourani urban mean A2G loss, correlated log-normal
shadowing and analytical Rayleigh block-fading success probability. Mode power
and spectral efficiency are abstract study choices, not a 3GPP MCS table. The
UAV is a base station and the receiver is terrestrial; this is not a full 3GPP
system-level simulator.

Every RL agent observes both the static mean/base link loss and instantaneous
CSI. This separation is necessary because the shadow term follows AR(1); total
CSI alone would not identify the next-state distribution.

Physical utility has distinct success and failure outcomes:

```text
U_success = weighted(rate + reliability + service_equity - energy)
U_failure = weighted(service_equity - energy)
```

`service_equity` is reward shaping, not Jain fairness.

For selected user `u`, Prospect Theory uses:

```text
v_u(x) = x^alpha_u                     if x >= 0
         -k_u * (-x)^beta_u            if x < 0
```

where `x = U - U_ref,u`. No probability-weighting function is added. Scalar
centers `alpha=beta=0.2`, `k=1.2` follow the supplied Sensors 2020 setting. The
heterogeneous truncated-normal spreads and bounds are sensitivity assumptions,
not empirically validated human-population distributions.

PT rewards are divided by the fixed declared constant `1.60`. The divisor does
not depend on the largest sampled `k_u`; changing user count cannot silently
change the reward scale.

## PT profiles and generalization

Behavioral draws use independent seeds for every replicate, parameter and user.
For each replicate, `(alpha_u,beta_u,k_u,U_ref,u)` is resolved once at the
maximum 60-user population. Exact prefixes are then reused in the 5-, 15-, 30-
and 45-user runs. Thus the same user retains all four behavioral descriptors
across the sweep. `user_count_profile_nesting_check.csv` verifies each field.

Physical scenario streams are nested as well: for a common replicate and
episode, existing users retain identical positions, initial packet counts,
per-link initial shadowing, arrivals and shadow innovations; newly added users
are appended. This is enforced with per-episode/per-slot seed namespaces and
checked by the result validator.

With `--training-profile-count N`, where `N > 1`, every RL method cycles through
the same separate bank of `N` randomized profiles. Evaluation uses a frozen
profile from a disjoint seed namespace, so it is held out. The training bank is
saved in `training_pt_profiles.csv`; the evaluation profile is in
`pt_user_profiles.csv`.

The training bank randomizes `alpha_u`, `beta_u` and `k_u` from the declared
sensitivity distributions and perturbs each calibrated `U_ref,u` uniformly by
`±20%` by default (`--training-reference-jitter`). This reference perturbation
is also a robustness-design assumption, not an empirically estimated behavioral
distribution. Thus held-out evaluation covers all four observed profile fields.

For a homogeneous source-backed control:

```powershell
py pt_d3qn_uav.py `
  --homogeneous-pt-profiles `
  --reference-mode qos_aspiration `
  --seeds 7,17,27,37,47,57,67,77,87,97 `
  --output homogeneous_control
```

The CLI defaults to randomized profiles when no explicit vector is provided
and rejects incompatible random, homogeneous and explicit combinations.

## Reference endpoints

`random_policy_calibration` estimates each `U_ref,u` on independent
pre-training scenarios. Training and evaluation outcomes are excluded. Because
a calibrated reference can change with scenario size, fixed QoS endpoints are
also reported:

- `qos_reference_satisfaction`;
- `qos_loss_event`;
- `qos_utility_shortfall`.

The canonical `reference_satisfaction`, `loss_event` and `utility_shortfall`
are scheduling-opportunity metrics. A transmission uses the selected user's
realized utility. Idle with backlog compares zero service with the mean
reference of active users. Idle without demand is excluded because no service
opportunity exists. Separate `attempt_*` columns retain transmission-attempt-
conditional diagnostics. `utility_shortfall` is the prespecified primary
behavioral endpoint in `paired_pt_vs_eut_d3qn.csv`.

## Fair random numbers and statistical unit

Within a replicate and evaluation episode, all methods receive the same user
placement, initial queues, shadowing innovations, arrivals and one uniform
outcome draw per slot. Different actions can still have different outcomes
because they map the common draw through different success probabilities.
`evaluation_step_raw.csv` exposes the draw for direct checking.

The independent statistical unit is a Monte Carlo replicate, not a slot,
episode or user. The package reports:

- replicate-level Student-t 95% confidence intervals;
- paired PT-D3QN minus EUT-D3QN Student-t tests;
- exact paired sign-flip tests for at most 20 nonzero pairs;
- Holm-adjusted p-values for a declared non-redundant secondary family;
- a flag identifying `utility_shortfall` as prespecified primary;
- the exchangeability/sign-reversal assumption of the exact test.

Do not use one quick run or one unadjusted p-value to claim PT improves every
metric.

## Memory and inference reporting

Replay masks are packed to one bit per action. Only the online network allocates
Adam state; the target network does not. Agents are trained and released one at
a time. The declared update frequency is shared by every RL method; target
updates count optimizer updates, not raw environment slots.
`algorithm_complexity.csv` reports dimensions, parameter count, mask size, an
explicitly named unshared replay-array payload estimate and a hardware-dependent
decision-time microbenchmark.

## Outputs

Each `users_NNN` folder contains:

- configuration, code hash and seed manifests;
- `training_history.csv`;
- `evaluation_raw.csv` (episode level);
- `evaluation_step_raw.csv` (slot level; disable with `-NoStepDetails`);
- `evaluation_user_raw.csv` (user/episode level);
- `evaluation_selection_raw.csv` (action marginals including Idle);
- replicate summaries and confidence intervals;
- `paired_pt_vs_eut_d3qn.csv`;
- evaluation and training PT profile tables;
- `algorithm_complexity.csv`;
- `plot_*.csv` data and PNG diagnostics;
- parameter, algorithm, fairness and model-scope notes.

The root contains combined `user_count_*.csv` tables. Raw slot data remain in
their corresponding user-count folder.

The user-count sweep jointly increases population, aggregate offered traffic,
initial backlog capacity, state dimension and action dimension. Describe it as
a joint scaling/load experiment, not an isolated action-space intervention.
Separate sweeps are needed to isolate traffic, channel uncertainty, UAV count
or channel count.

## Verification

`test_pt_d3qn_uav.py` checks functional and protocol invariants.
`verify_algorithms_independently.py` independently recomputes the link budget,
Rayleigh probability, utilities, PT score, Adam update, finite-difference
gradients for plain and dueling networks, profile nesting, common random
numbers, packet conservation, energy mask, mask packing and sign-flip test.
`validate_results.py OUTPUT_FOLDER` reconstructs PDR, packet conservation,
throughput, energy, common random numbers, row counts, profile holdout/nesting
and statistical-output fields from a completed run.

`--quick` is only a software smoke test. Its results are not paper results.

## Main sources

- Mitsis et al., Sensors 2020: <https://doi.org/10.3390/s20082434>
- Al-Hourani et al., IEEE WCL 2014: <https://doi.org/10.1109/LWC.2014.2342736>
- Mnih et al., DQN: <https://doi.org/10.1038/nature14236>
- van Hasselt et al., Double DQN: <https://doi.org/10.1609/aaai.v30i1.10295>
- Wang et al., Dueling DQN: <https://proceedings.mlr.press/v48/wangf16.html>
