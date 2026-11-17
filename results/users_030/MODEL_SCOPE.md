# Physical-model and provenance scope

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
to this UAV scheduling utility. `U_ref,u` is the average, over independent pre-training calibration states, of user `u`'s expected physical utility under a uniform choice among that user's feasible UAV/channel/mode actions. A separate uniform random feasible action advances each calibration trajectory. No training or held-out evaluation outcome enters this estimate. The recorded calibration population size is `60`; user-count sweeps reuse exact reference prefixes.

For each Monte Carlo replicate, alpha, beta and k are sampled once per user from the declared truncated-normal sensitivity distributions (alpha and beta independently) and then frozen for all methods and episodes. Only the distribution centers are source-backed; standard deviations and bounds are study assumptions. For the written first replicate, the resolved reference
vector is `[0.547662, 0.545256, 0.544179, 0.550692, 0.550834, 0.543139, 0.547606, 0.545932, 0.547085, 0.55038, 0.549355, 0.545164, 0.54693, 0.550053, 0.549147, 0.547057, 0.548656, 0.550922, 0.545435, 0.54667, 0.549903, 0.548253, 0.547368, 0.544272, 0.548725, 0.546112, 0.550899, 0.545444, 0.549123, 0.544863]` and its mean is `0.547570499172`.
The complete replicate-specific table is in `pt_user_profiles.csv`.

All RL agents observe the same per-user profile in their state. EUT methods use
`U - U_ref,u`; PT methods additionally apply the selected user's
`alpha_u`, `beta_u` and `k_u`. Thus EUT-D3QN and PT-D3QN still differ only by
the nonlinear reward transformation. PT uses the fixed declared reward divisor
`1.6` across every population and seed. The supplied implementation uses
objective probabilities and no probability weighting, so this code does the
same. PT is applied to each one-step realized utility outcome, not to a learned
distribution of whole-episode returns.

When `training_profile_count=20` exceeds one, all RL
methods train on the same separately seeded behavioral profile bank, while the
written evaluation profile is held out. The training bank also applies an
independent uniform relative perturbation of up to
`20%` to each `U_ref,u`; this is
a declared robustness assumption, not a fitted human distribution. A value of one is a fixed-profile
control and does not establish behavioral generalization.

## Interpretation of runs

`--quick` changes training to 8 episodes, evaluation to 3 episodes, horizon to
25 slots, batch size to 32, warm-up to 64 transitions and target update to 30
updates. It is a software smoke test only. Scientific conclusions require a
declared training budget, convergence checks, multiple independent Monte Carlo
replicates, uncertainty intervals and sensitivity analyses.
