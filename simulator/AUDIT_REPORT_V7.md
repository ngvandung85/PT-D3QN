# Line-by-line and algorithm-by-algorithm audit — V7

Date: 2026-08-28  
Release: `6.1.0`  
Output schema: `4`

## 1. Verdict

The V7 implementation is **correct for the explicitly declared mathematical
model and comparison protocol**. No critical or major numerical defect remains
in the nine policy definitions, physical transition equations, Prospect Theory
transformation, replay targets, seed pairing, statistical aggregation or output
schema.

The audit did find one executable-wrapper defect in V6: PowerShell `-Quick`
silently passed the full defaults (3,000 training episodes, 100 evaluation
episodes and 50 calibration episodes) back to Python, overriding Python's quick
values. V7 fixes this and adds a regression check. V7 also streams the two large
detail tables to prevent avoidable RAM exhaustion in the 60-user experiment.

This verdict establishes software and internal mathematical correctness. It
does **not** establish that every modeling assumption is empirically valid or
that PT-D3QN will outperform other methods; those claims require the completed
multi-seed experiment and sensitivity analysis.

## 2. Code coverage of the manual audit

All executable and verification files were parsed and read by logical block:

| File | Lines | Audited role |
|---|---:|---|
| `pt_d3qn_uav.py` | 4,725 | Configuration, environment, nine methods, training, evaluation, statistics, sweep and CLI |
| `test_pt_d3qn_uav.py` | 1,210 | Functional and regression tests |
| `verify_algorithms_independently.py` | 543 | Independently recomputed equations, gradients and Bellman targets |
| `validate_results.py` | 420 | Completed-output reconstruction and cross-count checks |
| `run_user_count_sweep.ps1` | 119 | Windows orchestration and final-run defaults |

The core line map is:

| Core lines | Functionality | Result |
|---|---|---|
| 44–208 | Profile resolution, seed derivation and configuration | Pass; invalid RL boundary values now fail early |
| 211–1,106 | Physical environment, state, action mask, PT/EUT rewards, queue and energy transitions | Pass |
| 1,109–1,585 | Evaluation/training PT profiles, calibration and seed manifests | Pass |
| 1,588–1,797 | Adam, plain/dueling Q network and DQN/DDQN target logic | Pass |
| 1,876–1,965 | Four baselines and five-agent training loop | Pass |
| 1,968–2,345 | Streamed CSV sinks and evaluation metrics | Pass |
| 3,154–3,665 | Replicate summaries, paired inference, Holm correction and plots | Pass |
| 3,882–4,367 | Nested user-count sweep, run orchestration and cross-count outputs | Pass |
| 4,370–4,721 | CLI validation and dispatch | Pass |

## 3. Independent check of each method

Every method was checked against a separately constructed score or Bellman
target rather than merely calling the method and accepting its own output.

| Method | Implemented definition | Independent result |
|---|---|---|
| `Random` | Uniform draw over the current feasible action indices, including idle | Every sampled action valid; definition pass |
| `Greedy_Physical_Utility` | `argmax U_success(a)` over the feasible mask; deliberately ignores outage risk | Exhaustive independent argmax pass |
| `EUT_Myopic_Oracle` | `argmax [p U_success + (1-p) U_failure - U_ref,u]`, with declared idle score | Independent expected-outcome enumeration pass |
| `PT_Myopic_Oracle` | `argmax [p v_u(U_success-U_ref,u)+(1-p)v_u(U_failure-U_ref,u)]/1.6` | Independent user-indexed PT enumeration pass |
| `EUT_DQN` | Plain Q-network; target-network maximum in the Bellman target | Flags and numerical target pass |
| `EUT_DDQN` | Plain Q-network; online argmax and target-network evaluation | Selection/evaluation decoupling pass |
| `EUT_Dueling_DQN` | `Q=V+A-mean(A)` with vanilla DQN target | Forward aggregation, finite-difference gradient and target pass |
| `EUT_D3QN` | Dueling network plus Double-DQN target and EUT reward | Architecture and target pass |
| `PT_D3QN` | Same D3QN architecture/initialization/profile/scenarios as EUT-D3QN; only realized reward uses `v_u(.)/1.6` | Parameter identity, trajectory identity under fixed actions and reward-only ablation pass |

Terminal transitions suppress bootstrap. Invalid next actions are masked before
both online selection and target evaluation. The stored replay mask belongs to
the next state, and packed/unpacked masks are bit-identical.

## 4. Physical and behavioral equations

The independent audit reconstructed the following from primitive variables:

- Al-Hourani mean air-to-ground path loss, elevation-dependent LoS probability
  and free-space loss;
- channel noise `N0 + 10 log10(B_channel) + noise figure`;
- received SNR and Rayleigh block-fading success probability
  `exp[-(2^SE-1)/SNR]`;
- successful, failed and expected physical utility;
- user-indexed PT gain/loss branches with separate `alpha_u`, `beta_u` and
  `k_u`, without probability weighting;
- FIFO partial packet service, packet completion delay, arrival admission/drop
  conservation and queue-bit/deque equality;
- radio energy `P_tx * slot_duration`, energy affordability masks and monotonic
  non-negative energy;
- Jain's index bounds and its undefined value before any delivered service.

All reconstructed quantities match the implementation. Across random
trajectories, `offered = admitted + dropped`, delivered packets never exceed
admitted packets, and every success bit equals the common uniform outcome draw
compared with the selected action's success probability.

## 5. PT profile sampling and common random numbers

For each Monte Carlo replicate, each user's `alpha_u`, `beta_u` and `k_u` is
sampled from an independent per-parameter/per-user seed namespace exactly once.
The evaluation profile is then frozen for all nine methods and all evaluation
episodes. When the training profile count exceeds one, all five RL methods use
the same separately seeded training bank and the evaluation profile is held
out.

For the 5/15/30/45/60-user sweep, all four behavioral descriptors
`(alpha_u,beta_u,k_u,U_ref,u)` are first resolved at 60 users and smaller
experiments receive exact prefixes. Geometry, initial packet queues, initial
shadowing, arrivals, shadow innovations and the link-outcome uniform are also
nested prefixes under a common seed. Independent dynamic-prefix checks pass.

## 6. Metrics and statistical inference

- Episode rows are first averaged within each Monte Carlo replicate. The
  replicate—not the slot, episode or user—is the inferential unit.
- PT-D3QN and EUT-D3QN are joined by common replicate seed before differences
  are computed.
- Student-t paired confidence intervals and paired tests use only finite pairs.
- The exact two-sided sign-flip test is correctly enumerated for at most 20
  nonzero pairs and records its sign-exchangeability assumption.
- `utility_shortfall` is the declared primary endpoint. Holm correction is
  applied only to the declared non-redundant secondary family.
- Offered PDR, admitted PDR, packet-drop rate, link success/outage and
  delivered-packet delay remain distinct quantities.
- Canonical reference-risk metrics include idle-with-backlog opportunities;
  attempt-only diagnostics are separately named and remain undefined for idle.

The validator reconstructs PDR, packet conservation, link Bernoulli outcomes,
energy, throughput, selection totals and risk identities from raw CSV rows.

## 7. Executed evidence

| Check | Result |
|---|---:|
| Unit tests | 49/49 pass |
| Independent equation/algorithm checks | 45/45 pass |
| Additional property assertions | 19,041 pass |
| Full requested count set in smoke mode | 5/5 folders pass validator |
| V6/V7 numerical equivalence after storage refactor | 10 tables, 315 columns exactly equal |
| Python syntax/AST parsing | Pass for all four Python files |
| ZIP CRC/read-back and re-executed extracted package | Pass (10 files) |

The audit host did not contain native PowerShell. The corrected wrapper was
therefore inspected line by line and checked by two platform-neutral regression
guards; the first Windows action should still be the documented `-Quick` run.

## 8. Remaining modeling assumptions—not code errors

These points must be stated in the manuscript and, where material, tested by
sensitivity analysis:

1. `w_fairness` is a custom service-equity shaping term, not Jain's index, and
   it contributes to both successful and failed transmission utility.
2. Idle with backlog uses the equally weighted mean reference of currently
   backlogged users. This is a declared system-level opportunity-cost rule, not
   an empirically validated behavioral law.
3. FIFO packet ages are retained internally for delay measurement but the full
   age vector is not exposed to the feed-forward agent. The optimized reward
   dynamics remain observable; delivered-packet delay is an evaluation metric,
   not a fully observed optimization state variable.
4. One global transmission is scheduled per slot. The code does not implement
   simultaneous multi-UAV transmissions or co-channel interference.
5. PT uses a value function with objective success probabilities and no
   probability weighting, matching the declared source adaptation.
6. Truncated-normal profile spreads/bounds and reference jitter are sensitivity
   assumptions, not fitted human-behavior distributions.
7. Increasing user count simultaneously increases aggregate offered load,
   state dimension and action dimension; it is not an isolated action-space
   intervention.

## 9. Release decision

V7 is approved for the configured full simulation. Run the PowerShell smoke
test first, confirm `PASS` from both verification suites and the result
validator, then start the final output in a new empty folder. Quick results are
software checks only and must not be used as paper evidence.
