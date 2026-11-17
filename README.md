# PT-D3QN: reference-dependent reward transformation for UAV-assisted IoT resource selection

Code, configurations and evaluation records for the manuscript
"Reference-Dependent Reward Transformation with a Prospect-Theoretic Value
Function in Dueling Double Deep Q-Learning for UAV-Assisted IoT Resource
Selection" (AI, manuscript ai-4658936).

The simulator and all learning code are a single NumPy module. No GPU is used.

## Contents

| Path | What it is |
|---|---|
| `simulator/pt_d3qn_uav.py` | Environment, agents, training, evaluation, statistics, plotting. SHA-256 `b5bb44186843fe4efe3c86e7039edcb398b35c23187f152641e815530e31ad91`, identical to the file that produced the reported results (recorded as `code_sha256` in every `results/users_NNN/reproducibility_manifest.json`) |
| `simulator/test_pt_d3qn_uav.py` | 49 unit tests |
| `simulator/verify_algorithms_independently.py` | Independent numerical audit of the learning rule, masks and statistics |
| `simulator/validate_results.py` | Consistency checks on a finished results folder |
| `simulator/run_user_count_sweep.ps1` | Launcher of the reported experiment |
| `revision_experiments/` | Drivers of the revision experiments. They import the simulator unmodified |
| `figures/` | MATLAB figure generators |
| `results/` | Archived outputs (see `results/README.md`) |
| `requirements-lock.txt` | Library versions of the recorded run |
| `LICENSE` | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| `SHA256SUMS.txt` | Digests of every file in this repository |
| `RELEASE_ASSETS_SHA256.txt` | Digests of the release assets |

## Licence

The code, configurations and records in this repository and its release assets
are released under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0), see `LICENSE`. Please cite the manuscript when using them.

## Large records (release assets)

Files above 25 MB are not stored in the Git tree. They are attached to the
release and included in the archived record:

| Asset | Content | Restore to |
|---|---|---|
| `evaluation_step_raw_uNNN.csv.gz` (NNN = 005, 015, 030, 045, 060) | Slot-level transition records of every evaluation episode: 9 policies x 10 seeds x 100 episodes x 80 slots | `results/users_NNN/evaluation_step_raw.csv` |
| `other_raw_records_uNNN.zip` | Episode-level, user-level and selection-level evaluation records and the training history | `results/users_NNN/` |
| `robustness_user_records.zip` | Per-user records of the two traffic robustness conditions | `results/robustness/` |

PowerShell, from the repository root, for one user count:

```
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('evaluation_step_raw_u060.csv.gz','rb'), open('results/users_060/evaluation_step_raw.csv','wb'))"
Expand-Archive other_raw_records_u060.zip -DestinationPath results\users_060
```

## Where each reported result comes from

| Manuscript item | File |
|---|---|
| Primary utility-shortfall comparison, paired statistics | `results/user_count_paired_pt_vs_eut_d3qn.csv`, `results/user_count_evaluation_seed_summary.csv` |
| Throughput, packet completion, fairness, energy by policy and user count | `results/user_count_evaluation_seed_summary.csv`, `results/user_count_user_seed_summary.csv` |
| Training trajectories | `results/user_count_training_curves.csv` |
| Reward-distribution, attainment, energy and feasibility analyses | slot-level records `evaluation_step_raw_uNNN.csv.gz` |
| Scale-matched linear control | `results/scale_control/` |
| Traffic robustness conditions | `results/robustness/` |
| Packet-aware baseline and discounted return | `results/packet_baseline/` (returns at all user counts: `archived_discounted_returns_all.csv`, `discounted_return_contrasts_all.csv`) |
| Packet-level service and delay diagnostics | `results/packet_diagnostics/` |
| Resolved configuration of every replicate | `results/users_NNN/config.json` |

## Check the installation

Python 3.14.5 was used for the recorded run.

```
pip install -r requirements-lock.txt
cd simulator
python -m unittest test_pt_d3qn_uav.py
python verify_algorithms_independently.py
```

## Regenerate the reported experiment

Five user counts (5, 15, 30, 45, 60), ten replicate seeds, 3000 training and
100 evaluation episodes per replicate, nine policies:

```
cd simulator
.\run_user_count_sweep.ps1 -Seeds "7,17,27,37,47,57,67,77,87,97" -Output ..\results_regenerated
```

which is equivalent to

```
python pt_d3qn_uav.py --user-counts 5,15,30,45,60 --seeds 7,17,27,37,47,57,67,77,87,97 --random-pt-profiles --reference-mode random_policy_calibration --training-profile-count 20 --training-reference-jitter 0.20 --initial-radio-energy-j 2.0 --replay-size 50000 --update-frequency 4 --train-episodes 3000 --eval-episodes 100 --reference-calibration-episodes 50 --output ../results_regenerated
```

The run writes one folder per user count (`users_005` ... `users_060`) with the
resolved replicate configurations (`config.json`), seed manifest, per-seed and
per-user summaries and, unless `--no-step-details` is given, the slot-level
transition records (`evaluation_step_raw.csv`). On the authors' workstation,
training one D3QN agent for one replicate took about 7, 14 and 27 minutes at
5, 15 and 30 users, and longer at 45 and 60 users. The complete experiment
trains five learned policies for ten replicates at each of five user counts.

Exact reproduction was verified on the authors' platform (Windows 10,
Python 3.14.5, the pinned library versions): rerunning EUT-D3QN and PT-D3QN on
replicate seed 7 reproduced the archived seed-level utility shortfall,
throughput and offered PDR to within 1.2e-16 at every user count
(`results/scale_control/calibration.json`, field `regression_vs_archive`).
Other platforms or library versions may give different floating-point results.

A quick end-to-end check that takes a few minutes:

```
python pt_d3qn_uav.py --user-counts 5,8 --seeds 7,17,27 --random-pt-profiles --train-episodes 60 --eval-episodes 3 --output ../results_quick
```

The evaluation of the non-learning policies is deterministic and can be checked
against the archive without training: `revision_experiments/run_packet_baseline.py`
with `--also PT_Myopic_Oracle` re-evaluates that policy on the archived
scenarios and reproduces its archived seed-level metrics.

## Revision experiments

All drivers read the archived `results/users_NNN/config.json` and leave the
simulator untouched.

```
cd revision_experiments

# Scale-matched linear reward control (one additional training arm)
python run_scale_control.py calibrate --simulator ../simulator --results ../results --seed-metrics ../results/user_count_evaluation_seed_summary.csv --user-counts 5 15 30 45 60 --out ../results/scale_control
python run_scale_control.py run       --simulator ../simulator --results ../results --user-counts 5 15 30 45 60 --out ../results/scale_control
python run_scale_control.py analyze   --seed-metrics ../results/user_count_evaluation_seed_summary.csv --user-seed-metrics ../results/user_count_user_seed_summary.csv --out ../results/scale_control

# Matched aggregate demand, and 1 Mbit packets at fixed offered bit volume
python run_robustness.py run --simulator ../simulator --results ../results --condition matched --demand-users 15 --user-counts 60 --out ../results/robustness
python run_robustness.py run --simulator ../simulator --results ../results --condition small_packet --user-counts 60 --out ../results/robustness
python run_robustness.py analyze --seed-metrics ../results/user_count_evaluation_seed_summary.csv --user-seed-metrics ../results/user_count_user_seed_summary.csv --out ../results/robustness

# Completion-aware scheduling rule with a fairness constraint (evaluation only)
python run_packet_baseline.py run --simulator ../simulator --results ../results --user-counts 5 15 30 45 60 --also PT_Myopic_Oracle --out ../results/packet_baseline

# Packet-level service and delay diagnostics by replaying the recorded actions
# (needs results/users_NNN/evaluation_step_raw.csv, see "Large records")
python replay_packet_diagnostics.py --simulator ../simulator --results ../results --user-count 60 --out ../results/packet_diagnostics/u060
python rule_packet_diagnostics.py --simulator ../simulator --results ../results --user-counts 5 15 30 45 60 --out ../results/packet_diagnostics/rule

# Realised discounted returns of the nine policies from the slot-level records
python discounted_returns.py --results ../results --user-counts 5 15 30 45 60 --out ../results/packet_baseline/archived_discounted_returns_all.csv
```

`run_scale_control.py`, `run_robustness.py` and `run_packet_baseline.py` each
have a `selftest` command that needs no archived results.


## Implementation details referred to in the manuscript

### Random streams

Every stream is a NumPy `default_rng` generator seeded by
`SeedSequence([base, part_1, part_2, ...])` (`_derived_seed`). Exogenous
streams are indexed by replicate, episode and slot. No exogenous quantity is
drawn from a stream whose state depends on the actions taken.

| Quantity | Seed tuple |
|---|---|
| Evaluation scenario seed `s` of episode `e` | `(100000, replicate, e, 0)` |
| Evaluation policy stream (Random policy only) | `(100000, replicate, e, 1)` |
| Training environment seed `s` | the replicate seed; the episode index counts training episodes |
| User positions | `(610000, s, episode, 0)` |
| Initial queue contents | `(610000, s, episode, 1)` |
| Initial shadowing | `(610000, s, episode, 2)` |
| Transmission success variate of slot `t` | `(610000, s, episode, 3, t)` |
| Packet arrivals of slot `t` | `(610000, s, episode, 4, t)` |
| Shadowing innovation of slot `t` | `(610000, s, episode, 5, t)` |
| Evaluation profile of user `u`: gain exponent, loss exponent, loss weight | `(300000, replicate, 0, k, u)` with `k = 0, 1, 2` |
| Training profile `p` of user `u`: gain exponent, loss exponent, loss weight, reference factor | `(350000, replicate, p, k, u)` with `k = 0, 1, 2, 3` |
| Reference calibration episode `e`: scenario, policy | `(200000, replicate, e, 0)` and `(200000, replicate, e, 1)` |
| Network initialisation `a` | `(400000, replicate)`; target network `a + 1000003`; exploration and replay sampling `a + 2000003` |

One success variate is drawn per slot and compared with the success
probability of the selected action, so two policies that select the same
action in the same slot of the same scenario obtain the same outcome.

### Order of events within a slot

1. The policy selects an action from the feasible set.
2. The transmission outcome, delivered bits, energy and reward are determined.
3. Arrivals are drawn per user from a Poisson distribution with mean
   `arrival_rate_pps * slot_duration_s`, admitted while the queue has room for
   a whole packet, and otherwise counted as dropped.
4. The shadowing state is advanced.

Packets present at the start of an episode carry timestamp 0. A packet
admitted in slot `t` carries timestamp `t + 1` and cannot be served in slot `t`.

### Observation vector

For `M` UAVs, `C` channels and `U` users the observation has
`M + 6U + MU + MCU + 1` entries, in this order:

| Entries | Feature | Normalisation |
|---|---|---|
| `M` | residual transmit energy of each UAV | divided by the initial allowance, 2 J |
| `U` | queue backlog of each user | divided by the queue capacity, 20 Mbit |
| `U` | bits delivered to each user so far | divided by `T * B_c * 4.0 bps/Hz * slot` = 80 Mbit, clipped to [0, 1] |
| `U` | gain exponent | none |
| `U` | loss exponent | none |
| `U` | loss weight `k` | `k / (1 + k)` |
| `U` | reference utility `r` | `tanh(r)` |
| `MU` | mean link quality: SNR in dB at the maximum transmit power with distance-dependent path loss only | `tanh(SNR / 20)` |
| `MCU` | instantaneous link quality: the same SNR including the current shadowing term | `tanh(SNR / 20)` |
| `1` | slot index | divided by the horizon, 80 |

All constants are fixed. None is estimated from training or evaluation data.

### Training profiles

Each replicate has a bank of 20 training profiles, used in rotation by training
episode. Gain exponent, loss exponent and loss weight are drawn per user from
the truncated normal distributions of the evaluation profile; the reference of
each user is the calibrated reference multiplied by a factor drawn uniformly
from [0.8, 1.2]. The evaluation profile is not in the bank.
