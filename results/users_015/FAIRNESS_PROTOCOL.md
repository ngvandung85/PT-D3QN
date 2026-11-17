# Fair comparison protocol

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
