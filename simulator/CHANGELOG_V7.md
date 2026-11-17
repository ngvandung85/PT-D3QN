# Changelog V7 (release 6.1.0)

## Correctness fixes

- Fixed the PowerShell `-Quick` contract. Full-run defaults for training,
  evaluation and reference calibration are no longer passed in quick mode
  unless the caller explicitly supplies the corresponding parameter.
- Added regression checks for all three PowerShell quick-mode override guards.
- Added explicit validation for episode counts, discount factor, learning rate,
  batch/replay/warm-up sizes, hidden width, target-update interval and the
  epsilon schedule. Invalid programmatic configurations now fail before a run.

## Full-sweep reliability

- User-level and selection-level raw rows are streamed directly to CSV instead
  of accumulating millions of Python dictionaries in memory.
- Every streamed user, selection and optional slot CSV is flushed, read back and
  checked against its analytically expected row count before summarization.
- The output schema and numerical definitions are unchanged from V6.

## Verification

- 49 unit tests pass.
- 45 independent equation/algorithm checks pass.
- 19,041 additional trajectory, conservation, CRN, nesting and architecture
  property assertions pass.
- A complete 5/15/30/45/60-user smoke sweep passes `validate_results.py`.
- Ten raw/statistical tables from matched V6 and V7 runs are numerically
  identical (315 exact column-level comparisons); V7 changes storage mechanics,
  validation and wrapper behavior, not algorithm outputs.

