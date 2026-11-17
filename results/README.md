# results/

Outputs of the recorded run and of the revision experiments.

| Path | Content |
|---|---|
| `user_count_*.csv`, `user_count_sweep_config.json`, `user_count_sweep_manifest.csv` | Aggregates over the five user counts: seed-level and user-level summaries, paired PT-D3QN minus EUT-D3QN statistics, training curves, evaluation and training profiles |
| `users_NNN/` | One folder per user count: resolved replicate configurations (`config.json`), `reproducibility_manifest.json`, `seed_manifest.csv`, seed-level and user-level summaries, profile tables, protocol and provenance documents. The raw records of each folder are release assets (see the top-level README) |
| `scale_control/` | Scale-matched linear control: frozen constants and regression check (`calibration.json`, `calibration.sha256`), seed-level results, training-reward statistics, paired contrasts, and one set of files per run in `units/` |
| `robustness/` | Matched-demand and small-packet conditions at 60 users: seed-level results, policy means, paired contrasts, and per-run files in `matched_d015/` and `small_packet_r4/` |
| `packet_baseline/` | Completion-First-Fair rule and re-evaluated PT Myopic Oracle: seed-level results, paired contrasts, policy means, the realised discounted returns of the nine policies at every user count (`archived_discounted_returns_all.csv`), and the paired return contrasts (`discounted_return_contrasts_all.csv`) |
| `packet_diagnostics/` | Packet-level service and delay diagnostics from the action replay: `uNNN/` per user count for the nine policies, and `service_diagnostics_seed_all.csv` with all policies including Completion-First-Fair |

`user_count_sweep_config.json` and the `output` field of the manifests record
the absolute output path of the original run. The folders it names are the
`users_NNN/` folders here.
