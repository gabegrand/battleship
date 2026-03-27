Guess Who graphing assets live here.

The notebook expects local-only data and does not require those experiment dumps to be tracked in git.

Default local data layout:

```text
experiments/guess_who/local_data/
  experiments/
    game_*.json
    experiment_summary_*.json
  posthoc_eig_results.csv
```

Optional overrides:

- `GUESS_WHO_LOCAL_DATA_DIR`
- `GUESS_WHO_EXPERIMENTS_DIR`
- `GUESS_WHO_POSTHOC_CSV`
- `GUESS_WHO_EXPORT_DIR`

By default the notebook writes exports into `experiments/guess_who/`.
