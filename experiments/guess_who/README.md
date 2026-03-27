Guess Who graphing assets live here.

This directory now includes the tracked experiment outputs needed to reproduce the graph.

Tracked repo assets:

- `guesswho_graphing.ipynb`
- `data/characters.json`
- `data/characters_50.json`
- `data/characters_100.json`
- `data/characters_original.json`
- `data/experiments/game_*.json`
- `data/posthoc_eig_results.csv`

Default tracked data layout:

```text
experiments/guess_who/data/
  characters.json
  characters_50.json
  characters_100.json
  characters_original.json
  experiments/
    game_*.json
  posthoc_eig_results.csv
```

Optional overrides:

- `GUESS_WHO_DATA_DIR`
- `GUESS_WHO_EXPERIMENTS_DIR`
- `GUESS_WHO_POSTHOC_CSV`
- `GUESS_WHO_EXPORT_DIR`

By default the notebook writes exports into `experiments/guess_who/`.
