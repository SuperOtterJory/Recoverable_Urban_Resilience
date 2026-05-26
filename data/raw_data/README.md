# Raw Data

Place local raw data here.

This directory is ignored by git because the current raw files include multi-GB speed CSVs, large ZIP archives, OSM extracts, and executable simulation artifacts.

The expected structure is:

```text
data/raw_data/
  speed/<city>/
  demand/<city>/
```

Analysis scripts read from this directory and write compact, tracked outputs to `results/data_mining/`.
