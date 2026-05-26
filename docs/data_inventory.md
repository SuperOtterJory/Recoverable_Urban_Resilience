# Data Inventory

This document is updated by `scripts/run_data_mining.py`.

The raw data are stored locally under `data/raw_data/` and are not tracked by git because several files exceed normal GitHub size limits.

Expected raw-data groups:

- `speed/<city>/`: link/TMC speed observations, rainfall series, TMC identification, and existing resilience-index files when present.
- `demand/<city>/`: DTALite-style demand, link, node, route assignment, link performance, OD performance, and system performance outputs.

The first data-mining pass records:

- city coverage by speed and demand data;
- file availability and approximate size;
- rainfall-event coverage;
- speed-deficit and recovery proxies;
- demand-network concentration and accessibility proxies;
- whether each dataset can support model parameters such as `b_t`, `Q_t`, `p_i`, and `A_t`.
