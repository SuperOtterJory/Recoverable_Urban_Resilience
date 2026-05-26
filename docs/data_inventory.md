# Data Inventory

Raw data are stored locally under `data/raw_data/` and are not tracked by git because several files exceed normal GitHub size limits.

## City Coverage

| city | speed_csv | speed_csv_present | speed_csv_size_gb | rainfall_csv | tmc_identification_csv | resilience_index_csv | demand_csv | link_performance_csv | demand_dir_size_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| New York | True | True | 4.944 | True | True | True | True | True | 469.3 |
| Los Angeles | False | True | 0 | True | True | False | True | True | 323.5 |
| Chicago | True | True | 2.775 | True | True | True | True | True | 416.7 |
| Houston | True | True | 2.644 | True | True | True | True | True | 511.3 |
| Phoenix | False | False | 0 | True | False | False | True | True | 500 |
| Philadelphia | True | True | 3.443 | True | True | False | True | True | 503.6 |
| San Antonio | True | True | 1.994 | True | True | False | True | True | 417.2 |
| San Diego | False | False | 0 | True | False | False | True | True | 317.1 |
| Dallas | True | True | 4.396 | True | True | False | True | True | 450.4 |
| San Jose | False | False | 0 | True | False | False | True | True | 327.6 |
| Austin | True | True | 1.133 | True | True | False | True | True | 285 |

## Notes

- `speed_csv_size_gb` reports the size of the largest speed CSV selected for each city.
- Demand directories include DTALite-style OD, link, node, route, and performance outputs.
- The data-mining pipeline writes compact tracked outputs to `results/data_mining/`.
