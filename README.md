# Recoverable Urban Resilience

This repository supports the paper project **Recoverable Urban Resilience: Learning the Laws of Managed Recovery from Empirical Urban Disruptions**.

The project studies whether observed urban disruption and mobility data contain enough empirical structure to support a decision-centered resilience question:

> Given observed urban functional loss, limited intervention resources, response delays, and equity constraints, what portion of the loss is recoverable through intelligent management?

## Repository Layout

- `high-level idea/`: conceptual notes and paper framing.
- `writing/draft/`: working drafts, including optimization-model notes.
- `writing/paper/`: LaTeX manuscript source.
- `src/recoverable_resilience/`: reusable Python code for data inventory, mining, metrics, and plotting.
- `scripts/`: command-line entry points for reproducible analyses.
- `configs/`: analysis configuration.
- `data/raw_data/`: local raw data, not tracked by git because files are multi-GB.
- `data/interim/`: reproducible intermediate outputs.
- `data/processed/`: cleaned or analysis-ready outputs.
- `results/data_mining/`: tables, figures, and written reports from exploratory data mining.
- `docs/`: project documentation, data inventory, research plan, and analysis notes.

## Data Policy

Raw data live locally under `data/raw_data/` and are intentionally ignored by git. Several speed CSV files are larger than GitHub's normal file limits. The repository tracks code, configuration, data dictionaries, compact summaries, figures, and reports so the analysis can be rerun from the local raw data.

## Quick Start

From the repository root:

```powershell
python -m pip install -e .
python scripts/run_data_mining.py --config configs/data_mining.yml
```

In the Codex desktop environment, use the bundled Python runtime if available.

## Current Analytical Goal

The current data-mining stage asks:

1. Do the data contain measurable disruption and recovery episodes?
2. Are speed deficits, rainfall shocks, network demand, and OD dependence aligned enough to estimate functional loss?
3. Is there cross-city heterogeneity in disruption severity, recovery speed, and network concentration that could support a recoverability law?
4. Which parts can be supported directly by data, and which parts require the optimization/counterfactual model?
