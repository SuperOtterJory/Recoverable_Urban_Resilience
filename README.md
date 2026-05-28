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
python scripts/analyze_rainfall_event_impacts.py
python scripts/calibrate_event_dynamics.py
python scripts/run_event_optimization.py --config configs/optimization.yml
python scripts/analyze_event_city_structure.py
```

In the Codex desktop environment, use the bundled Python runtime if available.

## Current Analytical Goal

The current data-mining stage asks:

1. Do the data contain measurable disruption and recovery episodes?
2. Are speed deficits, rainfall shocks, network demand, and OD dependence aligned enough to estimate functional loss?
3. Is there cross-city heterogeneity in disruption severity, recovery speed, and network concentration that could support a recoverability law?
4. Which parts can be supported directly by data, and which parts require the optimization/counterfactual model?

## Current Data-Mining Outputs

- Main Chinese report: `results/data_mining/reports/data_mining_report_zh.md`
- Final conclusion memo: `docs/final_conclusion_zh.md`
- Tables: `results/data_mining/tables/`
- Figures: `results/data_mining/figures/`

The current conclusion is that the data provide a strong empirical basis for observed disruption, endogenous recovery proxies, spatial heterogeneity, demand/network dependence, and potential targeting leverage. The central recoverable-resilience claim, however, requires the optimization/counterfactual layer because intervention effectiveness, budgets, response delay, and alternative allocation decisions are not directly observed in the raw data.

## Current Event-Level Optimization Outputs

- LP implementation: `src/recoverable_resilience/recovery_lp.py`
- Event calibration utilities: `src/recoverable_resilience/event_calibration.py`
- Optimization config: `configs/optimization.yml`
- Dynamic calibration outputs: `results/event_calibration/`
- Observed-event optimization outputs: `results/event_optimization/`
- Event city-structure outputs: `results/event_city_structure/`

The current LP keeps the draft model's continuous structure while adding two credibility refinements that remain linear:

- primitive-specific continuous deployment caps;
- concave piecewise-linear diminishing returns through continuous segment variables.

The canonical optimization now uses all available OD zones for the seven cities with usable speed-overlap data, with a sparse functional-dependence matrix so the LP remains tractable at hundreds to thousands of units. Each positive rainfall-impact event becomes its own 12-hour LP scenario. The older one-city-one-representative-scenario workflow remains in `scripts/run_optimization.py` for comparison only, not as the main analysis path.

## Current City-Structure Analysis

- Event city-structure report: `results/event_city_structure/reports/event_city_structure_report_zh.md`
- Event city-structure tables and figures: `results/event_city_structure/`
- Calibration explanation: `docs/calibration_explanation_zh.md`
- Rainfall-event definitions: `docs/rainfall_event_definitions_zh.md`

The current structural analysis fixes the recovery regime and asks whether cross-city structural variables explain event-level recoverability after moving away from aggregate rainfall scenarios. The strongest hypotheses are expected to involve functional-dependence scale and sparsity, event abnormal-speed-loss shape, congestion exposure, and whether optimized R/C/S resources target high-activity zones or less obvious dependence positions. These remain preliminary because the speed-overlap city sample currently contains seven cities.
