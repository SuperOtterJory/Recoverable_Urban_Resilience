"""Audit neural action-value surrogates and identity leakage.

The learning plan describes a recoverability decision surrogate, but the paper's
scientific object is the compact law rather than a black-box predictor. This
script trains lightweight MLP regressors on action-value tokens and compares
them with ridge baselines under leave-city and random-event splits. It also
tests city/event identity features in non-strict splits to quantify how much
apparently strong performance can come from leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor
from sklearn.preprocessing import StandardScaler

from analyze_factorized_action_surrogate import (
    DEFICIT_FEATURES,
    EVENT_CONTEXT_FEATURES,
    EVENT_KEYS,
    EXPOSURE_FEATURES,
    FACTORIZED_BASE,
    STRUCTURE_FEATURES,
    SUBSTITUTION_FEATURES,
    TIME_FEASIBILITY_FEATURES,
    fit_ridge,
    predict_ridge,
    prepare_tokens,
)
from recoverable_resilience.paths import find_repo_root


EPS = 1e-12
RIDGE_ALPHA = 2.0
RANDOM_EVENT_FOLDS = 5
TOKEN_TEST_SHARE = 0.20
SEED = 20260529

FULL_ADDITIVE = list(
    dict.fromkeys(
        DEFICIT_FEATURES
        + EXPOSURE_FEATURES
        + STRUCTURE_FEATURES
        + SUBSTITUTION_FEATURES
        + TIME_FEASIBILITY_FEATURES
        + EVENT_CONTEXT_FEATURES
    )
)


MODEL_SPECS = [
    {
        "model_id": "R1_factorized_ridge",
        "family": "ridge",
        "feature_set": "factorized",
        "description": "ridge on seven factorized law features",
        "base_features": FACTORIZED_BASE,
        "split_roles": {"leave_city", "random_event"},
        "include_city_id": False,
        "include_event_id": False,
    },
    {
        "model_id": "R2_full_ridge",
        "family": "ridge",
        "feature_set": "full_additive",
        "description": "ridge on full additive action features",
        "base_features": FULL_ADDITIVE,
        "split_roles": {"leave_city", "random_event", "token_random"},
        "include_city_id": False,
        "include_event_id": False,
    },
    {
        "model_id": "N1_factorized_mlp",
        "family": "mlp",
        "feature_set": "factorized",
        "description": "MLP on seven factorized law features",
        "base_features": FACTORIZED_BASE,
        "split_roles": {"leave_city"},
        "include_city_id": False,
        "include_event_id": False,
    },
    {
        "model_id": "N2_full_mlp",
        "family": "mlp",
        "feature_set": "full_additive",
        "description": "MLP on full additive action features",
        "base_features": FULL_ADDITIVE,
        "split_roles": {"leave_city", "random_event", "token_random"},
        "include_city_id": False,
        "include_event_id": False,
    },
    {
        "model_id": "N3_full_city_id_mlp",
        "family": "mlp",
        "feature_set": "full_additive_plus_city_id",
        "description": "MLP with full additive features and city identity; evaluated only where cities appear in train",
        "base_features": FULL_ADDITIVE,
        "split_roles": {"random_event"},
        "include_city_id": True,
        "include_event_id": False,
    },
    {
        "model_id": "N4_full_event_id_mlp",
        "family": "mlp",
        "feature_set": "full_additive_plus_event_id",
        "description": "MLP with full additive features and event identity under random token split",
        "base_features": FULL_ADDITIVE,
        "split_roles": {"token_random"},
        "include_city_id": False,
        "include_event_id": True,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", nargs="*", type=int, default=[64, 32])
    parser.add_argument("--output-dir", default="results/neural_surrogate_leakage")
    args = parser.parse_args()

    root = find_repo_root()
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    set_seeds(SEED)

    tokens = load_tokens(root)
    validate_features(tokens)
    splits = build_splits(tokens)
    metrics, event_metrics = run_models(
        tokens,
        splits,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden=list(args.hidden),
    )
    summary = summarize(metrics)
    diagnostics = build_diagnostics(summary)

    write_table(splits, table_dir / "neural_split_summary.csv")
    write_table(metrics, table_dir / "neural_surrogate_metrics.csv")
    write_table(event_metrics, table_dir / "neural_surrogate_event_metrics.csv")
    write_table(summary, table_dir / "neural_surrogate_summary.csv")
    (table_dir / "neural_surrogate_leakage_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_figures(metrics, summary, figure_dir)
    write_report(
        report_dir / "neural_surrogate_leakage_report_zh.md",
        diagnostics,
        summary,
        metrics,
    )
    print(f"Wrote neural surrogate leakage analysis to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    tokens = pd.read_csv(path)
    tokens = prepare_tokens(tokens)
    tokens["event_key"] = event_key(tokens)
    return tokens


def validate_features(tokens: pd.DataFrame) -> None:
    missing: list[str] = []
    for feature in sorted(set(FACTORIZED_BASE + FULL_ADDITIVE)):
        if feature not in tokens:
            missing.append(feature)
    if missing:
        raise KeyError(f"Missing neural surrogate features: {missing}")


def build_splits(tokens: pd.DataFrame) -> pd.DataFrame:
    events = (
        tokens[["city", "event_id", "event_start", "event_key"]]
        .drop_duplicates("event_key")
        .sort_values(["city", "event_start", "event_id"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    all_events = set(events["event_key"])

    for city in sorted(events["city"].unique()):
        test_events = set(events.loc[events["city"].eq(city), "event_key"])
        train_events = all_events - test_events
        rows.append(
            split_row(
                split_role="leave_city",
                split_id=f"leave_{slug(city)}",
                heldout=str(city),
                train_events=train_events,
                test_events=test_events,
                full_event_eval=True,
                tokens=tokens,
                events=events,
            )
        )

    event_folds = assign_random_event_folds(events)
    for fold in range(RANDOM_EVENT_FOLDS):
        test_events = set(events.loc[event_folds.eq(fold), "event_key"])
        train_events = all_events - test_events
        rows.append(
            split_row(
                split_role="random_event",
                split_id=f"fold_{fold}",
                heldout=f"random_event_fold_{fold}",
                train_events=train_events,
                test_events=test_events,
                full_event_eval=True,
                tokens=tokens,
                events=events,
            )
        )

    rng = np.random.default_rng(SEED)
    token_test_mask = rng.random(len(tokens)) < TOKEN_TEST_SHARE
    rows.append(
        {
            "split_role": "token_random",
            "split_id": "token_random_20pct",
            "heldout": "random_action_tokens",
            "train_event_keys": ";".join(sorted(all_events)),
            "test_event_keys": ";".join(sorted(all_events)),
            "n_train_events": int(events["event_key"].nunique()),
            "n_test_events": int(events["event_key"].nunique()),
            "n_train_cities": int(events["city"].nunique()),
            "n_test_cities": int(events["city"].nunique()),
            "n_train_tokens": int((~token_test_mask).sum()),
            "n_test_tokens": int(token_test_mask.sum()),
            "full_event_eval": False,
            "token_random_seed": SEED,
            "token_test_share": TOKEN_TEST_SHARE,
        }
    )
    return pd.DataFrame(rows)


def split_row(
    *,
    split_role: str,
    split_id: str,
    heldout: str,
    train_events: set[str],
    test_events: set[str],
    full_event_eval: bool,
    tokens: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, Any]:
    train_event_frame = events[events["event_key"].isin(train_events)]
    test_event_frame = events[events["event_key"].isin(test_events)]
    return {
        "split_role": split_role,
        "split_id": split_id,
        "heldout": heldout,
        "train_event_keys": ";".join(sorted(train_events)),
        "test_event_keys": ";".join(sorted(test_events)),
        "n_train_events": int(len(train_event_frame)),
        "n_test_events": int(len(test_event_frame)),
        "n_train_cities": int(train_event_frame["city"].nunique()),
        "n_test_cities": int(test_event_frame["city"].nunique()),
        "n_train_tokens": int(tokens["event_key"].isin(train_events).sum()),
        "n_test_tokens": int(tokens["event_key"].isin(test_events).sum()),
        "full_event_eval": bool(full_event_eval),
        "token_random_seed": "",
        "token_test_share": "",
    }


def assign_random_event_folds(events: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(SEED)
    folds = pd.Series(index=events.index, dtype=int)
    for _, city_events in events.groupby("city", sort=True):
        idx = city_events.index.to_numpy()
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        for pos, event_idx in enumerate(shuffled):
            folds.loc[event_idx] = pos % RANDOM_EVENT_FOLDS
    return folds.astype(int)


def run_models(
    tokens: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    hidden: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    feature_cache: dict[str, pd.DataFrame] = {}
    for spec in MODEL_SPECS:
        feature_key = feature_cache_key(spec)
        if feature_key not in feature_cache:
            feature_cache[feature_key] = build_feature_frame(tokens, spec)
        features = feature_cache[feature_key]
        for split in splits.itertuples(index=False):
            if split.split_role not in spec["split_roles"]:
                continue
            train_mask, test_mask = split_masks(tokens, split)
            train = tokens.loc[train_mask].copy()
            test = tokens.loc[test_mask].copy()
            if train.empty or test.empty:
                continue
            x_train = features.loc[train.index]
            x_test = features.loc[test.index]
            if spec["family"] == "ridge":
                model = fit_ridge(x_train, train["target_log"], alpha=RIDGE_ALPHA)
                pred_log = predict_ridge(model, x_test)
                train_loss = np.nan
            else:
                pred_log, train_loss = fit_predict_mlp(
                    x_train,
                    train["target_log"],
                    x_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    hidden=hidden,
                    seed=split_seed(spec["model_id"], split.split_id),
                )
            test["predicted_value"] = np.expm1(np.maximum(pred_log, 0.0)) / 1_000.0
            base = {
                "split_role": split.split_role,
                "split_id": split.split_id,
                "heldout": split.heldout,
                "model_id": spec["model_id"],
                "family": spec["family"],
                "feature_set": spec["feature_set"],
                "description": spec["description"],
                "n_features": int(x_train.shape[1]),
                "n_train_tokens": int(len(train)),
                "n_test_tokens": int(len(test)),
                "n_train_events": int(split.n_train_events),
                "n_test_events": int(split.n_test_events),
                "full_event_eval": bool(split.full_event_eval),
                "train_loss": safe_float(train_loss),
            }
            metric_rows.append({**base, **prediction_metrics(test, "predicted_value", bool(split.full_event_eval))})
            if bool(split.full_event_eval):
                event_rows.extend(event_metric_rows(test, base))
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows)


def build_feature_frame(tokens: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    frame = tokens[list(spec["base_features"])].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    parts = [frame]
    if spec.get("include_city_id"):
        city_dummies = pd.get_dummies(tokens["city"].astype(str).map(slug), prefix="city_id", dtype=float)
        parts.append(city_dummies)
    if spec.get("include_event_id"):
        event_dummies = pd.get_dummies(tokens["event_key"].astype(str), prefix="event_id", dtype=float)
        parts.append(event_dummies)
    frame = pd.concat(parts, axis=1)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_predict_mlp(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    hidden: list[int],
    seed: int,
) -> tuple[np.ndarray, float]:
    set_seeds(seed)
    x_train_arr = x_train.to_numpy(dtype=np.float64)
    x_test_arr = x_test.to_numpy(dtype=np.float64)
    y_arr = y_train.to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    x_train_arr = scaler.fit_transform(np.nan_to_num(x_train_arr, nan=0.0, posinf=0.0, neginf=0.0))
    x_test_arr = scaler.transform(np.nan_to_num(x_test_arr, nan=0.0, posinf=0.0, neginf=0.0))
    model = SklearnMLPRegressor(
        hidden_layer_sizes=tuple(int(width) for width in hidden),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=batch_size,
        learning_rate_init=1e-3,
        max_iter=max(1, epochs),
        random_state=seed,
        early_stopping=False,
        n_iter_no_change=max(epochs + 1, 10),
        tol=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train_arr, y_arr)
    return model.predict(x_test_arr), safe_float(getattr(model, "loss_", np.nan))


def prediction_metrics(frame: pd.DataFrame, score_col: str, full_event_eval: bool) -> dict[str, float]:
    y = frame["target_value"].to_numpy(dtype=float)
    pred = frame[score_col].to_numpy(dtype=float)
    out = {
        "n_tokens": int(len(frame)),
        "n_events": int(frame[EVENT_KEYS].drop_duplicates().shape[0]),
        "pearson": safe_corr(y, pred),
        "spearman": safe_float(frame["target_value"].corr(frame[score_col], method="spearman")),
        "mae": safe_float(np.mean(np.abs(y - pred))),
        "top_5pct_value_capture": np.nan,
        "top_5pct_ndcg": np.nan,
        "top_5pct_precision": np.nan,
        "top_5pct_regret": np.nan,
    }
    if full_event_eval:
        event_metric = [event_top_metrics(group, score_col, 0.05) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        event_df = pd.DataFrame(event_metric)
        if not event_df.empty:
            out["top_5pct_value_capture"] = safe_float(event_df["value_capture"].mean())
            out["top_5pct_ndcg"] = safe_float(event_df["ndcg"].mean())
            out["top_5pct_precision"] = safe_float(event_df["precision"].mean())
            out["top_5pct_regret"] = 1.0 - safe_float(event_df["value_capture"].mean())
    return out


def event_metric_rows(frame: pd.DataFrame, base: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (city, event_id), group in frame.groupby(EVENT_KEYS, sort=True):
        metric = event_top_metrics(group, "predicted_value", 0.05)
        rows.append(
            {
                **base,
                "city": city,
                "event_id": int(event_id),
                "n_tokens": int(len(group)),
                "spearman": safe_float(group["target_value"].corr(group["predicted_value"], method="spearman")),
                "top_5pct_value_capture": metric["value_capture"],
                "top_5pct_ndcg": metric["ndcg"],
                "top_5pct_precision": metric["precision"],
            }
        )
    return rows


def event_top_metrics(group: pd.DataFrame, score_col: str, frac: float) -> dict[str, float]:
    if group.empty or group["target_value"].sum() <= EPS:
        return {"value_capture": np.nan, "ndcg": np.nan, "precision": np.nan}
    k = max(1, int(np.ceil(len(group) * frac)))
    chosen = group.nlargest(k, score_col)
    ideal = group.nlargest(k, "target_value")
    chosen_values = chosen["target_value"].to_numpy(dtype=float)
    ideal_values = ideal["target_value"].to_numpy(dtype=float)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(chosen_values * discount[: len(chosen_values)]))
    idcg = float(np.sum(ideal_values * discount[: len(ideal_values)]))
    return {
        "value_capture": safe_div(float(chosen_values.sum()), float(ideal_values.sum())),
        "ndcg": safe_div(dcg, idcg),
        "precision": len(set(chosen.index) & set(ideal.index)) / k,
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.groupby(["split_role", "model_id", "family", "feature_set", "description"], as_index=False)
        .agg(
            n_splits=("split_id", "nunique"),
            n_features=("n_features", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_pearson=("pearson", "mean"),
            mean_mae=("mae", "mean"),
            mean_top5_capture=("top_5pct_value_capture", "mean"),
            min_top5_capture=("top_5pct_value_capture", "min"),
            mean_top5_ndcg=("top_5pct_ndcg", "mean"),
            mean_top5_precision=("top_5pct_precision", "mean"),
            mean_train_loss=("train_loss", "mean"),
        )
        .sort_values(["split_role", "mean_top5_capture", "mean_spearman"], ascending=[True, False, False])
    )


def build_diagnostics(summary: pd.DataFrame) -> dict[str, Any]:
    leave_factorized_mlp = one_row(summary, split_role="leave_city", model_id="N1_factorized_mlp")
    leave_full_mlp = one_row(summary, split_role="leave_city", model_id="N2_full_mlp")
    leave_factorized_ridge = one_row(summary, split_role="leave_city", model_id="R1_factorized_ridge")
    leave_full_ridge = one_row(summary, split_role="leave_city", model_id="R2_full_ridge")
    random_full_mlp = one_row(summary, split_role="random_event", model_id="N2_full_mlp")
    random_city_mlp = one_row(summary, split_role="random_event", model_id="N3_full_city_id_mlp")
    random_full_ridge = one_row(summary, split_role="random_event", model_id="R2_full_ridge")
    token_full_mlp = one_row(summary, split_role="token_random", model_id="N2_full_mlp")
    token_event_mlp = one_row(summary, split_role="token_random", model_id="N4_full_event_id_mlp")
    token_full_ridge = one_row(summary, split_role="token_random", model_id="R2_full_ridge")
    return {
        "leave_city_factorized_mlp_top5_capture": safe_float(leave_factorized_mlp.get("mean_top5_capture")),
        "leave_city_full_mlp_top5_capture": safe_float(leave_full_mlp.get("mean_top5_capture")),
        "leave_city_factorized_ridge_top5_capture": safe_float(leave_factorized_ridge.get("mean_top5_capture")),
        "leave_city_full_ridge_top5_capture": safe_float(leave_full_ridge.get("mean_top5_capture")),
        "leave_city_full_mlp_minus_ridge_top5": safe_float(leave_full_mlp.get("mean_top5_capture"))
        - safe_float(leave_full_ridge.get("mean_top5_capture")),
        "leave_city_factorized_mlp_minus_ridge_top5": safe_float(leave_factorized_mlp.get("mean_top5_capture"))
        - safe_float(leave_factorized_ridge.get("mean_top5_capture")),
        "random_event_full_mlp_top5_capture": safe_float(random_full_mlp.get("mean_top5_capture")),
        "random_event_full_ridge_top5_capture": safe_float(random_full_ridge.get("mean_top5_capture")),
        "random_event_city_id_mlp_top5_capture": safe_float(random_city_mlp.get("mean_top5_capture")),
        "random_event_city_id_minus_no_id_top5": safe_float(random_city_mlp.get("mean_top5_capture"))
        - safe_float(random_full_mlp.get("mean_top5_capture")),
        "random_event_minus_leave_city_full_mlp_top5": safe_float(random_full_mlp.get("mean_top5_capture"))
        - safe_float(leave_full_mlp.get("mean_top5_capture")),
        "token_random_full_mlp_spearman": safe_float(token_full_mlp.get("mean_spearman")),
        "token_random_event_id_mlp_spearman": safe_float(token_event_mlp.get("mean_spearman")),
        "token_random_full_ridge_spearman": safe_float(token_full_ridge.get("mean_spearman")),
        "token_random_event_id_minus_no_id_spearman": safe_float(token_event_mlp.get("mean_spearman"))
        - safe_float(token_full_mlp.get("mean_spearman")),
    }


def make_figures(metrics: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path) -> None:
    make_split_model_figure(summary, figure_dir / "neural_split_model_summary.png")
    make_leakage_gap_figure(summary, figure_dir / "neural_identity_leakage_gaps.png")
    make_fold_scatter(metrics, figure_dir / "neural_random_vs_leave_city.png")


def make_split_model_figure(summary: pd.DataFrame, path: Path) -> None:
    keep = summary[summary["split_role"].isin(["leave_city", "random_event"])].copy()
    if keep.empty:
        return
    keep["label"] = keep["split_role"] + "\n" + keep["model_id"]
    keep = keep.sort_values(["split_role", "mean_top5_capture"], ascending=[True, True])
    colors = keep["family"].map({"ridge": "#94a3b8", "mlp": "#2563eb"}).fillna("#64748b")
    fig, ax = plt.subplots(figsize=(11.0, max(5.2, 0.34 * len(keep))))
    ax.barh(keep["label"], keep["mean_top5_capture"], color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean held-out-event top-5% value capture")
    ax.set_title("Neural and ridge action-value surrogates under strict and non-strict splits")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_leakage_gap_figure(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    random_full = one_row(summary, split_role="random_event", model_id="N2_full_mlp")
    random_city = one_row(summary, split_role="random_event", model_id="N3_full_city_id_mlp")
    token_full = one_row(summary, split_role="token_random", model_id="N2_full_mlp")
    token_event = one_row(summary, split_role="token_random", model_id="N4_full_event_id_mlp")
    leave_full = one_row(summary, split_role="leave_city", model_id="N2_full_mlp")
    rows.append(
        {
            "comparison": "random event: +city ID",
            "delta": safe_float(random_city.get("mean_top5_capture")) - safe_float(random_full.get("mean_top5_capture")),
            "metric": "top-5% capture",
        }
    )
    rows.append(
        {
            "comparison": "random token: +event ID",
            "delta": safe_float(token_event.get("mean_spearman")) - safe_float(token_full.get("mean_spearman")),
            "metric": "Spearman",
        }
    )
    rows.append(
        {
            "comparison": "random event - leave city",
            "delta": safe_float(random_full.get("mean_top5_capture")) - safe_float(leave_full.get("mean_top5_capture")),
            "metric": "top-5% capture",
        }
    )
    plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = np.where(plot["delta"] >= 0, "#2563eb", "#ef4444")
    ax.bar(plot["comparison"], plot["delta"], color=colors)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_ylabel("Delta")
    ax.set_title("Identity and split-leakage audit")
    for idx, row in enumerate(plot.itertuples(index=False)):
        ax.text(idx, row.delta + (0.005 if row.delta >= 0 else -0.005), f"{row.delta:+.3f}", ha="center", va="bottom" if row.delta >= 0 else "top")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_fold_scatter(metrics: pd.DataFrame, path: Path) -> None:
    keep = metrics[metrics["model_id"].isin(["R2_full_ridge", "N2_full_mlp"]) & metrics["full_event_eval"]].copy()
    if keep.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for model_id, color in [("R2_full_ridge", "#94a3b8"), ("N2_full_mlp", "#2563eb")]:
        plot = keep[keep["model_id"].eq(model_id)]
        ax.scatter(plot["spearman"], plot["top_5pct_value_capture"], label=model_id, color=color, alpha=0.82, s=52)
    ax.set_xlabel("Token Spearman on held-out split")
    ax.set_ylabel("Held-out-event top-5% value capture")
    ax.set_title("Fold-level neural surrogate performance")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Neural Surrogate Leakage Audit V26",
        "",
        "## 这一版做了什么",
        "",
        "V26 用轻量 MLP action-value surrogate 检查两个问题：第一，神经网络是否明显超过当前低维/线性 law；第二，随机事件切分、随机 token 切分和 city/event identity 特征会不会带来虚高表现。主科学结论仍以 leave-city、leave-regime、temporal holdout 和 symbolic law 为准；这里的 neural surrogate 是审计工具，不是新的黑箱政策。",
        "",
        "## 主要指标",
        "",
        f"- leave-city full MLP top-5% capture = {diagnostics['leave_city_full_mlp_top5_capture']:.4f}; full ridge = {diagnostics['leave_city_full_ridge_top5_capture']:.4f}; MLP minus ridge = {diagnostics['leave_city_full_mlp_minus_ridge_top5']:+.4f}.",
        f"- leave-city factorized MLP top-5% capture = {diagnostics['leave_city_factorized_mlp_top5_capture']:.4f}; factorized ridge = {diagnostics['leave_city_factorized_ridge_top5_capture']:.4f}; MLP minus ridge = {diagnostics['leave_city_factorized_mlp_minus_ridge_top5']:+.4f}.",
        f"- random-event full MLP top-5% capture = {diagnostics['random_event_full_mlp_top5_capture']:.4f}; random-event full ridge = {diagnostics['random_event_full_ridge_top5_capture']:.4f}.",
        f"- adding city identity under random-event split changes top-5% capture by {diagnostics['random_event_city_id_minus_no_id_top5']:+.4f}.",
        f"- random-event minus leave-city full MLP top-5% gap = {diagnostics['random_event_minus_leave_city_full_mlp_top5']:+.4f}.",
        f"- under random token split, adding event identity changes token Spearman by {diagnostics['token_random_event_id_minus_no_id_spearman']:+.4f}.",
        "",
        "## Interpretation",
        "",
        "如果 MLP 在 leave-city 上没有显著超过 ridge 或低维 factorized law，说明当前 recoverability law 的主要结构已经被低维 activated variables 捕捉；神经网络可以作为 sanity check，但不是必要的科学产品。如果 random-event 或 random-token 设置明显更好，或者 identity 特征带来增益，就说明随机切分确实可能混入 city/event memorization，论文中应继续坚持 leave-city、regime holdout 和 within-city chronological holdout 作为主验证。",
        "",
        "## Summary Table",
        "",
        table_to_markdown(summary),
        "",
        "## Fold Metrics",
        "",
        table_to_markdown(metrics),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_masks(tokens: pd.DataFrame, split: Any) -> tuple[pd.Series, pd.Series]:
    if split.split_role == "token_random":
        rng = np.random.default_rng(int(split.token_random_seed))
        test_mask_values = rng.random(len(tokens)) < float(split.token_test_share)
        test_mask = pd.Series(test_mask_values, index=tokens.index)
        train_mask = ~test_mask
        return train_mask, test_mask
    train_keys = set(str(split.train_event_keys).split(";"))
    test_keys = set(str(split.test_event_keys).split(";"))
    train_mask = tokens["event_key"].isin(train_keys)
    test_mask = tokens["event_key"].isin(test_keys)
    return train_mask, test_mask


def feature_cache_key(spec: dict[str, Any]) -> str:
    return f"{spec['feature_set']}|city={spec.get('include_city_id')}|event={spec.get('include_event_id')}"


def event_key(frame: pd.DataFrame) -> pd.Series:
    return frame["city"].astype(str) + "||" + pd.to_numeric(frame["event_id"], errors="coerce").fillna(-1).astype(int).astype(str)


def split_seed(model_id: str, split_id: str) -> int:
    digest = hashlib.sha256(f"{model_id}|{split_id}|{SEED}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def one_row(df: pd.DataFrame, **filters: Any) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if column not in df:
            return pd.Series(dtype=float)
        mask &= df[column].astype(str).eq(str(value))
    if not mask.any():
        return pd.Series(dtype=float)
    return df.loc[mask].iloc[0]


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except Exception:
        return float("nan")


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) <= EPS or np.std(right) <= EPS:
        return float("nan")
    return safe_float(np.corrcoef(left, right)[0, 1])


def safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(den) > EPS else float("nan")


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def table_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_empty_"
    compact = df.head(max_rows).copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


if __name__ == "__main__":
    main()
