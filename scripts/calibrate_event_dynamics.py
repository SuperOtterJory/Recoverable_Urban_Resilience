"""Estimate event-level recovery dynamics from matched abnormal speed deficits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from recoverable_resilience.paths import find_repo_root


RAIN_LAG_HOURS = 6
TRAIN_SHARE = 0.70


def main() -> None:
    root = find_repo_root()
    input_path = root / "results" / "data_mining" / "tables" / "speed_hourly_abnormal_deficit.csv"
    output_dir = root / "results" / "event_calibration"
    table_dir = output_dir / "tables"
    report_dir = output_dir / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_csv(input_path, parse_dates=["hour"])
    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for city, city_hourly in hourly.groupby("city", sort=True):
        summary, predictions = fit_city_dynamics(city, city_hourly)
        summary_rows.append(summary)
        prediction_frames.append(predictions)

    summary = pd.DataFrame(summary_rows).sort_values("city")
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary_path = table_dir / "event_dynamic_calibration_summary.csv"
    prediction_path = table_dir / "event_dynamic_calibration_predictions.csv"
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    write_report(report_dir / "event_dynamic_calibration_report_zh.md", summary)
    print(f"Wrote {summary_path}")
    print(f"Wrote {prediction_path}")


def fit_city_dynamics(city: str, city_hourly: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    city_hourly = city_hourly.sort_values("hour").copy()
    city_hourly["target"] = pd.to_numeric(city_hourly["positive_abnormal_deficit"], errors="coerce").fillna(0.0)
    city_hourly["precipitation"] = pd.to_numeric(city_hourly["precipitation"], errors="coerce").fillna(0.0)
    city_hourly["target_next"] = city_hourly["target"].shift(-1)
    for lag in range(RAIN_LAG_HOURS + 1):
        city_hourly[f"rain_lag_{lag}"] = city_hourly["precipitation"].shift(lag).fillna(0.0)

    feature_cols = ["target", *[f"rain_lag_{lag}" for lag in range(RAIN_LAG_HOURS + 1)]]
    model_df = city_hourly.dropna(subset=["target_next"]).copy()
    model_df = model_df[np.isfinite(model_df[["target_next", *feature_cols]].to_numpy(dtype=float)).all(axis=1)]
    if len(model_df) < 48 or model_df["target"].max() <= 0:
        summary = fallback_summary(city, len(model_df), "insufficient_positive_hourly_signal")
        model_df["predicted_next"] = np.nan
        model_df["split"] = "unused"
        return summary, prediction_output(city, model_df)

    x = model_df[feature_cols].to_numpy(dtype=float)
    y = model_df["target_next"].to_numpy(dtype=float)
    rain_lag_sum = model_df[[f"rain_lag_{lag}" for lag in range(RAIN_LAG_HOURS + 1)]].sum(axis=1).to_numpy(dtype=float)
    weights = 1.0 + 4.0 * (rain_lag_sum > 0) + 2.0 * (model_df["target"].to_numpy(dtype=float) > 0) + 2.0 * (y > 0)
    weights = np.sqrt(weights)
    split_idx = int(np.clip(round(len(model_df) * TRAIN_SHARE), 24, len(model_df) - 12))

    lower = np.array([0.50, *([0.0] * (RAIN_LAG_HOURS + 1))], dtype=float)
    upper = np.array([0.995, *([1.0] * (RAIN_LAG_HOURS + 1))], dtype=float)
    result = lsq_linear(
        x[:split_idx] * weights[:split_idx, None],
        y[:split_idx] * weights[:split_idx],
        bounds=(lower, upper),
        lsmr_tol="auto",
        max_iter=2000,
    )
    coef = result.x
    predicted = np.clip(x @ coef, 0.0, 1.0)
    model_df["predicted_next"] = predicted
    model_df["split"] = np.where(np.arange(len(model_df)) < split_idx, "train", "test")

    train_metrics = metrics(y[:split_idx], predicted[:split_idx])
    test_metrics = metrics(y[split_idx:], predicted[split_idx:])
    rain_betas = {f"rain_beta_lag_{lag}h": float(coef[lag + 1]) for lag in range(RAIN_LAG_HOURS + 1)}
    a_retention = float(coef[0])
    half_life = np.log(0.5) / np.log(a_retention) if 0 < a_retention < 1 else np.nan
    summary = {
        "city": city,
        "fit_status": "estimated",
        "n_hourly_rows": int(len(model_df)),
        "n_train_rows": int(split_idx),
        "n_test_rows": int(len(model_df) - split_idx),
        "rain_lag_hours": RAIN_LAG_HOURS,
        "a_retention": a_retention,
        "natural_half_life_hours": float(half_life),
        "rain_kernel_sum": float(np.sum(coef[1:])),
        "rain_kernel_peak_lag_h": int(np.argmax(coef[1:])),
        "rain_exposure_row_share": float(np.mean(rain_lag_sum > 0)),
        "positive_target_row_share": float(np.mean(y > 0)),
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_corr": test_metrics["corr"],
        "solver_cost": float(result.cost),
        "solver_optimality": float(result.optimality),
        **rain_betas,
    }
    return summary, prediction_output(city, model_df)


def fallback_summary(city: str, n_rows: int, status: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "city": city,
        "fit_status": status,
        "n_hourly_rows": int(n_rows),
        "n_train_rows": 0,
        "n_test_rows": 0,
        "rain_lag_hours": RAIN_LAG_HOURS,
        "a_retention": 0.90,
        "natural_half_life_hours": float(np.log(0.5) / np.log(0.90)),
        "rain_kernel_sum": 0.0,
        "rain_kernel_peak_lag_h": 0,
        "rain_exposure_row_share": np.nan,
        "positive_target_row_share": np.nan,
        "train_rmse": np.nan,
        "train_mae": np.nan,
        "train_r2": np.nan,
        "test_rmse": np.nan,
        "test_mae": np.nan,
        "test_r2": np.nan,
        "test_corr": np.nan,
        "solver_cost": np.nan,
        "solver_optimality": np.nan,
    }
    for lag in range(RAIN_LAG_HOURS + 1):
        row[f"rain_beta_lag_{lag}h"] = 0.0
    return row


def prediction_output(city: str, model_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "hour",
        "target",
        "target_next",
        "precipitation",
        "predicted_next",
        "split",
        "rain_hour",
        "rain_influence_hour",
    ]
    available = [col for col in cols if col in model_df.columns]
    out = model_df[available].copy()
    out.insert(0, "city", city)
    return out


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "corr": np.nan}
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 1e-12 else np.nan
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 and np.std(y_pred) > 0 and np.std(y_true) > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "corr": corr}


def write_report(path: Path, summary: pd.DataFrame) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "| {city} | {status} | {a:.4f} | {half:.2f} | {kernel:.4f} | {lag} | {rmse:.4f} | {r2:.3f} |".format(
                city=row["city"],
                status=row["fit_status"],
                a=row["a_retention"],
                half=row["natural_half_life_hours"],
                kernel=row["rain_kernel_sum"],
                lag=int(row["rain_kernel_peak_lag_h"]),
                rmse=row["test_rmse"],
                r2=row["test_r2"],
            )
        )
    text = "\n".join(
        [
            "# Event-Level Dynamic Calibration",
            "",
            "本表用 matched temporal baseline 后的正异常速度损失来估计动态：",
            "",
            "`positive_abnormal_deficit[t+1] = a * positive_abnormal_deficit[t] + sum_l beta_l * precipitation[t-l]`",
            "",
            "`a` 表示没有新降雨冲击时异常损失保留到下一小时的比例；`beta_l` 是降雨滞后核，之后会被用于每个真实降雨事件的 LP disturbance `h[t]`。",
            "",
            "| city | status | a retention | natural half-life h | rain kernel sum | peak lag h | test RMSE | test R2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "解释：这个标定仍不是严格因果识别，但它避免了“前 6 小时 baseline”把早晚高峰周期误当成降雨影响的问题，并且用时间递推关系把自然恢复项和外部降雨冲击项分离开。",
        ]
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
