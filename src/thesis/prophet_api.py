import itertools
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np

from thesis.dataloading import (
    ELECTRICITY_URL,
    TRAFFIC_URL,
    download_and_extract_zip,
    load_electricity,
    load_traffic,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric

from .metrics import compute_metrics

ParamDict: TypeAlias = dict[str, Any]
ParamGrid: TypeAlias = dict[str, list[Any]]


def flatten_grid(grid: ParamGrid) -> list[ParamDict]:
    return [dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())]


class Series:
    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        freq: pd.Timedelta,
        horizon: pd.Timedelta,
    ):
        assert set(data.columns) <= {"ds", "y", "cap", "floor"}
        self.name = name
        self.data = data
        self.freq = freq
        self.horizon = horizon

    @property
    def train_cutoff(self) -> pd.Timestamp:
        return self.data["ds"].max() - self.horizon

    @property
    def train(self) -> pd.DataFrame:
        return self.data[self.data["ds"] <= self.train_cutoff]

    @property
    def test(self) -> pd.DataFrame:
        return self.data[self.data["ds"] > self.train_cutoff]


def save_model_results(path: str | PathLike, series: Series, params: ParamDict) -> None:
    path = Path(path)

    model = Prophet(**params)
    model.fit(series.train)

    future = model.make_future_dataframe(
        periods=len(series.test), freq=series.freq  # type: ignore
    )
    if "floor" in series.test.columns:
        assert series.test["floor"].nunique() == 1
        future["floor"] = series.test["floor"].iloc[0]
    if "cap" in series.test.columns:
        assert series.test["cap"].nunique() == 1
        future["cap"] = series.test["cap"].iloc[0]

    forecast = model.predict(future)

    # Performance csv
    y_pred = forecast.iloc[-len(series.test) :]["yhat"].values
    y_true = series.test["y"].values
    performance = compute_metrics(y_true, y_pred)
    pd.Series(performance).to_frame().T.to_csv(path / "performance.csv", index=False)

    # Model image
    fig_model = model.plot(forecast, include_legend=True)
    _ = add_changepoints_to_plot(fig_model.gca(), model, forecast)

    fig_model.savefig(path / "model.png")
    # fig_model.savefig(path / "model.pdf")
    plt.close(fig_model)

    # Components image
    fig_components = model.plot_components(forecast)

    fig_components.savefig(path / "components.png")
    # fig_components.savefig(path / "components.pdf")
    plt.close(fig_components)

    # Forecasts image (Conforming with Pytorch Forecasting plots)
    fig_forecasts, ax = plt.subplots()

    y = series.data["y"].values[-8 * len(series.test) :]
    y_hat = forecast["yhat"].values[-len(series.test) :]
    n_pred = y_hat.shape[0]
    x_obs = np.arange(-(y.shape[0] - n_pred), 0)
    x_pred = np.arange(n_pred)
    prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
    obs_color = next(prop_cycle)["color"]
    pred_color = next(prop_cycle)["color"]

    ax.plot(x_obs, y[:-n_pred], label="observed", c=obs_color)
    ax.plot(x_pred, y[-n_pred:], label=None, c=obs_color)
    ax.plot(x_pred, y_hat, label="predicted", c=pred_color)
    ax.set_xlabel("Time index")
    fig_forecasts.legend()

    fig_forecasts.savefig(path / "forecasts.png")
    # fig_forecasts.savefig(path / "forecasts.pdf")
    plt.close(fig_forecasts)


@dataclass
class CvResult:
    params: ParamDict
    result: pd.DataFrame

    def performance(self) -> pd.DataFrame:
        return performance_metrics(self.result, rolling_window=1.0)  # type: ignore

    def save(self, path: str | PathLike) -> None:
        path = Path(path)

        # CV
        self.result.to_csv(path / "cv.csv")

        # Params
        with (path / "params.json").open("w") as f:
            json.dump(self.params, f, indent=2)

        # CV image
        fig_cv = plot_cross_validation_metric(self.result, metric="smape")
        fig_cv.gca().set_title(f"CV")
        fig_cv.savefig(path / "cv.png")
        # fig_cv.savefig(path / "cv.pdf")
        plt.close(fig_cv)


def cross_validate_(
    series: Series,
    params: ParamDict,
    initial_horizons: int,
) -> CvResult:
    m = Prophet(**params).fit(series.train)
    return CvResult(
        params,
        cross_validation(
            m,
            initial=initial_horizons * series.horizon,
            horizon=series.horizon,
            parallel="processes",
        ),
    )


@dataclass
class GridSearchCvResult:
    result: list[CvResult]

    def best(self, metric="rmse") -> CvResult:
        return min(self.result, key=lambda cv: cv.performance()[metric].item())


def gridsearch_cv_(
    series: Series,
    param_grid: ParamGrid,
    initial_horizons: int,
) -> GridSearchCvResult:
    return GridSearchCvResult(
        [
            cross_validate_(series, params, initial_horizons)
            for params in flatten_grid(param_grid)
        ]
    )


def run(
    dataset_name: Literal["electricity", "traffic"],
    input_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
):
    import logging

    import tqdm

    # Disable Stan spam
    logger = logging.getLogger("cmdstanpy")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.WARNING)

    input_dir = Path(input_dir, dataset_name)
    output_dir = Path(output_dir, dataset_name, "prophet")

    if dataset_name == "electricity":
        url = ELECTRICITY_URL
        loader = load_electricity
        assigner: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df
        param_grid = {
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_mode": ["multiplicative"],
        }
        initial_horizons = 205  # 210
    elif dataset_name == "traffic":
        url = TRAFFIC_URL
        loader = load_traffic
        assigner: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df.assign(
            floor=0.0, cap=1.0
        )
        param_grid = {
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_mode": ["additive"],
            "growth": ["logistic"],
        }
        initial_horizons = 169  # 174
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if not input_dir.is_dir():
        input_dir.mkdir(parents=True)
        print(f"Dataset {dataset_name} not found in {input_dir}")
        print(f"Downloading dataset {dataset_name} from {url} to {input_dir}")
        download_and_extract_zip(url, input_dir)
        print(f"Downloaded dataset {dataset_name} to {input_dir}")

    df = loader(input_dir)
    df = df.replace(0.0, np.nan)

    series_iter = (
        Series(
            str(name),
            s.reset_index().rename(columns={"date": "ds", name: "y"}).pipe(assigner),
            pd.Timedelta(df.index.freq),  # type: ignore
            pd.Timedelta(1, "day"),
        )
        for name, s in df.items()
    )
    series_iter_with_progress_bar = tqdm.tqdm(
        series_iter,
        total=len(df.columns),
        desc=f"Prophet {dataset_name}",
    )
    for series in series_iter_with_progress_bar:
        series_iter_with_progress_bar.set_postfix_str(f"series: {series.name}")

        path = output_dir / series.name
        path.mkdir(parents=True, exist_ok=False)

        cv_result = gridsearch_cv_(
            series, param_grid, initial_horizons=initial_horizons
        ).best()
        cv_result.save(path)

        save_model_results(path, series, cv_result.params)

    results = pd.DataFrame.from_dict(
        {
            p.parent.stem: pd.read_csv(p).iloc[0]
            for p in output_dir.glob("**/performance.csv")
        },
        orient="index",
    ).sort_index()
    results.to_csv(output_dir / "results.csv")
