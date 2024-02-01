import itertools
import json
import warnings
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric

ParamDict: TypeAlias = dict[str, Any]
ParamGrid: TypeAlias = dict[str, list[Any]]


def flatten_grid(grid: ParamGrid) -> list[ParamDict]:
    return [dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())]


class Series:
    def __init__(
        self,
        data: pd.DataFrame,
        freq: pd.Timedelta,
        horizon: pd.Timedelta,
    ):
        assert set(data.columns) == {"ds", "y"}
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
    forecast = model.predict(future)
    forecast["y"] = series.data["y"]
    forecast["cutoff"] = series.train_cutoff

    # Performance csv
    performance: pd.DataFrame = performance_metrics(forecast, rolling_window=1.0)  # type: ignore
    performance.to_csv(path / "performance.csv")

    # Model image
    fig_model = model.plot(forecast, include_legend=True)
    _ = add_changepoints_to_plot(fig_model.gca(), model, forecast)

    fig_model.savefig(path / "model.png")
    fig_model.savefig(path / "model.pdf")
    plt.close(fig_model)

    # Components image
    fig_components = model.plot_components(forecast)

    fig_components.savefig(path / "components.png")
    fig_components.savefig(path / "components.pdf")
    plt.close(fig_components)

    # Forecasts image
    fig_forecasts = plt.figure(figsize=(10, 6))
    series.train.set_index("ds")["y"][-3 * len(series.test) :].rename("training").plot(
        legend=True
    )
    series.test.set_index("ds")["y"].rename("actual").plot(legend=True)
    forecast.set_index("ds")["yhat"][-len(series.test) :].rename("predicted").plot(
        legend=True
    )
    fig_forecasts.tight_layout()

    fig_forecasts.savefig(path / "forecasts.png")
    fig_forecasts.savefig(path / "forecasts.pdf")
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
        fig_cv = plot_cross_validation_metric(self.result, metric="mape")
        fig_cv.gca().set_title(f"CV")
        fig_cv.savefig(path / "cv.png")
        fig_cv.savefig(path / "cv.pdf")
        plt.close(fig_cv)


def cross_validate_(
    series: Series, params: ParamDict, initial_horizons: int
) -> CvResult:
    m = Prophet(**params).fit(series.train)
    return CvResult(
        params,
        cross_validation(
            m,
            initial=initial_horizons * series.horizon,
            horizon=series.horizon,
            # parallel="processes",
        ),
    )


@dataclass
class GridSearchCvResult:
    result: list[CvResult]

    def best(self, metric="rmse") -> CvResult:
        return min(self.result, key=lambda cv: cv.performance()[metric].item())


def gridsearch_cv_(
    series: Series, param_grid: ParamGrid, initial_horizons: int
) -> GridSearchCvResult:
    return GridSearchCvResult(
        [
            cross_validate_(series, params, initial_horizons)
            for params in flatten_grid(param_grid)
        ]
    )
