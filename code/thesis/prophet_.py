import itertools
import os
import pickle
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric


@dataclass
class ProphetSeries:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    horizon: pd.Timedelta
    freq: pd.Timedelta

    def __post_init__(self):
        for df in self.train, self.test:
            assert df.columns.to_list() == ["ds", "y"]

    def cross_validate(self, params: dict[str, Any]) -> pd.DataFrame:
        m = Prophet(**params).fit(self.train)
        cv = cross_validation(
            m,
            initial=30 * self.horizon,
            horizon=self.horizon,
            parallel="processes",
        )

        return cv

    def gridsearch_cv(
        self, param_grid: dict[str, list[Any]], metric="rmse"
    ) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
        all_params = [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
        ]

        cvs = list(map(self.cross_validate, all_params))
        perfs: list[pd.DataFrame] = [performance_metrics(cv, rolling_window=1.0) for cv in cvs]  # type: ignore

        perf, params, cv = min(
            zip(perfs, all_params, cvs), key=lambda t: t[0][metric].item()
        )

        return perf, params, cv

    def fit(self, params: dict[str, Any]) -> tuple[Prophet, pd.DataFrame]:
        m = Prophet(**params)
        m.fit(self.train)

        future = m.make_future_dataframe(periods=len(self.test), freq=self.freq)  # type: ignore
        forecast = m.predict(future)

        return m, forecast

    def fit_best(
        self,
        param_grid: dict[str, list[Any]],
        metric="rmse",
    ) -> "BestFit":
        perf, params, cv = self.gridsearch_cv(param_grid, metric=metric)
        model, forecast = self.fit(params)

        return BestFit(self, model, forecast, params, cv, metric, perf)


@dataclass
class BestFit:
    series: ProphetSeries
    model: Prophet
    forecast: pd.DataFrame
    params: dict[str, Any]
    cv: pd.DataFrame
    metric: str
    perf: pd.DataFrame

    def save(self, root_dir: str | os.PathLike[str]):
        path = Path(root_dir) / self.series.name

        path.mkdir(parents=True, exist_ok=False)

        with open(path / "bestfit.pkl", "wb") as f:
            pickle.dump(self, f)

        # Model
        fig_model = self.model.plot(self.forecast, include_legend=True)
        _ = add_changepoints_to_plot(fig_model.gca(), self.model, self.forecast)
        fig_model.suptitle(self.series.name)
        fig_model.savefig(path / "model.png")
        fig_model.savefig(path / "model.pdf")
        plt.close(fig_model)

        # CV
        fig_cv = plot_cross_validation_metric(self.cv, metric="mape")
        fig_cv.gca().set_title(f"{self.series.name} CV")
        fig_cv.savefig(path / "cv.png")
        fig_cv.savefig(path / "cv.pdf")
        plt.close(fig_cv)

        # Components
        fig_components = self.model.plot_components(self.forecast)
        fig_components.suptitle(self.series.name)
        fig_components.savefig(path / "components.png")
        fig_components.savefig(path / "components.pdf")
        plt.close(fig_components)

        # Forecasts
        fig_forecasts = plt.figure(figsize=(10, 6))
        self.series.train.set_index("ds")["y"][-3 * len(self.series.test) :].rename(
            "training"
        ).plot(legend=True)
        self.series.test.set_index("ds")["y"].rename("actual").plot(legend=True)
        self.forecast.set_index("ds")["yhat"][-len(self.series.test) :].rename(
            "predicted"
        ).plot(legend=True)
        fig_forecasts.gca().set_title(self.series.name)
        fig_forecasts.tight_layout()
        fig_forecasts.savefig(path / "forecasts.png")
        fig_forecasts.savefig(path / "forecasts.pdf")
        plt.close(fig_forecasts)


def make_series(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> Sequence[ProphetSeries]:
    freq = pd.Timedelta(pd.infer_freq(df_train.index))  # type: ignore
    horizon = len(df_test) * freq

    return [
        ProphetSeries(
            str(name),
            s_train.reset_index().rename(columns={"time_idx": "ds", name: "y"}),
            s_test.reset_index().rename(columns={"time_idx": "ds", name: "y"}),
            horizon,
            freq,
        )
        for ((name, s_train), (_, s_test)) in zip(df_train.items(), df_test.items())
    ]


def main():
    from thesis.dataloading import load_eld

    root_dir = Path("output", "eld", "prophet")
    data_path = Path(sys.argv[1])
    param_grid = {
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_mode": ["multiplicative"],
        "weekly_seasonality": [True],
    }
    df_train, df_test = load_eld(data_path)

    perfs: dict[str, pd.Series] = {}
    for series in make_series(df_train, df_test):
        best_fit = series.fit_best(param_grid)
        perfs[series.name] = best_fit.perf.iloc[0]
        best_fit.save(root_dir)

    perfs_df = pd.DataFrame.from_dict(perfs, orient="index")
    perfs_df.to_csv(root_dir / "scores.csv")
    perfs_df.mean(axis=0).to_csv(root_dir / "mean_scores.csv")


if __name__ == "__main__":
    main()

    # root_dir = Path("output", "eld", "prophet")
    # for p in root_dir.glob("**/bestfit.pkl"):
    #     with p.open("rb") as f:
    #         bestfit = pickle.load(f)
