from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from .dataloading import load_traffic
from .prophet_ import Series, gridsearch_cv_, save_model_results

ROOT_DIR = Path("output", "traffic", "prophet")

PARAM_GRID = {
    "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
    "seasonality_mode": ["additive"],
    "weekly_seasonality": [True],
}
# HORIZON = pd.Timedelta(7, "day")
# INITIAL_HORIZONS = 23
HORIZON = pd.Timedelta(1, "day")
INITIAL_HORIZONS = 174


def fit_one(name: str, series: Series) -> None:
    path = ROOT_DIR / name
    path.mkdir(parents=True, exist_ok=False)

    cv_result = gridsearch_cv_(
        series, PARAM_GRID, initial_horizons=INITIAL_HORIZONS
    ).best()
    cv_result.save(path)

    save_model_results(path, series, cv_result.params)

    print(f"FINISHED {name}")


def main():
    data_path = Path("./datasets/traffic/")
    traffic, freq = load_traffic(data_path)

    named_series_list = [
        (
            str(name),
            Series(
                s.reset_index()
                .rename(columns={"date": "ds", name: "y"})
                .assign(cap=1.0),
                freq,
                HORIZON,
            ),
        )
        for name, s in traffic.items()
    ]

    with Pool() as p:
        p.starmap(fit_one, named_series_list)

    results = pd.DataFrame.from_dict(
        {
            p.parent.stem: pd.read_csv(p).iloc[0]
            for p in ROOT_DIR.glob("**/performance.csv")
        },
        orient="index",
    ).sort_index()
    results.to_csv(ROOT_DIR / "results.csv")


if __name__ == "__main__":
    main()
