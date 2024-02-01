from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from thesis.dataloading import load_eld
from thesis.prophet_ import Series, gridsearch_cv_, save_model_results

pd.set_option("mode.copy_on_write", True)
pd.set_option("future.infer_string", True)
pd.set_option("future.no_silent_downcasting", True)


ROOT_DIR = Path("output", "eld", "prophet")
PARAM_GRID = {
    "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
    "seasonality_mode": ["multiplicative"],
    "weekly_seasonality": [True],
}


def fit_one(name: str, series: Series) -> None:
    path = ROOT_DIR / name
    path.mkdir(parents=True, exist_ok=False)

    cv_result = gridsearch_cv_(series, PARAM_GRID, initial_horizons=30).best()
    cv_result.save(path)

    save_model_results(path, series, cv_result.params)

    print(f"FINISHED {name}")


def main():
    data_path = Path("./datasets/LD2011_2014.txt")
    eld, freq = load_eld(data_path)

    horizon = pd.Timedelta(7, "day")

    named_series_list = [
        (
            str(name),
            Series(
                s.reset_index().rename(columns={"date": "ds", name: "y"}),
                freq,
                horizon,
            ),
        )
        for name, s in eld.items()
    ]

    with Pool() as p:
        p.starmap(fit_one, named_series_list)


if __name__ == "__main__":
    main()
