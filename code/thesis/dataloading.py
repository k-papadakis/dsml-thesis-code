from os import PathLike

import pandas as pd

pd.set_option("mode.copy_on_write", True)
pd.set_option("future.infer_string", True)
pd.set_option("future.no_silent_downcasting", True)


def load_eld(
    file_path: str | PathLike[str],
    freq=pd.Timedelta(2, "hour"),
    start_time=pd.Timestamp(2014, 1, 2),
    end_time=pd.Timestamp(2014, 9, 1),
    components=[f"MT_{i:03}" for i in range(1, 51)],
) -> tuple[pd.DataFrame, pd.Timedelta]:
    # TODO: Start from 2014-01-01
    # TODO: Resample to 1 hour
    # TODO: Predict 1 day using 1 week
    df = pd.read_csv(
        file_path,
        sep=";",
        index_col=0,
        parse_dates=True,
        decimal=",",
        engine="pyarrow",
    )
    df = df.loc[start_time:end_time, components]
    df = df.resample(freq).mean()

    df.index.name = "date"

    assert df.notna().all(axis=None), "There are missing values in the dataset."
    assert df.index.diff()[1:].nunique() == 1, "The dataset is not uniformly sampled."  # type: ignore

    return df, freq
