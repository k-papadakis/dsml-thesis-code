from os import PathLike

import numpy as np
import pandas as pd


def load_eld(
    file_path: str | PathLike[str],
    freq=pd.offsets.Hour(2),
    horizon=pd.offsets.Week(1),
    start_time=pd.Timestamp(2014, 1, 2),
    end_time=pd.Timestamp(2014, 9, 1),
    components=[f"MT_{i:03}" for i in range(1, 51)],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Data points every 2 hours. Test length 1 week"""
    df = pd.read_csv(
        file_path, sep=";", index_col=0, parse_dates=True, decimal=","
    ).sort_index()
    df = df.loc[start_time:end_time, components]
    df = df.resample(freq).mean()
    df = df.astype(np.float32)

    df.index.name = "time_idx"

    df_train = df.loc[: end_time - horizon]
    df_test = df.loc[end_time - horizon + freq :]

    assert len(df_train) + len(df_test) == len(df)

    return (df_train, df_test)
