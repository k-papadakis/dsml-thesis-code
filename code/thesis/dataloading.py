from collections.abc import Hashable
from os import PathLike

import numpy as np
import pandas as pd

from thesis.constants import ELD_FREQ, ELD_TEST_LEN


def load_eld(
    file_path: str | PathLike,
    freq=ELD_FREQ,
    test_length=ELD_TEST_LEN,
    start_time="2014-01-02 00:00:00",
    end_time="2014-09-01 00:00:00",
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

    df_train = df.iloc[:-test_length]
    df_test = df.iloc[-test_length:]

    return df_train, df_test


def load_eld_prophet(
    file_path: str | PathLike,
) -> dict[Hashable, tuple[pd.DataFrame, pd.DataFrame]]:
    df_train, df_test = load_eld(file_path)

    return {
        name: (
            s_train.reset_index().rename(columns={"time_idx": "ds", name: "y"}),
            s_test.reset_index().rename(columns={"time_idx": "ds", name: "y"}),
        )
        for ((name, s_train), (_, s_test)) in zip(df_train.items(), df_test.items())
    }
