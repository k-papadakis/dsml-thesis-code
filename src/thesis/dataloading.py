from io import BytesIO
from os import PathLike
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

pd.set_option("mode.copy_on_write", True)
pd.set_option("future.infer_string", True)
pd.set_option("future.no_silent_downcasting", True)


ELECTRICITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
TRAFFIC_URL = "https://archive.ics.uci.edu/static/public/204/pems+sf.zip"


def download_and_extract_zip(zipurl: str, path: str | PathLike[str]) -> None:
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(path)


def load_electricity(
    path: str | PathLike[str],
    freq=pd.Timedelta(1, "hour"),
    start_time=pd.Timestamp(2014, 2, 1),
    end_time=pd.Timestamp(2014, 9, 1),
    components=slice(50),
) -> pd.DataFrame:

    path = Path(path, "LD2011_2014.txt")

    df = pd.read_csv(
        path,
        sep=";",
        index_col=0,
        parse_dates=True,
        decimal=",",
    )
    df = df.loc[start_time:end_time].iloc[:, components]
    df = df.resample(freq).mean()

    df.index.name = "date"

    assert df.notna().all(axis=None), "There are missing values in the dataset."
    assert df.index.diff()[1:].nunique() == 1, "The dataset is not uniformly sampled."  # type: ignore

    return df


def load_traffic(
    path: str | PathLike[str],
    freq=pd.Timedelta(1, "hour"),
    start_time=pd.Timestamp(2008, 1, 1),
    end_time=pd.Timestamp(2008, 6, 25),
    components=slice(50),
) -> pd.DataFrame:

    def load_perms(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
        with open(path) as f:
            permutation = np.fromstring(next(f).strip("[]\n"), dtype=int, sep=" ") - 1
        inv_permutation = np.empty_like(permutation)
        inv_permutation[permutation] = np.arange(permutation.size)
        return permutation, inv_permutation

    def load_tensor(path: str | PathLike) -> np.ndarray:
        with open(path) as f:
            mats = []
            for line in f:
                rows = line.strip("[]\n").split(";")
                mat = np.loadtxt(rows, delimiter=" ", dtype=float)
                mats.append(mat)

        return np.array(mats).swapaxes(1, 2)  # (day, time, sensor)

    def load_labels(path: str | PathLike) -> np.ndarray:
        with open(path) as f:
            labels = np.fromstring(next(f).strip("[]\n"), dtype=int, sep=" ") - 1
        return labels

    def load_stations(path: str | PathLike) -> np.ndarray:
        with open(path) as f:
            stations = np.fromstring(next(f).strip("[]\n"), dtype=int, sep=" ")
        return stations

    path = Path(path)

    _, inv_permutation = load_perms(path / "randperm")

    data = np.concatenate(
        [
            load_tensor(path / "PEMS_train"),
            load_tensor(path / "PEMS_test"),
        ],
        axis=0,
    )[inv_permutation]

    labels = np.concatenate(
        [
            load_labels(path / "PEMS_trainlabels"),
            load_labels(path / "PEMS_testlabels"),
        ],
        axis=0,
    )[inv_permutation]

    stations = load_stations(path / "stations_list")

    diffs = np.diff(labels, prepend=labels[0], append=labels[-1] + 1) % 7
    date_index = pd.Timestamp("2008-01-01") + pd.to_timedelta(np.cumsum(diffs), "D")
    datetime_index = (
        date_index.to_series()
        .asfreq("10min")
        .index.to_series()
        .dt.date.reset_index()
        .set_index(0)
        .loc[date_index]
        .reset_index(drop=True)
        .set_index("index")[:-1]
        .index.rename("date")
    )

    df = pd.DataFrame(
        data.reshape((-1, data.shape[-1])),
        index=datetime_index,
        columns=stations.astype(str),
    )
    df = df.loc[start_time:end_time].iloc[:, components]
    df = df.resample(freq).mean()

    return df
