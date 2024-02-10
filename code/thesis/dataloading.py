from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Literal
from urllib.request import urlopen
from zipfile import ZipFile

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from pytorch_forecasting import EncoderNormalizer, TimeSeriesDataSet
from torch.utils.data import DataLoader

pd.set_option("mode.copy_on_write", True)
pd.set_option("future.infer_string", True)
pd.set_option("future.no_silent_downcasting", True)


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
) -> tuple[pd.DataFrame, pd.Timedelta]:
    """https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"""

    path = Path(path, "LD2011_2014.txt")

    df = pd.read_csv(
        path,
        sep=";",
        index_col=0,
        parse_dates=True,
        decimal=",",
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    df = df.loc[start_time:end_time].iloc[:, components]
    df = df.resample(freq).mean()

    df.index.name = "date"

    assert df.notna().all(axis=None), "There are missing values in the dataset."
    assert df.index.diff()[1:].nunique() == 1, "The dataset is not uniformly sampled."  # type: ignore

    return df, freq


def load_traffic(
    path: str | PathLike[str],
    freq=pd.Timedelta(1, "hour"),
    start_time=pd.Timestamp(2008, 1, 1),
    end_time=pd.Timestamp(2008, 6, 25),
    components=slice(50),
) -> tuple[pd.DataFrame, pd.Timedelta]:
    """https://archive.ics.uci.edu/dataset/204/pems-sf"""

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
                line = line.strip("[]\n").split(";")
                mat = np.loadtxt(line, delimiter=" ", dtype=float)
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
        columns=stations,
    )
    df = df.loc[start_time:end_time].iloc[:, components]
    df = df.resample(freq).mean()

    return df, freq


class SeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: Literal["electricity", "traffic"],
        path: str | PathLike,
        multivariate: bool,
        with_covariates: bool,
        transform: Literal["auto", "logit"],
        batch_size: int,
    ):
        super().__init__()

        self.name = name
        self.path = Path(path)
        self.multivariate = multivariate
        self.with_covariates = with_covariates
        self.transform = transform
        self.batch_size = batch_size

    def prepare_data(self):
        if self.path.is_dir():
            return

        if self.name == "electricity":
            url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
        elif self.name == "traffic":
            url = "https://archive.ics.uci.edu/static/public/204/pems+sf.zip"
        else:
            raise ValueError(f"Unknown dataset name: {self.name}")

        print(f"Dataset {self.name} not found in {self.path}.")
        print(f"Downloading dataset {self.name} from {url} to {self.path}.")
        download_and_extract_zip(url, self.path)
        print(f"Downloaded dataset {self.name} to {self.path}.")

    def setup(self, stage: str):
        if self.name == "electricity":
            loader = load_electricity
        elif self.name == "traffic":
            loader = load_traffic
        else:
            raise ValueError(f"Unknown dataset name: {self.name}")

        data, freq = loader(self.path)
        data = (
            data.dropna()
            .reset_index()
            .reset_index()
            .rename(columns={"index": "time_idx"})
            .set_index(["time_idx", "date"])
            .rename_axis("series", axis="columns")
            .stack()
            .rename("value")  # type: ignore
            .reset_index()
        )
        data["weekday"] = data["date"].dt.weekday.astype("string").astype("category")
        data["hour"] = data["date"].dt.hour.astype("string").astype("category")
        data["series"] = data["series"].astype("string").astype("category")

        horizon = pd.Timedelta(1, "day")

        output_length = horizon // freq
        input_length = 7 * output_length
        validation_cutoff = data["time_idx"].max() - output_length
        training_cutoff = validation_cutoff - 21 * output_length

        if self.transform == "logit":
            target_normalizer = EncoderNormalizer(
                method="identity", transformation="logit"
            )
        elif self.transform == "auto":
            target_normalizer = "auto"
        else:
            raise ValueError(f"Unknown transform method: {self.transform}")

        self.train = TimeSeriesDataSet(
            data[data["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target="value",
            group_ids=["series"],
            time_varying_unknown_reals=["value"],
            max_encoder_length=input_length,
            max_prediction_length=output_length,
            time_varying_known_categoricals=(
                ["hour", "weekday"] if self.with_covariates else []
            ),
            static_categoricals=["series"] if self.with_covariates else [],
            target_normalizer=target_normalizer,
        )
        self.val = TimeSeriesDataSet.from_dataset(
            self.train,
            data[data["time_idx"] <= validation_cutoff],
            min_prediction_idx=training_cutoff + 1,
        )
        self.test = TimeSeriesDataSet.from_dataset(
            self.train,
            data,
            # min_prediction_idx=validation_cutoff + 1,
            predict=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.train.to_dataloader(
            train=True,
            batch_sampler="synchronized" if self.multivariate else None,  # type: ignore
            batch_size=self.batch_size,
            num_workers=2,
        )

    def val_dataloader(self) -> DataLoader:
        return self.val.to_dataloader(
            train=False,
            batch_sampler="synchronized" if self.multivariate else None,  # type: ignore
            batch_size=self.batch_size,
            num_workers=2,
        )

    def test_dataloader(self) -> DataLoader:
        return self.test.to_dataloader(
            train=False,
            batch_sampler="synchronized" if self.multivariate else None,  # type: ignore
            batch_size=self.batch_size,
            num_workers=0,
        )
