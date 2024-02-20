from os import PathLike
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_median(df: pd.DataFrame, dataset: Literal["electricity", "traffic"]) -> Figure:
    if dataset == "electricity":
        title = "Median Electricity Consumption"
        ylabel = "Electricity Consumption (kW per 15 minutes)"
    elif dataset == "traffic":
        title = "Median Occupancy Rate"
        ylabel = "Occupancy Rate"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    fig = plt.figure(figsize=(15, 7))

    df.sum(axis=1).plot(linewidth=1)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    return fig


def plot_corr(df: pd.DataFrame, dataset: Literal["electricity", "traffic"]) -> Figure:
    if dataset == "electricity":
        title = "Correlation Heatmap of Electricity Consumption"
        axis_label = "Household ID"
        cbar_label = "Correlation Coefficient"
    elif dataset == "traffic":
        title = "Correlation Heatmap of Lane Occupancy Rates"
        axis_label = "Lane ID"
        cbar_label = "Correlation Coefficient"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    fig = plt.figure(figsize=(20, 15))

    sns.heatmap(
        df.corr(),
        center=0.0,
        annot=False,
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
    )

    plt.title(title)
    plt.xlabel(axis_label)
    plt.ylabel(axis_label)
    plt.tight_layout()

    return fig


def plot_heatmap(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"]
) -> Figure:

    if dataset == "electricity":
        title = "Heatmap of Electricity Consumption (One Week Period)"
        ylabel = "Household ID"
        cbar_label = "Log Electricity Consumption (kW per 15 minutes)"
        min_date, max_date = "2014-02-01", "2014-02-07"
    elif dataset == "traffic":
        title = "Heatmap of Lane Occupancy Rates (One Week Period)"
        ylabel = "Lane ID"
        cbar_label = "Log Lane Occupancy Rate"
        min_date, max_date = "2008-01-07", "2008-01-14"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    df = df.copy()
    df = df.loc[min_date:max_date]
    df.index = df.index.strftime("%Y-%m-%d %H:%M")  # type: ignore

    fig = plt.figure(figsize=(20, 10))

    sns.heatmap(
        np.log(df.T), cmap="viridis", robust=True, cbar_kws={"label": cbar_label}
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.tight_layout()

    return fig


def plot_hourly_boxplot(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"]
) -> Figure:
    if dataset == "electricity":
        title = "Hourly Electricity Consumption Boxplot"
        ylabel = "Electricity Consumption (kW per 15 minutes)"
    elif dataset == "traffic":
        title = "Hourly Lane Occupancy Rate Boxplot"
        ylabel = "Lane Occupancy Rate"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    fig = plt.figure(figsize=(20, 10))

    sns.boxplot(df.groupby(df.index.hour).mean().T)  # type: ignore

    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel("Hour of the Day")
    plt.ylabel(ylabel)
    plt.tight_layout()

    return fig


def plot_weekly_boxplot(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"]
) -> Figure:
    if dataset == "electricity":
        title = "Weekly Electricity Consumption Boxplot"
        ylabel = "Electricity Consumption (kW per 15 minutes)"
    elif dataset == "traffic":
        title = "Weekly Lane Occupancy Rate Boxplot"
        ylabel = "Lane Occupancy Rate"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    days_of_week = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    df = df.groupby(df.index.weekday).mean()  # type: ignore
    df.index = days_of_week  # type: ignore

    fig = plt.figure(figsize=(20, 10))

    sns.boxplot(df.T)

    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel("Day of the Week")
    plt.ylabel(ylabel)
    plt.tight_layout()

    return fig


def plot_hourly_median(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"]
) -> Figure:
    if dataset == "electricity":
        title = "Median Hourly Electricity Consumption"
        ylabel = "Electricity Consumption (kW per 15 minutes)"
    elif dataset == "traffic":
        title = "Median Hourly Lane Occupancy Rate"
        ylabel = "Lane Occupancy Rate"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    fig = plt.figure(figsize=(20, 10))

    df.groupby(df.index.hour).mean().median(axis=1).plot(kind="line", marker="o")  # type: ignore

    plt.title(title)
    plt.xlabel("Hour of the Day")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.tight_layout()

    return fig


def plot_weekly_median(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"]
) -> Figure:
    if dataset == "electricity":
        title = "Median Weekly Electricity Consumption"
        ylabel = "Electricity Consumption (kW per 15 minutes)"
    elif dataset == "traffic":
        title = "Median Weekly Lane Occupancy Rate"
        ylabel = "Lane Occupancy Rate"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    days_of_week = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    df = df.groupby(df.index.weekday).mean().median(axis=1)  # type: ignore
    df.index = days_of_week  # type: ignore

    fig = plt.figure(figsize=(20, 10))

    df.plot(kind="line", marker="o", color="green")

    plt.title(title)
    plt.xlabel("Day of the Week")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def save_plots(
    dataset: Literal["electricity", "traffic"],
    input_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
):
    from .dataloading import (
        ELECTRICITY_URL,
        TRAFFIC_URL,
        download_and_extract_zip,
        load_electricity,
        load_traffic,
    )

    input_dir = Path(input_dir, dataset)
    output_dir = Path(output_dir, "visualizations", dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "electricity":
        url = ELECTRICITY_URL
        loader = load_electricity
    elif dataset == "traffic":
        url = TRAFFIC_URL
        loader = load_traffic
    else:
        raise ValueError(f"Unknown dataset name: {dataset}")

    if not input_dir.is_dir():
        input_dir.mkdir(parents=True)
        print(f"Dataset {dataset} not found in {input_dir}")
        print(f"Downloading dataset {dataset} from {url} to {input_dir}")
        download_and_extract_zip(url, input_dir)
        print(f"Downloaded dataset {dataset} to {input_dir}")

    df = loader(input_dir)

    plot_median(df, dataset).savefig(
        output_dir / "median.png",
    )
    plot_hourly_boxplot(df, dataset).savefig(
        output_dir / "hourly_boxplot.png",
    )
    plot_hourly_median(df, dataset).savefig(
        output_dir / "hourly_median.png",
    )
    plot_hourly_boxplot(df, dataset).savefig(
        output_dir / "hourly_boxplot.png",
    )
    plot_weekly_boxplot(df, dataset).savefig(
        output_dir / "weekly_boxplot.png",
    )
    plot_weekly_median(df, dataset).savefig(
        output_dir / "weekly_median.png",
    )
    plot_corr(df, dataset).savefig(
        output_dir / "corr.png",
    )
    plot_heatmap(df, dataset).savefig(
        output_dir / "heatmap.png",
    )
