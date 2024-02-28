from os import PathLike
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_median(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"], add_title: bool
) -> Figure:
    if dataset == "electricity":
        title = "Median Electricity Consumption"
        ylabel = "Electricity Consumption (kW per 15 minutes)"
    elif dataset == "traffic":
        title = "Median Occupancy Rate"
        ylabel = "Occupancy Rate"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    fig = plt.figure(figsize=(10, 5))

    df.sum(axis=1).plot(linewidth=1)

    if add_title:
        plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    return fig


def plot_corr_heatmap(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"], add_title: bool
) -> Figure:
    if dataset == "electricity":
        title = "Correlation Heatmap of Electricity Consumption"
        axis_label = "Household ID"
    elif dataset == "traffic":
        title = "Correlation Heatmap of Lane Occupancy Rates"
        axis_label = "Lane ID"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    fig = plt.figure(figsize=(8, 6))

    sns.heatmap(
        df.corr(),
        center=0.0,
        annot=False,
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
    )

    if add_title:
        plt.title(title)
    plt.xlabel(axis_label)
    plt.ylabel(axis_label)
    plt.tight_layout()

    return fig


def plot_corr_hist(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"], add_title: bool
) -> Figure:
    if dataset == "electricity":
        title = "Correlation Histogram of Electricity Consumption"
    elif dataset == "traffic":
        title = "Correlation Histogram of Lane Occupancy Rates"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    fig = plt.figure(figsize=(6.4, 4.8))

    corr = df.corr()
    sns.histplot(corr.values[corr.values < 1.0], kde=True)

    if add_title:
        plt.title(title)
    plt.xlabel("Correlation Coefficient")
    plt.tight_layout()

    return fig


def plot_heatmap(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"], add_title: bool
) -> Figure:

    if dataset == "electricity":
        min_date, max_date = "2014-02-03", "2014-02-10"
        title = f"Heatmap of Electricity Consumption ({min_date} to {max_date})"
        ylabel = "Household ID"
        cbar_label = "Log Electricity Consumption (kW per 15 minutes)"
    elif dataset == "traffic":
        min_date, max_date = "2008-01-07", "2008-01-14"
        title = f"Heatmap of Lane Occupancy Rates ({min_date} to {max_date})"
        ylabel = "Lane ID"
        cbar_label = "Log Lane Occupancy Rate"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    df = df.copy()
    df = df.loc[min_date:max_date]
    df.index = df.index.strftime("%-H")  # type: ignore

    fig = plt.figure(figsize=(10, 5))

    sns.heatmap(
        np.log(df.T),
        cmap="viridis",
        robust=True,
        cbar_kws={"label": cbar_label},
        xticklabels=6,
    )

    plt.xticks(rotation=0)
    if add_title:
        plt.title(title)
    plt.xlabel("Hour of the Day")
    plt.ylabel(ylabel)
    plt.tight_layout()

    return fig


def plot_hourly_boxplot(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"], add_title: bool
) -> Figure:
    if dataset == "electricity":
        title = "Hourly Electricity Consumption Boxplot"
        ylabel = "Electricity Consumption (kW per 15 minutes)"
    elif dataset == "traffic":
        title = "Hourly Lane Occupancy Rate Boxplot"
        ylabel = "Lane Occupancy Rate"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    fig = plt.figure(figsize=(6.4, 4.8))

    sns.boxplot(df.groupby(df.index.hour).mean().T)  # type: ignore

    plt.xticks(rotation=45)
    if add_title:
        plt.title(title)
    plt.xlabel("Hour of the Day")
    plt.ylabel(ylabel)
    plt.tight_layout()

    return fig


def plot_weekly_boxplot(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"], add_title: bool
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

    fig = plt.figure(figsize=(6.4, 4.8))

    sns.boxplot(df.T)

    plt.xticks(rotation=45)
    if add_title:
        plt.title(title)
    plt.xlabel("Day of the Week")
    plt.ylabel(ylabel)
    plt.tight_layout()

    return fig


def plot_hourly_median(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"], add_title: bool
) -> Figure:
    if dataset == "electricity":
        title = "Median Hourly Electricity Consumption"
        ylabel = "Electricity Consumption (kW per 15 minutes)"
    elif dataset == "traffic":
        title = "Median Hourly Lane Occupancy Rate"
        ylabel = "Lane Occupancy Rate"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    fig = plt.figure(figsize=(6.4, 4.8))

    df.groupby(df.index.hour).mean().median(axis=1).plot(kind="line", marker="o")  # type: ignore

    if add_title:
        plt.title(title)
    plt.xlabel("Hour of the Day")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.tight_layout()

    return fig


def plot_weekly_median(
    df: pd.DataFrame, dataset: Literal["electricity", "traffic"], add_title: bool
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

    fig = plt.figure(figsize=(6.4, 4.8))

    df.plot(kind="line", marker="o", color="green")

    if add_title:
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
    add_titles: bool,
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

    plot_median(df, dataset, add_titles).savefig(
        output_dir / "median.png",
    )
    plot_hourly_boxplot(df, dataset, add_titles).savefig(
        output_dir / "hourly_boxplot.png",
    )
    plot_hourly_median(df, dataset, add_titles).savefig(
        output_dir / "hourly_median.png",
    )
    plot_hourly_boxplot(df, dataset, add_titles).savefig(
        output_dir / "hourly_boxplot.png",
    )
    plot_weekly_boxplot(df, dataset, add_titles).savefig(
        output_dir / "weekly_boxplot.png",
    )
    plot_weekly_median(df, dataset, add_titles).savefig(
        output_dir / "weekly_median.png",
    )
    plot_corr_heatmap(df, dataset, add_titles).savefig(
        output_dir / "corr_heatmap.png",
    )
    plot_corr_hist(df, dataset, add_titles).savefig(
        output_dir / "corr_hist.png",
    )
    plot_heatmap(df, dataset, add_titles).savefig(
        output_dir / "heatmap.png",
    )
