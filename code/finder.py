import logging
import sys
from argparse import ArgumentParser

import lightning.pytorch as pl
import optuna
import torch

from thesis.objectives import deepar_objective, nbeats_objective, tft_objective


def main():
    argparser = ArgumentParser()
    argparser.add_argument(
        "dataset",
        choices=["electricity", "traffic"],
    )
    argparser.add_argument(
        "model",
        choices=["tft", "nbeats", "deepvar", "deepar"],
    )
    argparser.add_argument(
        "--storage",
        default="sqlite:///experiments.db",
        type=str,
        required=False,
    )
    args = argparser.parse_args()

    objective = {
        ("electricity", "nbeats"): nbeats_objective("electricity"),
        ("electricity", "deepvar"): deepar_objective("traffic", "multinormal"),
        ("electricity", "tft"): tft_objective("electricity"),
        ("traffic", "nbeats"): nbeats_objective("traffic"),
        ("traffic", "deepvar"): deepar_objective("traffic", "multinormal"),
        ("traffic", "deepar"): deepar_objective("traffic", "beta"),
        ("traffic", "tft"): tft_objective("traffic"),
    }.get((args.dataset, args.model), None)

    if objective is None:
        raise NotImplementedError(
            f"Model {args.model} for dataset {args.dataset} not implemented"
        )

    study = optuna.create_study(
        study_name=f"{args.dataset}-{args.model}-study",
        storage=args.storage,
        direction="minimize",
    )

    study.optimize(
        nbeats_objective("electricity"),
        n_trials=30,
        timeout=2 * 60 * 60,
        show_progress_bar=True,
        gc_after_trial=True,
    )


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    main()
