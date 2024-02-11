import logging
import sys
from argparse import ArgumentParser

import lightning.pytorch as pl
import optuna
import torch

from .objectives import deepar_objective, nbeats_objective, tft_objective


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
    argparser.add_argument(
        "--n-trials",
        default=30,
        type=int,
        required=False,
    )
    argparser.add_argument(
        "--timeout",
        default=1 * 60 * 60,
        type=int,
        required=False,
    )
    argparser.add_argument(
        "--seed",
        type=int,
        required=False,
    )
    argparser.add_argument(
        "--input-dir",
        default="datasets",
        type=str,
    )
    argparser.add_argument(
        "--output-dir",
        default="output",
        type=str,
    )
    args = argparser.parse_args()

    objective = {
        ("electricity", "nbeats"): nbeats_objective(
            "electricity", args.input_dir, args.output_dir
        ),
        ("electricity", "deepvar"): deepar_objective(
            "traffic", "multinormal", args.input_dir, args.output_dir
        ),
        ("electricity", "tft"): tft_objective(
            "electricity", args.input_dir, args.output_dir
        ),
        ("traffic", "nbeats"): nbeats_objective(
            "traffic", args.input_dir, args.output_dir
        ),
        ("traffic", "deepvar"): deepar_objective(
            "traffic", "multinormal", args.input_dir, args.output_dir
        ),
        ("traffic", "deepar"): deepar_objective(
            "traffic", "beta", args.input_dir, args.output_dir
        ),
        ("traffic", "tft"): tft_objective("traffic", args.input_dir, args.output_dir),
    }.get((args.dataset, args.model), None)

    if objective is None:
        raise NotImplementedError(
            f"Model {args.model} for dataset {args.dataset} not implemented"
        )

    if args.seed is not None:
        pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    study = optuna.create_study(
        study_name=f"{args.dataset}-{args.model}",
        storage=args.storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        gc_after_trial=True,
    )


if __name__ == "__main__":
    main()
