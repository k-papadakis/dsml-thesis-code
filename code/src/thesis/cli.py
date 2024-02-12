from argparse import ArgumentParser


def run(args):
    import lightning.pytorch as pl
    import torch

    if args.seed is not None:
        pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    from .defaults import (
        electricity_deepar,
        electricity_deepvar,
        electricity_nbeats,
        electricity_tft,
        traffic_deepar,
        traffic_deepvar,
        traffic_nbeats,
        traffic_tft,
    )

    setting_creator = {
        ("electricity", "nbeats"): electricity_nbeats,
        ("electricity", "deepvar"): electricity_deepvar,
        ("electricity", "deepar"): electricity_deepar,
        ("electricity", "tft"): electricity_tft,
        ("traffic", "nbeats"): traffic_nbeats,
        ("traffic", "deepvar"): traffic_deepvar,
        ("traffic", "deepar"): traffic_deepar,
        ("traffic", "tft"): traffic_tft,
    }.get((args.dataset, args.model), None)

    if setting_creator is None:
        raise NotImplementedError(
            f"Model {args.model} for dataset {args.dataset} not implemented"
        )

    setting = setting_creator(args.input_dir, args.output_dir)
    if args.find_lr:
        setting.find_lr()
    setting.run()


def find(args):
    import logging
    import sys

    import lightning.pytorch as pl
    import optuna
    import torch

    if args.seed is not None:
        pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    from .objectives import deepar_objective, nbeats_objective, tft_objective

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


def main():
    argparser = ArgumentParser()

    subparsers = argparser.add_subparsers()

    # Runner
    runner_parser = subparsers.add_parser("run")
    runner_parser.add_argument(
        "dataset",
        choices=["electricity", "traffic"],
    )
    runner_parser.add_argument(
        "model",
        choices=["tft", "nbeats", "deepvar", "deepar"],
    )
    runner_parser.add_argument(
        "--seed",
        type=int,
        required=False,
    )
    runner_parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets",
    )
    runner_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
    )
    runner_parser.add_argument(
        "--find-lr",
        action="store_true",
    )
    runner_parser.set_defaults(func=run)

    # Finder
    finder_parser = subparsers.add_parser("find")
    finder_parser.add_argument(
        "dataset",
        choices=["electricity", "traffic"],
    )
    finder_parser.add_argument(
        "model",
        choices=["tft", "nbeats", "deepvar", "deepar"],
    )
    finder_parser.add_argument(
        "--storage",
        default="sqlite:///experiments.db",
        type=str,
        required=False,
    )
    finder_parser.add_argument(
        "--n-trials",
        default=30,
        type=int,
        required=False,
    )
    finder_parser.add_argument(
        "--timeout",
        default=1 * 60 * 60,
        type=int,
        required=False,
    )
    finder_parser.add_argument(
        "--seed",
        type=int,
        required=False,
    )
    finder_parser.add_argument(
        "--input-dir",
        default="datasets",
        type=str,
    )
    finder_parser.add_argument(
        "--output-dir",
        default="output",
        type=str,
    )
    finder_parser.set_defaults(func=find)

    args = argparser.parse_args()
    args.func(args)
