from argparse import ArgumentParser


def parse_args():
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
        "--seed",
        type=int,
        required=False,
    )
    argparser.add_argument(
        "--input-dir",
        type=str,
        default="datasets",
    )
    argparser.add_argument(
        "--output-dir",
        type=str,
        default="output",
    )
    argparser.add_argument(
        "--find-lr",
        action="store_true",
    )
    args = argparser.parse_args()
    return args


def main(args):
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
