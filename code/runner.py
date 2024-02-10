from argparse import ArgumentParser

import lightning.pytorch as pl
import torch

from thesis.defaults import (
    electricity_deepvar,
    electricity_nbeats,
    electricity_tft,
    traffic_deepar,
    traffic_deepvar,
    traffic_nbeats,
    traffic_tft,
)


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
    args = argparser.parse_args()

    setting_creator = {
        ("electricity", "nbeats"): electricity_nbeats,
        ("electricity", "deepvar"): electricity_deepvar,
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

    setting = setting_creator()
    setting.run()


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    main()
