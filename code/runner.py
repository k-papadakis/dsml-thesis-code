from argparse import ArgumentParser

import lightning.pytorch as pl
import torch

from thesis.configs import (
    default_electricity_deepvar,
    default_electricity_nbeats,
    default_electricity_tft,
    default_traffic_deepar,
    default_traffic_deepvar,
    default_traffic_nbeats,
    default_traffic_tft,
)


def main():
    argparser = ArgumentParser()
    argparser.add_argument(
        "--dataset", choices=["electricity", "traffic"], required=True
    )
    argparser.add_argument(
        "--model", choices=["tft", "nbeats", "deepvar", "deepar"], required=True
    )
    args = argparser.parse_args()

    d = {
        ("electricity", "nbeats"): default_electricity_nbeats,
        ("electricity", "deepvar"): default_electricity_deepvar,
        ("electricity", "tft"): default_electricity_tft,
        ("traffic", "nbeats"): default_traffic_nbeats,
        ("traffic", "deepvar"): default_traffic_deepvar,
        ("traffic", "deepar"): default_traffic_deepar,
        ("traffic", "tft"): default_traffic_tft,
    }

    setting_getter = d.get((args.dataset, args.model), None)

    if setting_getter is None:
        raise NotImplementedError(
            f"Model {args.model} for dataset {args.dataset} not implemented"
        )

    setting = setting_getter()
    setting.run()


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    main()
