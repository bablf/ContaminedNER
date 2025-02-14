import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from with_argparse import with_argparse


def prefix_dict(inp: dict, prefix: str):
    return OrderedDict({prefix + ke: va for ke, va in inp.items()})


@with_argparse(use_glob={"file"})
def parse_diffusionner_csv_as_metrics_json(files: list[Path]):
    for file in files:
        df = pd.read_csv(file, header=0, sep=";")

        metrics_dict = df.iloc[0].to_dict()
        if "unseen" in file.name:
            twin_df = pd.read_csv(
                file.with_name(file.name.replace("unseen", "seen")), header=0, sep=";"
            )
            metrics_dict = prefix_dict(metrics_dict, "unseen_")
            metrics_dict.update(prefix_dict(twin_df.iloc[0].to_dict(), "seen_"))

            test_df = pd.read_csv(file.with_stem("eval_test"), header=0, sep=";")
            metrics_dict.update(test_df.iloc[0].to_dict())

        with open(file.with_name("metrics.json"), "w") as f:
            json.dump({"test_metrics": metrics_dict}, f)
    pass


parse_diffusionner_csv_as_metrics_json()
