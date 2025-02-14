import copy
import json
from pathlib import Path

from with_argparse import with_argparse


@with_argparse
def generate_iter_configs(
    base: Path,
    n_splits: int,
    n_contaminated: int = 10,
    output_dir: Path | None = None,
    use_split_in_name: bool = False,
):
    contamination_step = 100 // n_contaminated
    output_dir = (output_dir or base.parent) / "dataset_contamination"
    output_dir.mkdir(parents=True, exist_ok=True)

    with base.open() as f:
        base_config = json.load(f)

    dataset_name = base_config["dataset"]["name"]
    for contamination_level in range(0, 100 + contamination_step, contamination_step):
        for split in range(n_splits):
            split_dataset_name = dataset_name
            if use_split_in_name:
                split_dataset_name = split_dataset_name.replace("%split%", str(split))
            split_config = copy.deepcopy(base_config)
            split_config["dataset"]["name"] = split_dataset_name
            split_files = {
                "train": f"{split_dataset_name}_conta{contamination_level}_split{split}_train.json",
                "test": {
                    "normal": f"{split_dataset_name}_test.json",
                    "seen": f"{split_dataset_name}_conta{contamination_level}_split{split}_seen_test.json",
                    "unseen": f"{split_dataset_name}_conta{contamination_level}_split{split}_unseen_test.json",
                },
            }
            for key, value in split_files.items():
                split_config["dataset"]["splits"][key] = value

            split_file = (
                output_dir
                / f"{split_dataset_name}_conta{contamination_level}_split{split}_iter.json"
            )
            with split_file.open("w") as f:
                json.dump(split_config, f)
    pass


generate_iter_configs()
