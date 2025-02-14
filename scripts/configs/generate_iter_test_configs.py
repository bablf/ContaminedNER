import copy
import json
from pathlib import Path

from with_argparse import with_argparse


@with_argparse
def generate_iter_configs(
    base: Path,
    output_dir: Path | None = None,
):
    output_dir = output_dir or base.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with base.open() as f:
        base_config = json.load(f)

    dataset_name = base_config["dataset"]["name"]
    dataset_config = copy.deepcopy(base_config)
    split_files = {
        "test": {
            "normal": f"{dataset_name}_test.json",
            "seen": f"{dataset_name}_seen_test.json",
            "unseen": f"{dataset_name}_unseen_test.json",
        }
    }
    for key, value in split_files.items():
        dataset_config["dataset"]["splits"][key] = value

    dataset_file = output_dir / f"{dataset_name}_clean_contaminated_iter.json"
    with dataset_file.open("w") as f:
        json.dump(dataset_config, f, indent=2)


generate_iter_configs()
