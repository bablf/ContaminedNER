import json
from pathlib import Path

from with_argparse import with_argparse


@with_argparse(use_glob={"configs"})
def generate_configs(configs: list[Path], template: Path, data_dir: Path):
    with template.open() as f:
        template = f.read()

    for config in configs:
        with config.open() as f:
            base_config = json.load(f)

        config_data_dir = data_dir / base_config["dataset"]["data_dir"]
        assert data_dir.exists(), data_dir.as_posix()
        test_paths = (
            config_data_dir / base_config["dataset"]["splits"]["test"]["normal"],
            config_data_dir / base_config["dataset"]["splits"]["test"]["seen"],
            config_data_dir / base_config["dataset"]["splits"]["test"]["unseen"],
        )
        test_path = ",".join((p.absolute().as_posix() for p in test_paths))
        train_path = config_data_dir / base_config["dataset"]["splits"]["train"]
        eval_path = config_data_dir / base_config["dataset"]["splits"]["eval"]
        types_path = config_data_dir / base_config["dataset"]["splits"]["types"]

        reformatted_template = template.format(
            test_path=test_path,
            train_path=train_path.absolute(),
            valid_path=eval_path.absolute(),
            types_path=types_path.absolute(),
        )
        with config.with_name(config.name.replace("iter", "diffusion")).with_suffix(".conf").open(
            "w"
        ) as f:
            f.write(reformatted_template)


generate_configs()
