import json
from pathlib import Path

from with_argparse import with_argparse


@with_argparse(use_glob={"configs"})
def generate_configs(configs: list[Path], template: Path, data_dir: Path):
    with template.open() as f:
        template = f.read()
    inner_template = template[len("training {") : template.index("\n}\n")]
    outer_template_pre = "training {"
    outer_template_post = template[template.index("\n}\n") :]

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
        train_path = config_data_dir / base_config["dataset"]["splits"]["train"]
        eval_path = config_data_dir / base_config["dataset"]["splits"]["eval"]
        types_path = config_data_dir / base_config["dataset"]["splits"]["types"]

        reformatted_template = inner_template.format(
            data_dir=config_data_dir.absolute(),
            test_path=test_paths[0].absolute(),
            seen_path=test_paths[1].absolute(),
            unseen_path=test_paths[2].absolute(),
            train_path=train_path.absolute(),
            dev_path=eval_path.absolute(),
            types_path=types_path.absolute(),
        )
        reformatted_template = (
            outer_template_pre + reformatted_template + outer_template_post
        )
        asp_config = config.with_name(config.name.replace("iter", "asp")).with_suffix(".conf")
        with asp_config.open("w") as f:
            f.write(reformatted_template)

generate_configs()
