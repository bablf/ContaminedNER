import json
import pickle
from pathlib import Path

from paper_dataset_contamination.utils import InputDataset, parse_single_dataset

from with_argparse import with_argparse


@with_argparse(dataset=parse_single_dataset)
def generate_mincut_splits(
    dataset: InputDataset,
    mincut_file: Path,
):
    docs = dataset.train_docs + dataset.test_docs + dataset.dev_docs

    with mincut_file.open("rb") as f:
        mincut = pickle.load(f)
    if len(mincut) != len(docs):
        raise ValueError(len(mincut), len(docs))

    test_part = 2 if 2 in mincut else 1
    dev_part = 1 if 2 in mincut else 2
    mincut = {
        "train": [i for i, part in enumerate(mincut) if part == 0],
        "test": [i for i, part in enumerate(mincut) if part == test_part],
        "dev": [i for i, part in enumerate(mincut) if part == dev_part],
    }

    mincut_train = [docs[i] for i in mincut["train"]]
    mincut_test = [docs[i] for i in mincut["test"]]
    mincut_dev = [docs[i] for i in mincut["dev"]]

    for split_file, split in zip(
        (dataset.train_split, dataset.test_split, dataset.dev_split),
        (mincut_train, mincut_test, mincut_dev),
    ):
        if split_file is None:
            dataset_name = dataset.train_split.stem.split("_")[0]
            split_file = dataset.train_split.with_stem(dataset_name + "_dev")

        split_file = split_file.with_stem(split_file.stem + "_mincut")
        with split_file.open("w") as f:
            json.dump(split, f)
        print(split_file.as_posix())
    print(dataset.train_split.parent.as_posix())


generate_mincut_splits()
