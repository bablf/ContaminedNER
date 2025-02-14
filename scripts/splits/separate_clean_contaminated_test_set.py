import json

from paper_dataset_contamination.utils import InputDataset, parse_dataset

from with_argparse import with_argparse


@with_argparse(datasets=parse_dataset)
def generate_contamination_splits(
    datasets: list[InputDataset],
):
    for dataset in datasets:
        clean, contaminated = dataset.filter_for_clean_and_contaminated_samples_in_split(
            dataset.test_docs,
            dataset.train_entities,
        )
        ds_name = dataset.test_split.stem.split("_")[0]
        if ds_name == "ade":
            ds_name = dataset.test_split.stem.split("_")[:3]
            ds_name = "_".join(ds_name)
        with dataset.test_split.with_stem(ds_name + "_unseen_test").open("w") as f:
            json.dump(clean, f)
        with dataset.test_split.with_stem(ds_name + "_seen_test").open("w") as f:
            json.dump(contaminated, f)


generate_contamination_splits()
