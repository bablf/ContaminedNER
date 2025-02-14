from paper_dataset_contamination.utils import InputDataset, parse_dataset
from with_argparse import with_argparse


@with_argparse(datasets=parse_dataset)
def split_ratios(
    datasets: list[InputDataset],
):
    for dataset in datasets:
        entities_per_doc = (
            dataset.train_entities_per_doc
            + dataset.test_entities_per_doc
            + dataset.dev_entities_per_doc
        )

        total_len = len(entities_per_doc)
        lengths = tuple(
            len(subset)
            for subset in (dataset.train_docs, dataset.dev_docs, dataset.test_docs)
        )
        ratios = (length / total_len for length in lengths)
        print(dataset.name, "elements", *lengths, "ratios", *tuple(f"{ratio * 100:.3f}" for ratio in ratios))


split_ratios()
