import json

from tqdm import tqdm
from with_argparse import with_argparse

from paper_dataset_contamination import (
    InputDataset,
    parse_dataset,
)


@with_argparse(datasets=parse_dataset)
def generate_contamination_splits(
    datasets: list[InputDataset],
    n_splits: int = 5,
    n_contaminated: int = 10,
):
    contamination_step = 100 // n_contaminated
    for dataset in tqdm(datasets, position=1, leave=False):
        clean_docs, contaminated_docs = (
            dataset.separate_clean_and_contaminated_samples_for_split(
                dataset.train_docs,
                dataset.train_entities_per_doc,
                dataset.train_entities,
                dataset.test_entities,
            )
        )
        total_size = min(len(clean_docs), len(contaminated_docs))

        for contamination_level in range(
            0, 100 + contamination_step, contamination_step
        ):
            for split_id, split in enumerate(
                dataset.get_partly_contaminated_splits_given_level(
                    clean_docs,
                    contaminated_docs,
                    contamination_level,
                    total_size,
                    n_splits,
                )
            ):
                contamination_filename_stem = "%s_conta%d_split%d" % (
                    dataset.train_split.stem.split("_")[0],
                    contamination_level,
                    split_id,
                )
                with dataset.train_split.with_stem(
                    contamination_filename_stem + "_train"
                ).open("w") as f:
                    json.dump(split, f)

                split_unique_entities = dataset._unique_entities(split)
                clean, contaminated = (
                    dataset.filter_for_clean_and_contaminated_samples_in_split(
                        dataset.test_docs,
                        split_unique_entities,
                    )
                )
                contaminated_entities = list()
                for doc in contaminated:
                    contaminated_entities.extend([
                        dict(entity, tokens=doc["tokens"]) for entity in doc["entities"]
                    ])

                assert len(clean) == len(dataset.test_docs)

                with dataset.train_split.with_stem(
                    contamination_filename_stem + "_seen_test"
                ).open("w") as f:
                    json.dump(contaminated, f)

                with dataset.train_split.with_stem(
                    contamination_filename_stem + "_unseen_test"
                ).open("w") as f:
                    json.dump(clean, f)

                with dataset.train_split.with_stem(
                    contamination_filename_stem + "_contaminated"
                ).open("w") as f:
                    json.dump(contaminated_entities, f)


generate_contamination_splits()
