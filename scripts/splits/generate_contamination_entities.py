import json

from with_argparse import with_argparse

from paper_dataset_contamination import parse_dataset, InputDataset


@with_argparse(datasets=parse_dataset)
def generate_contamination_entities(datasets: list[InputDataset]):
    for dataset in datasets:
        contaminated_entities = list()
        for dev_test_entities_in_doc, dev_test_doc in zip(
            dataset.test_entities_per_doc + dataset.dev_entities_per_doc,
            dataset.test_docs + dataset.dev_docs,
        ):
            contaminated_entities_in_doc = dataset.train_entities & dev_test_entities_in_doc
            for entity in contaminated_entities_in_doc:
                literal, typ, start, end = entity
                contaminated_entities.append({
                    "start": start,
                    "end": end,
                    "type": typ,
                    "words": dev_test_doc["tokens"]
                })

        pos = dataset.train_split.name.find("train")
        prefix = dataset.train_split.name[:pos]

        contaminated_filename = dataset.train_split.with_stem(
            prefix + "contaminated"
        )
        with open(contaminated_filename, "w") as f:
            json.dump({
                "train": contaminated_entities
            }, f)
    print(f"ok")


generate_contamination_entities()