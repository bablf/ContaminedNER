import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from with_argparse import with_argparse


@with_argparse
def generate_sred_fm(dataset_dir: Path, max_samples: int = 100000):
    dataset_dict = load_dataset('Babelscape/SREDFM', 'en')
    (dataset_dir / "sred").mkdir(parents=True, exist_ok=True)

    for split, dataset in dataset_dict.items():
        if len(dataset) > max_samples:
            dataset = (
                dataset
                .shuffle(seed=42)
                .select(range(max_samples))
            )

        elements = list()
        for elem in tqdm(dataset):
            elements.append(map_to_json_conll_format(elem))
        with open(dataset_dir / "sred" / f"sred_{split}.json", "w") as f:
            json.dump(elements, f)


def map_to_json_conll_format(example):
    text = example["text"]
    filtered_entities = [entity for entity in example["entities"] if entity["type"] == "NUMBER"]
    split_at = {entity["start"] for entity in filtered_entities}
    split_at |= {entity["end"] for entity in filtered_entities}
    for i in split_at:
        if i + 1 in split_at and text[i] == ' ':
            pass

    split_at = list(sorted(split_at))

    words = list(map(lambda x: text[slice(*x)], zip(split_at, split_at[1:] + [None])))
    words = list(map(lambda x: x.strip(), words))
    entities = [
        dict(
            entity,
            start=split_at.index(entity["start"]),
            end=split_at.index(entity["end"])
        )
        for entity in filtered_entities
    ]
    for entity in entities:
        if entity["surfaceform"] != " ".join(words[entity["start"]:entity["end"]]):
            raise ValueError
    return dict(example, tokens=words, entities=entities)


generate_sred_fm()
