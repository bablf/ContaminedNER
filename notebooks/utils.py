import json
import os
import warnings
from copy import deepcopy

import pandas as pd
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained(
    "bert-base-cased", use_fast=True, skip_special_tokens=True
)


def load_dataset_file(dir_path, file):
    with open(os.path.abspath(dir_path + "/" + file), "r") as f:
        json_obj = json.load(f)
        df = pd.DataFrame.from_records(json_obj)
    if "extended" in df.columns and "ace05" in file and "old_entities" in df.columns:
        if all(
            [
                1 if x == y else 0
                for x, y in zip(df.entities.tolist(), df.old_entities.tolist())
            ]
        ):
            df["old_tokens"] = deepcopy(df.tokens.tolist())
            df["tokens"] = df["extended"]
            old_entities = deepcopy(df.entities.tolist())
            for i, row in df.iterrows():
                if len(row.entities) == 0:
                    continue
                sent_start = row.tokens.index("<extra_id_22>")
                for entity in row.entities:
                    entity["start"] += sent_start
                    entity["end"] += sent_start
                    assert entity["start"] >= 0
                    assert entity["end"] >= 0
                    assert entity["end"] <= len(row.tokens), print(row)
            df["old_entities"] = old_entities
            assert df.old_entities.str != df.entities.str

    if isinstance(df["tokens"].values[0], str):
        df["text"] = df["tokens"]
    else:
        df["text"] = df["tokens"].apply(lambda x: tk.convert_tokens_to_string(x))
    df["text"] = df["text"].astype("string")
    return df


def find_file_with_substring(directory, substring):
    if not os.path.isdir(directory):
        raise ValueError(f"The path '{directory}' is not a valid directory.")

    # List all files in the directory
    matching_files = [f for f in os.listdir(directory) if substring in f]

    # Check the number of matching files
    if len(matching_files) == 0:
        warnings.warn(f"No files containing '{substring}' were found in '{directory}'.")
        return None
    elif len(matching_files) > 1:
        raise RuntimeError(
            f"Multiple files containing '{substring}' were found: {matching_files}"
        )

    # Return the matching file name (not full path)
    return matching_files[0]


def get_dataset(
    dataset_name,
    train_file="",
    dev_file="",
    test_file="",
    type_file="",
    data_dir="data/",
) -> dict:
    if train_file == "":
        train_file = find_file_with_substring(data_dir + dataset_name, "train.json")
    elif train_file is not None and os.path.exists(
        os.path.abspath(data_dir + dataset_name + "/" + train_file)
    ):
        train_df = load_dataset_file(data_dir + dataset_name, train_file)
    else:
        train_df = None
        warnings.warn("Train dataset not found." + dataset_name)
    if dev_file == "":
        dev_file = find_file_with_substring(data_dir + dataset_name, "dev.json")
    elif dev_file is not None and os.path.exists(
        os.path.abspath(data_dir + dataset_name + "/" + dev_file)
    ):
        eval_df = load_dataset_file(data_dir + dataset_name, dev_file)
    else:
        eval_df = None
        warnings.warn("Dev dataset not found." + dataset_name)
    if test_file == "":
        test_file = find_file_with_substring(data_dir + dataset_name, "test.json")
    if test_file is not None and os.path.exists(
        os.path.abspath(data_dir + dataset_name + "/" + test_file)
    ):
        test_df = load_dataset_file(data_dir + dataset_name, test_file)
    else:
        test_df = None
        warnings.warn("Test dataset not found." + dataset_name)
    if type_file == "":
        type_file = find_file_with_substring(data_dir + dataset_name, "types.json")
    if type_file is not None and os.path.exists(
        os.path.abspath(data_dir + dataset_name + "/" + type_file)
    ):
        with open(os.path.abspath(data_dir + dataset_name + "/" + type_file), "r") as f:
            types = json.load(f)
    else:
        types = create_types_file(dataset_name, train_df, eval_df, test_df)
        warnings.warn(
            "Types file not found. Trying to create it automatically. Double check your data_dir"
            + dataset_name
        )
    return {"train_df": train_df, "dev_df": eval_df, "test_df": test_df, "types": types}


def create_types_file(dataset_name, train_df, dev_df, test_df):
    ents, rels = set(), set()
    for df in [train_df, dev_df, test_df]:
        if df is None:
            continue

        for entities in df.entities:
            for ent in entities:
                ents.add(ent["type"])
        if "relations" not in df.columns:
            continue
        for relations in df.relations:
            for rel in relations:
                rels.add(rel["type"])
    with open("data/" + dataset_name + "/" + dataset_name + "_types.json", "w") as f:
        json.dump(
            {
                "entities": {tp: {"short": tp, "verbose": tp} for tp in ents},
                "relations": {
                    tp: {"short": tp, "verbose": tp, "symmetric": False} for tp in rels
                },
            },
            f,
        )
    return {
        "entities": {tp: {"short": tp, "verbose": tp} for tp in ents},
        "relations": {
            tp: {"short": tp, "verbose": tp, "symmetric": False} for tp in rels
        },
    }
