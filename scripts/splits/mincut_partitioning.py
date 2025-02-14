import itertools
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import metis
from networkx.convert_matrix import from_scipy_sparse_array
from scipy.sparse import dok_array
from tqdm import tqdm
from with_argparse import with_argparse

from paper_dataset_contamination import InputDataset, parse_single_dataset


@with_argparse(dataset=parse_single_dataset)
def graph_partitioning(
    dataset: InputDataset,
    seed: int,
    output_file: Path,
    partition_sizes: Optional[list[float]] = None,
):
    entities_per_doc = (
        dataset.train_entities_per_doc
        + dataset.test_entities_per_doc
        + dataset.dev_entities_per_doc
    )
    n_documents = len(entities_per_doc)

    if not partition_sizes:
        print(f"Inferring partition sizes from the original dataset ...")
        print(
            f"Got {n_documents} documents in total, thereof "
            f"{dataset.train_size}, {dataset.test_size} and {dataset.dev_size} documents in train, test and dev splits"
        )
        partition_sizes = [len(split) / n_documents for split in [dataset.train_docs, dataset.dev_docs, dataset.test_docs]]
        print(f"Partitioning into {partition_sizes}")

    entity_in_docs = defaultdict(set)
    entity_types = set()
    print(f"Collecting entity information for {dataset.name} ....")
    for doc_id, doc in enumerate(
        tqdm(entities_per_doc, leave=False, desc=dataset.name)
    ):
        for entity in doc:
            entity_types.add(entity[1])
            entity_in_docs[entity].add(doc_id)

    interactions = dok_array((len(entities_per_doc), len(entities_per_doc)), dtype=bool)
    print(f"Constructing adjacency matrix")
    for entity, doc_ids in tqdm(
        entity_in_docs.items(), leave=False, total=len(entity_in_docs), position=1, desc=dataset.name
    ):
        if len(doc_ids) < 2:
            continue

        for inner_id, outer_id in tqdm(
            itertools.product(doc_ids, doc_ids),
            position=0,
            leave=False,
            total=len(doc_ids) ** 2,
            desc=entity[0] + ", " + entity[1] + " occurs in " + str(len(doc_ids)) + " docs"
        ):
            if inner_id < outer_id:
                interactions[outer_id, inner_id] = True

    print(f"Creating graph ...")
    G = from_scipy_sparse_array(interactions)
    print(f"Done")

    print(f"Partitioning into {len(partition_sizes)} partitions: {partition_sizes}")

    objval, parts = metis.part_graph(
        G, nparts=len(partition_sizes), tpwgts=partition_sizes, objtype="cut", seed=seed
    )
    if len(parts) != G.number_of_nodes():
        raise ValueError

    print("Objective value (edge cuts):", objval)
    print("Partition assignments:", parts)

    # Count crossing edges (unweighted):
    crossing_count = 0
    for u, v in G.edges():
        if parts[u] != parts[v]:
            crossing_count += 1
    print("Number of edges crossing partitions:", crossing_count)

    # Sum crossing edge weights
    weighted_cut = 0.0
    for u, v, data in G.edges(data=True):
        if parts[u] != parts[v]:
            weighted_cut += data.get("weight", 1.0)

    print("Weighted cut (sum of crossing edge weights):", weighted_cut)
    partition_counts = Counter(parts)
    print("Partition counts:", partition_counts)

    with open(output_file, "wb") as f:
        pickle.dump(parts, f)


graph_partitioning()
