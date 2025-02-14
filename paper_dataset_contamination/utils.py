import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar

from tqdm import tqdm

T_IN = TypeVar("T_IN")
T_OUT = TypeVar("T_OUT")
ENTITY_T = tuple[str, str]
T = TypeVar("T")

DOCS_T = list[dict]
ENTITIES_T = set[ENTITY_T]


@dataclass(repr=False)
class InputDataset:
    train_split: Path
    test_split: Path
    name: str
    dev_split: Optional[Path] = None

    train_docs: DOCS_T = dataclasses.field(default_factory=list)
    train_entities: set[ENTITY_T] = dataclasses.field(default_factory=set)
    train_entities_per_doc: list[set[ENTITY_T]] = dataclasses.field(
        default_factory=list
    )
    test_docs: DOCS_T = dataclasses.field(default_factory=list)
    test_entities: set[ENTITY_T] = dataclasses.field(default_factory=set)
    test_entities_per_doc: list[set[ENTITY_T]] = dataclasses.field(default_factory=list)
    dev_docs: DOCS_T = dataclasses.field(default_factory=list)
    dev_entities: set[ENTITY_T] = dataclasses.field(default_factory=set)
    dev_entities_per_doc: list[set[ENTITY_T]] = dataclasses.field(default_factory=list)

    def __repr__(self):
        return f"Dataset({self.name})"

    def load(self):
        self.train_entities, self.train_entities_per_doc, self.train_docs = (
            self.load_split(self.train_split)
        )
        self.test_entities, self.test_entities_per_doc, self.test_docs = (
            self.load_split(self.test_split)
        )
        self.dev_entities, self.dev_entities_per_doc, self.dev_docs = (
            self.load_split(self.dev_split)
            if self.dev_split is not None
            else ([], [], [])
        )

    def __post_init__(self):
        self.load()

    @property
    def train_size(self):
        return len(self.train_entities_per_doc)

    @property
    def test_size(self):
        return len(self.test_entities_per_doc)

    @property
    def dev_size(self):
        return len(self.dev_entities_per_doc)

    @property
    def entities_per_doc(self):
        return (
            self.train_entities_per_doc
            + self.test_entities_per_doc
            + self.dev_entities_per_doc
        )

    @staticmethod
    def load_split(
        split_file: Path,
    ) -> tuple[set[ENTITY_T], list[ENTITY_T], list[dict]]:
        split_entities = set()
        split_entities_per_doc = list()
        with split_file.open() as f:
            split_json = json.load(f)
        for elem in tqdm(split_json, leave=False, desc=split_file.name):
            elem_text = elem["tokens"]
            elem_entities = {
#                ("_".join([str(entity["start"]), str(entity["end"])]), entity["type"])
                ("_".join(elem_text[entity["start"]: entity["end"]]), entity["type"])
                for entity in elem["entities"]
            }
            split_entities.update(elem_entities)
            split_entities_per_doc.append(elem_entities)
        return split_entities, split_entities_per_doc, split_json

    @staticmethod
    def swap_if(a: T, b: T, cond: bool) -> tuple[T, T]:
        if cond:
            return b, a
        return a, b

    @staticmethod
    def _compute_doc_contamination(
        source,
        sink,
        whole_doc_contaminated: bool = False,
        normalize: bool = True,
    ):
        num_total = len(source)
        if whole_doc_contaminated:
            num_contaminated = sum(
                (sum(ent in sink for ent in doc) == len(doc)) and len(doc) > 0
                for doc in source
            )
            if num_contaminated > num_total:
                raise ValueError
        else:
            num_contaminated = sum(any(ent in sink for ent in doc) for doc in source)

        return num_contaminated / (not normalize or num_total)

    @staticmethod
    def _unique_entities(split: DOCS_T) -> ENTITIES_T:
        split_entities = set()
        for elem in split:
            elem_text = elem["tokens"]
            elem_entities = {
                ("_".join(elem_text[entity["start"] : entity["end"]]), entity["type"])
                for entity in elem["entities"]
            }
            split_entities.update(elem_entities)
        return split_entities

    @staticmethod
    def _split_entities(split: DOCS_T) -> list[ENTITIES_T]:
        split_entities = list()
        for elem in split:
            elem_text = elem["tokens"]
            elem_entities = {
                ("_".join(elem_text[entity["start"] : entity["end"]]), entity["type"])
                for entity in elem["entities"]
            }
            split_entities.append(elem_entities)
        return split_entities

    @staticmethod
    def _doc_entities(doc: dict, position_aware: bool = False, container_type: type[list | set] = set) -> ENTITIES_T | list[ENTITY_T]:
        entities = container_type()
        words = doc["tokens"]
        for entity in doc["entities"]:
            if not position_aware:
                name, typ = "_".join(words[entity["start"]:entity["end"]]), entity["type"]
            else:
                name, typ = "_".join(map(str, [entity["start"], entity["end"]])), entity["type"]
            entities += container_type([(name, typ)])
        return entities

    @staticmethod
    def _compute_entity_contamination(
        source,
        sink,
        normalize: bool = True,
    ):
        return len(source & sink) / (not normalize or len(sink))

    def compute_contamination(
        self,
        split: Literal["train", "test"],
        entity_level: bool = False,
        whole_doc_contaminated: bool = False,
        normalize: bool = True,
    ):
        if entity_level:
            if whole_doc_contaminated:
                raise ValueError(
                    f"Cannot compute whole_doc_contaminated with {entity_level=}"
                )
            source, sink = self.swap_if(
                self.train_entities, self.test_entities, split != "train"
            )
            return self._compute_entity_contamination(source, sink, normalize=normalize)
        else:
            ((source, _), (_, sink)) = self.swap_if(
                (self.train_entities_per_doc, self.train_entities),
                (self.test_entities_per_doc, self.test_entities),
                split != "train",
            )
            return self._compute_doc_contamination(
                source,
                sink,
                whole_doc_contaminated=whole_doc_contaminated,
                normalize=normalize,
            )
    def get_split(self, split: str):
        if split == "train":
            return self.train_entities_per_doc, self.train_docs
        elif split == "test":
            return self.test_entities_per_doc, self.test_docs
        elif split == "dev":
            return self.dev_entities_per_doc, self.dev_docs
        else:
            raise ValueError(split)

    def get_partly_contaminated_splits_given_level(
        self,
        clean_docs,
        contaminated_docs,
        contamination_level: int,
        total_size: int,
        n_splits: int,
    ):
        assert total_size <= len(clean_docs)
        assert total_size <= len(contaminated_docs)

        for split_id in range(n_splits):
            rand = random.Random(split_id)

            clean_size = int(round(total_size * (1 - (contamination_level / 100)), 0))
            contaminated_size = total_size - clean_size

            clean_split = rand.sample(clean_docs, clean_size)
            contaminated_split = rand.sample(contaminated_docs, contaminated_size)

            assert len(clean_split) == clean_size
            assert len(contaminated_split) == contaminated_size
            assert len(clean_split) + len(contaminated_split) == total_size

            split = clean_split + contaminated_split
            entity_contamination = self._compute_entity_contamination(
                self._unique_entities(split),
                self.test_entities,
                normalize=True,
            )

            print(
                "split",
                split_id,
                "total_size",
                total_size,
                "contamination_level",
                contamination_level,
                "entity_contamination",
                entity_contamination,
            )
            print(f"{contaminated_size / total_size:.5f}")

            yield split

    @classmethod
    def filter_for_clean_and_contaminated_samples_in_split(
        cls,
        split_docs: DOCS_T,
        opposing_unique_entities: ENTITIES_T,
    ) -> tuple[list[DOCS_T], list[DOCS_T]]:
        """
        For a split of documents, return two sets of all documents, but with filtered entities:
        - Once with all clean entities
        - Once with all contaminated entities
        Args:
            split_docs (list): aaabbb
        Returns:
            Two similar test sets with filtered entities
        """
        docs_with_clean_entities = list()
        docs_with_contaminated_entities = list()
        for doc in split_docs:
            clean_entities = list()
            contaminated_entities = list()
            for pos, entity in enumerate(cls._doc_entities(doc, False, list)):
                if entity in opposing_unique_entities:
                    contaminated_entities.append(doc["entities"][pos])
                else:
                    clean_entities.append(doc["entities"][pos])
            assert len(clean_entities) + len(contaminated_entities) == len(doc["entities"])
            clean_doc = dict(doc, entities=clean_entities)
            contaminated_doc = dict(doc, entities=contaminated_entities)
            docs_with_clean_entities.append(clean_doc)
            docs_with_contaminated_entities.append(contaminated_doc)

        assert len(docs_with_clean_entities) == len(split_docs)
        assert len(docs_with_contaminated_entities) == len(split_docs)
        return docs_with_clean_entities, docs_with_contaminated_entities

    @staticmethod
    def separate_clean_and_contaminated_samples_for_split(
        split_docs: DOCS_T,
        split_entities: list[ENTITIES_T],
        split_unique_entities: ENTITIES_T,
        opposing_unique_entities: ENTITIES_T,
    ):
        """
        Separate the split into two sub splits: one that contains only clean samples while the other contains
            just contaminated training/test samples
        """
        (
            all_contaminated,
            contaminated,
            no_contamination,
            numb_ents_no_contamination,
            empty,
        ) = (0, 0, 0, 0, 0)

        clean_dataset = []
        contaminated_dataset = []
        contaminated_entities = split_unique_entities & opposing_unique_entities
        for doc, entities in zip(split_docs, split_entities):
            if len(entities) == 0:
                clean_dataset.append(doc)
                empty += 1
                continue
            # assert something pandas
            num_contaminated = len(entities & opposing_unique_entities)
            if num_contaminated == len(entities):
                all_contaminated += 1
            if num_contaminated > 0:
                contaminated_dataset.append(doc)
            else:
                no_contamination += 1
                numb_ents_no_contamination += len(entities)
                clean_dataset.append(doc)

        print(
            f"All Samples ({len(clean_dataset) + len(contaminated_dataset)}) have {0} NE contaminated samples ({all_contaminated} with all ents in sample)."
            f" {len(clean_dataset)} are clean ({numb_ents_no_contamination} Entities), of which {empty} are empty samples."
        )

        for sample in clean_dataset:
            for ent in sample["entities"]:
                if len(sample["tokens"]) < ent["end"]:
                    print("BAD")
            assert all(
                [
                    (
                        1
                        if (
                            " ".join(sample["tokens"][ent["start"] : ent["end"]]),
                            ent["type"],
                        )
                        not in contaminated_entities
                        else 0
                    )
                    for ent in sample["entities"]
                ]
            )
        return clean_dataset, contaminated_dataset


def outer(t_in: T_IN, t_out: T_OUT, enabled=True):
    if not enabled:

        def wrapper(fn: Callable[[t_in], t_out]):
            return fn

        return wrapper

    def parse_as_list(fn: Callable[[t_in], t_out]):
        def inner(inp: list[t_in]) -> list[t_out]:
            return [fn(elem) for elem in inp]

        return inner

    return parse_as_list


T_DS = TypeVar("T_DS", bound=InputDataset)


def parse_dataset_subclass(subclass: type[T_DS], sep=":", as_list=True):
    @outer(str, subclass, as_list)
    def _parse_dataset(inp: str) -> subclass:
        if sep in inp:
            args = inp.split(sep, 3)
        elif "," in inp:
            args = inp.split(",", 3)
        else:
            raise ValueError(inp)
        if len(args) < 3:
            raise ValueError(args)
        if len(args) > 3:
            return subclass(Path(args[0]), Path(args[1]), args[3], Path(args[2]))
        return subclass(Path(args[0]), Path(args[1]), args[2])

    return _parse_dataset


parse_dataset = parse_dataset_subclass(InputDataset)
parse_single_dataset = parse_dataset_subclass(InputDataset, as_list=False)
