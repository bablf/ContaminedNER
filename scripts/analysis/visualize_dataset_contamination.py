from pathlib import Path

import matplotlib.colors
from bs4 import BeautifulSoup, Tag
from spacy import displacy
from tqdm import tqdm
from with_argparse import with_argparse

from plot_dataset_contamination import InputDataset, parse_dataset

contaminated_gradient = (
    "linear-gradient(to right, "
    + matplotlib.colors.to_hex(matplotlib.colormaps["tab20c"].colors[3])
    + ", "
    + matplotlib.colors.to_hex(matplotlib.colormaps["tab20b"].colors[18])
    + ");"
)
clean_gradient = (
    "linear-gradient(to right, "
    + matplotlib.colors.to_hex(matplotlib.colormaps["tab20c"].colors[9])
    + ", "
    + matplotlib.colors.to_hex(matplotlib.colormaps["tab20c"].colors[11])
    + ");"
)

colors = {
    "ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
    "PEOP": "linear-gradient(to right, #2980b9, #6dd5fa, #fff);",
    "LOC": "linear-gradient(to right, #fffbd5, #b20a2c);",
    "OTHER": "linear-gradient(to right, #23074d, #cc5333);",
    "CONTAMINATED": contaminated_gradient,
    "CLEAN": clean_gradient,
}


def is_entity_fine(words, entity: dict, flt):
    return ("_".join(words[entity["start"]: entity["end"]]), entity["type"]) not in flt


def entity_label(words, entity: dict, flt):
    return "CLEAN" if is_entity_fine(words, entity, flt) else "CONTAMINATED"


def to_spacy_doc(inp: dict, overlaps: set):
    words = inp["tokens"]
    text = " ".join(words)
    offsets = [len(" ".join(words[:k])) + bool(k) for k, word in enumerate(words)]
    offsets.append(len(text))
    for ent in inp["entities"]:
        if ent["type"].upper() not in colors:
            raise ValueError(ent["type"])

    doc = {
        "text": text,
        "ents": [
            {  # noqa
                "start": offsets[entity["start"]],
                "end": offsets[entity["end"]],
                "label": entity_label(words, entity, overlaps),
            }
            for entity in inp["entities"]
        ],
    }
    return doc


@with_argparse(
    datasets=parse_dataset,
)
def visualize_dataset_contamination(
    datasets: list[InputDataset],
    output_dir: Path,
    min_contamination_length: int = 5,
):
    # fig, axis = plt.subplots(1, 1, layout="constrained", figsize=(7, 4.5))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    options = {"ents": list(colors.keys()), "colors": colors}

    for col, dataset in enumerate(datasets):
        train_contamination = dataset.compute_contamination("train", False, False, True)
        train_entity_contamination = dataset.compute_contamination(
            "train", True, normalize=True
        )

        filtered_overlaps = list()
        for i, train_entities in enumerate(tqdm(dataset.train_entities_per_doc)):
            train_entities = {
                ent for ent in train_entities if len(ent[0]) >= min_contamination_length
            }
            for j, other_entities in enumerate(dataset.test_entities_per_doc):
                if train_entities & other_entities:
                    filtered_overlaps.append((i, j))

        for k, (i, j) in enumerate(tqdm(filtered_overlaps)):
            train_entities = dataset.train_entities_per_doc[i]
            train_entities = {
                ent for ent in train_entities if len(ent[0]) >= min_contamination_length
            }
            test_entities = dataset.test_entities_per_doc[j]
            overlaps = train_entities & test_entities

            displacy_kwargs = {
                "style": "ent",
                "options": options,
                "manual": True,
                "jupyter": False,
                "minify": True,
            }
            source_html = displacy.render(
                to_spacy_doc(dataset.train_docs[i], overlaps), **displacy_kwargs
            )
            write_to_file(source_html, output_dir / f"{i:03d}_source.html")
            sink_html = displacy.render(
                to_spacy_doc(dataset.test_docs[j], overlaps), **displacy_kwargs
            )
            sink_html = combine_source_sink(
                dataset,
                i,
                source_html,
                j,
                sink_html,
                train_contamination=train_contamination,
                train_entity_contamination=train_entity_contamination,
                prev_page=None if k == 0 else filtered_overlaps[k - 1],
                next_page=(
                    None
                    if k + 1 == len(filtered_overlaps)
                    else filtered_overlaps[k + 1]
                ),
            )
            write_to_file(sink_html, output_dir / f"{i:03d}_sink{j:03d}.html")
        pass

    # fig.savefig("spacy_contamination.pdf")


def write_to_file(html: str, output_path: Path):
    with output_path.open("w") as f:
        f.write(html)


def combine_source_sink(
    dataset: InputDataset,
    source_idx: int,
    source: str,
    sink_idx: int,
    sink: str,
    **kwargs,
) -> str:
    source = BeautifulSoup(source)
    sink = BeautifulSoup(sink)

    apply_half_width_style(source, True)
    apply_half_width_style(sink, False)

    source = source.prettify()
    sink = sink.prettify()

    executable_dir = Path(__file__).parent
    with open(executable_dir / "static" / "visualize_dataset_contamination.html") as f:
        outer = f.read()
        outer = outer.format(source, sink)

    outer = BeautifulSoup(outer)
    for elem in outer.find_all(class_="dataset"):
        elem.string = dataset.name
    outer.find(id="source").string = str(source_idx)
    outer.find(id="sink").string = str(sink_idx)
    outer.find(id="contamination").string = (
        f"{kwargs.get('train_contamination') * 100:.1f}%"
    )
    outer.find(id="entity-contamination").string = (
        f"{kwargs.get('train_entity_contamination') * 100:.1f}%"
    )
    page_fmt = "%03d_sink%03d.html"
    if kwargs.get("next_page"):
        outer.find(id="next")["href"] = page_fmt % kwargs.get("next_page")
    else:
        outer.find(id="next")["disabled"] = "disabled"
    if kwargs.get("prev_page"):
        outer.find(id="prev")["href"] = page_fmt % kwargs.get("prev_page")
    else:
        outer.find(id="prev")["disabled"] = "disabled"
    return outer.prettify()


def apply_half_width_style(tag: Tag, is_source: bool):
    outer_div = tag.find("div")
    outer_div["class"].append("source" if is_source else "sink")


if __name__ == "__main__":
    visualize_dataset_contamination()
