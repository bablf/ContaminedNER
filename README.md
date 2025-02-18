# Entity Contamination inflates Named Entity Recognition Performance: Quantifying the Memorization-Generalization Gap in NER Models

This repository contains the code for the OpenReview submission "*Entity Contamination inflates Named Entity Recognition
Performance: Quantifying the Memorization-Generalization Gap in NER Models*"

### Repository Structure

This repository is structured as follows:

├─ [1. Visualizations](#dataset-contamination-visualizations)

├─ [2. Recreating Experiments](#recreating-experiments)

├─ [3. Creating Clean Splits using Minimum Cut](#minimum-cut)

├─ [4. Create ACE05 Clean and Contaminated Test set.](#create-clean-and-contaminated-test-split-for-any-dataset)



### Setup

```shell
git clone xyz && cd xyz
pip install .
```

### Dataset Contamination Visualizations

##### Plots

To visualize the contamination of any dataset, run
```shell
make -f plots.Makefile contamination.pdf
```
and open the resulting file

For the other plots, run the notebooks/experiments_contamination_analysis.ipynb Notebook.

##### Dataset Contamination HTML Example Generator

To re-create Figure 1 in our paper, run
```shell
make -f plots.Makefile contamination_html
```
and open any file in [scripts/analysis/visuals](scripts/analysis/visuals) in your Browser


### Recreating Experiments

For our experiments, we used ITER[^iter], ASP[^asp] and DiffusionNER[^diff].
To recreate our experiments, configs have to be generated first:

```shell
make -f configs.Makefile all
```

Then, we can simply run all our experiments via:

```shell
python3 scripts/paper_dataset_contamination/run_experiment \
 --model {model} \
 --dataset configs/*/dataset_contamination/*_{asp.conf,iter.json,diffusion.conf} \
 --n_splits 5 \
 --experiment_dir experiments/ \
 --architecture {asp,iter,diffusionner} --split_as_dataset
```

for either ASP, ITER or DiffusionNER at a time.
For models, we used:

| Model        | Transformer                  |
|--------------|------------------------------|
| ASP          | `google/flan-t5-base`        |
| ITER         | `microsoft/deberta-v3-small` |
| DiffusionNER | `bert-large-cased`           |

[^iter]: https://aclanthology.org/2024.findings-emnlp.655/https://aclanthology.org/2024.findings-emnlp.655/

[^asp]: https://arxiv.org/pdf/2210.14698

[^diff]: https://arxiv.org/abs/2305.13298

### Minimum Cut

To create new splits using our minimum cut algorithm, you need to install the `metis` package on Linux via

```shell
sudo apt install metis
```

All that is required is to run

```shell
make -f plots.Makefile mincut_dataset
```

which will create 80/10/10 minimum cut splits for all datasets. **To re-create the plot from our paper**, run

```shell
make -f plots.Makefile mincut_contamination.pdf
```

### Create Clean and Contaminated Test Split for any dataset

Since we are not allowed to share ACE05, we can only share a script to recreate the clean
and contaminated test splits for all/any dataset.

0. (Optional) Put the original ACE05 dataset into `datasets/ace05/*/English`
1. Run `bash scripts/datasets/load_datasets.sh`. This will load and preprocess all datasets.
2. Run `make -f plots.Makefile separated_test_set_files`.
The `{dataset}_{seen,unseen}_test.json` for each dataset can be found in the datasets/{dataset} folder

If you want to create the clean and contaminated test split for any dataset, you need to convert it into the required format,
where *start* and *end* are the token indices.
```json
[ {"tokens": ["Peripheral", "neuropathy", "associated", "with", "capecitabine", "."],
"entities": [{"type": "Adverse-Effect", "start": 0, "end": 2}, {"type": "Drug", "start": 4, "end": 5}],
    ...
]
```

You can create the clean and contaminated test splits with
```shell
make -f plots.Makefile separated_test_set_files DATASETS=datasets/YOURDATASET/YOURDATASET_train.json,datasets/YOURDATASET/YOUR_DATASET_test.json,YOURDATASET
```
