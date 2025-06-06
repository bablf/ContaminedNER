COMMA := ,
SEED := 42
PARTITION_SIZE := 0.8 0.1 0.1
# comment out individual datasets if they are not available, f.e. ACE05
DATASET_NAMES := ADE CoNLL03 CoNLL04 GENIA NYT SciERC
ifneq ($(wildcard datasets/ace05/ace05_train.json),)
	DATASETS := datasets/ace05/ace05_train.json,datasets/ace05/ace05_test.json,datasets/ace05/ace05_dev.json,ACE05
	DATASET_NAMES := ACE05 $(DATASET_NAMES)
else
	DATASETS :=
endif
DATASETS += datasets/ade/ade_split_0_train.json,datasets/ade/ade_split_0_test.json,ADE
DATASETS += datasets/conll03/conll03_train.json,datasets/conll03/conll03_test.json,datasets/conll03/conll03_dev.json,CoNLL03
DATASETS += datasets/conll04/conll04_train.json,datasets/conll04/conll04_test.json,datasets/conll04/conll04_dev.json,CoNLL04
DATASETS += datasets/genia/genia_train_dev.json,datasets/genia/genia_test.json,GENIA
DATASETS += datasets/nyt/nyt_train.json,datasets/nyt/nyt_test.json,datasets/nyt/nyt_dev.json,NYT
DATASETS += datasets/scierc/scierc_train.json,datasets/scierc/scierc_test.json,datasets/scierc/scierc_dev.json,SciERC

DATASET_FILES := $(filter-out $(DATASET_NAMES),$(subst $(COMMA), ,$(DATASETS)))
DATASET_TEST_FILES := $(filter %_test.json,$(DATASET_FILES))
DATASET_CONTAMINATION_FILES := $(subst _test.json,_contaminated.json,$(filter-out datasets/ade/%.json,$(DATASET_TEST_FILES)))
SEPARATED_DATASET_FILES := $(subst _test,_seen_test,$(DATASET_TEST_FILES)) $(subst _test,_unseen_test,$(DATASET_TEST_FILES))

MINCUT_DATASETS := $(subst .json,_mincut.json,$(DATASETS))
MINCUT_FILES := $(subst .json,_mincut.json,$(DATASET_FILES))
MINCUT_DIR :=

CONTAMINATION_LEVELS := 0 10 80 90 100
SPLIT_DATASET := scierc
SPLIT_DATASET_NAME := SciERC
SPLIT_FILES := $(foreach CONTA,$(CONTAMINATION_LEVELS),datasets/$(SPLIT_DATASET)/$(SPLIT_DATASET)_conta$(CONTA)_split0_train.json datasets/$(SPLIT_DATASET)/$(SPLIT_DATASET)_test.json)
SPLIT_DATASETS := $(foreach CONTA,$(CONTAMINATION_LEVELS),datasets/$(SPLIT_DATASET)/$(SPLIT_DATASET)_conta$(CONTA)_split0_train.json$(COMMA)datasets/$(SPLIT_DATASET)/$(SPLIT_DATASET)_test.json$(COMMA)$(SPLIT_DATASET_NAME))

mincut_contamination.pdf: $(MINCUT_FILES) $(DATASET_FILES)
	python3 scripts/analysis/plot_dataset_contamination.py --dataset $(DATASETS) --mincut_dataset $(MINCUT_DATASETS) --mincut_cmap "tab20c:4:8" --test_cmap "tab20b:4:16" --train_cmap "tab20c:4:0" --with_label --distance_between_bars 1 --bar_width 0.7 --scale 1 --y_scale 1 --filename mincut_contamination.pdf

contamination.pdf: $(DATASET_FILES)
	python3 scripts/analysis/plot_dataset_contamination.py --dataset $(DATASETS) --test_cmap "tab20b:4:16" --train_cmap "tab20c:4:0" --with_label --distance_between_bars 1 --bar_width 0.7 --scale 1 --y_scale 1

split_contamination.pdf: scripts/analysis/plot_split_contamination.py $(SPLIT_FILES)
	python3 scripts/analysis/plot_split_contamination.py --dataset $(SPLIT_DATASETS) --test_cmap "tab20c:4:4" --train_cmap "tab20c:4:4" --with_label --distance_between_bars 1 --bar_width 0.7 --scale 1 --y_scale 1 --filename $@

default_ratios:
	python3 scripts/analysis/split_ratios.py --dataset $(DATASETS)

contamination_html:
	python3 scripts/analysis/visualize_dataset_contamination.py --output_dir scripts/analysis/visuals --dataset $(filter %CoNLL04,$(DATASETS))

contamination_splits: $(DATASET_FILES)
	python3 scripts/splits/generate_contamination_splits.py --n_splits 5 --n_contaminated 10 --dataset $(DATASETS)

contaminated_entity_files: $(DATASET_CONTAMINATION_FILES)
$(DATASET_CONTAMINATION_FILES): $(DATASET_FILES)
	@python3 scripts/splits/generate_contamination_entities.py --dataset $(filter %$(word 2,$(subst /, ,$@)),$(shell echo $(DATASETS) | tr A-Z a-z))

$(SEPARATED_DATASET_FILES): $(DATASET_TEST_FILES)
	@python3 scripts/splits/separate_clean_contaminated_test_set.py --dataset $(filter %$(word 2,$(subst /, ,$@)),$(shell echo $(DATASETS) | tr A-Z a-z))

separated_test_set_files: $(SEPARATED_DATASET_FILES);

mincut_files: $(MINCUT_FILES);

dataset_files: $(DATASET_FILES);

mincut_dataset: $(MINCUT_DATASETS);


$(MINCUT_DATASETS): $(MINCUT_FILES);

$(DATASETS): $(DATASET_FILES);

$(DATASET_FILES):
	@echo $@

%_train_mincut.json %_train_dev_mincut.json: %.pkl
	@python3 scripts/splits/generate_mincut_splits.py --dataset $(filter %$(word 2,$(subst /, ,$@)),$(shell echo $(DATASETS) | tr A-Z a-z)) --mincut_file $<

%.pkl:
	@python3 scripts/splits/mincut_partitioning.py --dataset $(filter %$(word 2,$(subst /, ,$@)),$(shell echo $(DATASETS) | tr A-Z a-z)) --partition_size $(PARTITION_SIZE) --seed $(SEED) --output_file $@
	@echo need to make $@

clean:
	find datasets -name "*_mincut.json" -print -delete
	find datasets -name "*seen*.json" -print -delete
	rm -fv *.pdf
