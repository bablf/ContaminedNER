DATASETS := ace05 conll04 genia nyt scierc
CONFIGS_DIR := configs

SPLITS := 0 1 2 3 4
CONTAMINATION_LEVELS := 0 10 20 30 40 50 60 70 80 90 100
ITER_CONFIGS := $(foreach D,$(DATASETS),$(foreach S,$(SPLITS),$(foreach C,$(CONTAMINATION_LEVELS),$(CONFIGS_DIR)/$D/dataset_contamination/$D_conta$C_split$S_iter.json)))
ASP_CONFIGS := $(subst _iter.json,_asp.conf,$(ITER_CONFIGS))
DIFF_CONFIGS := $(subst _iter.json,_diffusion.conf,$(ITER_CONFIGS))

ITER_TEST_CONFIGS := $(foreach D,$(DATASETS),configs/$D/$D_clean_contaminated_iter.json)
ASP_TEST_CONFIGS := $(subst _iter.json,_asp.conf,$(ITER_TEST_CONFIGS))
DIFF_TEST_CONFIGS := $(subst _iter.json,_diffusion.conf,$(ITER_TEST_CONFIGS))

ls:
	@echo $(DIFF_TEST_CONFIGS)

all: iter_configs asp_configs diffusion_configs;

iter_configs: $(ITER_CONFIGS) $(ITER_TEST_CONFIGS);

asp_configs: $(ASP_CONFIGS) $(ASP_TEST_CONFIGS);

diffusion_configs: $(DIFF_CONFIGS) $(DIFF_TEST_CONFIGS);

$(ITER_CONFIGS):
	python3 scripts/configs/generate_iter_configs.py --base $(CONFIGS_DIR)/templates/ITER/$(lastword $(subst /, ,$(subst /dataset_contamination,,$(dir $@)))).json --output_dir $(CONFIGS_DIR)/$(lastword $(subst /, ,$(subst /dataset_contamination,,$(dir $@)))) --n_splits 5

$(ITER_TEST_CONFIGS):
	python3 scripts/configs/generate_iter_test_configs.py --base $(CONFIGS_DIR)/templates/ITER/$(lastword $(subst /, ,$(dir $@))).json --output_dir $(CONFIGS_DIR)/$(lastword $(subst /, ,$(dir $@)))

%_diffusion.conf: %_iter.json
	python3 scripts/configs/generate_diffusion_ner_configs.py --template $(CONFIGS_DIR)/templates/DiffusionNER/$(lastword $(subst /, ,$(subst /dataset_contamination,,$(dir $@)))).conf --data_dir datasets --config $<

%_asp.conf: %_iter.json
	python3 scripts/configs/generate_asp_configs.py --template $(CONFIGS_DIR)/templates/ASP/$(lastword $(subst /, ,$(subst /dataset_contamination,,$(dir $@)))).conf --data_dir datasets --config $<

clean:
	rm -vf configs/*/dataset_contamination/*.json
	rm -vf configs/*/dataset_contamination/*.conf
	rm -vf configs/*/*.json
	rm -vf configs/*/*.conf
	rm -rvf configs/*/dataset_contamination

#.PHONY: seen_unseen_configs
