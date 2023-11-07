.PHONY: hpc_env local_env prepare_kaggle_food_data

hpc_env:
	if [ ! -d "env" ]; then \
		@module load python3/3.10.11; \
		python3 -m venv env; \
	fi
	@source env/bin/activate; \
	python3 -m pip install -r requirements.txt

local_env:
	if [ ! -d "env" ]; then \
		python3 -m venv env; \
	fi
	@source env/bin/activate; \
	python3 -m pip install -r requirements.txt

finetune_CLIP:
	@source env/bin/activate; \
	python3 -m src.models.finetune_CLIP


prepare_kaggle_food_data:
	@echo "Step 1: Check if archive.zip exists in data/raw"
	@if [ ! -f "data/raw/archive.zip" ]; then \
		echo "Cannot find file archive.zip in data/raw. Do you need to rename the .zip file?"; \
		exit 1; \
	fi

	@echo "Step 2: Rename archive.zip to KaggleFoodDataset.zip"
	mv data/raw/archive.zip data/raw/KaggleFoodDataset.zip
	
	@echo "Step 3: Unzip KaggleFoodDataset.zip"
	unzip -q data/raw/KaggleFoodDataset.zip -d data/raw/
	
	@echo "Step 4: Check if the folder 'Food Images' exists in data/raw"
	@if [ ! -d "data/raw/Food Images" ]; then \
		echo "Cannot find folder Food Images in data/raw. Do you need to rename the folder?"; \
		exit 1; \
	fi

	@echo "Step 5: Rename 'Food Images' to 'images'"
	mv data/raw/'Food Images' data/raw/images
	
	@echo "Step 6: Move all images and delete the folder 'Food Images'"
	mv data/raw/images/'Food Images'/* data/raw/images/
	rmdir data/raw/images/'Food Images'

	@echo "Step 7: Rename 'Food Ingredients and Recipe Dataset with Image Name Mapping.csv' to 'data.csv'"
	mv data/raw/'Food Ingredients and Recipe Dataset with Image Name Mapping.csv' data/raw/data.csv

	@echo "Step 8: Move images and data.csv to data/processed/KaggleFoodDataset"
	mkdir -p data/processed/KaggleFoodDataset
	mv data/raw/images data/processed/KaggleFoodDataset/
	mv data/raw/data.csv data/processed/KaggleFoodDataset/

	@echo "Data preparation complete!"
