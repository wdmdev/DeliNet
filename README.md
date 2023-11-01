# DeliNet
Deeplearning project for image2text conversion for images of food to textual recipes.

## Data
* The Kaggle `Food Ingredients and Recipes Dataset with Images`
    * [Download here](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)

### Prepare Kaggle Food Dataset
After downloading the Kaggle `Food Ingredients and Recipes Dataset with Images` we need it to be processed to fit our dataloader.
To do this perform the following steps:
1. Save the `archive.zip` file that you have downloaded with the Kaggle `Food Ingredients and Recipes Dataset with Images` in the project folder `data/raw`
2. Run the command `make prepare_kaggle_food_data`

Then you're good to go.
The raw data will be kept but renamed from `archive.zip` to `KaggleFoodDataset.zip` and the preprocessed data will be placed in `data/preprocessed/KaggleFoodDataset`.

## Project Structure
```
── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```
**NB the folders `data` and `models` in the root folder are gitignored and should only be held locally to avoid pushing large files with `git`**
