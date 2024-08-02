from os import path
import pathlib

ROOT_DIR=path.dirname(path.abspath(__file__))
dataset_path = pathlib.Path(ROOT_DIR).joinpath("data/kagglecatsanddogs_3367a/PetImages")
model_path=pathlib.Path(ROOT_DIR).joinpath("model/binary_classification.keras")
