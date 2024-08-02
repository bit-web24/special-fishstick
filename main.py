import os
import tensorflow as tf
from src.data_preprocessing import load_data
from src.evaluate import model_evaluate
from src.model import create_model
from src.visualize import plot_history
from src.predictions import visualize_predictions
from src.train import model_train_and_save
from definitions import dataset_path

def main():
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("GPU Reuired for training this model.")
        exit(-1)

    # Load data
    train_generator, validation_generator = load_data(dataset_path)

    # Create and compile the model
    history = model_train_and_save(train_generator, validation_generator, epochs=5)

    # Evaluate the model
    model, test_loss, test_acc = model_evaluate(train_generator, validation_generator)
    print("Test Accuracy:", test_acc)

    # Plot training & validation accuracy values
    plot_history(history)

    # Visualize predictions
    class_names = list(validation_generator.class_indices.keys())
    visualize_predictions(model, validation_generator, class_names)

if __name__ == "__main__":
    main()
