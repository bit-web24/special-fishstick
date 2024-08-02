import tensorflow as tf
from src.data_preprocessing import load_data
from definitions import dataset_path, model_path

def model_evaluate(train_generator, validation_generator):
    train_generator, validation_generator = load_data(dataset_path)

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    test_loss, test_acc = model.evaluate(validation_generator)
    return model, test_loss, test_acc
