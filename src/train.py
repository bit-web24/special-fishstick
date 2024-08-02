import os
from definitions import dataset_path, model_path
import tensorflow as tf
from src.data_preprocessing import load_data
from src.model import create_model

def model_train_and_save(train_generator, validation_generator, epochs):
    model = create_model()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Save the trained model
    model.save(model_path)

    return history
