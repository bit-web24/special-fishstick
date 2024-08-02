import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def visualize_predictions(model, validation_generator, class_names, num_images=5):
    validation_batch = next(validation_generator)
    images, true_labels = validation_batch

    predictions = model.predict(images)
    predicted_labels = (predictions > 0.5).astype(int)

    # Set up the figure
    plt.figure(figsize=(15, 15))

    # Display images with their predictions
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {class_names[int(true_labels[i])]}\nPredicted: {class_names[int(predicted_labels[i])]}")
        plt.axis('off')

    plt.show()
