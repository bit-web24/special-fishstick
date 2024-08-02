from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
from definitions import dataset_path, model_path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def load_image(image_path, target_size=(64, 64)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

def predict(image_path):
    validation_image = load_image(image_path)

    model = tf.keras.models.load_model(model_path)
    class_names = ['cat', 'dog']

    predictions = model.predict(validation_image)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    predicted_label = class_names[predicted_labels[0]]

    return predicted_label

def home(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_file_url = fs.url(filename)
            uploaded_file_path = fs.path(filename)
            predicted_label = predict(uploaded_file_path)

            return render(request, 'prediction.html', {
                'image': {
                    'name': filename,
                    'url': uploaded_file_url,
                    },
                'predicted_label': predicted_label,
            })
    else:
        form = ImageUploadForm()

    return render(request, 'home.html', {'form': form})
