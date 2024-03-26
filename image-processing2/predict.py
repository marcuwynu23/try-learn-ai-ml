from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

if len(sys.argv) != 2:
      print("Usage: python predict.py <path_to_your_image.jpg>")
      sys.exit(1)

# Load the trained model
model = load_model('model1.h5')

# Preprocess input image
img_path = str(sys.argv[1])  # Replace 'path_to_your_image.jpg' with the path to your input image
img = image.load_img(img_path, target_size=(64, 64))  # Resize image to match model input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize pixel values

# Make prediction
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("Predicted class: Cat")
else:
    print("Predicted class: Dog")
