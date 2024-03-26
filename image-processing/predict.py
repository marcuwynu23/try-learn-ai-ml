 
import tensorflow as tf
import numpy as np
import sys
import os
import cv2

# Load the trained model
model = tf.keras.models.load_model('model.h5')  # Update with the path to your saved model

# Define the classes (you may need to adjust this based on your dataset)
# read the class names from the file animal.names it is integer labels converted to class names
with open("animal.names", "r") as f:
		content = f.read().strip().split("\n")
		names = {}
		classes = []
		for line in content:
				label, name = line.split()
				names[int(label)] = name
				classes.append(int(label))


# Function to preprocess the input image
def preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image
    image = cv2.resize(image, target_size)
    # Normalize the pixel values
    image = image / 255.0
    # Expand the dimensions to match the input shape expected by the model
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image_path):
    # Preprocess the input image
    processed_image = preprocess_image(image_path)
    # Make predictions
    predictions = model.predict(processed_image)
    # Get the predicted class label
    predicted_class = np.argmax(predictions[0])
    # Get the class name from the classes list
    class_name = classes[predicted_class]
    return class_name

if __name__ == "__main__":
    # Check if an image file is provided as argument
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_file>")
        sys.exit(1)

    # Get the image file path from command line argument
    image_file = sys.argv[1]

    # Check if the image file exists
    if not os.path.isfile(image_file):
        print("Error: Image file not found.")
        sys.exit(1)

    # Make prediction
    predicted_class = predict(image_file)
    print("Predicted Class:", predicted_class)
    print("Predicted Class Name:", names[predicted_class])
    
