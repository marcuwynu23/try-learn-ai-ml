from flask import Flask, render_template, Response
import cv2
import numpy as np
from predict import predict
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # Set the camera index (0 for the default camera)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process the frame and make predictions
            predicted_class = predict_frame(frame)
            # Display prediction on the frame
            cv2.putText(frame, f"Predicted Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def predict_frame(frame):
    try:
        # Preprocess the frame
        processed_frame = cv2.resize(frame, (224, 224))  # Resize frame to match model input size
        processed_frame = processed_frame / 255.0  # Normalize pixel values
        processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension
        # Make prediction
        predicted_class = predict(processed_frame)
        return predicted_class
    except Exception as e:
        print(f"Error predicting frame: {e}")
        return "Unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
