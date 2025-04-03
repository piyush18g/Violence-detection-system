from __future__ import print_function
import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from flask import Flask, render_template, request, jsonify, Response
from keras.models import load_model
import logging
from collections import deque
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure uploads directory exists
IMAGE_HEIGHT = 64          
IMAGE_WIDTH = 64           
SEQUENCE_LENGTH = 16       
CLASSES_LIST = ["Non-violent","Violent"] 

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}  # Extend as needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your pre-trained ML model from model.h5
model = load_model(r'C:\Users\HP\projects\ML\ML_Project\public\model.h5')

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_video(video_file_path, SEQUENCE_LENGTH):
    """
    Processes the video file and predicts whether it is violent or non-violent.
    Uses OpenCV to extract frames, preprocesses them, and uses the loaded model.
    """
    video_reader = cv2.VideoCapture(video_file_path)

    # Get original video dimensions (if needed).
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # List to store preprocessed frames.
    frames_list = []

    # Get total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval for frame sampling.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Loop over the number of frames defined by SEQUENCE_LENGTH.
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the video position.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break

        # Resize frame to the expected dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Normalize the frame.
        normalized_frame = resized_frame / 255.0
        # Append to list.
        frames_list.append(normalized_frame)

    # Expand dimensions to add a batch dimension and get prediction probabilities.
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
    # Find the index of the highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
    # Get the class name from CLASSES_LIST.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Print prediction details.
    confidence = predicted_labels_probabilities[predicted_label]
    print(f'Predicted: {predicted_class_name}\nConfidence: {confidence}')

    video_reader.release()
    conf = float("{:.2f}".format(confidence*100))
    return predicted_class_name, conf

def real():
    cap = cv2.VideoCapture(0)
    SEQUENCE_LENGTH = 16
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)
    while True:
        ret , frame = cap.read()
        if not ret:  # Check if frame was read successfully
            print("Error reading frame. Exiting loop.")
            break  
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            # Normalize the resized frame.
        normalized_frame = resized_frame / 255
            # Passing the  pre-processed frames to the model and get the predicted probabilities.
        frames_queue.append(normalized_frame)
            # We Need at Least number of SEQUENCE_LENGTH Frames to perform a prediction.
            # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis = 0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
            if predicted_class_name == "Violent":
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 2, 255), 5)
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 2), 5)
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        #cv2.imshow('Kya aap hinsak ho? ', frame)
        _, buffer = cv2.imencode('.jpg', frame)  # Convert frame to JPEG format
        frame_bytes = buffer.tobytes()  # Convert JPEG image to bytes

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def home():
    print("Hello, Flask!", flush=True)
    app.logger.info('try yaar')
    return render_template('index.html')

@app.route('/video', methods=['GET'])
def vid():
    return render_template('video.html')

@app.route('/real_time', methods=['GET'])
def web_p():
        return render_template('real.html')
    
@app.route('/show_real', methods=['GET'])
def time_r():
    return Response(real(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/final_result', methods=['POST'])
def predict_video_route():
    # Check if the request contains a video file
    if 'video' not in request.files:
        return render_template('video.html', 
                               prediction="No video file provided", 
                               confidence=0.0)
    
    file = request.files['video']
    
    if file.filename == '':
        return render_template('video.html', 
                               prediction="No file selected", 
                               confidence=0.0)

    if file and allowed_file(file.filename):
       # Secure and save the file.
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call the predict_video function.
        predicted_class, confidence = predict_video(file_path, SEQUENCE_LENGTH)
        # Remove the file after processing.
        os.remove(file_path)
        return render_template('video.html', 
                               prediction=predicted_class, 
                               confidence=confidence)
    else:
        return render_template('video.html', 
                               prediction="Invalid file type", 
                               confidence=0.0)


if __name__ == '__main__':
    app.run(debug=True)
