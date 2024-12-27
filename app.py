from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Initialize the Flask app
app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = tf.keras.models.load_model('imagedetection_modelFor128.h5')

# Define allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocess the image to make it compatible with the model input.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))  # Adjust the size according to the model input
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    """
    Render the main HTML page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save the file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image and predict
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)

        # Assuming binary classification: 0 - Real, 1 - Synthetic
        result = 'Synthetic' if prediction[0][0] > 0.5 else 'Real'
        confidence = float(prediction[0][0]) if result == 'Synthetic' else float(1 - prediction[0][0])

        # Clean up uploaded file
        os.remove(file_path)

        return jsonify({'result': result, 'confidence': round(confidence * 100, 2)})
    else:
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files (if needed for debugging or frontend purposes).
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)