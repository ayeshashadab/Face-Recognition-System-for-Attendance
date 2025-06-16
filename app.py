from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np
from utils.test1 import test
import datetime
import time

app = Flask(__name__)

# Folder paths
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading the image
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file to the 'uploads' folder
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load image
        image = cv2.imread(file_path)

        # Run anti-spoofing detection
        label = test(image, 'resources/anti_spoof_models', device_id=0)
        
        # Display result based on the label
        if label == 1:  # Real person
            result = 'Real person detected'
        else:  # Fake (spoof) detected
            result = 'Fake (spoof) detected'

        # Render the result page with the prediction
        return render_template('result.html', filename=filename, result=result)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
