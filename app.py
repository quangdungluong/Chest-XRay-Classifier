"""Flask app"""
import os
import pandas as pd

import torch
from flask import Flask, render_template, request, send_from_directory

from config import CFG
from models import create_model
from utils import process_image, predict

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
UPLOAD_FOLDER = 'uploads'

# Create model
model = create_model("efficientnet_b2", num_classes=CFG.num_classes)
model.eval()

def allowed_file(filename):
    """Check allow file"""
    return filename.split('.')[-1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    """Home"""
    return render_template('index.html', label='Hiiii', imagesource='./static/img/b.png', returnJson={})


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Upload and process file"""
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image = process_image(image_path=file_path, display=False)
            output, probs = predict(model, image)
        ## Get Ground truth label
        return_json = {}
        for i, prob in enumerate(probs):
            return_json[CFG.labels_map[i]] = round(prob*100,3)
        return_json = dict(sorted(return_json.items(), key=lambda item: item[1], reverse=True))
    return render_template('index.html', label=CFG.labels_map[output], imagesource=file_path, returnJson=return_json)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Save uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
