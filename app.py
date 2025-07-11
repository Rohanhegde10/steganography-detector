from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('model/stego_cnn.h5')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

IMG_SIZE = 64

def prepare_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = prepare_image(filepath)
            prediction = model.predict(img)[0]

            result = "Stego Image Detected" if np.argmax(prediction) == 1 else "Cover (Clean) Image"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)