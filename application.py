from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from PIL import Image
from io import BytesIO
import os
import sys
import base64


# https://github.com/tensorflow/tensorflow/issues/24828
from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# dimensions of our images.
img_width, img_height = 224, 224

final_model_path ='models/final_model_InceptionV3.h5'

class_dictionary = np.load('models/class_indices.npy').item()

global model, graph
graph = tf.get_default_graph()
model = load_model(final_model_path)

def classify_image(image):
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255.0

    # add a new axis to make the image array confirm with
    # the (samples, height, width, depth) structure
    image = np.expand_dims(image, axis=0)

    # get the probabilities for the prediction
    with graph.as_default():
        probabilities = model.predict(image)

    prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]

    class_predicted = np.argmax(probabilities, axis=1)

    inID = class_predicted[0]

    # invert the class dictionary in order to get the label for the id
    inv_map = {v: k for k, v in class_dictionary.items()}
    label = inv_map[inID]

    print("Predicted: {}, Confidence: {}".format(label, prediction_probability))

    return label, prediction_probability

def get_iamge_thumbnail(image):
    image = image.convert("RGB")
    with BytesIO() as buffer:
        image.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def index():
    if request.method == 'GET':
        with application.app_context():
            return render_template('index.html')

    if request.method == 'POST':
        f = request.files['bird_image']
        sec_filename = secure_filename(f.filename)
        sec_filename = sec_filename.replace(" ", "_")
        image_path = './uploads/' + sec_filename
        f.save(image_path)

        image = load_img(image_path, target_size=(img_width, img_height), interpolation='lanczos')

        label, prediction_probability = classify_image(image=image)

        image_data = get_iamge_thumbnail(image=image)

        os.remove(image_path)

        with application.app_context():
            return render_template('index.html', label=label, prob=prediction_probability, image=image_data)

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# add a rule for the index page.
application.add_url_rule('/', 'index', index, methods=['GET', 'POST'])

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()