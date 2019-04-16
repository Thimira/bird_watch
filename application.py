from flask import Flask
from flask import request
from werkzeug.utils import secure_filename

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import os
import sys

# https://github.com/tensorflow/tensorflow/issues/24828
from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# dimensions of our images.
img_width, img_height = 224, 224

final_model_path ='data/models/final_model_InceptionV3.h5'

class_dictionary = np.load('data/models/class_indices.npy').item()

global model, graph
graph = tf.get_default_graph()
model = load_model(final_model_path)

# print a nice greeting.
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username

def run_pred():
    # print(final_model_path)
    # model = load_model(final_model_path)
    if request.method == 'POST':
        f = request.files['bird_image']
        sec_filename = secure_filename(f.filename)
        image_path = './uploads/' + sec_filename
        f.save(image_path)

        image = load_img(image_path, target_size=(img_width, img_height))
        image = img_to_array(image)

        # important! otherwise the predictions will be '0'
        image = image / 255.0

        # add a new axis to make the image array confirm with
        # the (samples, height, width, depth) structure
        image = np.expand_dims(image, axis=0)

        # get the bottleneck prediction from the pre-trained VGG16 model
        # bottleneck_prediction = model.predict(image)

        # use the bottleneck prediction on the top model to get the final classification
        # class_predicted = model.predict_classes(image)

        # get the probabilities for the prediction
        with graph.as_default():
            probabilities = model.predict(image)

        prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]

        class_predicted = np.argmax(probabilities, axis=1)

        inID = class_predicted[0]

        # invert the class dictionary in order to get the label for the id
        inv_map = {v: k for k, v in class_dictionary.items()}
        label = inv_map[inID]

        print(label)
        print(prediction_probability)

    return label

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>Bird Watch</title> </head>\n<body>'''
instructions = '''
        <p>Select image to upload:</p>\n
        <form action="/run" method="post" enctype="multipart/form-data">
        <input type="file" name="bird_image" id="bird_image">
        <input type="submit" value="Upload Image" name="submit">
        </form>'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# add a rule for the index page.
application.add_url_rule('/', 'index', (lambda: header_text +
    say_hello() + instructions + footer_text))

# add a rule when the page is accessed with a name appended to the site
# URL.
application.add_url_rule('/<username>', 'hello', (lambda username:
    header_text + say_hello(username) + home_link + footer_text))

application.add_url_rule('/run', 'run', run_pred, methods=['GET', 'POST'])

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()