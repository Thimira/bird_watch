from flask import Flask, request, render_template, url_for, make_response, send_from_directory
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
import uuid
from datetime import datetime, timedelta
import configparser

config = configparser.ConfigParser()
config.read('conf/application.ini')

app_config = config['default']

# https://github.com/tensorflow/tensorflow/issues/24828
from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

'''The domain name we will be using for our website. This will be used for SEO'''
site_domain = app_config.get('site_domain')

# dimensions of our images.
img_width = app_config.getint('img_width')
img_height = app_config.getint('img_height')

final_model_path = app_config.get('final_model_path')

class_dictionary = np.load(app_config.get('class_dictionary_path')).item()

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

    print("[Info] Predicted: {}, Confidence: {}".format(label, prediction_probability))

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
        file_extension = os.path.splitext(sec_filename)[1]
        file_tempname = uuid.uuid4().hex
        image_path = './uploads/' + file_tempname + file_extension
        f.save(image_path)
        file_size = os.path.getsize(image_path)

        file_size_str = str(file_size) + " bytes"
        if (file_size >= 1024):
            if (file_size >= 1024 * 1024):
                file_size_str = str(file_size // (1024 * 1024)) + " MB"
            else:
                file_size_str = str(file_size // 1024) + " KB"

        image = load_img(image_path, target_size=(img_width, img_height), interpolation='lanczos')

        orig_width, orig_height = Image.open(image_path).size

        label, prediction_probability = classify_image(image=image)
        prediction_probability = np.around(prediction_probability * 100, decimals=4)

        image_data = get_iamge_thumbnail(image=image)

        os.remove(image_path)

        with application.app_context():
            return render_template('index.html', 
                                    label=label, 
                                    prob=prediction_probability, 
                                    image=image_data,
                                    file_name=sec_filename,
                                    file_size=file_size_str,
                                    width=orig_width,
                                    height=orig_height
                                    )

def about():
    return render_template('about.html')

def howitworks():
    return render_template('howitworks.html')

def sitemap():
    try:
        """Generate sitemap.xml. Makes a list of urls and date modified."""
        pages=[]
        one_week = (datetime.now() - timedelta(days=7)).date().isoformat()
        # static pages
        for rule in application.url_map.iter_rules():
            if "GET" in rule.methods and len(rule.arguments)==0:
                pages.append(
                            [site_domain + str(rule.rule), one_week]
                            )

        sitemap_xml = render_template('sitemap_template.xml', pages=pages)
        response = make_response(sitemap_xml)
        response.headers["Content-Type"] = "application/xml"    

        return response
    except Exception as e:
        return(str(e))

def robots():
    return send_from_directory(application.static_folder, 'robots.txt')

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# add a rule for the index page.
application.add_url_rule('/', 'index', index, methods=['GET', 'POST'])

application.add_url_rule('/about', 'about', about, methods=['GET'])
application.add_url_rule('/howitworks', 'howitworks', howitworks, methods=['GET'])

application.add_url_rule('/sitemap.xml', 'sitemap.xml', sitemap, methods=['GET'])
application.add_url_rule('/robots.txt', 'robots.txt', robots, methods=['GET'])

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = app_config.getboolean('debug')
    application.run()