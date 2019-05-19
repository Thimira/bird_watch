from flask import Flask, request, render_template, url_for, make_response, send_from_directory, flash, redirect, jsonify
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
import time
from datetime import datetime, timedelta
import configparser
import boto3
from decimal import Decimal

config = configparser.ConfigParser()
config.read('conf/application.ini')

app_config = config['default']

# https://github.com/tensorflow/tensorflow/issues/24828
from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# Get the DynamoDB service resource.
dynamodb = boto3.resource('dynamodb')
dynamoDBClient = boto3.client('dynamodb')

predictions_log = dynamodb.Table('birdwatch_predictions_log')
customer_feedback_tbl = dynamodb.Table('birdwatch_customer_feedback')

'''The domain name we will be using for our website. This will be used for SEO'''
site_domain = app_config.get('site_domain')

# dimensions of our images.
img_width = app_config.getint('img_width')
img_height = app_config.getint('img_height')

final_model_path = app_config.get('final_model_path')

class_dictionary = np.load(app_config.get('class_dictionary_path')).item()

# Google analytics property ID
analytics_id = app_config.get('analytics_id')

global model, graph
graph = tf.get_default_graph()
model = load_model(final_model_path)

ALLOWED_FILETYPES = set(['.jpg',' .jpeg', '.gif', '.png'])

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
            return render_template('index.html', analytics_id=analytics_id)

    if request.method == 'POST':
        # check if the post request has the file part
        if 'bird_image' not in request.files:
            print("[Error] No file uploaded.")
            flash('No file uploaded.')
            return redirect(url_for('index'))
        
        f = request.files['bird_image']

        # if user does not select file, browser also
        # submit an empty part without filename
        if f.filename == '':
            print("[Error] No file selected to upload.")
            flash('No file selected to upload.')
            return redirect(url_for('index'))

        sec_filename = secure_filename(f.filename)
        file_extension = os.path.splitext(sec_filename)[1]

        if f and file_extension.lower() in ALLOWED_FILETYPES:
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

            prediction_id = log_prediction(prediction_label=label, prediction_confidence=prediction_probability)

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
                                        height=orig_height,
                                        analytics_id=analytics_id,
                                        prediction_id=prediction_id
                                        )
        else:
            print("[Error] Unauthorized file extension: {}".format(file_extension))
            flash("The file type you selected is not supported. Please select a '.jpg', '.jpeg', '.gif', or a '.png' file.")
            return redirect(url_for('index'))

def log_prediction(prediction_label, prediction_confidence):
    prediction_id = str(uuid.uuid4())
    timestamp = int(time.time())
    prediction_confidence = Decimal(str(prediction_confidence))
    predictions_log.put_item(
        Item={
            'prediction_id': prediction_id,
            'timestamp': timestamp,
            'prediction_label': prediction_label,
            'prediction_confidence': prediction_confidence,
            'correctness': -1,
        }
    )
    
    item_count = predictions_log.item_count
    print("[Info] Item count: {}".format(item_count))
    # response = dynamoDBClient.describe_table(TableName='birdwatch_predictions_log')
    # print(response['Table']['ItemCount'])
    return prediction_id

def set_correctness():
    req_json = request.get_json()
    prediction_id = req_json.get('prediction_id')
    correctness = req_json.get('correctness')

    if (req_json and prediction_id and correctness):
        try:
            correctness = int(correctness)
            update_correctness(prediction_id=prediction_id, correctness=correctness)
        except Exception as e:
            print("[Error] Error updating correctness: {}".format(e))

    return jsonify(success=True)

def update_correctness(prediction_id, correctness):
    predictions_log.update_item(
        Key={
            'prediction_id': prediction_id
        },
        UpdateExpression='SET correctness = :val1',
        ExpressionAttributeValues={
            ':val1': correctness
        }
    )

def customer_feedback():
    req_json = request.get_json()
    feedback_id = str(uuid.uuid4())
    timestamp = int(time.time())

    feedback = req_json.get('feedback')
    rating = req_json.get('rating')

    if (req_json and feedback and rating):
        try:
            rating = int(rating)

            customer_feedback_tbl.put_item(
                Item={
                    'feedback_id': feedback_id,
                    'timestamp': timestamp,
                    'feedback': feedback,
                    'rating': rating,
                }
            )
        except Exception as e:
            print("[Error] Error setting feedback: {}".format(e))

    return jsonify(success=True)

def about():
    return render_template('about.html', analytics_id=analytics_id)

def howitworks():
    return render_template('howitworks.html', analytics_id=analytics_id)

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

def http_413(e):
    print("[Error] Uploaded file too large.")
    flash('Uploaded file too large.')
    return redirect(url_for('index'))


# EB looks for an 'application' callable by default.
application = Flask(__name__)
application.secret_key = app_config.get('application_secret')

# add a rule for the index page.
application.add_url_rule('/', 'index', index, methods=['GET', 'POST'])

# AJAX routes
application.add_url_rule('/correctness', 'correctness', set_correctness, methods=['POST'])
application.add_url_rule('/feedback', 'feedback', customer_feedback, methods=['POST'])


application.add_url_rule('/about', 'about', about, methods=['GET'])
application.add_url_rule('/howitworks', 'howitworks', howitworks, methods=['GET'])

application.add_url_rule('/sitemap.xml', 'sitemap.xml', sitemap, methods=['GET'])
application.add_url_rule('/robots.txt', 'robots.txt', robots, methods=['GET'])

application.register_error_handler(413, http_413)
application.config['MAX_CONTENT_LENGTH'] = app_config.getint('max_upload_size') * 1024 * 1024

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = app_config.getboolean('debug')
    application.run()