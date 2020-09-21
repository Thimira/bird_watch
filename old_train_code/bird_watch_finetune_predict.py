import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
from keras import optimizers
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import os
import sys
import configparser

config = configparser.ConfigParser()
config.read('conf/application.ini')

app_config = config['training']

# dimensions of our images.
img_width = app_config.getint('img_width')
img_height = app_config.getint('img_height')

final_model_path = app_config.get('final_model_path')
class_indices_path = app_config.get('class_dictionary_path')

def get_top_predictions(preds, class_map, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_map[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results[0]

def predict():
    class_dictionary = np.load(class_indices_path).item()

    model = load_model(final_model_path)

    print("[INFO] loading and preprocessing image...")
    eval_image_path = './data/eval/'

    image_paths = [os.path.join(eval_image_path, f) for f in os.listdir(eval_image_path)]

    for image_path in image_paths:
        orig = cv2.imread(image_path)
        image = load_img(image_path, target_size=(img_width, img_height), interpolation='lanczos')
        image = img_to_array(image)

        # important! otherwise the predictions will be '0'
        image = image / 255.0

        # add a new axis to make the image array confirm with
        # the (samples, height, width, depth) structure
        image = np.expand_dims(image, axis=0)

        # get the probabilities for the prediction
        probabilities = model.predict(image)

        prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]

        class_predicted = np.argmax(probabilities, axis=1)

        inID = class_predicted[0]

        # invert the class dictionary in order to get the label for the id
        inv_map = {v: k for k, v in class_dictionary.items()}
        label = inv_map[inID]

        results = get_top_predictions(probabilities, class_map=inv_map, top=5)

        print("\n")

        print("[Info] Top 5 Predictions: {}".format(results))

        image_filename = os.path.basename(image_path)
        image_filename = os.path.splitext(image_filename)[0]
        image_filename = image_filename.replace("_", " ")

        # display the prediction in the console
        print("Image ID: {}, Label: {}, Confidence: {}, Actual: {}".format(
            inID, label, prediction_probability, image_filename))

        # Resize the display image
        resized_img = cv2.resize(orig, (600, 600), interpolation = cv2.INTER_CUBIC)

        # display the prediction in OpenCV window
        cv2.putText(resized_img, "Predicted: {}".format(label), (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (43, 99, 255), 2, cv2.LINE_AA)

        cv2.imshow("Classification", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

predict()

cv2.destroyAllWindows()