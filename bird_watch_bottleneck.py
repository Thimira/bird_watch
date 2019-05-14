'''
Train bottleneck features,
python bird_watch_bottleneck.py --bottleneck-save 1 --top-model-train 1

Evaluate,
python bird_watch_bottleneck.py --predict 1
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import os
import sys
import argparse

# Setup the argument parser to parse out command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bottleneck-save", type=int, default=-1,
                help="Save the bolleneck features")
ap.add_argument("-t", "--top-model-train", type=int, default=-1,
                help="Train the top model using the saved bottleneck features")
ap.add_argument("-p", "--predict", type=int, default=-1,
                help="Evaluate the model by running predictions the evaluation samples")
args = vars(ap.parse_args())

# dimensions of our images.
img_width, img_height = 400, 400

bottleneck_features_train_path = 'data/models/bottleneck_features_train.npy'
bottleneck_features_validation_path = 'data/models/bottleneck_features_validation.npy'

top_model_weights_path = 'data/models/bottleneck_fc_model_006.h5'

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

eval_image_path = './data/eval/'

class_indices_path = 'data/models/class_indices_006.npy'
train_data_shape_path = 'data/models/train_data_shape.npy'

# number of epochs to train top model
epochs = 100
# batch size used by flow_from_directory and predict_generator
batch_size = 64

datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=40,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    validation_split=0.1)

train_generator = datagen.flow_from_directory(
                    train_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=False,
                    interpolation='lanczos',
                    subset='training')

validation_generator = datagen.flow_from_directory(
                    train_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=False,
                    interpolation='lanczos',
                    subset='validation')

nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
num_classes = len(train_generator.class_indices)

train_labels = train_generator.classes
validation_labels = validation_generator.classes

print("[Info] Dataset stats: Training: {} Validation: {}".format(nb_train_samples, nb_validation_samples))
print("[Info] Number of Classes: {}".format(num_classes))
print("[Info] Class Labels: {}".format(train_generator.class_indices))

# save the class indices for use in the predictions
np.save(class_indices_path, train_generator.class_indices)

def save_bottlebeck_features():
    # build the InceptionV3 network
    model = InceptionV3(include_top=False, weights='imagenet')

    ### save the bottleneck features for the training data ###

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    print("[Info] Bottleneck Training Batches: {}".format(predict_size_train))

    bottleneck_features_train = model.predict_generator(
        train_generator, predict_size_train, verbose=1)

    np.save(bottleneck_features_train_path,
            bottleneck_features_train)

    ### save the bottleneck features for the validation data ###

    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

    print("[Info] Bottleneck Validation Batches: {}".format(predict_size_validation))

    bottleneck_features_validation = model.predict_generator(
        validation_generator, predict_size_validation, verbose=1)

    np.save(bottleneck_features_validation_path,
            bottleneck_features_validation)

def build_top_model(input_shape, num_classes):
    i = Input(shape=input_shape)
    x = GlobalAveragePooling2D()(i)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    pred = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=i, outputs=pred)

    return model

def train_top_model():
    # load the bottleneck features for the train data saved earlier
    train_data = np.load(bottleneck_features_train_path)

    train_data_shape = train_data.shape[1:]
    np.save(train_data_shape_path, train_data_shape)
    print("[INFO] Train data shape: {}".format(train_data_shape))

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels_cat = to_categorical(train_labels, num_classes=num_classes)

    # load the bottleneck features for the validation data saved earlier
    validation_data = np.load(bottleneck_features_validation_path)
    
    validation_labels_cat = to_categorical(validation_labels, num_classes=num_classes)

    model = build_top_model(input_shape=train_data_shape, num_classes=num_classes)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels_cat,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels_cat))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels_cat, batch_size=batch_size, verbose=1)

    print("\n")

    print("[INFO] Accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    print("\n")

    plt.style.use('ggplot')

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()


def predict():
    class_dictionary = np.load(class_indices_path).item()

    train_data_shape = np.load(train_data_shape_path)
    print("[INFO] Data shape parameter loaded: {}".format(train_data_shape))

    num_classes = len(class_dictionary)

    # build the InceptionV3 network
    model = InceptionV3(include_top=False, weights='imagenet')

    model_top = build_top_model(input_shape=train_data_shape, num_classes=num_classes)

    model_top.load_weights(top_model_weights_path)

    print("[INFO] loading and preprocessing images...")
    
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

        # get the bottleneck prediction from the pre-trained InceptionV3 model
        bottleneck_prediction = model.predict(image)

        # use the bottleneck prediction on the top model to get the final classification
        # first, get the probabilities for the prediction
        probabilities = model_top.predict(bottleneck_prediction)

        prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]

        # use the max probability to get the class predicted
        class_predicted = np.argmax(probabilities, axis=1)
        
        inID = class_predicted[0]

        # invert the class dictionary in order to get the label for the id
        inv_map = {v: k for k, v in class_dictionary.items()}
        label = inv_map[inID]

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


if args["bottleneck_save"] > 0:
    save_bottlebeck_features()

if args["top_model_train"] > 0:
    train_top_model()

if args["predict"] > 0:
    predict()


cv2.destroyAllWindows()
