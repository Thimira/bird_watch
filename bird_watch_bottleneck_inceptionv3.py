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
img_width, img_height = 224, 224

bottleneck_features_train_path = 'data/models/bottleneck_features_train-X.npy'
bottleneck_features_validation_path = 'data/models/bottleneck_features_validation-X.npy'

top_model_weights_path = 'data/models/bottleneck_fc_model_InceptionV3-X.h5'

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

eval_image_path = './data/eval/'

class_indices_path = 'data/models/class_indices.npy'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def save_bottlebeck_features():
    # build the InceptionV3 network
    model = InceptionV3(include_top=False, weights='imagenet')

    ### save the bottleneck features for the training data ###

    datagen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    generator = datagen_train.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train, verbose=1)

    np.save(bottleneck_features_train_path,
            bottleneck_features_train)

    ### save the bottleneck features for the validation data ###

    datagen_validation = ImageDataGenerator(rescale=1. / 255)

    generator = datagen_validation.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation, verbose=1)

    np.save(bottleneck_features_validation_path,
            bottleneck_features_validation)

def build_top_model(input_shape, num_classes):
    i = Input(shape=input_shape)
    x = GlobalAveragePooling2D()(i)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    pred = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=i, outputs=pred)

    return model

def train_top_model():
    # use a data generator to load the labels for the training and validation data
    datagen_top = ImageDataGenerator(rescale=1./255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    num_classes = len(generator_top.class_indices)

    # save the class indices for use in the predictions
    np.save(class_indices_path, generator_top.class_indices)

    # load the bottleneck features for the train data saved earlier
    train_data = np.load(bottleneck_features_train_path)

    # get the class labels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    # load the bottleneck features for the validation data saved earlier
    validation_data = np.load(bottleneck_features_validation_path)

    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    model = build_top_model(input_shape=train_data.shape[1:], num_classes=num_classes)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("\n")

    print("[INFO] Accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    print("\n")

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def predict():
    class_dictionary = np.load(class_indices_path).item()

    train_data_orig = np.load(bottleneck_features_train_path)

    num_classes = len(class_dictionary)

    # build the InceptionV3 network
    model = InceptionV3(include_top=False, weights='imagenet')

    model_top = build_top_model(input_shape=train_data_orig.shape[1:], num_classes=num_classes)

    model_top.load_weights(top_model_weights_path)

    print("[INFO] loading and preprocessing images...")
    
    image_paths = [os.path.join(eval_image_path, f) for f in os.listdir(eval_image_path)]

    for image_path in image_paths:
        orig = cv2.imread(image_path)
        image = load_img(image_path, target_size=(img_width, img_height))
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
