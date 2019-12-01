# TODO: This is to be an optimized version of the trainig script. Still the work is in progress.
# The target is to reduce the time taken for training

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
import cv2
import os
import sys
import configparser
import warnings

# filter out unwanted "Corrupt EXIF data" warnings from PIL
# https://github.com/python-pillow/Pillow/issues/518
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

config = configparser.ConfigParser()
config.read('conf/application.ini')

app_config = config['training']

# dimensions of our images.
img_width = app_config.getint('img_width')
img_height = app_config.getint('img_height')

train_data_dir = 'data/train'

class_indices_path = app_config.get('class_dictionary_path')
initial_model_path = app_config.get('initial_model_path')
final_model_path = app_config.get('final_model_path')

# number of epochs to train top model
train_epochs = app_config.getint('train_epochs')
fine_tune_epochs = app_config.getint('fine_tune_epochs')
# batch size used by flow_from_directory and fit_generator
batch_size = app_config.getint('train_batch_size')

datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=40,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    validation_split=0.25)

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for the data augmentation object
imagenet_mean = np.array([123.68, 116.779, 103.939], dtype="float32")
datagen.mean = imagenet_mean

train_generator = datagen.flow_from_directory(
                    train_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode='categorical',
                    interpolation='lanczos',
                    subset='training')

validation_generator = datagen.flow_from_directory(
                    train_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode='categorical',
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

def plot_history(history, save_fig=False, save_path='data/models-new/training.png'):
    plt.rcParams["figure.figsize"] = (12, 9)

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
    
    if save_fig:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    # clear and close the current figure
    plt.clf()
    plt.close()

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_width, img_height, 3)))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_steps = int(math.ceil(nb_train_samples / batch_size))
validation_steps = int(math.ceil(nb_validation_samples / batch_size))

bn_start_time = datetime.now()
print("[Info] Model Bottlenecking started at: {}".format(bn_start_time))

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# history = model.fit_generator(
#             train_generator,
#             steps_per_epoch=train_steps,
#             epochs=train_epochs,
#             validation_data=validation_generator,
#             validation_steps=validation_steps,
#             callbacks=[early_stop])

history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=train_epochs,
            callbacks=[early_stop])

plot_history(history, save_fig=True, save_path='data/models-new/bottleneck.png')

print("\n")

bn_end_time = datetime.now()
print("[Info] Model Bottlenecking completed at: {}".format(bn_end_time))

bn_duration = bn_end_time - bn_start_time
print("[Info] Total time for Bottlenecking: {}".format(bn_duration))

(eval_loss, eval_accuracy) = model.evaluate_generator(
                                validation_generator,
                                steps=validation_steps)

print("\n")

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

print("[Info] Saving initial model to disk: {}".format(initial_model_path))
model.save(initial_model_path)

# reset our data generators
train_generator.reset()
validation_generator.reset()

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True
    
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

filepath="data/models-new/checkpoints/model-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

early_stop_ft = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

callbacks_list = [early_stop_ft, checkpoint]

ft_start_time = datetime.now()
print("[Info] Model Fine-tuning started at: {}".format(ft_start_time))

history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=fine_tune_epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks_list)

print("[Info] Saving final model to disk: {}".format(final_model_path))
model.save(final_model_path)

print("\n")

plot_history(history, save_fig=True, save_path='data/models-new/finetune.png')

(eval_loss, eval_accuracy) = model.evaluate_generator(
                                validation_generator,
                                steps=validation_steps)

print("\n")

ft_end_time = datetime.now()
print("[Info] Model Fine-tuning completed at: {}".format(ft_end_time))
ft_duration = ft_end_time - ft_start_time
print("[Info] Total time for Fine-tuning: {}".format(ft_duration))

tot_duration = ft_end_time - bn_start_time
print("[Info] Total time for training: {}".format(tot_duration))

print("\n")

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

