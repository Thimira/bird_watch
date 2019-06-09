import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import math
import configparser
from datetime import datetime, timedelta
import os

config = configparser.ConfigParser()
config.read('conf/application.ini')

app_config = config['training']

# get the initial epoch number from the checkpoint name
def get_init_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    filename = os.path.splitext(filename)[0]
    init_epoch = filename.split("-")[1]
    return int(init_epoch)

# dimensions of our images.
img_width = app_config.getint('img_width')
img_height = app_config.getint('img_height')

top_model_weights_path = app_config.get('top_model_weights_path')
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

final_model_path = app_config.get('final_model_path')

# number of epochs to train top model
epochs = app_config.getint('fine_tune_epochs')

# batch size used by flow_from_directory and predict_generator
batch_size = app_config.getint('fine_tune_batch_size')

# the checkpoint to load and continue from
checkpoint_to_load = "data/models/checkpoints/model-68-0.81.h5"
# get the epoch number to continue from
init_epoch = get_init_epoch(checkpoint_to_load)

# prepare data augmentation configuration
datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=40,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    validation_split=0.25)

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

print("[Info] Dataset stats: Training: {} Validation: {}".format(nb_train_samples, nb_validation_samples))
print("[Info] Number of Classes: {}".format(num_classes))
print("[Info] Class Labels: {}".format(train_generator.class_indices))

train_steps = int(math.ceil(nb_train_samples / batch_size))
validation_steps = int(math.ceil(nb_validation_samples / batch_size))

model = load_model(checkpoint_to_load)

filepath = "data/models/checkpoints/model-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callbacks_list = [checkpoint]

ft_start_time = datetime.now()
print("[Info] Model Fine-tuning started at: {}".format(ft_start_time))

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    initial_epoch=init_epoch,
    callbacks=callbacks_list)

(eval_loss, eval_accuracy) = model.evaluate_generator(
    validation_generator,
    steps=validation_steps)

ft_end_time = datetime.now()
print("[Info] Model Fine-tuning completed at: {}".format(ft_end_time))
ft_duration = ft_end_time - ft_start_time
print("[Info] Total time for Fine-tuning: {}".format(ft_duration))

print("[Info] Saving final model to disk: {}".format(final_model_path))
model.save(final_model_path)

print("\n")

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

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
# plt.show()
plt.savefig('Fine-tune-031.png', bbox_inches='tight', dpi=300)