# TODO: This is to be an optimized version of the trainig script. Still the work is in progress.
# The target is to reduce the time taken for training
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from birdwatch.callbacks import TrainingMonitor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
import cv2
import os
import os.path
import sys
import configparser
import warnings

# filter out unwanted "Corrupt EXIF data" warnings from PIL
# https://github.com/python-pillow/Pillow/issues/518
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# util function to plot the training and validation history
def plot_history(history, save_fig=False, save_path=None):
    if save_path is None:
        save_path = os.path.join('data', 'models', 'training.png')
    
    plt.rcParams["figure.figsize"] = (12, 9)

    plt.style.use('ggplot')

    plt.figure(1)

    # subplot for accuracy
    plt.subplot(211)
    plt.plot(history.history['acc'])
    if 'val_acc' in history.history:
        plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # subplot for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
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

# util function to get the initial epoch number from the checkpoint name
def get_init_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    filename = os.path.splitext(filename)[0]
    init_epoch = filename.split("-")[1]
    return int(init_epoch)

# util function to calculate the class weights based on the number of samples on each class
# this is useful with datasets that are higly skewed datasets (data sets where the number of samples in each class differs vastly)
def get_class_weights(class_data_dir):
    labels_count = dict()
    for img_class in [ic for ic in os.listdir(class_data_dir) if ic[0] != '.']:
        labels_count[img_class] = len(os.listdir(os.path.join(class_data_dir, img_class)))
    total_count = sum(labels_count.values())
    class_weights = {cls: total_count / count for cls, count in 
                    enumerate(labels_count.values())}
    return class_weights

run_training = True
run_finetune = True

config = configparser.ConfigParser()
config.read('conf/application.ini')

app_config = config['training']

# dimensions of our images.
img_width = app_config.getint('img_width')
img_height = app_config.getint('img_height')

train_data_dir = os.path.join('data', 'train_f')
validation_data_dir = os.path.join('data', 'validation_f')
model_dir = app_config.get('model_dir')

class_indices_path = app_config.get('class_dictionary_path')
initial_model_path = app_config.get('initial_model_path')
final_model_path = app_config.get('final_model_path')

# number of epochs to train top model
train_epochs = app_config.getint('train_epochs')
fine_tune_epochs = app_config.getint('fine_tune_epochs')
# batch size used by flow_from_directory and fit_generator
batch_size = app_config.getint('train_batch_size')

# check which of the training steps still need to complete
if os.path.isfile(initial_model_path):
    run_training = False
    print("[Info] Initial model exists {}. Skipping training step.".format(initial_model_path))

if os.path.isfile(final_model_path):
    run_finetune = False
    print("[Info] Fine-tuned model exists {}. Skipping fine-tuning step.".format(final_model_path))

load_from_checkpoint_train = False

training_checkpoint_dir = os.path.join(model_dir, 'checkpoints', 'train')
if len(os.listdir(training_checkpoint_dir)) > 0:
    # the checkpoint to load and continue from
    training_checkpoint = os.path.join(training_checkpoint_dir, os.listdir(training_checkpoint_dir)[len(os.listdir(training_checkpoint_dir))-1])
    load_from_checkpoint_train = True

init_epoch_train = 0
if load_from_checkpoint_train:
    # get the epoch number to continue from
    print(training_checkpoint)
    init_epoch_train = get_init_epoch(training_checkpoint)
    print("[Info] Training checkpoint found for epoch {}. Will continue from that step.".format(init_epoch_train))


load_from_checkpoint_finetune = False

finetune_checkpoint_dir = os.path.join(model_dir, 'checkpoints', 'finetune')
if len(os.listdir(finetune_checkpoint_dir)) > 0:
    # the checkpoint to load and continue from
    finetune_checkpoint = os.path.join(finetune_checkpoint_dir, os.listdir(finetune_checkpoint_dir)[len(os.listdir(finetune_checkpoint_dir))-1])
    load_from_checkpoint_finetune = True

init_epoch_finetune = 0
if load_from_checkpoint_finetune:
    # get the epoch number to continue from
    init_epoch_finetune = get_init_epoch(finetune_checkpoint)
    print("[Info] Training checkpoint found for epoch {}. Will continue from that step.".format(init_epoch_finetune))

datagen_train = ImageDataGenerator(
                        rescale=1/255,
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest'
                    )

datagen_validation = ImageDataGenerator(
                        rescale=1/255
                    )

train_generator = datagen_train.flow_from_directory(
                    train_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode='categorical',
                    interpolation='lanczos')

validation_generator = datagen_validation.flow_from_directory(
                        validation_data_dir,
                        target_size=(img_width, img_height),
                        batch_size=batch_size,
                        class_mode='categorical',
                        interpolation='lanczos')

nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
num_classes = len(train_generator.class_indices)

train_steps = int(math.ceil(nb_train_samples / batch_size))
validation_steps = int(math.ceil(nb_validation_samples / batch_size))
# train_steps = nb_train_samples // batch_size
# validation_steps = nb_validation_samples // batch_size

train_labels = train_generator.classes
validation_labels = validation_generator.classes

print("[Info] Dataset stats: Training: {} Validation: {}".format(nb_train_samples, nb_validation_samples))
print("[Info] Number of Classes: {}".format(num_classes))
print("[Info] Class Labels: {}".format(train_generator.class_indices))

# save the class indices for use in the predictions
np.save(class_indices_path, train_generator.class_indices)

# get the class weights
class_weights = get_class_weights(train_data_dir)
print(class_weights)

bn_start_time = datetime.now()

if run_training:
    if load_from_checkpoint_train:
        model = load_model(training_checkpoint)
    else:
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
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc', 'top_k_categorical_accuracy'])

    bn_start_time = datetime.now()
    print("[Info] Model Bottlenecking started at: {}".format(bn_start_time))

    filepath = training_checkpoint_dir + "/model-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(patience=3)

    figPath = os.path.join(model_dir, 'checkpoints', 'progress', 'train.png')
    jsonPath = os.path.join(model_dir, 'checkpoints', 'progress', 'train.json')
    train_monitor = TrainingMonitor(figPath, jsonPath=jsonPath, startAt=init_epoch_train)
    # TypeError: Object of type float32 is not JSON serializable
    # train_monitor = TrainingMonitor(figPath)

    callbacks_list = [early_stop, reduce_lr, checkpoint, train_monitor]

    history = model.fit_generator(
                train_generator,
                steps_per_epoch=train_steps,
                epochs=train_epochs,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                class_weight=class_weights,
                initial_epoch=init_epoch_train,
                max_queue_size=15,
                workers=8,
                callbacks=callbacks_list)

    plot_history(history, save_fig=True, save_path=os.path.join('data', 'models', 'bottleneck.png'))

    print("\n")

    bn_end_time = datetime.now()
    print("[Info] Model Bottlenecking completed at: {}".format(bn_end_time))

    bn_duration = bn_end_time - bn_start_time
    print("[Info] Total time for Bottlenecking: {}".format(bn_duration))

    print(model.metrics_names)

    (eval_loss, eval_accuracy, top_k) = model.evaluate_generator(
                                    validation_generator,
                                    steps=validation_steps)

    print("\n")

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Top-5-accuracy: {:.2f}%".format(top_k * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    print("[Info] Saving initial model to disk: {}".format(initial_model_path))
    model.save(initial_model_path)
else:
    model = load_model(initial_model_path)


if run_finetune:
    # reset our data generators
    train_generator.reset()
    validation_generator.reset()

    if load_from_checkpoint_finetune:
        model = load_model(finetune_checkpoint)
    else:
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
                    metrics=['acc', 'top_k_categorical_accuracy'])

    filepath = finetune_checkpoint_dir + "/model-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.h5"
    checkpoint_ft = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

    early_stop_ft = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    figPath = os.path.join(model_dir, 'checkpoints', 'progress', 'finetune.png')
    jsonPath = os.path.join(model_dir, 'checkpoints', 'progress', 'finetune.json')
    finetune_monitor = TrainingMonitor(figPath, jsonPath=jsonPath, startAt=init_epoch_train)
    # finetune_monitor = TrainingMonitor(figPath)

    callbacks_list = [early_stop_ft, checkpoint_ft, finetune_monitor]

    ft_start_time = datetime.now()
    print("[Info] Model Fine-tuning started at: {}".format(ft_start_time))

    history = model.fit_generator(
                train_generator,
                steps_per_epoch=train_steps,
                epochs=fine_tune_epochs,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                class_weight=class_weights,
                initial_epoch=init_epoch_finetune,
                max_queue_size=15,
                workers=8,
                callbacks=callbacks_list)

    print("[Info] Saving final model to disk: {}".format(final_model_path))
    model.save(final_model_path)

    print("\n")

    plot_history(history, save_fig=True, save_path=os.path.join('data', 'models', 'finetune.png'))

    (eval_loss, eval_accuracy, top_k) = model.evaluate_generator(
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
    print("[INFO] Top-5-accuracy: {:.2f}%".format(top_k * 100))
    print("[INFO] Loss: {}".format(eval_loss))

