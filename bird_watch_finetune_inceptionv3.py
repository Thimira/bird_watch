import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'data/models/bottleneck_fc_model_InceptionV3.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

final_model_path ='data/models/final_model_InceptionV3.h5'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

nb_train_samples = len(train_generator.filenames)

num_classes = len(train_generator.class_indices)

train_steps = int(math.ceil(nb_train_samples / batch_size))

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

nb_validation_samples = len(validation_generator.filenames)

validation_steps = int(math.ceil(nb_validation_samples / batch_size))

input_tensor = Input(shape=(img_width, img_height, 3))

# build the InceptionV3 network
base_model = InceptionV3(weights='imagenet', include_top= False, input_tensor=input_tensor)
print('Model loaded.')

i = Input(shape=base_model.output_shape[1:])
x = GlobalAveragePooling2D()(i)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(num_classes, activation='softmax')(x)
top_model = Model(inputs=i, outputs=pred)

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 280 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:280]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

(eval_loss, eval_accuracy) = model.evaluate_generator(
    validation_generator,
    steps=validation_steps)

model.save(final_model_path)

print("\n")

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

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
