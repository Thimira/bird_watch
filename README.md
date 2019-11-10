# Bird Watch

A Deep Learning based Bird Image Identification System, using Keras, TensorFlow, OpenCV, and Flask.

## Introduction

The *'Bird Watch'* project, created by an amateur photographer and a machine learning enthusiast, is a solution to a simple problem faced by fellow wildlife photographers: a way to identify birds in photographs. The application is developed using Keras and TensorFlow, with Flask for the web application. InceptionV3 was used as the base model and was trained using transfer learning and fine-tuning techniques.

The live application can be found at [https://www.birdwatch.photo/](https://www.birdwatch.photo/)

## Usage

### Setup

The libraries required to run the Flask app can be installed via the following commands.

Using PIP:
```bash
pip install -r requirements.txt
```

Using Conda:
```bash
conda install numpy scipy h5py Pillow Click Flask itsdangerous Jinja2 MarkupSafe Werkzeug tensorflow
pip install keras
```

Note: You can install `tensorflow-gpu` (instead of tensorflow) if you have a CUDA capable GPU.

### Running the App

First, head over to the [Releases page](https://github.com/Thimira/bird_watch/releases/latest) and grab the latest `final_model_*.h5` and `class_indices_*.npy` files, and place them in the `models` directory.

You can then start the Flask app can be run by running,

```bash
python application.py
```

The app would by default run on `http://127.0.0.1:5000/`

### Training with your own data

In order to train with you own images, create a `data/train` directory and place your images within sub-directories for each class within the train directory (as required by the flow_from_directory function of Keras: [https://keras.io/preprocessing/image/](https://keras.io/preprocessing/image/) ). Create a `data/models` directory for the bottleneck features and the trained models to be saved. You can also create a `data/eval` directory and place few sample images there to evaluate the model after training.

Once you have the data ready, you can run,

```bash
python bird_watch_bottleneck.py --bottleneck-save 1 --top-model-train 1
```

This will save the bottleneck features and train the base model on them.

You can evaluate the base model by running,

```bash
python bird_watch_bottleneck.py --predict 1
```

Once you have the base model working, you can fine-tune it for greater accuracy by running,

```bash
python bird_watch_finetune.py
```

Note: Fine-tuning may take hours to run, even on a GPU.

Once the fine-tuning is over, you will have `final_model_*.h5` and `class_indices_*.npy` in your `data/models` directory. Copy them over to your top level `models` directory and you'll be good to go.

## Dependencies

### Runtime

The main requirements to run the Flask application are:

- TensorFlow
- Keras
- Flask

The full set of runtime dependencies are in the requirements.txt

### Training

In order to re-train the model, the following additional libs are needed:

- OpenCV
- Matplotlib
- Pillow

## Author

  - [Thimira Amaratunga](https://github.com/Thimira)
