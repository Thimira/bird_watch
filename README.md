# Bird Watch

A Deep Learning based Bird Image Identification System, using Keras, TensorFlow, OpenCV, and Flask.

## Introduction

The *'Bird Watch'* project, created by an amateur photographer and a machine learning enthusiast, is a solution to a simple problem faced by fellow wildlife photographers: a way to identify birds in photographs. The application is developed using Keras and TensorFlow, with Flask for the web application. InceptionV3 was used as the base model and was trained using transfer learning and fine-tuning techniques. 

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
pip install tensorflow
```

### Running the App

The Flask app can be run by:

```bash
python application.py
```

The app would by default run on `http://127.0.0.1:5000/`

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