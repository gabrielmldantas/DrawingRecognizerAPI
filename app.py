import numpy as np
import json
import pathlib
from flask import Flask, make_response, request, render_template
from PIL import Image
from cnn import CNN_PATH
from keras.models import load_model

app = Flask(__name__)

MAX_IMAGE_SIZE = 512 * 1024
CATEGORIES_FILE = pathlib.Path('resources/categories.json')
MODEL = load_model(CNN_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    image_size = request.files['image'].content_length
    if image_size > MAX_IMAGE_SIZE:
        return 'Tamanho máximo excedido', 400

    image = Image.open(request.files['image'].stream)
    if image.size != (28, 28):
        return 'As dimensões da imagem devem ser 28x28', 400
    if image.mode != 'L':
        return 'A imagem deve estar em escala de cinza', 400

    probabilities = MODEL.predict(np.array(image).reshape([1, 28, 28, 1]))
    result = _get_best_category(probabilities)
    return make_response(result)

def _load_categories():
    with open(CATEGORIES_FILE, 'r') as f:
        data = json.load(f)
    return {int(key): data[key] for key in data}

def _get_best_category(probabilities):
    probability_group = probabilities[0]
    best_probability = -1
    best_index = -1
    for index, probability in enumerate(probability_group):
        if probability > best_probability:
            best_probability = probability
            best_index = index
    return {'class': _load_categories()[best_index], 'probability': round(best_probability.item() * 100, 2)}

