from api.mlp import MLP
from utils.typeFormatter import formatter
from flask import Flask, request

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pickle
import pandas as pd
import os
import json


app = Flask(__name__)


# Fungsi untuk evaluasi ekspresi dari string
def evaluate_expression(expr, values):
    # Pisahkan elemen-elemen dalam key
    elements = expr.split()
    # Jika key mengandung '^2'
    if '^2' in expr:
        base = expr.replace('^2', '')
        return values[base] * values[base]
    # Jika key mengandung perkalian dua elemen
    elif len(elements) == 2:
        return values[elements[0]] * values[elements[1]]
    # Jika key adalah elemen tunggal
    else:
        return values[expr]


# Fungsi untuk normalisasi Min-Max
def min_max_normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def normalized_values(data):
    with open('./min_max_values.json', 'r') as file:
        min_max_json = json.load(file)
    
    # Lakukan normalisasi pada setiap item dalam data_to_normalize
    normalized_data = {}
    for key, value in data.items():

        min_value = min_max_json["min"][key]
        max_value = min_max_json["max"][key]
        normalized_data[key] = min_max_normalize(value, min_value, max_value)

    return normalized_data


def denorm(data):
    with open('./min_max_values.json', 'r') as file:
        min_max = json.load(file)
    min_ = min_max['min']['average_adg']
    max_ = min_max['max']['average_adg']

    return data * (max_ - min_) + min_


@app.route('/api/predict', methods=['POST'])
def predict():
    features = request.get_json()

    with open('./feature_store/mi_features.json', 'r') as file:
        mi_features = json.load(file)

    result_dict = {key: evaluate_expression(key, features) for key in mi_features}
    result_dict_norm = normalized_values(result_dict)

    input_x = list(result_dict_norm.values())

    # Muat model menggunakan torch.load
    model = MLP(36, 64, 1)
    model.load_state_dict(torch.load('base_model.pth'))
    model.eval()

    with torch.no_grad():
        input_tensor = torch.tensor([input_x], dtype=torch.float32)
        predict_normalized = model(input_tensor)

    # return predict_normalized.numpy()
    return {
        "value": denorm(predict_normalized.numpy()[0][0])
        # "value": str(predict_normalized.numpy()[0][0])
    }


if __name__=='__main__':
    app.run(debug=True)
