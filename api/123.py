from .mlp import MLP
from utils.typeFormatter import formatter
from flask import Flask
from flask_restful import request
import pandas as pd
import os


app = Flask(__name__)


@app.route('/api/adg-calculate', methods=['GET'])
def ResourceADGCalculate():
    # ADG = (berat_akhir - berat_awal) / n_cycle * 100
    cycle_id = request.args.get('cycle_id', None)
    if not cycle_id:
        return {"error": "No Cycle ID"}, 400

    try:
        cycle_id = formatter(cycle_id)
    except BaseException as e:
        return {"error": str(e)}

    df = pd.read_csv(f"{os.getcwd()}/datasets/samplings.csv")
    df = df.sort_values('sampled_at')
    df['sampled_at'] = pd.to_datetime(df['sampled_at'])

    sample = df[df['cycle_id'] == cycle_id]
    if sample.empty:
        return {"message": "Cycle ID not found"}, 404

    w_diff = sample['average_weight'].diff()
    d_diff = sample['sampled_at'].diff().dt.days

    adg = w_diff / d_diff

    return {
        "result": round(adg.mean(), 2) * 100,
        "unit": "percent"
    }


@app.route('/api/sr-calculate', methods=['POST'])
def SRCalculate():
    # SR = (n_akhir - n_awal) * 100
    result = .4
    return {
        "result": result
    }


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



@app.route('/api/predict', methods=['POST'])
def predict():
    features = request.get_json()
    
    return features


if __name__=='__main__':
    app.run(debug=True)
