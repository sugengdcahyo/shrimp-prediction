from utils.typeFormatter import formatter
from flask import Flask, request
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

@app.route('/api/predict', methods=['GET'])
def predict():

    result = {
        "predict": [x for x in range(100)]
    }

    return result


if __name__=='__main__':
    app.run(debug=True)
