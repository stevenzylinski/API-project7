from flask import Flask, request, redirect, url_for, flash, jsonify
import json, pickle
import pandas as pd
import numpy as np
import shap
from utils import *

app = Flask(__name__)
model = pickle.load(open('./model/modelv3.pkl','rb'))
explainer = pickle.load(open('./model/explainerv2.pkl','rb'))

@app.route('/api/makecalc/', methods=['POST'])

def makepred():
    jsonfile = request.get_json()
    data = pd.read_json(json.dumps(jsonfile),orient = 'index').fillna(value=np.nan)
    ypred = model.predict_proba(data)
    return(jsonify(ypred.tolist()))

@app.route('/api/shap_imp/', methods = ['POST'])

def makeimp():
    jsonfile = request.get_json()
    data = pd.read_json(json.dumps(jsonfile),orient = 'index').fillna(value=np.nan)
    object_columns = data.select_dtypes('object').columns.to_list()
    numeric_columns = data.select_dtypes(exclude = 'object').columns.to_list()
    transformer = model['preprocess']
    dff_transform = transformer.transform(data)
    dff = data[object_columns+numeric_columns]
    sv = convert_sv(explainer(dff_transform))
    exp = shap.Explanation(sv.values[:],
                          sv.base_values[:],
                          data = dff,
                          feature_names = dff.columns)
    return(jsonify(Value = exp.values.tolist(),Base_Value = exp.base_values.tolist()))




if __name__ == '__main__':
    app.run(debug = True)