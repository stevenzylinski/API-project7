from flask import Flask, request, redirect, url_for, flash, jsonify
import json, pickle
import pandas as pd
import numpy as np
import shap

app = Flask(__name__)

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
    model = pickle.load(open('model/modelv3.pkl','rb'))
    explainer = pickle.load(open('model/explainerv2.pkl','rb'))
    def sigmoid_function(x):
        y = np.exp(x)/(1+np.exp(x))
        return(y)
    def convert_sv(sv):
        """
        This function take a shap.Explanation object in input and convert all values to plot values according
        to the probability predicted.
        """
    
        # Total contribution of every feature
        total_values = [sum(x) for x in sv.values]
    
        # Base value predicted convert to probability
        proba_base = sigmoid_function([x for x in sv.base_values])
    
        # Final value predicted convert to probability
        proba_predict = sigmoid_function([x for x in sv.base_values+total_values])
    
        # Converting each contribution of feature regarding probability
        contrib = np.ndarray(shape = sv.values.shape)
        for x in range(0,len(sv.values)):
            contrib[x] = [(value/sum((sv.values[x]))*(proba_predict[x]-proba_base[x])) for value in sv.values[x]]
    
        sv.values = contrib
        sv.base_values = proba_base
    
        return(sv)
    print("Loading OK")
    app.run(debug = True)