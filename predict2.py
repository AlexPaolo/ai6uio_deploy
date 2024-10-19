import pickle
from flask import Flask
from flask import request
from flask import jsonify
 
model_file = 'model_C=0.9.bin'

with open(model_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

#X = dv.transform([cliente3])
app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    #Tomo la predicción de que sea 1
    y_predict = model.predict_proba(X)[0,1]
    churn = y_predict >= 0.5


    result = {
        'probabilidad_de_abandono': float(y_predict),
        'abandono': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)
