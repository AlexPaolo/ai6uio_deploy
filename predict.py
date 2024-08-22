import pickle

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


cliente3 = {
    'customerid': '8879-zkjort',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'yes',
    'tenure': 10,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 18,
    'totalcharges': 180
}

X = dv.transform([cliente3])

y_predict = model.predict_proba(X)[0,1]

print('input', cliente3)
print('churn probability', y_predict)
