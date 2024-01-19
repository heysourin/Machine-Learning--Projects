from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

car = pd.read_csv('clean_car.csv')
model = pickle.load(open('LinearRegressionModell.pkl', "rb"))


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())  # Years reversely sorted
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())

    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_type=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    car_model = request.form['model']
    year = int(request.form['year'])
    fuel_type = request.form['fuel_type']
    kms_driven = int(request.form['kilo_driven'])
    print(company, car_model, year, fuel_type, kms_driven)

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=[
                               'name', 'company', 'year', 'kms_driven', 'fuel_type']))
    print("hehe", prediction)

    # Convert the NumPy array to a standard Python type (float or int)
    # result = str(prediction[0])

    # return jsonify({'prediction': prediction})

    # return result
    prediction_list = prediction.tolist()

    return jsonify({'prediction': prediction_list})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
