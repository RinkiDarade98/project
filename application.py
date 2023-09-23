import numpy as np
from flask import Flask, render_template,request
import pandas as pd
import pickle
import sklearn



from xgboost import XGBRegressor


app = Flask(__name__, template_folder='templates')

model = pickle.load(open("pipe.pkl",'rb'))
car = pd.read_csv("car price predictor/cars_clean_data.csv")


@app.route('/')
def index():

    Year = sorted(car['Year'].unique(), reverse=True)
    Mileage = sorted(car['Mileage'].unique())
    Make = sorted(car['Make'].unique())
    Model = sorted(car['Model'].unique())

    Make.insert(0, 'Select Company')
    Model.insert(0, 'Select Model')

    return render_template('index.html', Year=Year, Mileage=Mileage, Make=Make, Model=Model)


@app.route('/predict', methods=['POST'])


def predict():


    Year = int(request.form.get('Year'))
    Mileage = int(request.form.get('Mileage'))
    Make = request.form.get('Make')
    Model = request.form.get('Model')

    print(Year, Mileage, Make, Model)

    prediction = model.predict(pd.DataFrame(columns=['Year', 'Mileage', 'Make', 'Model'],
                                            data=np.array([Year, Mileage, Make, Model]).reshape(1, 4)))
    print(prediction)


    return str(prediction[0])


if __name__ == '__main__':
    app.run()
