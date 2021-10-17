from flask import Flask, request
from flask_restx import Api, Resource, fields
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

app = Flask(__name__)

app_x = Api(app = app, 
		  version = "1.0", 
		  title = "Income Prediction API", 
		  description = "Predict income based on various attributes")

namespace = app_x.namespace('income', description="""
		The classification task is to determine whether a person makes over 50K a year
		""")

list_of_names = {}

@namespace.route("/")
class IncomeClass(Resource):

	@app_x.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error' }, 
			 params={ 'age': 'continuous' 
			 , 'workclass': 'State-gov, Self-emp-not-inc, Private, Federal-gov, Local-gov, Self-emp-inc'
			 , 'education-num': 'continuous'
			 , 'marital-status': 'Never-married, Married-civ-spouse, Divorced, Married-spouse-absent, Separated, Married-AF-spouse, Widowed' 
			 , 'occupation': 'Adm-clerical, Exec-managerial, Handlers-cleaners, Prof-specialty, Other-service, Sales, Transport-moving, Farming-fishing, Machine-op-inspct, Tech-support, Craft-repair, Protective-serv, Armed-Forces, Priv-house-serv'
			 , 'relationship': 'Not-in-family, Husband, Wife, Own-child, Unmarried, Other-relative'
			 , 'race': 'White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other'
			 , 'sex': 'Male, Female' 
			 , 'capital-gain': 'continuous'
			 , 'capital-loss': 'continuous'
			 , 'hours-per-week': 'continuous'
			 , 'native-country': """United-States, Cuba, Jamaica, India, Mexico, Puerto-Rico,
				Honduras, England, Canada, Germany, Iran, Philippines, Poland,
				Columbia, Cambodia, Thailand, Ecuador, Laos, Taiwan, Haiti,
				Portugal, Dominican-Republic, El-Salvador, France, Guatemala,
				Italy, China, South, Japan, Yugoslavia, Peru,
				Outlying-US(Guam-USVI-etc), Scotland, Trinadad&Tobago, Greece,
				Nicaragua, Vietnam, Hong, Ireland, Hungary, Holand-Netherlands""" })

	def get(self):
		names = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
		numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
		categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

		age = float(request.args.get('age', default = '39'))
		workclass = request.args.get('workclass', default = 'State-gov')
		education_num = float(request.args.get('education-num', default = '13.0'))
		marital_status = request.args.get('marital-status', default = 'Never-married')
		occupation = request.args.get('occupation', default = 'Adm-clerical')
		relationship = request.args.get('relationship', default = 'Not-in-family')
		race = request.args.get('race', default = 'White')
		sex = request.args.get('sex', default = 'Male')
		capital_gain = float(request.args.get('capital-gain', default = '2174.0'))
		capital_loss = float(request.args.get('capital-loss', default = '0.0'))
		hours_per_week = float(request.args.get('hours-per-week', default = '40.0'))
		native_country = request.args.get('native-country', default = 'United-States')

		input_list = [age, workclass, education_num, marital_status, 
				occupation, relationship, race, sex, capital_gain, 
				capital_loss, hours_per_week, native_country]

		df = pd.DataFrame([input_list], columns = names)

		enc = np.load('model/one_hot_encoder.npy', allow_pickle = True).item()
		array_dummies = enc.transform(df[categorical_cols]).toarray()
		X = np.concatenate((df[numerical_cols].values, array_dummies), axis = 1)

		clf = joblib.load('model/gradient_boosting_clf')

		y = int(clf.predict(X)[0])
		target_dict = {0:'<=50K', 1:'>50K'}
		income_prediction = {'income' : target_dict[y]}


		try:
			return {
				"status": "Income prediction",
				"Income" : income_prediction
			}
		except KeyError as e:
			growscore_space.abort(500, e.__doc__, status = "Could not retrieve information", statusCode = "500")
		except Exception as e:
			growscore_space.abort(400, e.__doc__, status = "Could not retrieve information", statusCode = "400")

if __name__=='__main__':
   app.run(host='127.0.0.1', port=8080, debug=True)