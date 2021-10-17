import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib



input_list = [39, 'State-gov', 13.0, 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male', 2174.0, 0.0, 40.0, 'United-States']

names = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

df = pd.DataFrame([input_list], columns = names)

enc = np.load('one_hot_encoder.npy', allow_pickle = True).item()
array_dummies = enc.transform(df[categorical_cols]).toarray()
X = np.concatenate((df[numerical_cols].values, array_dummies), axis = 1)

clf = joblib.load('random_forest_clf')

y = clf.predict(X)
print(y)