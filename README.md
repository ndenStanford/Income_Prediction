# Income Prediction

In this assignment you will be evaluating UCI’s Census Income dataset. The classification task is to determine whether a person makes over 50K a year. Please examine the prediction using a minimum of 2 machine learning model endpoints and return a brief analysis of the best result. The feature space is as follows, and contains a mixture of numerical and categorical inputs. 

## API Dashboard 
Please see the swagger UI here

https://income-prediction-involve-ai.herokuapp.com/

Example API 

https://income-prediction-involve-ai.herokuapp.com/income/?age=35&workclass=Self-emp-not-inc&education=Some-college&marital-status=Married-spouse-absent&occupation=Armed-Forces&relationship=Own-child&race=Amer-Indian-Eskimo&sex=Female&capital-gain=0&capital-loss=0&hours-per-week=40&native-country=Jamaica

## Dataset 

- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Tasks
- Use any two ML models to predict whether the income will be >50k or <=50k for each adult in the test dataset. Please include all the procedures and code as well as the results.
  - During the data preprocessing stage of the workflow, please include at least one data visualization of your feature space. 

- Please set up an endpoint API for predicting the result from your best ML model. Your endpoint should take the input data in the test dataset (single row of data or a batch) and return the predictions. Please include your code and steps to use them.


