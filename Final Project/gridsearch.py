import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

wells = pd.read_excel('UT Completion and Sequencing.xlsx')
counter = 0
for value in wells['Compl. Type']:
    if type(value) != float and value[0] in '1234567890':
        wells.loc[counter, 'Compl. Type'] = 'Mixed'
    counter += 1

wells['Compl. Type'].replace(['P & P (cmt.)', 'P & P (After sleeves mill out)', 'P & P (After milling out sleeves)'], 'P & P', inplace=True)
wells['Compl. Type'].replace('Sleeves and P & P', 'Mixed', inplace=True)
wells['Compl. Type'].replace('Sleeves (cmt.)', 'Sleeves', inplace=True)
wells['Compl. Type'].replace(['Cemented Liner', 'Screen', 'Perforated Liner'], 'Liner', inplace=True)
wells['Compl. Type'].replace(['CT', 'Coil Tubing Frac'], 'Coiled Tubing', inplace=True)
wells['Compl. Type'].replace(['unknown, probably hybrid', 'Not indicated', 'Frac Ports', 'Frac ports'], 'No Data', inplace=True)

wells['Formation'].replace('TF1', 'TFH', inplace=True)
wells['Formation'].replace(['UTFH', 'MTFH', 'TFSH', 'TF4', 'TF2.5'], 'Other', inplace=True)
formation = pd.get_dummies(wells['Formation'], prefix = 'FORM', dtype = np.float64)
wells = pd.concat([wells, formation], axis=1)
wells.drop(['Formation'], axis=1, inplace=True)
completions_type = pd.get_dummies(wells['Compl. Type'], prefix = 'COMP', dtype = np.float64)
wells = pd.concat([wells, completions_type], axis=1)
wells.drop(['Compl. Type'], axis=1, inplace=True)
wells.drop(['Date Fracd','Operator', 'Well Name', 'Township ', 'Range', 'Section', 'Best1 Mo BOPD', 'Best3 Mo BOPD', 'Best6 Mo BOPD', 'Best9 Mo BOPD', 'Best12 Mo BOPD', 'Fluid Type from DI'], axis=1, inplace=True)
wells.dropna(inplace = True)
wells = wells[wells['Stages'] != 1.0]
wells = wells[(wells.loc[:,'Lateral Length':'12 month Cum Prod'] != 0).all(1)]

scaler = MinMaxScaler((0, 1))
model = MLPRegressor(random_state = 42, max_iter = 100, verbose=True)
pipeline = Pipeline(steps = [('scaler', scaler), ('model', model)])

params = {'model__activation': ['identity', 'logistic', 'relu','tanh'],
          'model__solver':['lbfgs','sgd', 'adam'], 'model__alpha':np.arange(0.0001, 0.1, 0.005)}

GSCV = GridSearchCV(pipeline, params, cv = 3)
features = wells.drop('12 month Cum Prod', axis = 1, inplace = False)
label = wells['12 month Cum Prod']
featuresScaled = scaler.fit_transform(features)
labelScaled = scaler.fit_transform(label.values.reshape(-1, 1))
GSCV.fit(featuresScaled, labelScaled)
best_params = GSCV.best_params_
print(best_params)

#x_train, x_test, y_train, y_test = train_test_split(featuresScaled, labelScaled, train_size = .9, random_state = 42)
#model.fit(x_train, y_train)
#y_pred = model.predict(x_test)



