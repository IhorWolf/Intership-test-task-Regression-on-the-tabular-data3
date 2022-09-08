#Import libraries

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import sklearn.model_selection
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

#upload train data
train = pd.read_csv(r'C:\Users\igora\Desktop\ТЕСТОВІ ЗАВДАННЯ\Quantum internship\internship_train.csv')

# Preparing to modeling

valid_part = 0.3
target_name = 'target'
# For boosting model
train_b = train
train_target_b = train_b[target_name]
train_b = train_b.drop([target_name], axis=1)
# Synthesis valid as test for selection models
trainb, testb, targetb, target_testb = train_test_split(train_b, train_target_b, test_size=valid_part, random_state=0)

# For linear/tree model 
train_target = train[target_name]
train = train.drop([target_name], axis=1)
#standardization of models
scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns = train.columns)

# Synthesis valid as test for selection models
train, test, target, target_test = train_test_split(train, train_target, test_size=valid_part, random_state=0)

# collect scores
acc_train_r2 = []
acc_test_r2 = []
acc_train_rmse = []
acc_test_rmse = []

# RMSE between predicted y_pred and measured y_meas values
def acc_rmse(y_meas, y_pred):
    return (mean_squared_error(y_meas, y_pred))**0.5

#Function for boosting model
def acc_boosting_model(num,model,train,test,num_iteration=0):
    global acc_train_r2, acc_test_r2, acc_train_rmse, acc_test_rmse
    
    if num_iteration > 0:
        ytrain = model.predict(train, num_iteration = num_iteration)  
        ytest = model.predict(test, num_iteration = num_iteration)
    else:
        ytrain = model.predict(train)  
        ytest = model.predict(test)

    print('target = ', targetb[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(targetb, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_rmse_num = round(acc_rmse(targetb, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_testb[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_testb, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_rmse_num = round(acc_rmse(target_testb, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)
    
#Function for other model linear/tree
def acc_model(num,model,train,test): 
    global acc_train_r2, acc_test_r2, acc_train_rmse, acc_test_rmse
    
    ytrain = model.predict(train)  
    ytest = model.predict(test)

    print('target = ', target[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_rmse_num = round(acc_rmse(target, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_test[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_rmse_num = round(acc_rmse(target_test, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)
    
# Linear Regression
linreg = LinearRegression()
linreg.fit(train, target)
acc_model(0,linreg,train,test)

# Decision Tree Regression
decision_tree = DecisionTreeRegressor()
decision_tree.fit(train, target)
acc_model(1,decision_tree,train,test)

# LightGBM
#split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(trainb, targetb, test_size=0.2, random_state=0)
train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': False,
        'seed':0,        
    }
modelL = lgb.train(params, train_set = train_set, num_boost_round=10000,
                   early_stopping_rounds=2000,verbose_eval=500, valid_sets=valid_set)
acc_boosting_model(2,modelL,trainb,testb,modelL.best_iteration)

#Create dataframe with resul for decide which is the best
models = pd.DataFrame({
    'Model': ['Linear Regression',
              'Decision Tree Regressor','LGBM'
             ],
    
    'r2_train': acc_train_r2,
    'r2_test': acc_test_r2,
    'rmse_train': acc_train_rmse,
    'rmse_test': acc_test_rmse
                     })
pd.options.display.float_format = '{:,.2f}'.format
print('Prediction accuracy for models by R2 criterion - r2_test')
models.sort_values(by=['r2_test', 'r2_train'], ascending=False)

#Upload test data
test = pd.read_csv(r'C:\Users\igora\Desktop\ТЕСТОВІ ЗАВДАННЯ\Quantum internship\internship_hidden_test.csv')
# Predict result in test data
y_pred = decision_tree.predict(test)
test['target'] = y_pred
test.to_csv('intership_result.csv', index=False)
