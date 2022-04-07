#!/usr/bin/env python
# coding: utf-8

# In[156]:


#Importing necessary Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from sklearn.tree import DecisionTreeRegressor

#use this for one vacancy
#data = pd.read_csv("/Users/adityabora/Downloads/PythonGenOneVac.csv")
#data = pd.read_csv("/Users/adityabora/Downloads/One_vac_216.csv")
#data1 = pd.read_csv("/Users/adityabora/Downloads/One_vacABCDE.csv")
#data1 = pd.read_csv("/Users/adityabora/Downloads/NWOrdTarun.csv")
# data = pd.read_csv("/Users/adityabora/Downloads/One_vacUpdated.csv")
# data1 = pd.read_csv("/Users/adityabora/Downloads/One_vacABCDE.csv")

#use this for two vacs
#data = pd.read_csv("/Users/adityabora/Downloads/Two_vac_216.csv")
#data = pd.read_csv("/Users/adityabora/Downloads/Two_vac_227.csv")



#data1 = pd.read_csv("/Users/adityabora/Downloads/Two_vac_results_216.csv")

#use this for three vacancies
data = pd.read_csv("/Users/adityabora/Downloads/Three_vac_2-16Data.csv")
data1 = pd.read_csv("/Users/adityabora/Downloads/Three_vac_results_2-16Data2123.csv")


#use this for one vacancy
#one_hot_encoded_data = pd.get_dummies(data)
#one_hot_test = pd.get_dummies(data1)
#features = one_hot_encoded_data.drop(['sub1_element', 'vac1_element', 'sub2_distance', 'sub1_location_x', 'sub1_location_z', 'num_substitution', 'sub3_location_x',  'sub3_location_y', 'vacancy3_location_y', 'sub3_location_z', 'vac2_distance', 'sub3_element','mo_vacancies','band_gap',  'vacancy1_location_x',  'num_vacancies', 'vacancy2_location_y', 'sub1_distance', 'vacancy2_location_z', 'vacancy2_location_x', 'vacancy3_location_z',  'sub2_location_x','sub2_location_y', 'vac3_distance', 'vacancy3_location_x', 's_vacancies'], axis = 1)
#featuresABCDE = one_hot_test.drop(['sub1_element', 'vac1_element', 'sub2_distance', 'sub1_location_x','sub1_location_z', 'num_substitution', 'sub3_location_x','sub3_location_y','vacancy3_location_y', 'sub3_location_z', 'vac2_distance', 'sub3_element','mo_vacancies', 'vacancy1_location_x',  'num_vacancies', 'vacancy2_location_y', 'sub1_distance', 'vacancy2_location_z', 'vacancy2_location_x', 'vacancy3_location_z', 'sub2_location_x','sub2_location_y', 'vac3_distance', 'vacancy3_location_x', 's_vacancies'], axis = 1)
# one_hot_encoded_data = pd.get_dummies(data, columns = ['sub1_element','sub2_element','vac1_element', 'vac2_element','vac3_element']) #use this setting for 3vacs
# one_hot_test = pd.get_dummies(data1, columns = ['sub1_element','sub2_element','vac1_element', 'vac2_element','vac3_element'])
# features = one_hot_encoded_data.drop(['num_substitution', 'sub2_location_z', 'sub1_location_z', 'sub1_location_x', 'sub1_location_y', 'sub2_distance', 'sub1_element_Se', 'sub3_location_x', 'sub2_element_None', 'sub3_location_y', 'vacancy3_location_y', 'sub3_location_z', 'vac2_distance', 'sub1_element_W', 'sub2_element_Se', 'vac1_element_Mo', 'vac2_element_None', 'sub3_element','mo_vacancies','band_gap','vacancy1_location_z', 'sub1_element_None', 'num_vacancies', 'vacancy2_location_y', 'sub1_distance', 'vacancy2_location_z', 'vacancy2_location_x', 'vacancy3_location_z', 'vac3_element_None', 'sub2_location_x','sub2_location_y', 'vac3_distance', 'vacancy3_location_x', 's_vacancies'], axis = 1)
# featuresABCDE = one_hot_test.drop(['num_substitution', 'sub2_location_z', 'sub1_location_z', 'sub1_location_x', 'sub1_location_y', 'sub2_distance', 'sub1_element_Se', 'sub3_location_x','sub2_element_None','sub3_location_y','vacancy3_location_y', 'sub3_location_z', 'vac2_distance', 'sub1_element_W','sub2_element_Se', 'vac1_element_Mo', 'vac2_element_None', 'sub3_element','mo_vacancies','vacancy1_location_z', 'sub1_element_None',  'num_vacancies', 'vacancy2_location_y', 'sub1_distance', 'vacancy2_location_z', 'vacancy2_location_x', 'vacancy3_location_z', 'vac3_element_None', 'sub2_location_x','sub2_location_y', 'vac3_distance', 'vacancy3_location_x', 's_vacancies'], axis = 1)





#use this for two vacancies
# one_hot_encoded_data = pd.get_dummies(data, columns = ['sub1_element', 'vac1_element', 'vac2_element','vac3_element', 'sub2_element', 'sub3_element']) #use this setting for twoVacs
# one_hot_test = pd.get_dummies(data1)
# #features = one_hot_encoded_data[['sub1_element_W','vac_distance_between', 'vac_distance_between_xy', 'vac1_element_Mo']]
# featuresABCDE = one_hot_test[['vac_distance_between', 'vac_distance_between_xy', 'vac1_element_Mo' ]]
# features = one_hot_encoded_data[['vac_distance_between', 'vac_distance_between_xy', 'vac1_element_Mo']]



#use this for three vacancies
one_hot_encoded_data = pd.get_dummies(data)
one_hot_test = pd.get_dummies(data1)
features = one_hot_encoded_data[['mo_s2_xydistance', 'mo_s1_xydistance', 's1_s2_xydistance', 'mo_s2_yzdistance', 'mo_xy_angle', 's1_s2_xzdistance']]
featuresABCDE = one_hot_test[['mo_s2_xydistance', 'mo_s1_xydistance', 's1_s2_xydistance', 'mo_s2_yzdistance', 'mo_xy_angle', 's1_s2_xzdistance']]


# In[159]:


labels = one_hot_encoded_data['band_gap']
feature_list = list(features.columns)
labels = np.array(labels)
features = np.array(features)
featuresTest = np.array(featuresABCDE)
#df = pd.DataFrame(featuresTest)
#df


# In[160]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X_train,X_test,Y_train,Y_test = train_test_split(features,labels,test_size=0.05,random_state=5)
#X_test,X_val,Y_test,Y_val = train_test_split(X_tst,Y_tst,test_size=0.5,random_state=0)


# In[161]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# train_scaled = scaler.fit_transform(X_train)
# test_scaled = scaler.transform(X_test)

#scaleTestData = scaler.transform(featuresTest)

#when running 3 vac make sure to scale the data 
train_scaled = X_train
test_scaled = X_test
bs = featuresTest


from sklearn.model_selection import GridSearchCV

model = DecisionTreeRegressor()

gs = GridSearchCV(model,
                  param_grid = {'max_depth': range(1, 10),
                                'min_samples_split': range(10, 60, 10)},
                  cv=5,
                  n_jobs=1,
                  scoring='neg_mean_squared_error')

gs.fit(train_scaled, Y_train)

print(gs.best_params_)
print(-gs.best_score_)

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(batch_size=5, learning_rate_init=0.0001, solver = 'adam', max_iter = 3000)
model.fit(train_scaled, Y_train)


# In[162]:


#import libraries
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
#Fit the model
clf = LassoCV().fit(train_scaled, Y_train)
#Selected features
importance = np.abs(clf.coef_)
idx_third = importance.argsort()[-3]
threshold = importance[idx_third] + 0.01
idx_features = (-importance).argsort()[:10]
name_features = np.array(feature_list)[idx_features]
print('Selected features: {}'.format(name_features))


# In[163]:


from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

max_depths = range(1, 10)
training_error = []
for max_depth in max_depths:
    model_1 = DecisionTreeRegressor(max_depth=max_depth)
    model_1.fit(X_train, Y_train)
    training_error.append(mse(Y_train, model_1.predict(X_train)))
    
testing_error = []
for max_depth in max_depths:
    model_2 = DecisionTreeRegressor(max_depth=max_depth)
    model_2.fit(X_train, Y_train)
    testing_error.append(mse(Y_test, model_2.predict(X_test)))
    

plt.plot(max_depths, training_error, color='blue', label='Training error')
plt.plot(max_depths, testing_error, color='green', label='Testing error')
plt.legend()
plt.ylabel('Mean squared error')
plt.xlabel('Tree depth')


# In[164]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
tree_model = DecisionTreeRegressor()
rf_model = RandomForestRegressor(max_depth = 5)

tree_model.fit(train_scaled, Y_train)
rf_model.fit(train_scaled, Y_train)

df = pd.DataFrame(train_scaled)


# In[165]:


from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

std_slc = StandardScaler()
pca = decomposition.PCA()
dtreeReg = tree.DecisionTreeRegressor()
pipe = Pipeline(steps=[("std_slc", std_slc),
                           ("pca", pca),
                           ("dtreeReg", dtreeReg)])
n_components = list(range(1,12))
criterion = ["friedman_mse", "mse"]
max_depth = [4,6,8,10]
parameters = dict(pca__n_components=n_components,
                      dtreeReg__criterion=criterion,
                      dtreeReg__max_depth=max_depth)
clf = GridSearchCV(pipe, parameters)
clf.fit(train_scaled, Y_train)
fullPreds = clf.predict(test_scaled)

print("Best Number Of Components:", clf.best_estimator_.get_params()["pca__n_components"])
print(); print(clf.best_estimator_.get_params()["dtreeReg"])
CV_Result = cross_val_score(clf, train_scaled, Y_train, cv=3, n_jobs=-1, scoring="r2")
print(); print(CV_Result)
print(); print(CV_Result.mean())
print(); print(CV_Result.std())


# In[166]:


#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from math import sqrt
import matplotlib.pyplot as plt


# In[167]:


rmse_val = [] #to store rmse values for different k
for K in range(50):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, Y_train)  #fit the model
    kpred = model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(Y_test,kpred)) #calculate rmse
    absError = mean_absolute_error(Y_test,kpred)
    rmse_val.append(error) #store rmse values
    #print('RMSE value for k= ' , K , 'is:', error)
    print('AME value for k= ' , K , 'is:', absError)
    #plotting the rmse values against k values
    
kmodel = neighbors.KNeighborsRegressor(n_neighbors = 3)
kmodel.fit(X_train, Y_train)
kpred = kmodel.predict(X_test)
#finalKTwoTest = kmodel.predict(featuresTest)
curve = pd.DataFrame(rmse_val) #elbow curve    

curve.plot()


# In[168]:


# from tensorflow import keras
# from tensorflow.keras import layers
# inputs = keras.Input(shape=(9,))
# dense = layers.Dense(1, activation="relu")
# x = dense(inputs)
# x = layers.Dense(1, activation="relu")(x)
# outputs = layers.Dense(2)(x)
# dPmodel = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
# dPmodel.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.RMSprop(),
#     metrics=["accuracy"],
# )
# history = dPmodel.fit(train_scaled, Y_train, batch_size=64, epochs=2, validation_split=0.2)
# test_scores = dPmodel.evaluate(test_scaled, Y_test, verbose=2)


# In[169]:


#Initialising ANN
threeVac = tf.keras.models.Sequential()


# In[170]:


#Adding First Hidden Layer
threeVac.add(tf.keras.layers.Dense(units=24,activation="relu"))


# In[171]:


#Adding Second Hidden Layer
threeVac.add(tf.keras.layers.Dense(units=12,activation="relu"))


# In[172]:


#Adding Output Layer
threeVac.add(tf.keras.layers.Dense(units=1,activation="relu"))


# In[173]:


threeVac.compile(optimizer="adam",loss="mse")


# In[174]:


#Fitting ANN
threeVac.fit(X_train,Y_train,batch_size=25,epochs = 1);


# In[175]:


# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler()
# poly = PolynomialFeatures(degree=2, include_bias=False)
# poly_features = poly.fit_transform(train_scaled.reshape(-1, 1))
# degree=1
# polyreg_scaled=make_pipeline(PolynomialFeatures(degree),scaler,LinearRegression())
# polyreg_scaled.fit(poly_features,Y_train)


# In[273]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#use this for 1 vac
# params = {
#     "n_estimators": 1000,
#     "max_depth": 2,
#     "min_samples_split": 10,
#     "learning_rate": 0.01,
#     "random_state": 100,
#     "criterion": 'mae'
# }

# #use this for 2 vac
# params = {
#     "n_estimators": 1000,
#     "max_depth": 3,
#     "min_samples_split": 10,
#     "learning_rate": 0.7,
#     "random_state": 100,
#     "criterion": 'mse',
#     "subsample": .5
# }

# use this for 3 vac
params = {
    "n_estimators": 1000,
    "max_depth": 2,
    "min_samples_split": 20,
    "learning_rate": 0.03,
    "random_state": 100,
    "criterion": 'mae'
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(train_scaled, Y_train)
hope = reg.predict(X_test)
finalOne2 = reg.predict(bs)
pd.DataFrame(finalOne2).to_csv('threeHopefulImprove.csv')


# In[264]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV

data_tuples = []
for row in feature_list:
    data_tuples.append(row)


ridge = RidgeCV(alphas=np.logspace(-20, 20, num=5)).fit(train_scaled, Y_train)
newTester = ridge.predict(test_scaled)
importance = np.abs(ridge.coef_)



plt.bar(height=importance, x=data_tuples)
plt.xticks(rotation=90)
plt.title("Feature importances via coefficients")
plt.show()


# In[265]:


mse = mean_squared_error(Y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(Y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(feature_list)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    reg, test_scaled, Y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(feature_list)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()


# In[266]:


from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
estimators = [
     ('lr', RidgeCV()),
   ('svr', LinearSVR(random_state=42))
]
stackreg = StackingRegressor(
    estimators=estimators,
   final_estimator=RandomForestRegressor(n_estimators=30,
                                          random_state=42)
)
stackreg.fit(train_scaled, Y_train)
stackpredict = stackreg.predict(test_scaled)


# In[267]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=10, n_restarts_optimizer = 2).fit(train_scaled, Y_train)
gpr.score(train_scaled, Y_train)
gprPredict = gpr.predict(test_scaled)
gprTest = gpr.predict(bs)


# In[268]:


from lightgbm import LGBMRegressor
from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std


lgmmodel = LGBMRegressor()
cv = RepeatedKFold(n_splits=100, n_repeats=10, random_state=1)
n_scores = cross_val_score(lgmmodel, train_scaled, Y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
lgmmodel = LGBMRegressor(learning_rate=0.01)
lgmmodel.fit(train_scaled, Y_train)


# In[269]:


y_pred = model.predict(test_scaled)

y_pred1 = tree_model.predict(test_scaled)


y_pred2 = rf_model.predict(test_scaled)


y_pred3 = threeVac.predict(test_scaled)
y_pred5 = lgmmodel.predict(test_scaled)


y_pred6 = regr.predict(test_scaled)


#finalOne = tree_model.predict(scaleTestData)
#pd.DataFrame(finalOne).to_csv('NewTwoVacModelData.csv')


# In[270]:


import operator
def ewt(size, observed, predictions):


    preds = np.array([])
    for k in predictions:
        preds = np.append(preds, k)

    x=np.subtract(observed, preds)

    count=0
    for i in x:
        if i > -0.02 and i < 0.02:
            count=count+1
            
            
    return count/size


# In[271]:


ewt(Y_test.shape[0], Y_test, gprPredict)


# In[272]:


ewt(Y_test.shape[0], Y_test, hope )


# In[241]:


ewt(Y_test.shape[0], Y_test, newTester)


# In[242]:


ewt(Y_test.shape[0], Y_test, fullPreds )


# In[190]:


ewt(Y_test.shape[0], Y_test, y_pred5 )


# In[191]:


ewt(Y_test.shape[0], Y_test, kpred )


# In[192]:


ewt(Y_test.shape[0], Y_test, y_pred )
#pd.DataFrame(ann.predict(A).tolist()).to_csv('output1.csv')


# In[193]:


importances = list(tree_model.feature_importances_)
feature_importances = [(feature, importance) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[194]:


ewt(Y_test.shape[0], Y_test, y_pred1 ) 
df = pd.DataFrame(finalKTwoTest)
pd.DataFrame(finalKTwoTest).to_csv('TwoNewData.csv')


# In[ ]:


importances = list(rf_model.feature_importances_)
feature_importances = [(feature, importance) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[ ]:





# In[ ]:




