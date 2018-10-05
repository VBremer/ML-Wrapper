# wrapper for machine learning models
# adjusted example inspired by "https://machinelearningmastery.com/evaluate-machine-learning-algorithms-for-human-activity-recognition/" 
# import 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

### Specify task and folds for k-fold cross validation and create toy data
# specifiy task (either 'classification' or 'regression'
task = 'regression'
# set k for k-fold cross validation
k = 5
###

# create toy data
if task == 'classification':
	X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=1)
	X, y = pd.DataFrame(X),pd.DataFrame(y)
if task == 'regression':
	X, y = make_regression(n_samples=100, n_features=3, noise=0.3)
	X, y = pd.DataFrame(X),pd.DataFrame(y)
if task != 'regression' and task != 'classification':
	print('Please specify a task according to line 19.')

# plot data
sns.set(style="darkgrid", palette="muted")
fig, axs = plt.subplots(ncols=1, nrows=2)
axs[0].plot(X.iloc[:,0], color="blue", label="Feature 1", linestyle="-")
axs[0].plot(X.iloc[:,1], color="red", label="Feature 2", linestyle="-")
axs[0].plot(X.iloc[:,2], color="orange", label="Feature 3", linestyle="-")
axs[1].plot(y, color="black", label="Dependent variable", linestyle=":")
axs[0].set_title("Plot of independent variables")
axs[1].set_title("Plot of dependent variable")
axs[0].legend()
axs[1].legend()
plt.show()

# create folds for k-fold cross validation
index = []
k_fold = KFold(n_splits=k, random_state=0)
for train_indices, test_indices in k_fold.split(X):    
	index.append((train_indices, test_indices))

### define functions

def createModels(task, models=dict()):
	"""models to evaluate."""
	if task != 'regression' and task != 'classification':
		print('Please specify a task according to line 19.')
	if task == 'regression':
		models['knn'] = KNeighborsRegressor(n_neighbors=7)
		models['cart'] = DecisionTreeRegressor()
		models['svm'] = SVR()
		models['rf'] = RandomForestRegressor(n_estimators=100)
		models['gbm'] = GradientBoostingRegressor(n_estimators=100)
	if task == 'classification':
		models['knn'] = KNeighborsClassifier(n_neighbors=7)
		models['cart'] = DecisionTreeClassifier()
		models['svm'] = SVC()
		models['rf'] = RandomForestClassifier(n_estimators=100)
		models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models

def evalModel(X, y, model, task):
	"""evaluate a single model with k-fold cross validation."""
	if task != 'regression' and task != 'classification':
		print('Please specify a task according to line 19.')
	performance = []
	for i in range(len(index)):
		X_train = X.iloc[list(index[i][0])]
		X_test = X.iloc[list(index[i][1])]
		y_train = y.iloc[list(index[i][0])]
		y_test = y.iloc[list(index[i][1])]
		# fit model
		model.fit(X_train, y_train.values.ravel())
		# predict
		yPred = model.predict(X_test)
		# evaluate predictions
		if task == 'regression':
			performance.append(mean_absolute_error(y_test, yPred))
		if task == 'classification':
			performance.append(accuracy_score(y_test, yPred) * 100.0)
	perf = np.mean(performance)
	return perf
	
def fitModel(X, y, model):
	"""fit model on all data."""
	fit = model.fit(X, y.values.ravel())
	return fit

def evalModels(X, y, models):
	"""evaluate a dict of models, fit, and prepare results."""
	performance = dict()
	fit = dict()
	results = dict()
	for name, model in models.items():
		# evaluate the model and fit model to all data
		performance[name] = evalModel(X, y, model, task)
		fit[name] = fitModel(X, y, model)
	results['pred_performance'] = pd.DataFrame(list(performance.items()), columns=['Model','Performance'])
	results['model_fit'] = fit
	return results

### execute analysis
		
# get models
models = createModels(task)
# evaluate models
res = evalModels(X, y, models)
# get results (performance is accuracy for classification and MAE for regression)
res['pred_performance']
res['model_fit']
