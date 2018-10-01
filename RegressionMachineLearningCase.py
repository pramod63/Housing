# Load Libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
filename = ("Boston.csv")
names = ["crim", "zn", "indus", "chas",
         "nox", "rm", "age", "dis", "rad",
         "tax", "ptratio", "black", "lstat", "medv"]
dataset = read_csv(filename, names=names)

# Shape
print(dataset.shape)

# Types
print(dataset.dtypes)

# Head
print(dataset.head(20))

# Description
set_option('precision', 1)
print(dataset.describe())

# Correlation
set_option('precision', 2)
print(dataset.corr(method='pearson'))

# Data visualization
# Histogram
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()

# Density
dataset.plot(kind='density', subplots=True, layout=(4, 4), sharex=False,
             legend=False, fontsize=1)
pyplot.show()

# Box and whisker plot
dataset.plot(kind='box', subplots=True, layout=(4,4), 
             sharex=False, sharey=False, fontsize=8)
pyplot.show()

# Multimodal data visualization
# Scatter_plot_matrix
scatter_matrix(dataset)
pyplot.show()

# Correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0, 14, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()


# Split out validation dataset
array = dataset.values
X = array[:, 0:13]
Y = array[:, 13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
                                test_size=validation_size, random_state=seed)

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# Spot - Check Algorithm
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,
                                 scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Comapare algorithm
fig = pyplot.figure()
fig.suptitle('ALgorithm Comaparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Standardize the data
pipelines = []
pipelines.append(('scaledLR',
                  Pipeline([('Scaler', StandardScaler()),
                            ('LR', LinearRegression())])))
pipelines.append(('scaledLASSO',
                  Pipeline([('Scaler', StandardScaler()),
                            ('LASSO', Lasso())])))
pipelines.append(('scaledEN',
                  Pipeline([('Scaler', StandardScaler()),
                            ('EN', ElasticNet())])))
pipelines.append(('scaledKNN',
                  Pipeline([('Scaler', StandardScaler()),
                            ('KNN', KNeighborsRegressor())])))
pipelines.append(('scaledCART',
                  Pipeline([('Scaler', StandardScaler()),
                            ('CART', DecisionTreeRegressor())])))
pipelines.append(('scaledSVR',
                  Pipeline([('Scaler', StandardScaler()), ('SVRT', SVR())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train,
                                 Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s : %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare algotithm
fig = pyplot.figure()
fig.suptitle("Scale Algorithm Comparison")
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# KNN Algorithm Tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_,
                             grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f, (%f) with : %r" % (mean, stdev, param))

# Ensemble

ensembles = []
ensembles.append(('ScalesAB', Pipeline([('Scaler',
                                         StandardScaler()), ('AB', AdaBoostRegressor())])))
ensembles.append(('ScalesGBM', Pipeline([('Scaler',
                                          StandardScaler()), ('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScalesRF', Pipeline([('Scaler',
                                         StandardScaler()), ('RF', RandomForestRegressor())])))
ensembles.append(('ScalesET', Pipeline([('Scaler',
                                         StandardScaler()), ('ET', ExtraTreesRegressor())])))

results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train,
                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithm
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Tune ensemble methods
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50, 100,
                                            150, 200, 250, 300, 350, 400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best : %f using %s " % (grid_result.best_score_,
                               grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with : %r" % (mean, stdev, param))

# Prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)

# tranform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))
