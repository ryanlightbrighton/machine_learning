from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np

data = pd.read_csv("iris.data")

x = data.values[:, 0:3]  # this is our inputs (columns 0-3) the "x"
y = data.values[:, 4]  # this is our output (column 4) the "y"
folds = 10
training_test_split = 1 / folds
_x_train, _x_test, _y_train, _y_test = train_test_split(x, y, test_size=0.2)  # test_size ??? random_state ???
# setup cross validation
_cross_validation = ShuffleSplit(n_splits=folds, test_size=training_test_split, random_state=0)

print("CV: ", _cross_validation)

_classifier = KNeighborsClassifier(n_neighbors=3)

# for _ in range(10):
_classifier = _classifier.fit(_x_train, _y_train)

_y_prediction = _classifier.predict(_x_test)

# get accuracy
_acc = accuracy_score(_y_test, _y_prediction)
print("Accuracy: ", _acc)

# get precision
_prec = precision_score(_y_test, _y_prediction, average='macro')
print("Precision: ", _prec)

grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11],  # odd so there can be no tie (random choice would be made if weights is set to
    # uniform)
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

gs = GridSearchCV(
    KNeighborsClassifier(),
    grid_params,
    verbose=1,
    cv=_cross_validation,
    n_jobs=-1
)
# for _ in range(10):
gs_results = gs.fit(_x_train, _y_train)

print("gs_results.best_score_: ", gs_results.best_score_)
print("gs_results.best_estimator_: ", gs_results.best_estimator_)
print("gs_results.best_params_: ", gs_results.best_params_)