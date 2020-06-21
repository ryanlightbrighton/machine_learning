from warnings import simplefilter
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
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import EnsembleVoteClassifier
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
import nltk

print('NLTK: {}'.format(nltk.__version__))

simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv("iris.data")


def get_data_values():
    x = data.values[:, 0:3]  # this is our inputs (columns 0-3) the "attributes of the flowers"
    y = data.values[:, 4]  # this is our output (column 4) the "name of the flower"
    return x, y


def get_split_values():
    folds = 10
    tts = 1 / folds
    return folds, tts


# note: check that the test_size value should be equal to the tts value calculated above
def get_training_test_split():
    x, y = get_data_values()
    folds, tts = get_split_values()
    _x_train, _x_test, _y_train, _y_test = train_test_split(
        x,
        y,
        test_size=tts
    )
    return _x_train, _x_test, _y_train, _y_test


def get_cross_validation():
    folds, tts = get_split_values()
    cv = ShuffleSplit(
        n_splits=folds,
        test_size=tts,
        random_state=0
    )
    return cv


def get_score(clf):
    x, y = get_data_values()
    _score = cross_val_score(clf, x, y, cv=get_cross_validation())
    return _score.mean()


def get_acc(clf):
    x, y = get_data_values()
    _score = cross_val_score(clf, x, y, cv=get_cross_validation(), scoring='accuracy')
    return _score.mean()


#  BROKEN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def get_prec(clf):
    x, y = get_data_values()
    _score = cross_val_score(clf, x, y, cv=get_cross_validation(), scoring='precision')  # f1 racall
    return _score


#  check this function (how to run 5 times)
def fit_clf(clf):
    xtr, xte, ytr, yte = get_training_test_split()
    # for _ in range(5):
    clf = clf.fit(xtr, ytr)
    return clf


def predict_scores(clf):
    xtr, xte, ytr, yte = get_training_test_split()
    _y_prediction = clf.predict(xte)
    _acc = accuracy_score(yte, _y_prediction)
    _prec = precision_score(yte, _y_prediction, average='macro')
    return _acc, _prec


def run_grid_search_cv(clf, grid_params):
    xtr, xte, ytr, yte = get_training_test_split()
    gs = GridSearchCV(
        clf,
        grid_params,
        verbose=1,
        cv=get_cross_validation(),
        n_jobs=-1
    )
    gs_results = gs.fit(xtr, ytr)

    return gs_results.best_score_, gs_results.best_estimator_, gs_results.best_params_


def ensemble_picker(clfrs):
    best_combo = [None, None, None]
    threshold = -1
    x, y = get_data_values()
    args = [None, None, None]  # define len so it doesn't freak out
    for i in range(len(clfrs)):
        args[0] = clfrs[i]
        for j in range(len(clfrs)):
            args[1] = clfrs[j]
            for k in range(len(clfrs)):
                args[2] = clfrs[k]
                print("running another ensemble model")
                ensemble = EnsembleVoteClassifier(clfs=args, weights=[1, 1, 1],voting='soft')
                scores = model_selection.cross_val_score(
                    ensemble,
                    x,
                    y,
                    cv=get_cross_validation(),
                    scoring='accuracy'
                )
                if scores.mean() > threshold:
                    best_combo = args
                    threshold = scores.mean()
    print("-----------------------")
    print("best_combo[0]: ", best_combo[0])
    print("best_combo[1]: ", best_combo[1])
    print("best_combo[2]: ", best_combo[2])
    print("score: ", threshold)




