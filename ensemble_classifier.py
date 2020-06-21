from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import source


grid_params_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
    'n_jobs': [-1]
}

score_knn, clf_knn, _ = source.run_grid_search_cv(KNeighborsClassifier(), grid_params_knn)

grid_params_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
    'splitter': ['random', 'best']
}

score_dt, clf_dt, _ = source.run_grid_search_cv(DecisionTreeClassifier(), grid_params_dt)

grid_params_rf = {
    'criterion': ['gini', 'entropy'],
    # 'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
    'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'n_estimators': [200, 700],
    'n_jobs': [-1],
    'max_features': ['sqrt', 'log2'],
    'n_estimators': [100, 200]
    # 'warm_start': [True, False]
}

score_rf, clf_rf, _ = source.run_grid_search_cv(RandomForestClassifier(), grid_params_rf)

print("KNN.best_score_: ", score_knn)
print("DT.best_score_: ", score_dt)
print("RF.best_score_: ", score_rf)

#  NOTE:    MUST FIT THE DATA BEFORE GETTING THE BEST PARAMS!!!! (done in the run_grid_search_cv function)

#  NOTE:   https://elitedatascience.com/overfitting-in-machine-learning  OVERFITTING!

source.ensemble_picker(
    [
        clf_knn,
        clf_dt,
        clf_rf
    ]
)

#  I don't think I need to fit them here because it is done already
#  in the run_grid_search_cv() function

#source.ensemble_picker(
#    [
#        source.fit_clf(clf_knn),
#        source.fit_clf(clf_dt),
#        source.fit_clf(clf_rf)
#    ]
#)