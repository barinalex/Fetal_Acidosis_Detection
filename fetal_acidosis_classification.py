# -*- coding: utf-8 -*-
"""
    Dummy template for classification
    B(E)3M33UI - Support script for the first semester task

    Jiri Spilka, 2019
"""
"""
    Not so dummy classifier

    Aleksandr Barinov 2021
"""

import numpy as np

from sklearn import metrics
from sklearn import linear_model, metrics, svm, preprocessing, model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.metrics import make_scorer, plot_confusion_matrix, balanced_accuracy_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt
import plotting
import utils
import copy

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, SMOTENC, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.pipeline import Pipeline as imbPipe
from imblearn.combine import SMOTEENN

CSV_CTU = "Features_CTU_stat_spectral_figo_mfhurst_20190329.csv"
CSV_LYON = "NOT_AVAILABLE.csv"

our_scorer = make_scorer(utils.g_mean_score, greater_is_better=True)
imbalanced_scorer = make_scorer(utils.g_mean_score_imbalanced, greater_is_better=True)
balanced_score = make_scorer(balanced_accuracy_score, greater_is_better=True)

classifier = "lr"
gridsearch = False
balance = None
ratio = 1

def balance_dataset(X, y, approach='SMOTE', ratio=1):
    X_resampled = X
    y_resampled = y
    if approach == 'RAND-over':
        oversample = RandomOverSampler(sampling_strategy=ratio)
        X_resampled, y_resampled = oversample.fit_resample(X, y)
    elif approach == 'RAND-under':
        oversample = RandomUnderSampler(sampling_strategy=ratio)
        X_resampled, y_resampled = oversample.fit_resample(X, y)
    elif approach == 'SMOTE':
        oversample = SMOTE(sampling_strategy=ratio)
        X_resampled, y_resampled = oversample.fit_resample(X, y)
    elif approach == 'SVMSMOTE':
        oversample = SVMSMOTE(sampling_strategy=ratio)
        X_resampled, y_resampled = oversample.fit_resample(X, y)
    elif approach == 'ClusterCentroids':
        undersample = ClusterCentroids(sampling_strategy=ratio)
        X_resampled, y_resampled = undersample.fit_resample(X, y)
    elif approach == 'under-over':
        undersample = ClusterCentroids(sampling_strategy=ratio)
        oversample = SMOTE()
        pipe = imbPipe(steps=[("u",undersample), ("o",oversample)])
        X_resampled, y_resampled = pipe.fit_resample(X, y)
    elif approach == 'over-under':
        oversample = SMOTE(sampling_strategy=ratio)
        undersample = ClusterCentroids()
        pipe = imbPipe(steps=[("o",oversample), ("u",undersample)])
        X_resampled, y_resampled = pipe.fit_resample(X, y)
    return X_resampled, y_resampled


def cross_validation_score(X, y, folds=5):
    kf = KFold(n_splits=folds)
    kf.get_n_splits(X)
    trn_scores = []
    tst_scores = []
    X = np.array(X)
    y = np.array(y)
    cm = np.zeros((2,2), dtype=int)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = train_model(X_train, y_train)
        y_tr_pred = predict(clf, X_train)
        y_tst_pred = predict(clf, X_test)
        trn_scores.append(utils.g_mean_score(y_train, y_tr_pred))
        tst_scores.append(utils.g_mean_score(y_test, y_tst_pred))
        cm += confusion_matrix(y_test, y_tst_pred)
    print("\ncv mean train score:",np.mean(trn_scores))
    print("cv mean test score:",np.mean(tst_scores),"\n")
    return trn_scores, tst_scores, cm


def train_model(X, y):
    """
    Return a trained model.

    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert "X" in locals().keys()
    assert "y" in locals().keys()
    assert len(locals().keys()) == 2

    X = np.array(X)
    y = np.array(y)

    if balance is not None:
        X, y = balance_dataset(X, y, balance, ratio)

    if classifier == "rf":
        clf = RandomForestClassifier(n_estimators=8, max_depth=6) 
        parameters = {'clf__n_estimators':range(4,12,2), 'clf__max_depth':range(2,8,2)}
    elif classifier == "lr":
        clf = LogisticRegression(class_weight="balanced", max_iter=200, solver='lbfgs', C=1)
        parameters = {'clf__solver':['lbfgs', 'liblinear'], 'clf__C':[0.1,0.3,0.5,0.7]}
    elif classifier == "bc":
        clf = BaggingClassifier(base_estimator = LogisticRegression(class_weight="balanced", max_iter=200, solver='lbfgs', C=1),
                                    n_estimators=6)
        parameters = {'clf__n_estimators':range(2,8,2)}

    vt = VarianceThreshold(threshold=(.9 * (1 - .9)))
    sc = StandardScaler()
    pipe = Pipeline(steps=[("sc", sc), 
                           ("vt", vt), 
                           ("clf", clf)
                           ])

    if gridsearch and not parameters is None:
        clf = GridSearchCV(pipe, parameters, scoring=our_scorer)
        clf.fit(X, y)
        print(clf.best_params_)
        print(clf.best_score_)
        pipe.set_params(**clf.best_params_)

    pipe.fit(X, y)
    return pipe


def predict(model1, X):
    """
    Produce predictions for X using given filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert len(locals().keys()) == 2
    return model1.predict(X)
    '''
    try:
        probs = model1.predict_proba(X) 
        threshold = 0.15
        preds = [(1 if p[1] >= threshold else 0) for p in probs]
        return preds 
    except:
        return model1.predict(X)
    '''


def train_on_provided_data_test_on_external_data():
    """The model be tested using this script.
    External dataset will be used as a test data
    """

    select_stage = 0  # CAN BE CHANGED
    nr_seg = 1  # CAN BE CHANGED

    # load training (provided data)
    df = utils.load_data_binary(CSV_CTU)
    df = utils.load_data_stage_last_k_segments(df, select_stage=select_stage, nr_seg=nr_seg)
    df = utils.df_drop_features(df)
    # preprocessing (if necessary)
    x_train, y_train, _ = utils.get_X_y_from_dataframe(df)

    df = utils.load_data_binary(CSV_LYON)
    df = utils.load_data_stage_last_k_segments(df, select_stage=select_stage, nr_seg=nr_seg)
    df = utils.df_drop_features(df)
    # preprocessing (if necessary)
    x_test, y_test, _ = utils.get_X_y_from_dataframe(df)

    print("\nTraining data CTU")
    print(f"y == 0: {sum(y_train == 0)}")
    print(f"y == 1: {sum(y_train == 1)}")

    print("\nTest data LYON ")
    print(f"y == 0: {sum(y_test == 0)}")
    print(f"y == 1: {sum(y_test == 1)}")

    # Train the model
    filter1 = train_model(x_train, y_train)

    # Compute predictions for training data and report g-mean
    # Ideally replace this with cross-validation g-mean, i.e. run CV on the CTU data
    y_tr_pred = predict(filter1, x_train)
    print("\ng-mean on training data: ", utils.g_mean_score(y_train, y_tr_pred))

    # Compute predictions for testing data and report our g-mean
    y_tst_pred = predict(filter1, x_test)
    print("g-mean on test data: ", utils.g_mean_score(y_test, y_tst_pred))


def train_test_on_provided_data():
    """Demonstration of model training and testing using the provided data
    You can do whatever you want with the provided data.
    The most important things:
        1 - your results should be reproducible
        2 - small change in a training data should not lead to large change in results
    """
    select_stage = 0
    nr_seg = 1

    df = utils.load_data_binary(CSV_CTU)
    df = utils.load_data_stage_last_k_segments(df, select_stage=select_stage, nr_seg=nr_seg)
    df = utils.df_drop_features(df)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    X, y, _ = utils.get_X_y_from_dataframe(df)

    # custom train/test just for a purpose of demonstrations. Use CV!!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("\nTraining data")
    print(f"y == 0: {sum(y_train == 0)}")
    print(f"y == 1: {sum(y_train == 1)}")

    filter1 = train_model(X_train, y_train)

    if not gridsearch:
        _,_,cm = cross_validation_score(X, y)
        cm_dist = ConfusionMatrixDisplay(cm)
        cm_dist.plot(colorbar=False)
        plt.show()

    # Compute predictions for training data and report our g-mean
    y_tr_pred = predict(filter1, X_train)
    print("\ng-mean on training data: ", utils.g_mean_score(y_train, y_tr_pred))


    # Compute predictions for test data and report our g-mean
    y_tst_pred = predict(filter1, X_test)
    print("g-mean on test data: ", utils.g_mean_score(y_test, y_tst_pred))

    print("\nTesting data")
    print(f"y == 0: {sum(y_test == 0)}")
    print(f"y == 1: {sum(y_test == 1)}")

if __name__ == "__main__":
    print("RUN LOGISTIC REGRESSION")
    #gridsearch = True
    classifier = "lr"
    train_test_on_provided_data()


    print("\nRUN RANDOM FORESTS")
    #gridsearch = True
    balance = 'RAND-under'
    ratio=1
    classifier = "rf"
    train_test_on_provided_data()

    print("\nRUN BAGGING CLASSIFIER")
    #gridsearch = True
    balance = 'SMOTE'
    ratio=1
    classifier = "bc"
    train_test_on_provided_data()
    #train_on_provided_data_test_on_external_data()

