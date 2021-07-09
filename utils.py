"""
    Utilities for data load and selection

    B(E)3M33UI - Support script for the first semester task

    Jiri Spilka, 2019
"""
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PH_THR = 7.05


def load_data_binary(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["y"] = (df.pH <= 7.05).astype(int).ravel()
    return df


def load_data_stage_last_k_segments(df: pd.DataFrame, select_stage: int = 0, nr_seg: int = 1) -> pd.DataFrame:
    """Load k last segments from data

    :param df: pandas dataframe
    :param select_stage: 0 - all, 1 - first stage, 2 - second stage
    :param nr_seg: number of last segments to load
    :return:
    """

    if select_stage == 0:
        return df.loc[df.segIndex <= nr_seg, :]

    elif select_stage == 1:
        ind = np.logical_and(df.segStageI_index > 0, df.segStageI_index <= nr_seg)
        return df.loc[ind, :]

    elif select_stage == 2:
        ind = np.logical_and(df.segStageII_index > 0, df.segStageII_index <= nr_seg)
        return df.loc[ind, :]

    else:
        raise Exception(f"Unknown value select_stage={select_stage}")


def df_drop_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(
        columns=[
            "name",
            "pH",
            "year",
            "segStart_samp",
            "segEnd_samp",
            "segIndex",
            "segStageI_index",
            "segStageII_index",
        ]
    )

    # the stage information might be useful
    df = df.drop(columns=["segStage"])

    # other features that are probably not very useful (correlated to the other ones or irrelevant)
    df = df.drop(
        columns=[
            "bslnMean",
            "bslnSD",
            "decDeltaMedian",
            "decDeltaMad",
            "decDtrdPlus",
            "decDtrdMinus",
            "decDtrdMedian",
            "bslnAllBeta0",
            "bslnAllBeta1",
            "MF_hmin_noint",
            "H310",
            "MF_c1",
            "MF_c2",
            "MF_c3",
            "MF_c4",
        ]
    )

    return df


def get_X_y_from_dataframe(df: pd.DataFrame) -> Tuple[np.array, np.array, List[str]]:
    """Get feature matrix and labels"""
    y = df.y
    df = df.drop(columns=["y"])
    return df.values, y, list(df)


def g_mean(estimator, X, y):
    y_pred = estimator.predict(X)
    return g_mean_score(y, y_pred)


def g_mean_score(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""

    cm = confusion_matrix(y, y_pred)
    #print(cm)
    if cm.shape != (2, 2):
        raise ValueError("The ground truth values and the predictions may contain at most 2 values (classes).")

    tn = cm[0, 0]
    fn = cm[1, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]

    denominator = tp + fn
    se = 0
    if denominator > 0:
        se = tp / float(tp + fn)

    denominator = tn + fp
    sp = 0
    if denominator > 0:
        sp = tn / float(tn + fp)
    # metrics.
    g = np.sqrt(se * sp)
    return g

def g_mean_score_imbalanced(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""

    cm = confusion_matrix(y, y_pred)
    if cm.shape != (2, 2):
        raise ValueError("The ground truth values and the predictions may contain at most 2 values (classes).")

    tn = cm[0, 0]
    fn = cm[1, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]

    denominator = tp + fn
    se = 0
    if denominator > 0:
        se = tp / float(tp + 5*fn)

    denominator = tn + fp
    sp = 0
    if denominator > 0:
        sp = tn / float(tn + fp)
    # metrics.
    g = np.sqrt(se * sp)

    return g




def get_all_metrics_as_dict(dataset: str, y_true: np.array, y_pred: np.array):

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    se = tp / float(tp + fn)
    sp = tn / float(tn + fp)

    g = np.sqrt(se * sp)

    return {"dataset": dataset, "G-mean": g, "SE": se, "SP": sp, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def print_results(_df: pd.DataFrame):
    print(_df[["dataset", "G-mean", "SE", "SP", "TP", "FN", "TN", "FP"]])
