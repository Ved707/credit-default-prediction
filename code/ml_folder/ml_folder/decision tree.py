import sys

import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN


train_df = pd.read_csv(r'/home/vedant/Project/ml/data/train_data.csv',index_col=0)
X_train=train_df.drop('credit_card_default',axis=1)
Y_train=train_df['credit_card_default']

smote = ADASYN(n_neighbors=5)
X_samp, Y_samp = smote.fit_resample(X_train, Y_train)


test_df = pd.read_csv(r'/home/vedant/Project/ml/data/test_data.csv',index_col=0)
X_test=test_df.drop('credit_card_default',axis=1)
Y_test=test_df['credit_card_default']

import os
import warnings


from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss, precision_score, recall_score

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)

scoring = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']


min_samples_split = (sys.argv[0]) if len(sys.argv) > 1 else 3
max_depth = (sys.argv[1]) if len(sys.argv) > 1 else 7


with mlflow.start_run():

    tree = DecisionTreeClassifier(min_samples_split=min_samples_split,max_depth=max_depth, random_state=2022)

    # scores = cross_validate(tree, X_samp, Y_samp, scoring=scoring,cv=kfold)

    tree.fit(X_samp,Y_samp)
    Y_pred_proba=tree.predict_proba(X_test)
    Y_pred=tree.predict(X_test)


    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)

    mlflow.log_metric("roc_auc", roc_auc_score(Y_test,Y_pred_proba[:,1]))
    mlflow.log_metric("accuracy", accuracy_score(Y_test,Y_pred))
    mlflow.log_metric("f1", f1_score(Y_test,Y_pred))
    mlflow.log_metric("precision", precision_score(Y_test,Y_pred))
    mlflow.log_metric("recall", recall_score(Y_test,Y_pred))


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":

        mlflow.sklearn.log_model(tree, "model", registered_model_name="Decision tree")
    else:
        mlflow.sklearn.log_model(tree, "model")


