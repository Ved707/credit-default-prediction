import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN

train_df = pd.read_csv(r'/home/vedant/Project/ml/data/train_data.csv',index_col=0)
X_train=train_df.drop('credit_card_default',axis=1)
Y_train=train_df['credit_card_default']

smote = ADASYN(n_neighbors=5)
X_samp, Y_samp = smote.fit_resample(X_train, Y_train)


import os
import warnings
import sys

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
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

c = (sys.argv[0]) if len(sys.argv) > 1 else 1.0

with mlflow.start_run():
    svm=SVC(random_state=2022,C=c,kernel='linear',probability=True)

    scores = cross_validate(svm, X_samp, Y_samp, scoring=scoring)
    print(sorted(scores.keys()))

    print("  roc_auc: %s" % scores['test_roc_auc'].mean())
    print("  accuracy: %s" % scores['test_accuracy'].mean())
    print("  f1: %s" % scores['test_f1'].mean())
    print("  precision: %s" % scores['test_precision'].mean())
    print("  recall: %s" % scores['test_recall'].mean())

    mlflow.log_param("solver", c)
    mlflow.log_metric("roc_auc", scores['test_roc_auc'].mean())
    mlflow.log_metric("accuracy", scores['test_accuracy'].mean())
    mlflow.log_metric("f1", scores['test_f1'].mean())
    mlflow.log_metric("precision", scores['test_precision'].mean())
    mlflow.log_metric("recall", scores['test_recall'].mean())

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(svm, "model", registered_model_name="SVM")
    else:
        mlflow.sklearn.log_model(svm, "model")


