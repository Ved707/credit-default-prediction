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
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier,RandomForestClassifier

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
    np.random.seed(2022)


no_of_estimators = (sys.argv[0]) if len(sys.argv) > 1 else 100
learning_rate = (sys.argv[1]) if len(sys.argv) > 1 else 0.5
max_depth= (sys.argv[2]) if len(sys.argv) > 1 else 7

with mlflow.start_run():
    log_reg = LogisticRegression(solver='sag', random_state=2022)
    knn = KNeighborsClassifier(n_neighbors=3)
    tree = DecisionTreeClassifier(min_samples_split=3,max_depth=7, random_state=2022)
    forest = RandomForestClassifier(max_features=10, max_depth=10, random_state=2022)

    gb = GradientBoostingClassifier(learning_rate=0.5,max_depth=7 ,random_state=2022)

    models_considered = [('Logistic Regression', log_reg),
                         ('KneighboursClassifier', knn),('Decision Tree', tree),('Random forest',forest),('gradient boost',gb)]

    voting = VotingClassifier(estimators=models_considered, voting='soft')


    voting.fit(X_samp, Y_samp)
    Y_pred_proba = voting.predict_proba(X_test)
    Y_pred = voting.predict(X_test)


    mlflow.log_metric("roc_auc", roc_auc_score(Y_test, Y_pred_proba[:, 1]))
    mlflow.log_metric("accuracy", accuracy_score(Y_test, Y_pred))
    mlflow.log_metric("f1", f1_score(Y_test, Y_pred))
    mlflow.log_metric("precision", precision_score(Y_test, Y_pred))
    mlflow.log_metric("recall", recall_score(Y_test, Y_pred))

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":

        mlflow.sklearn.log_model(gb, "model", registered_model_name="voting")
    else:
        mlflow.sklearn.log_model(gb, "model")


